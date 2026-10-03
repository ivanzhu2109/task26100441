"""Async lifecycle manager for local model handles.

Only loading, deactivation and leasing are handled here. No inference,
persistence or crash recovery. All coordination is per-generation; no
global lock is ever held across an await of load/close/lease-exit.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable, Dict, Optional

from .errors import CapacityExceeded, NotActive, Superseded


class _Generation:
    """One activation generation of a single model.

    A generation owns at most one handle. It occupies exactly one capacity
    slot from just before its load starts until its load fails or its
    handle's close completes.
    """

    __slots__ = ("load_task", "handle", "superseded", "leases",
                 "close_started", "waiters")

    def __init__(self) -> None:
        self.load_task: Optional[asyncio.Task] = None
        self.handle: Any = None
        self.superseded: bool = False
        self.leases: int = 0
        self.close_started: bool = False
        self.waiters: list[asyncio.Future] = []


class ModelPool:
    """Manages activate/deactivate/lease lifecycles for local models."""

    def __init__(
        self,
        paths: Dict[str, str],
        load: Callable[[str], Awaitable[Any]],
        close: Callable[[Any], Awaitable[None]],
        capacity: int = 5,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._paths = dict(paths)
        self._load = load
        self._close = close
        self._capacity = capacity
        self._slots_used = 0
        self._current: Dict[str, _Generation] = {}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    async def activate(self, model_id: str) -> None:
        """Wait until the model is activated. Does not return the handle."""
        if model_id not in self._paths:
            raise KeyError(model_id)
        gen = self._current.get(model_id)
        if gen is not None and gen.handle is not None and not gen.superseded:
            return
        if gen is None or gen.superseded:
            gen = self._start_generation(model_id)
        # Share the in-flight load: each caller gets its own waiter future
        # so cancelling one waiter never cancels the shared load task.
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future = loop.create_future()
        gen.waiters.append(waiter)
        try:
            await waiter
        finally:
            if waiter in gen.waiters:
                gen.waiters.remove(waiter)

    def deactivate(self, model_id: str) -> None:
        """Immediately revoke the current activation intent.

        Never waits for loading, leases or closing. Idempotent.
        """
        gen = self._current.get(model_id)
        if gen is None or gen.superseded:
            return
        gen.superseded = True
        del self._current[model_id]
        self._notify_waiters(gen, exc=Superseded(
            f"activation of {model_id!r} was superseded"))
        self._maybe_close(gen)

    @asynccontextmanager
    async def lease(self, model_id: str):
        """Borrow the currently activated handle; fails fast otherwise."""
        gen = self._current.get(model_id)
        if gen is None or gen.superseded or gen.handle is None:
            raise NotActive(f"model {model_id!r} is not activated")
        gen.leases += 1
        try:
            yield gen.handle
        finally:
            gen.leases -= 1
            self._maybe_close(gen)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _start_generation(self, model_id: str) -> _Generation:
        if self._slots_used >= self._capacity:
            raise CapacityExceeded(
                f"no free slot to load {model_id!r} "
                f"({self._slots_used}/{self._capacity} in use)")
        self._slots_used += 1  # reserved before the load starts
        gen = _Generation()
        self._current[model_id] = gen
        gen.load_task = asyncio.ensure_future(self._run_load(model_id, gen))
        return gen

    async def _run_load(self, model_id: str, gen: _Generation) -> None:
        try:
            handle = await self._load(self._paths[model_id])
        except asyncio.CancelledError:
            # We never cancel load tasks ourselves; treat defensively as a
            # failure so the slot is not leaked.
            self._on_load_failure(model_id, gen,
                                  asyncio.CancelledError())
            return
        except BaseException as exc:
            self._on_load_failure(model_id, gen, exc)
            return
        self._on_load_success(model_id, gen, handle)

    def _on_load_success(self, model_id: str, gen: _Generation,
                         handle: Any) -> None:
        if gen.superseded or self._current.get(model_id) is not gen:
            # Late handle from a revoked generation: it may only be cleaned
            # up, never published. The slot is freed when this close done.
            gen.handle = handle
            gen.superseded = True
            self._maybe_close(gen)
            return
        gen.handle = handle
        self._notify_waiters(gen, result=None)

    def _on_load_failure(self, model_id: str, gen: _Generation,
                         exc: BaseException) -> None:
        # A failed load frees its slot immediately, whatever its generation.
        self._slots_used -= 1
        if not gen.superseded and self._current.get(model_id) is gen:
            del self._current[model_id]
            self._notify_waiters(gen, exc=exc)
        # A superseded generation's failure must not touch the new one.

    def _maybe_close(self, gen: _Generation) -> None:
        if (gen.superseded and gen.handle is not None
                and gen.leases == 0 and not gen.close_started):
            gen.close_started = True
            asyncio.ensure_future(self._run_close(gen, gen.handle))

    async def _run_close(self, gen: _Generation, handle: Any) -> None:
        try:
            await self._close(handle)  # close is guaranteed to succeed
        finally:
            self._slots_used -= 1

    @staticmethod
    def _notify_waiters(gen: _Generation, result: Any = None,
                        exc: Optional[BaseException] = None) -> None:
        waiters, gen.waiters = gen.waiters, []
        for waiter in waiters:
            if waiter.done():
                continue
            if exc is not None:
                waiter.set_exception(exc)
            else:
                waiter.set_result(result)
