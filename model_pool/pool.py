"""Async lifecycle manager for local model handles (load / deactivate / lease)."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional

from .errors import CapacityExceeded, NotActive, Superseded

LoadFn = Callable[[str], Awaitable[Any]]
CloseFn = Callable[[Any], Awaitable[None]]


class _Generation:
    """One activation generation of a model."""

    __slots__ = (
        "model_id",
        "path",
        "handle",
        "load_done",
        "load_error",
        "deactivated",
        "leases",
        "close_started",
        "waiters",
    )

    def __init__(self, model_id: str, path: str) -> None:
        self.model_id = model_id
        self.path = path
        self.handle: Any = None
        self.load_done = False
        self.load_error: Optional[BaseException] = None
        self.deactivated = False
        self.leases = 0
        self.close_started = False
        self.waiters: list[asyncio.Future] = []

    @property
    def active(self) -> bool:
        return self.load_done and self.load_error is None and not self.deactivated


class ModelPool:
    """Manages activation generations of local models.

    All state mutations happen synchronously between awaits, so no lock is
    ever held across ``load``, ``close`` or lease exits; different models
    never serialize each other.
    """

    def __init__(
        self,
        paths: Dict[str, str],
        load: LoadFn,
        close: CloseFn,
        capacity: int = 5,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._paths = dict(paths)
        self._load = load
        self._close = close
        self._capacity = capacity
        self._in_use = 0
        self._current: Dict[str, _Generation] = {}
        self._tasks: set[asyncio.Task] = set()

    @property
    def slots_in_use(self) -> int:
        """Resource slots currently held (loading, active, leased, closing)."""
        return self._in_use

    async def activate(self, model_id: str) -> None:
        path = self._paths[model_id]
        gen = self._current.get(model_id)
        if gen is not None and gen.active:
            return
        if gen is None:
            if self._in_use >= self._capacity:
                raise CapacityExceeded(model_id)
            self._in_use += 1
            gen = _Generation(model_id, path)
            self._current[model_id] = gen
            self._spawn(self._run_load(gen))
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        gen.waiters.append(fut)
        try:
            await fut
        finally:
            # Cancelling this waiter must not affect the shared load or the
            # activation intent; just detach from the waiter list.
            try:
                gen.waiters.remove(fut)
            except ValueError:
                pass

    async def deactivate(self, model_id: str) -> None:
        gen = self._current.pop(model_id, None)
        if gen is None:
            return
        gen.deactivated = True
        waiters, gen.waiters = gen.waiters, []
        for fut in waiters:
            if not fut.done():
                fut.set_exception(Superseded(model_id))
        self._maybe_close(gen)

    def lease(self, model_id: str) -> "_Lease":
        return _Lease(self, model_id)

    async def wait_idle(self) -> None:
        """Wait until all in-flight loads/closes settle (test/diagnostic aid)."""
        while True:
            pending = [t for t in self._tasks if not t.done()]
            if not pending:
                return
            await asyncio.gather(*pending)

    # -- internals ---------------------------------------------------------

    def _spawn(self, coro: Awaitable[None]) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _release_slot(self) -> None:
        self._in_use -= 1

    def _maybe_close(self, gen: _Generation) -> None:
        if (
            gen.deactivated
            and gen.load_done
            and gen.load_error is None
            and gen.leases == 0
            and not gen.close_started
        ):
            gen.close_started = True
            self._spawn(self._run_close(gen))

    async def _run_load(self, gen: _Generation) -> None:
        try:
            handle = await self._load(gen.path)
        except Exception as exc:
            gen.load_done = True
            gen.load_error = exc
            # A failed load leaves no resources behind: release the slot.
            self._release_slot()
            if not gen.deactivated:
                if self._current.get(gen.model_id) is gen:
                    del self._current[gen.model_id]
                waiters, gen.waiters = gen.waiters, []
                for fut in waiters:
                    if not fut.done():
                        fut.set_exception(exc)
            # A deactivated generation's late failure touches nothing.
            return
        gen.load_done = True
        gen.handle = handle
        if gen.deactivated:
            # Late success on a revoked generation: clean up, never publish.
            self._maybe_close(gen)
        else:
            waiters, gen.waiters = gen.waiters, []
            for fut in waiters:
                if not fut.done():
                    fut.set_result(None)

    async def _run_close(self, gen: _Generation) -> None:
        try:
            await self._close(gen.handle)
        finally:
            self._release_slot()


class _Lease:
    def __init__(self, pool: ModelPool, model_id: str) -> None:
        self._pool = pool
        self._model_id = model_id
        self._gen: Optional[_Generation] = None

    async def __aenter__(self) -> Any:
        # Synchronous check-and-increment: atomic with respect to the loop.
        gen = self._pool._current.get(self._model_id)
        if gen is None or not gen.active:
            raise NotActive(self._model_id)
        gen.leases += 1
        self._gen = gen
        return gen.handle

    async def __aexit__(self, *exc_info: object) -> bool:
        gen = self._gen
        if gen is not None:
            self._gen = None
            gen.leases -= 1
            self._pool._maybe_close(gen)
        return False
