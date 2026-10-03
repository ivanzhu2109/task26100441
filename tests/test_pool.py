"""Tests for model_pool.ModelPool.

The backend uses real temporary files as handles and asyncio.Event gates so
every critical interleaving is controlled deterministically (no sleeps).
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import os
import tempfile
import unittest

from model_pool import CapacityExceeded, ModelPool, NotActive, Superseded


class Handle:
    """A real readable handle backed by a temp file's contents."""

    def __init__(self, seq: int, path: str, data: bytes) -> None:
        self.seq = seq
        self.path = path
        self._data = data

    def read(self) -> bytes:
        return self._data


class Backend:
    """Controllable load/close backend recording all activity."""

    def __init__(self) -> None:
        self._seq = itertools.count()
        self.load_calls: list[str] = []
        self.close_calls: list[Handle] = []
        self.closed: list[Handle] = []
        self.occupancy = 0
        self.max_occupancy = 0
        self._load_gates: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._close_gates: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._load_errors: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._cond = asyncio.Condition()

    # -- scripting helpers (called by tests before the fact) ---------------

    def gate_next_load(self, path: str) -> asyncio.Event:
        gate = asyncio.Event()
        self._load_gates[path].append(gate)
        return gate

    def gate_next_close(self, path: str) -> asyncio.Event:
        gate = asyncio.Event()
        self._close_gates[path].append(gate)
        return gate

    def fail_next_load(self, path: str, exc: Exception) -> None:
        self._load_errors[path].append(exc)

    def load_count(self, path: str) -> int:
        return self.load_calls.count(path)

    async def wait_for(self, pred) -> None:
        async with self._cond:
            await self._cond.wait_for(pred)

    # -- pool-facing coroutines --------------------------------------------

    async def load(self, path: str) -> Handle:
        self.load_calls.append(path)
        self.occupancy += 1
        self.max_occupancy = max(self.max_occupancy, self.occupancy)
        async with self._cond:
            self._cond.notify_all()
        gates = self._load_gates[path]
        if gates:
            await gates.popleft().wait()
        errors = self._load_errors[path]
        if errors:
            self.occupancy -= 1
            async with self._cond:
                self._cond.notify_all()
            raise errors.popleft()
        with open(path, "rb") as fh:
            data = fh.read()
        return Handle(next(self._seq), path, data)

    async def close(self, handle: Handle) -> None:
        self.close_calls.append(handle)
        async with self._cond:
            self._cond.notify_all()
        gates = self._close_gates[handle.path]
        if gates:
            await gates.popleft().wait()
        self.closed.append(handle)
        self.occupancy -= 1
        async with self._cond:
            self._cond.notify_all()


async def spin_until(pred, tries: int = 10000) -> None:
    """Yield to the event loop until ``pred`` holds (deterministic FIFO)."""
    for _ in range(tries):
        if pred():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition not reached")


class PoolTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.backend = Backend()
        self.paths: dict[str, str] = {}

    def make_pool(self, model_ids, capacity: int = 5) -> ModelPool:
        for mid in model_ids:
            if mid not in self.paths:
                path = os.path.join(self._tmp.name, mid + ".bin")
                with open(path, "wb") as fh:
                    fh.write(("weights:" + mid).encode())
                self.paths[mid] = path
        return ModelPool(
            dict(self.paths), self.backend.load, self.backend.close, capacity=capacity
        )

    def rewrite(self, model_id: str, content: str) -> None:
        with open(self.paths[model_id], "wb") as fh:
            fh.write(content.encode())

    def waiter_count(self, pool: ModelPool, model_id: str) -> int:
        gen = pool._current.get(model_id)
        return len(gen.waiters) if gen is not None else 0


class ActivationTests(PoolTestCase):
    async def test_concurrent_activate_shares_single_load(self):
        pool = self.make_pool(["a"])
        gate = self.backend.gate_next_load(self.paths["a"])
        tasks = [asyncio.create_task(pool.activate("a")) for _ in range(3)]
        await self.backend.wait_for(lambda: self.backend.load_count(self.paths["a"]) == 1)
        await spin_until(lambda: self.waiter_count(pool, "a") == 3)
        gate.set()
        await asyncio.gather(*tasks)
        self.assertEqual(self.backend.load_count(self.paths["a"]), 1)
        async with pool.lease("a") as handle:
            self.assertEqual(handle.read(), b"weights:a")
        # Already active: activate returns immediately, no new load.
        await pool.activate("a")
        self.assertEqual(self.backend.load_count(self.paths["a"]), 1)

    async def test_cancel_some_waiters_keeps_shared_load(self):
        pool = self.make_pool(["a"])
        gate = self.backend.gate_next_load(self.paths["a"])
        t1 = asyncio.create_task(pool.activate("a"))
        t2 = asyncio.create_task(pool.activate("a"))
        t3 = asyncio.create_task(pool.activate("a"))
        await spin_until(lambda: self.waiter_count(pool, "a") == 3)
        t1.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await t1
        # The shared load and the activation intent survive the cancellation.
        self.assertIn("a", pool._current)
        gate.set()
        await asyncio.gather(t2, t3)
        async with pool.lease("a") as handle:
            self.assertEqual(handle.read(), b"weights:a")

    async def test_cancel_all_waiters_keeps_loading(self):
        pool = self.make_pool(["a"])
        gate = self.backend.gate_next_load(self.paths["a"])
        tasks = [asyncio.create_task(pool.activate("a")) for _ in range(2)]
        await spin_until(lambda: self.waiter_count(pool, "a") == 2)
        for task in tasks:
            task.cancel()
        for task in tasks:
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIn("a", pool._current)
        gate.set()
        await pool.wait_idle()
        # Load completed despite zero waiters; model is now active.
        await pool.activate("a")
        self.assertEqual(self.backend.load_count(self.paths["a"]), 1)
        async with pool.lease("a") as handle:
            self.assertEqual(handle.read(), b"weights:a")

    async def test_load_failure_propagates_and_allows_retry(self):
        pool = self.make_pool(["a"])
        boom = RuntimeError("boom")
        self.backend.fail_next_load(self.paths["a"], boom)
        results = await asyncio.gather(
            pool.activate("a"), pool.activate("a"), return_exceptions=True
        )
        self.assertEqual(results, [boom, boom])
        self.assertEqual(pool.slots_in_use, 0)
        self.assertEqual(self.backend.occupancy, 0)
        # Retry succeeds after the failure.
        await pool.activate("a")
        async with pool.lease("a") as handle:
            self.assertEqual(handle.read(), b"weights:a")

    async def test_hung_load_does_not_block_other_models(self):
        pool = self.make_pool(["a", "b"])
        gate = self.backend.gate_next_load(self.paths["a"])
        pending = asyncio.create_task(pool.activate("a"))
        await self.backend.wait_for(lambda: self.backend.load_count(self.paths["a"]) == 1)
        # Model b loads and leases fine while a's load hangs.
        await pool.activate("b")
        async with pool.lease("b") as handle:
            self.assertEqual(handle.read(), b"weights:b")
        await pool.deactivate("a")
        with self.assertRaises(Superseded):
            await pending
        gate.set()
        await pool.wait_idle()
        self.assertEqual([h.path for h in self.backend.closed], [self.paths["a"]])


class DeactivateTests(PoolTestCase):
    async def test_deactivate_supersedes_and_is_idempotent(self):
        pool = self.make_pool(["a"])
        gate = self.backend.gate_next_load(self.paths["a"])
        pending = asyncio.create_task(pool.activate("a"))
        await spin_until(lambda: self.waiter_count(pool, "a") == 1)
        await pool.deactivate("a")
        await pool.deactivate("a")  # no side effects
        with self.assertRaises(Superseded):
            await pending
        gate.set()
        await pool.wait_idle()
        self.assertEqual(len(self.backend.closed), 1)
        self.assertEqual(pool.slots_in_use, 0)

    async def test_late_old_load_cannot_replace_new_generation(self):
        pool = self.make_pool(["a"])
        old_gate = self.backend.gate_next_load(self.paths["a"])
        first = asyncio.create_task(pool.activate("a"))
        await self.backend.wait_for(lambda: self.backend.load_count(self.paths["a"]) == 1)
        self.rewrite("a", "weights:v2")
        await pool.deactivate("a")
        with self.assertRaises(Superseded):
            await first
        # New generation loads immediately and becomes current.
        await pool.activate("a")
        async with pool.lease("a") as new_handle:
            self.assertEqual(new_handle.read(), b"weights:v2")
            # Old load finishes late: it is cleaned up, never published.
            old_gate.set()
            await pool.wait_idle()
            self.assertEqual(len(self.backend.closed), 1)
            self.assertIsNot(self.backend.closed[0], new_handle)
        async with pool.lease("a") as again:
            self.assertIs(again, new_handle)

    async def test_lease_survives_deactivate_until_last_release(self):
        pool = self.make_pool(["a"])
        await pool.activate("a")
        lease1 = pool.lease("a")
        old_handle = await lease1.__aenter__()
        lease2 = pool.lease("a")
        await lease2.__aenter__()
        await pool.deactivate("a")
        # Old lease remains fully usable after deactivation.
        self.assertEqual(old_handle.read(), b"weights:a")
        self.assertEqual(self.backend.close_calls, [])
        # A new generation may start while old leases are open.
        self.rewrite("a", "weights:v2")
        await pool.activate("a")
        await lease1.__aexit__(None, None, None)
        self.assertEqual(self.backend.close_calls, [])  # lease2 still open
        await lease2.__aexit__(None, None, None)
        await pool.wait_idle()
        self.assertEqual(self.backend.closed, [old_handle])
        async with pool.lease("a") as new_handle:
            self.assertEqual(new_handle.read(), b"weights:v2")
        await pool.deactivate("a")
        await pool.wait_idle()
        self.assertEqual(len(self.backend.closed), 2)
        self.assertEqual(len({h.seq for h in self.backend.closed}), 2)


class LeaseTests(PoolTestCase):
    async def test_lease_requires_active_model(self):
        pool = self.make_pool(["a"])
        with self.assertRaises(NotActive):
            async with pool.lease("a"):
                pass
        gate = self.backend.gate_next_load(self.paths["a"])
        pending = asyncio.create_task(pool.activate("a"))
        await spin_until(lambda: self.waiter_count(pool, "a") == 1)
        with self.assertRaises(NotActive):  # still loading
            async with pool.lease("a"):
                pass
        gate.set()
        await pending
        await pool.deactivate("a")
        with self.assertRaises(NotActive):  # deactivated
            async with pool.lease("a"):
                pass
        await pool.wait_idle()

    async def test_lease_released_on_body_exception(self):
        pool = self.make_pool(["a"])
        await pool.activate("a")
        with self.assertRaises(ValueError):
            async with pool.lease("a"):
                await pool.deactivate("a")
                raise ValueError("user error")
        await pool.wait_idle()
        self.assertEqual(len(self.backend.closed), 1)
        self.assertEqual(pool.slots_in_use, 0)

    async def test_lease_released_on_cancellation(self):
        pool = self.make_pool(["a"])
        await pool.activate("a")
        entered = asyncio.Event()

        async def user():
            async with pool.lease("a"):
                await pool.deactivate("a")
                entered.set()
                await asyncio.Event().wait()  # hang until cancelled

        task = asyncio.create_task(user())
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        await pool.wait_idle()
        self.assertEqual(len(self.backend.closed), 1)
        self.assertEqual(pool.slots_in_use, 0)

    async def test_concurrent_leases_on_same_model(self):
        pool = self.make_pool(["a"])
        await pool.activate("a")
        async with pool.lease("a") as h1:
            async with pool.lease("a") as h2:
                self.assertIs(h1, h2)


class CapacityTests(PoolTestCase):
    async def test_five_slots_cover_inflight_leased_and_closing(self):
        pool = self.make_pool(list("abcdef"), capacity=5)
        p = self.paths

        # Slot 1: loaded, leased, then deactivated (lease still open).
        await pool.activate("a")
        lease_a = pool.lease("a")
        handle_a = await lease_a.__aenter__()
        await pool.deactivate("a")

        # Slot 2: in-flight load, deactivated while hanging.
        gate_b = self.backend.gate_next_load(p["b"])
        act_b = asyncio.create_task(pool.activate("b"))
        await self.backend.wait_for(lambda: self.backend.load_count(p["b"]) == 1)
        await pool.deactivate("b")
        with self.assertRaises(Superseded):
            await act_b

        # Slot 3: deactivated, close hangs.
        await pool.activate("c")
        gate_c_close = self.backend.gate_next_close(p["c"])
        await pool.deactivate("c")
        await self.backend.wait_for(lambda: len(self.backend.close_calls) == 1)

        # Slot 4: plain active model.
        await pool.activate("d")

        # Slot 5: load in flight, still activated.
        gate_e = self.backend.gate_next_load(p["e"])
        act_e = asyncio.create_task(pool.activate("e"))
        await self.backend.wait_for(lambda: self.backend.load_count(p["e"]) == 1)

        self.assertEqual(pool.slots_in_use, 5)

        # Sixth load must not start; failure is immediate.
        with self.assertRaises(CapacityExceeded):
            await pool.activate("f")
        self.assertEqual(self.backend.load_count(p["f"]), 0)
        # Already-active models need no capacity.
        await pool.activate("d")

        # Free exactly one slot: return a's lease, its close completes.
        await lease_a.__aexit__(None, None, None)
        await spin_until(lambda: pool.slots_in_use == 4)
        self.assertEqual(self.backend.closed, [handle_a])

        # Now a new load may start.
        gate_f = self.backend.gate_next_load(p["f"])
        act_f = asyncio.create_task(pool.activate("f"))
        await self.backend.wait_for(lambda: self.backend.load_count(p["f"]) == 1)
        gate_f.set()
        await act_f

        # Cleanup: release everything still in flight.
        gate_b.set()
        gate_e.set()
        await act_e
        gate_c_close.set()
        await pool.deactivate("d")
        await pool.deactivate("e")
        await pool.deactivate("f")
        await pool.wait_idle()

        # Every successfully loaded handle was closed exactly once.
        self.assertEqual(pool.slots_in_use, 0)
        self.assertEqual(self.backend.occupancy, 0)
        self.assertLessEqual(self.backend.max_occupancy, 5)
        seqs = [h.seq for h in self.backend.closed]
        self.assertEqual(len(seqs), len(set(seqs)))
        self.assertEqual(len(seqs), 6)  # a, b(late), c, d, e, f


if __name__ == "__main__":
    unittest.main()
