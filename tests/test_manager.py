"""Deterministic interleaving tests for modelpool.ModelPool.

No sleeps are used to guess at scheduling: every critical interleaving is
controlled by asyncio.Event gates on load/close calls.
"""

import asyncio
import os
import tempfile
import unittest

from modelpool import CapacityExceeded, ModelPool, NotActive, Superseded

TIMEOUT = 5.0


class LoadCall:
    def __init__(self, path):
        self.path = path
        self.release = asyncio.Event()
        self.error = None
        self.handle = None


class CloseCall:
    def __init__(self, handle):
        self.handle = handle
        self.release = asyncio.Event()
        self.done = asyncio.Event()


class FakeBackend:
    """load/close backed by real temp files, gated by events."""

    def __init__(self, root):
        self.root = root
        self.load_calls = []
        self.close_calls = []
        self._load_queue = asyncio.Queue()
        self._close_queue = asyncio.Queue()
        self.auto_release = False

    def make_file(self, name, content):
        path = os.path.join(self.root, name)
        with open(path, "wb") as fh:
            fh.write(content)
        return path

    async def next_load(self):
        return await asyncio.wait_for(self._load_queue.get(), TIMEOUT)

    async def next_close(self):
        return await asyncio.wait_for(self._close_queue.get(), TIMEOUT)

    async def load(self, path):
        call = LoadCall(path)
        self.load_calls.append(call)
        self._load_queue.put_nowait(call)
        await call.release.wait()
        if call.error is not None:
            raise call.error
        call.handle = open(path, "rb")  # real readable handle
        return call.handle

    async def close(self, handle):
        call = CloseCall(handle)
        self.close_calls.append(call)
        self._close_queue.put_nowait(call)
        if self.auto_release:
            call.release.set()
        await call.release.wait()
        handle.close()
        call.done.set()


class PoolTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.backend = FakeBackend(self._tmp.name)
        self.paths = {
            name: self.backend.make_file(name + ".bin", name.encode())
            for name in ("a", "b", "c", "d", "e", "f")
        }
        self.pool = ModelPool(self.paths, self.backend.load,
                              self.backend.close, capacity=5)

    async def asyncTearDown(self):
        # Revoke everything, open every gate, then drain the loop until
        # all slots are returned (bounded; no interleaving depends on it).
        self.backend.auto_release = True
        for model_id in list(self.pool._current):
            self.pool.deactivate(model_id)
        for call in self.backend.load_calls:
            call.release.set()
        for call in self.backend.close_calls:
            call.release.set()
        for _ in range(1000):
            await asyncio.sleep(0)
            if self.pool._slots_used == 0:
                break
        self.assertEqual(self.pool._slots_used, 0)

    # -- helpers --------------------------------------------------------
    def activate_task(self, model_id):
        return asyncio.ensure_future(self.pool.activate(model_id))

    async def start_load(self, model_id):
        """Begin activating and wait until the backend load starts."""
        task = self.activate_task(model_id)
        call = await self.backend.next_load()
        return task, call

    def assert_no_new_loads(self, n):
        self.assertEqual(len(self.backend.load_calls), n)

    # -- tests ----------------------------------------------------------
    async def test_concurrent_activate_shares_one_load(self):
        tasks = [self.activate_task("a") for _ in range(3)]
        call = await self.backend.next_load()
        call.release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), TIMEOUT)
        self.assert_no_new_loads(1)

    async def test_cancel_some_waiters_keeps_shared_load(self):
        tasks = [self.activate_task("a") for _ in range(3)]
        call = await self.backend.next_load()
        tasks[0].cancel()
        tasks[1].cancel()
        for t in tasks[:2]:
            with self.assertRaises(asyncio.CancelledError):
                await t
        self.assertFalse(call.release.is_set())  # load still in flight
        call.release.set()
        await asyncio.wait_for(tasks[2], TIMEOUT)
        self.assert_no_new_loads(1)

    async def test_cancel_all_waiters_keeps_loading(self):
        tasks = [self.activate_task("a") for _ in range(2)]
        call = await self.backend.next_load()
        for t in tasks:
            t.cancel()
        for t in tasks:
            with self.assertRaises(asyncio.CancelledError):
                await t
        call.release.set()
        # Activation intent survives: a later activate joins the same
        # generation and no second load ever happens.
        await asyncio.wait_for(self.pool.activate("a"), TIMEOUT)
        self.assert_no_new_loads(1)
        async with self.pool.lease("a") as handle:
            self.assertEqual(handle.read(), b"a")

    async def test_load_failure_propagates_and_allows_retry(self):
        task, call = await self.start_load("a")
        call.error = RuntimeError("boom")
        call.release.set()
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(task, TIMEOUT)
        # Slot was freed: retry starts a fresh load and succeeds.
        task2, call2 = await self.start_load("a")
        call2.release.set()
        await asyncio.wait_for(task2, TIMEOUT)
        self.assert_no_new_loads(2)

    async def test_hung_load_does_not_block_other_models(self):
        task_a, call_a = await self.start_load("a")  # never released yet
        task_b, call_b = await self.start_load("b")
        call_b.release.set()
        await asyncio.wait_for(task_b, TIMEOUT)
        async with self.pool.lease("b") as handle:
            self.assertEqual(handle.read(), b"b")
        self.assertFalse(task_a.done())
        # cleanup: revoke a, let its late load finish and be cleaned up
        self.pool.deactivate("a")
        with self.assertRaises(Superseded):
            await asyncio.wait_for(task_a, TIMEOUT)
        call_a.release.set()
        close = await self.backend.next_close()
        self.assertIs(close.handle, call_a.handle)
        close.release.set()
        await asyncio.wait_for(close.done.wait(), TIMEOUT)

    async def test_late_load_cannot_replace_new_generation(self):
        old_task, old_call = await self.start_load("a")
        self.pool.deactivate("a")
        with self.assertRaises(Superseded):
            await asyncio.wait_for(old_task, TIMEOUT)
        # New generation starts while the old load is still in flight.
        new_task, new_call = await self.start_load("a")
        self.assertIsNot(old_call, new_call)
        new_call.release.set()
        await asyncio.wait_for(new_task, TIMEOUT)
        # Old load finishes late: it must be cleaned up, not published.
        old_call.release.set()
        close = await self.backend.next_close()
        self.assertIs(close.handle, old_call.handle)
        close.release.set()
        await asyncio.wait_for(close.done.wait(), TIMEOUT)
        # The current handle is still the new generation's.
        async with self.pool.lease("a") as handle:
            self.assertIs(handle, new_call.handle)
        self.assertEqual(len(self.backend.close_calls), 1)

    async def test_lease_survives_deactivate_until_last_return(self):
        task, call = await self.start_load("a")
        call.release.set()
        await asyncio.wait_for(task, TIMEOUT)

        ctx1 = self.pool.lease("a")
        handle1 = await ctx1.__aenter__()
        ctx2 = self.pool.lease("a")
        handle2 = await ctx2.__aenter__()
        self.assertIs(handle1, handle2)

        self.pool.deactivate("a")
        self.pool.deactivate("a")  # idempotent
        with self.assertRaises(NotActive):
            async with self.pool.lease("a"):
                pass
        # Existing leases keep using the original handle.
        self.assertEqual(handle1.read(), b"a")
        self.assertEqual(len(self.backend.close_calls), 0)
        await ctx1.__aexit__(None, None, None)
        self.assertEqual(len(self.backend.close_calls), 0)
        await ctx2.__aexit__(None, None, None)
        # Last lease returned: exactly one close begins.
        close = await self.backend.next_close()
        self.assertIs(close.handle, call.handle)
        close.release.set()
        await asyncio.wait_for(close.done.wait(), TIMEOUT)
        self.assertEqual(len(self.backend.close_calls), 1)

    async def test_lease_requires_activated_model(self):
        with self.assertRaises(NotActive):
            async with self.pool.lease("a"):
                pass
        task, call = await self.start_load("a")
        with self.assertRaises(NotActive):  # still loading
            async with self.pool.lease("a"):
                pass
        call.release.set()
        await asyncio.wait_for(task, TIMEOUT)

    async def test_capacity_counts_all_held_slots(self):
        # Slot 1: active model with an unreturned lease.
        t1, c1 = await self.start_load("a")
        c1.release.set()
        await asyncio.wait_for(t1, TIMEOUT)
        held = self.pool.lease("a")
        held_handle = await held.__aenter__()

        # Slot 2: deactivated while its load is still in flight.
        t2, c2 = await self.start_load("b")
        self.pool.deactivate("b")
        with self.assertRaises(Superseded):
            await asyncio.wait_for(t2, TIMEOUT)

        # Slot 3: deactivated, close hung inside the backend.
        t3, c3 = await self.start_load("c")
        c3.release.set()
        await asyncio.wait_for(t3, TIMEOUT)
        self.pool.deactivate("c")
        hung_close = await self.backend.next_close()

        # Slot 4: load in flight.
        t4, c4 = await self.start_load("d")
        # Slot 5: plain active model.
        t5, c5 = await self.start_load("e")
        c5.release.set()
        await asyncio.wait_for(t5, TIMEOUT)

        # Sixth activation is rejected immediately, no load starts.
        with self.assertRaises(CapacityExceeded):
            await self.pool.activate("f")
        self.assert_no_new_loads(5)

        # Free exactly one slot: finish b's superseded load and its close.
        c2.release.set()
        b_close = await self.backend.next_close()
        self.assertIs(b_close.handle, c2.handle)
        b_close.release.set()
        await asyncio.wait_for(b_close.done.wait(), TIMEOUT)

        # Now a new load may start.
        t6, c6 = await self.start_load("f")
        c6.release.set()
        await asyncio.wait_for(t6, TIMEOUT)
        self.assert_no_new_loads(6)

        # Revoke everything and let all cleanup finish.
        self.pool.deactivate("a")  # leased: close waits for lease return
        self.pool.deactivate("d")  # in-flight load
        self.pool.deactivate("e")
        self.pool.deactivate("f")
        with self.assertRaises(Superseded):
            await asyncio.wait_for(t4, TIMEOUT)
        c4.release.set()
        # e and f close immediately; d's close starts only once its late
        # load finishes. Match close calls by handle, not by order.
        closes = {}
        for _ in range(3):
            close = await self.backend.next_close()
            closes[close.handle] = close
        self.assertEqual(set(closes), {c4.handle, c5.handle, c6.handle})
        self.assertEqual(len(self.backend.close_calls), 5)
        await held.__aexit__(None, None, None)
        a_close = await self.backend.next_close()
        self.assertIs(a_close.handle, held_handle)
        # Release every close gate; every handle closes exactly once.
        for call in [hung_close, a_close, *closes.values()]:
            call.release.set()
            await asyncio.wait_for(call.done.wait(), TIMEOUT)
        closed = [c.handle for c in self.backend.close_calls]
        self.assertEqual(len(closed), len(set(map(id, closed))))
        self.assertEqual(len(closed), 6)  # a..f, each exactly once
        self.assertEqual(self.pool._slots_used, 0)


if __name__ == "__main__":
    unittest.main()
