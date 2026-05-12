"""Tests for pineuro.gui.dispatch — MainThreadDispatcher, get/set_dispatcher."""

import threading
import time

import pytest

from pineuro.gui import dispatch as dispatch_mod
from pineuro.gui.dispatch import MainThreadDispatcher, get_dispatcher, set_dispatcher


@pytest.fixture(autouse=True)
def _reset_global_dispatcher():
    """Restore module-level _dispatcher between tests so they don't bleed state."""
    saved = dispatch_mod._dispatcher
    yield
    dispatch_mod._dispatcher = saved


class TestGlobalAccessors:
    def test_get_dispatcher_raises_when_uninitialised(self):
        dispatch_mod._dispatcher = None
        with pytest.raises(RuntimeError, match="not initialised"):
            get_dispatcher()

    def test_set_then_get_returns_same_instance(self):
        d = MainThreadDispatcher()
        set_dispatcher(d)
        assert get_dispatcher() is d


class TestSubmitAndRunForever:
    def _drive(self, dispatcher):
        """Run the dispatch loop in a background thread (since the unit test thread is the 'main' thread)."""
        t = threading.Thread(target=dispatcher.run_forever, daemon=True)
        t.start()
        return t

    def test_submit_runs_callable_and_returns_result(self):
        d = MainThreadDispatcher()
        t = self._drive(d)
        try:
            fut = d.submit(lambda x, y: x + y, 2, 3)
            assert fut.result(timeout=2.0) == 5
        finally:
            d.shutdown()
            t.join(timeout=2.0)

    def test_submit_propagates_exceptions(self):
        d = MainThreadDispatcher()
        t = self._drive(d)
        try:
            fut = d.submit(lambda: 1 / 0)
            with pytest.raises(ZeroDivisionError):
                fut.result(timeout=2.0)
        finally:
            d.shutdown()
            t.join(timeout=2.0)

    def test_submit_preserves_order(self):
        d = MainThreadDispatcher()
        t = self._drive(d)
        try:
            results = []
            futs = [d.submit(results.append, i) for i in range(5)]
            for f in futs:
                f.result(timeout=2.0)
            assert results == [0, 1, 2, 3, 4]
        finally:
            d.shutdown()
            t.join(timeout=2.0)

    def test_shutdown_stops_loop(self):
        d = MainThreadDispatcher()
        t = self._drive(d)
        d.shutdown()
        t.join(timeout=2.0)
        assert not t.is_alive()

    def test_submit_passes_kwargs(self):
        d = MainThreadDispatcher()
        t = self._drive(d)
        try:
            fut = d.submit(lambda *, a, b: a * b, a=3, b=4)
            assert fut.result(timeout=2.0) == 12
        finally:
            d.shutdown()
            t.join(timeout=2.0)

    def test_callable_runs_on_dispatch_thread_not_caller(self):
        """The callable executes on the thread that's running run_forever."""
        d = MainThreadDispatcher()
        t = self._drive(d)
        try:
            seen = {}
            fut = d.submit(lambda: seen.setdefault("tid", threading.get_ident()))
            fut.result(timeout=2.0)
            # The dispatch thread's ident should match the seen ident,
            # and NOT the caller (this test thread) ident.
            assert seen["tid"] == t.ident
            assert seen["tid"] != threading.get_ident()
        finally:
            d.shutdown()
            t.join(timeout=2.0)


class TestFlushCocoaEventsNonMacOs:
    def test_noop_on_non_darwin(self, monkeypatch):
        # On non-mac platforms _flush_cocoa_events must return immediately without raising.
        monkeypatch.setattr("sys.platform", "win32")
        # Should not raise or block
        start = time.monotonic()
        dispatch_mod._flush_cocoa_events()
        assert time.monotonic() - start < 0.5  # not blocking on a CFRunLoop
