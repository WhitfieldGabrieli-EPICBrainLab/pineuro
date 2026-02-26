"""Main-thread task dispatcher for macOS PsychoPy compatibility.

On macOS, Cocoa requires ``NSWindow`` creation on the main thread.  Since the
Flet GUI normally occupies the main thread, PsychoPy tasks launched from
background pipeline threads cannot create windows.

This module provides a simple dispatcher: the main thread runs
:meth:`MainThreadDispatcher.run_forever`, pulling callables from a queue.
Background threads submit work via :meth:`submit` and receive a
:class:`~concurrent.futures.Future` for the result.

Only used on macOS (``sys.platform == "darwin"``).  Windows and Linux continue
to use the existing thread-based approach.
"""

from __future__ import annotations

import queue
from concurrent.futures import Future
from typing import Any

_dispatcher: MainThreadDispatcher | None = None


def get_dispatcher() -> MainThreadDispatcher:
    """Return the global dispatcher.  Raises if not initialised."""
    if _dispatcher is None:
        raise RuntimeError("MainThreadDispatcher not initialised — is this macOS?")
    return _dispatcher


def set_dispatcher(dispatcher: MainThreadDispatcher) -> None:
    """Set the global dispatcher (called once from ``app.main()``)."""
    global _dispatcher
    _dispatcher = dispatcher


class MainThreadDispatcher:
    """Execute callables on the main thread from any background thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue[
            tuple[Any, tuple, dict, Future] | None
        ] = queue.Queue()

    def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Future:
        """Submit *fn* to run on the main thread.  Returns a ``Future``."""
        fut: Future = Future()
        self._queue.put((fn, args, kwargs, fut))
        return fut

    def shutdown(self) -> None:
        """Signal the dispatch loop to exit."""
        self._queue.put(None)

    def run_forever(self) -> None:
        """Block, executing submitted callables until :meth:`shutdown`.

        Must be called on the main thread.
        """
        while True:
            item = self._queue.get()
            if item is None:
                break
            fn, args, kwargs, fut = item
            try:
                result = fn(*args, **kwargs)
                fut.set_result(result)
            except BaseException as exc:
                fut.set_exception(exc)
            # After task execution (which may have opened/closed PsychoPy
            # windows), pump the Cocoa event loop so pending operations
            # like fullscreen exit and window close are finalised.
            _flush_cocoa_events()


def _flush_cocoa_events() -> None:
    """Briefly run the Core Foundation run loop to process pending events.

    On macOS, window operations (especially fullscreen exit and close)
    are processed asynchronously by Cocoa.  After a PsychoPy task calls
    ``win.close()``, the Cocoa run loop must pump for the window to
    actually disappear.  Since our dispatcher blocks on a Python queue
    (not the Cocoa run loop), we pump it explicitly here.
    """
    try:
        import ctypes
        import ctypes.util

        cf = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("CoreFoundation")
        )
        kCFRunLoopDefaultMode = ctypes.c_void_p.in_dll(
            cf, "kCFRunLoopDefaultMode"
        )
        cf.CFRunLoopRunInMode.restype = ctypes.c_int32
        cf.CFRunLoopRunInMode.argtypes = [
            ctypes.c_void_p,
            ctypes.c_double,
            ctypes.c_bool,
        ]
        # Process pending events for up to 0.5 s (returns earlier if idle).
        cf.CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, False)
    except Exception:
        pass
