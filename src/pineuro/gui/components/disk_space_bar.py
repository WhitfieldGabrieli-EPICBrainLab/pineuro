"""CONN-style free-disk-space readout pinned to the bottom of the window.

Shows free space on each volume the current screen writes to: every listed
study's data root on the Study Manager, and during a session that study's data
root (plus the scanner's DICOM folder when it sits on another volume). It turns
into a warning when free space falls below the projected need of one more
session (:func:`pineuro.storage.largest_session_size`). A daemon thread
refreshes it, as in :class:`FeedbackLogPoller`; UI writes go through
``page.run_task``.
"""

import contextlib
import logging
import threading
from collections.abc import Iterable
from pathlib import Path

import flet as ft

from pineuro.storage import DiskSpace, disk_space, format_bytes, low_space_reason

from .. import styles as st

logger = logging.getLogger(__name__)


def summarize(paths: Iterable[str | Path], projected: int | None = None) -> tuple[str, bool]:
    """Bar text for the volumes holding *paths*, and whether it is a warning.

    Paths on the same volume are reported once. *projected* is the expected
    size of one more session; a volume with less free space than that is
    flagged.
    """
    spaces: list[DiskSpace] = []
    seen: set[int] = set()
    for path in paths:
        space = disk_space(path)
        if space is None or space.volume in seen:
            continue
        seen.add(space.volume)
        spaces.append(space)
    if not spaces:
        return "Free disk space: unavailable", False

    parts = [f"{format_bytes(s.free)} free of {format_bytes(s.total)} ({s.path})" for s in spaces]
    reasons = [r for s in spaces if (r := low_space_reason(s, projected)) is not None]
    text = "Free disk space: " + "  ·  ".join(parts)
    if reasons:
        text += f" — {reasons[0]}"
    return text, bool(reasons)


class DiskSpaceBar:
    """Bottom app bar showing free disk space; one instance per app window.

    Assign :attr:`control` to ``page.bottom_appbar`` once: it lives on the root
    view, so it survives the ``page.controls.clear()`` screen swaps. Each screen
    then calls :meth:`watch` with the paths it writes to.
    """

    REFRESH_SECONDS = 60.0

    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self._lock = threading.Lock()
        self._paths: list[Path] = []
        self._projected: int | None = None
        self._status: tuple[str, bool] | None = None  # (text, is_warning), e.g. the last backup's outcome
        self._icon = ft.Icon(ft.Icons.STORAGE, size=st.ICON_SM, color=st.MUTED)
        self._text = ft.Text(
            "Free disk space: checking…",
            size=st.CAPTION,
            color=st.MUTED,
            max_lines=1,
            overflow=ft.TextOverflow.ELLIPSIS,
            expand=True,
        )
        self.control = ft.BottomAppBar(
            content=ft.Row(
                [self._icon, self._text],
                spacing=st.GAP_SM,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=36,
            padding=ft.Padding.symmetric(horizontal=st.PAD),
            bgcolor=st.SUBTLE_BG,
        )
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def watch(self, paths: Iterable[str | Path], projected_session_bytes: int | None = None) -> None:
        """Report free space for the volumes holding *paths* from now on.

        With *projected_session_bytes*, a volume with less free space than one
        more session would need is shown as a warning.
        """
        with self._lock:
            self._paths = [Path(p) for p in paths if p]
            self._projected = projected_session_bytes
        self.refresh()

    def set_status(self, text: str | None, warning: bool = False) -> None:
        """Show *text* (e.g. the last backup's outcome) after the free-space figures; None clears it."""
        with self._lock:
            self._status = (text, warning) if text else None
        self.refresh()

    def refresh(self) -> None:
        """Re-read free space now instead of at the next periodic refresh (thread-safe)."""
        self._wake.set()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="disk-space-bar", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        self._wake.set()

    def _run(self) -> None:
        while not self._stopped.is_set():
            self._wake.clear()
            with self._lock:
                paths, projected, status = list(self._paths), self._projected, self._status
            if paths:  # nothing to report until a screen has called watch()
                try:
                    text, warning = summarize(paths, projected)
                except Exception:
                    logger.exception("Disk space refresh failed")
                else:
                    if status is not None:
                        text = f"{text}   |   {status[0]}"
                        warning = warning or status[1]
                    self._publish(text, warning)
            self._wake.wait(self.REFRESH_SECONDS)

    def _publish(self, text: str, warning: bool) -> None:
        async def _apply() -> None:
            color = st.WARNING if warning else st.MUTED
            self._text.value = text
            self._text.color = color
            self._icon.icon = ft.Icons.WARNING_AMBER if warning else ft.Icons.STORAGE
            self._icon.color = color
            # The first refresh can beat the first render; the values above are
            # then shown when the bar mounts.
            with contextlib.suppress(Exception):
                self.control.update()

        try:
            self.page.run_task(_apply)
        except Exception:
            logger.debug("Disk space bar update skipped (page not ready)", exc_info=True)
