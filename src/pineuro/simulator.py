"""
Volume simulator for offline testing of real-time neurofeedback.

Summary
-------
Copies volume files (DICOM or NIfTI) from a source folder to a target folder
at TR-paced intervals to mimic scanner acquisition for testing pipelines.

Components
----------
- VolumeSimulator: Deadline-based copier that preserves timing and ordering

Workflow
--------
- Sort source volumes (auto-detect DICOM or NIfTI)
- Determine TR (argument, DICOM header, or NIfTI JSON sidecar)
- Copy to target at TR/speed deadlines, invoking optional callbacks
- For NIfTI files, also copy JSON sidecar if present
- Optional cleanup of target directory on completion

Dependencies
------------
pydicom, nibabel, pathlib, shutil

Notes
-----
Files are sorted to keep temporal order. TR can be provided, read from
DICOM headers, or read from NIfTI JSON sidecars. The speed parameter
accelerates or slows delivery (e.g., speed=2.0 for 2x). Deadline-based
timing avoids drift from variable copy durations, making this suitable
for end-to-end dry runs without a scanner.
Further reading: see docs/references.md.

Command-line usage:
    python -m pineuro.simulator --source /path/to/volumes --target /path/to/incoming --tr 1.5 --speed 1.0

References
----------
[1] DICOM Standard: https://www.dicomstandard.org/
"""

# Standard library imports
import argparse
import logging
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Union

TriggerMode = Literal["manual", "os_key"]

# Win32 SendInput structs. Defined unconditionally (ctypes/wintypes are stdlib and
# importable on all platforms; only ctypes.windll is Windows-only) so the regression
# test can verify their size on any CI runner — the union MUST include MOUSEINPUT
# (its 32-byte size on 64-bit / 24-byte on 32-bit dominates the union), otherwise
# ctypes.sizeof(INPUT) is 32/20 bytes and SendInput silently rejects every call
# (returns 0).
import ctypes
from ctypes import wintypes


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class _INPUT_I(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("i", _INPUT_I)]


# Third-party imports
# Local imports
from .utils import (
    get_json_sidecar_path,
    get_tr_from_volume,
    get_volume_files,
    is_nifti_file,
)

# Configure module-level logger
logger = logging.getLogger(__name__)


class VolumeSimulator:
    """
    Simulates real-time volume acquisition by copying files at TR intervals.

    Supports both DICOM and NIfTI input files. For NIfTI files, also copies
    the JSON sidecar file alongside each volume.

    Uses deadline-based timing to ensure consistent TR intervals regardless of
    copy operation duration. Each volume has an expected arrival time, and the
    simulator sleeps until that deadline rather than sleeping a fixed duration.

    Parameters
    ----------
    source_dir : str or Path
        Directory containing source volume files (DICOM or NIfTI)
    target_dir : str or Path
        Directory to copy volumes to
    tr : float, optional
        Repetition time in seconds. If None, reads from DICOM headers or
        NIfTI JSON sidecar.
    speed : float, default=1.0
        Speed multiplier for simulation
    clear_target : bool, default=True
        Whether to clear target directory before starting
    cleanup_on_complete : bool, default=False
        Whether to delete the entire target directory after simulation completes

    Attributes
    ----------
    source_type : str
        Detected source type: 'dicom', 'nifti', or 'mixed'
    start_time : float or None
        Unix timestamp when simulation started (set by start() or copy_volumes_iter)
    volume_copied_callback : callable or None
        Optional callback invoked after each volume is copied, with signature:
        callback(volume_num: int, elapsed: float, expected_time: float)

    Raises
    ------
    FileNotFoundError
        If source_dir does not exist
    ValueError
        If source_dir is not a directory or contains no volume files

    Notes
    -----
    **Timing Model:**

    Volume N should arrive at time N * TR (0-indexed). The simulator computes
    the expected arrival time for each volume and sleeps until that deadline.
    This compensates for variable copy duration and prevents cumulative drift.

    Example at TR=1.2s, speed=1x:
    - Volume 0: deadline = 0.0s (copy immediately after start)
    - Volume 1: deadline = 1.2s
    - Volume 2: deadline = 2.4s
    - etc.

    If a copy takes 0.1s, the next sleep is 1.1s (not 1.2s), maintaining
    the overall schedule.

    **NIfTI Support:**

    For NIfTI files (.nii, .nii.gz), the simulator also copies the JSON sidecar
    file if present (e.g., vol_001.nii + vol_001.json). TR is extracted from
    the JSON sidecar's RepetitionTime field.
    """

    def __init__(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        tr: float | None = None,
        speed: float = 1.0,
        clear_target: bool = True,
        cleanup_on_complete: bool = False,
        trigger_mode: TriggerMode = "manual",
        trigger_key: str = "t",
        trigger_event: threading.Event | None = None,
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.tr = tr
        self.speed = speed
        self.clear_target = clear_target
        self.cleanup_on_complete = cleanup_on_complete

        # Trigger control: "manual" blocks on trigger_event until the task
        # detects a real keypress; "os_key" injects a synthetic OS keypress
        # that the task receives via the same focus-dependent code path as
        # a real scanner. Both modes faithfully exercise the real-scan path.
        self.trigger_mode: TriggerMode = trigger_mode
        self.trigger_key: str = trigger_key
        self.trigger_event: threading.Event | None = trigger_event

        # Timing state
        self.start_time: float | None = None
        self.volume_copied_callback: Callable[[int, float, float], None] | None = None

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        if not self.source_dir.is_dir():
            raise ValueError(f"Source is not a directory: {self.source_dir}")

        # Auto-detect source type and get volume files
        self.volume_files = get_volume_files(self.source_dir)
        if len(self.volume_files) == 0:
            raise ValueError(f"No volume files (DICOM or NIfTI) found in: {self.source_dir}")

        # Determine source type based on first file
        if is_nifti_file(self.volume_files[0]):
            self.source_type = "nifti"
        else:
            self.source_type = "dicom"

        logger.info(f"Found {len(self.volume_files)} {self.source_type} files in source")

        if self.tr is None:
            self.tr = get_tr_from_volume(self.volume_files[0])
            logger.info(f"Read TR from {self.source_type}: {self.tr}s")

        self.delay = self.tr / self.speed
        logger.info(f"Effective delay: {self.delay:.3f}s (speed={self.speed}x)")

    def _sleep_until_deadline(self, deadline: float, stop_event=None) -> float:
        """
        Sleep until the specified deadline time.

        Parameters
        ----------
        deadline : float
            Absolute time (from time.monotonic()) to sleep until.
        stop_event : threading.Event, optional
            If provided, check this event periodically and return early if set.

        Returns
        -------
        float
            Actual sleep duration (may be 0 if deadline already passed).
        """
        now = time.monotonic()
        sleep_duration = deadline - now
        if sleep_duration > 0:
            if stop_event is not None:
                # Sleep in short intervals so we can react to stop_event.
                interval = 0.1  # Check every 100ms
                slept = 0.0
                while sleep_duration > 0:
                    if stop_event.is_set():
                        return slept
                    chunk = min(interval, sleep_duration)
                    time.sleep(chunk)
                    slept += chunk
                    sleep_duration -= chunk
                return slept
            time.sleep(sleep_duration)
            return sleep_duration
        else:
            # Deadline already passed - we're running behind
            lag = -sleep_duration
            if lag > self.delay * 0.1:  # Warn if >10% behind schedule
                logger.warning(f"Simulator running {lag:.3f}s behind schedule")
            return 0.0

    def _clear_target_directory(self) -> None:
        """Remove all files from target directory."""
        if self.target_dir.exists():
            for f in self.target_dir.iterdir():
                if f.is_file():
                    f.unlink()
            logger.info(f"Cleared target directory: {self.target_dir}")

    def cleanup(self) -> None:
        """
        Delete the entire target directory and its contents.

        This method removes the target directory created by the simulator,
        useful for cleaning up after a simulation run completes.
        """
        if self.target_dir.exists():
            shutil.rmtree(self.target_dir)
            logger.info(f"Deleted target directory: {self.target_dir}")

    # ------------------------------------------------------------------
    # OS-level synthetic keypress (for trigger_mode="os_key")
    # ------------------------------------------------------------------
    # Routes the simulated trigger through the real OS input queue so it
    # exercises the same focus-dependent code path as a USB-keyboard scanner
    # trigger. Without this, simulator runs cannot reproduce real-scan failures.

    _VK_WIN: dict[str, int] = {
        "0": 0x30,
        "1": 0x31,
        "2": 0x32,
        "3": 0x33,
        "4": 0x34,
        "5": 0x35,
        "6": 0x36,
        "7": 0x37,
        "8": 0x38,
        "9": 0x39,
        "t": 0x54,
        "equal": 0xBB,
        "+": 0x6B,
        "space": 0x20,
        "return": 0x0D,
    }
    _VK_MAC: dict[str, int] = {
        "0": 0x1D,
        "1": 0x12,
        "2": 0x13,
        "3": 0x14,
        "4": 0x15,
        "5": 0x17,
        "6": 0x16,
        "7": 0x1A,
        "8": 0x1C,
        "9": 0x19,
        "t": 0x11,
        "equal": 0x18,
        "+": 0x45,
        "space": 0x31,
        "return": 0x24,
    }
    _XKEYSYM: dict[str, str] = {
        "0": "0",
        "1": "1",
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "t": "t",
        "equal": "equal",
        "+": "plus",
        "space": "space",
        "return": "Return",
    }

    def _send_os_keypress(self, key: str) -> bool:
        """Send a synthetic OS-level key down+up. Returns True on success."""
        if sys.platform == "win32":
            return self._send_keypress_win32(key)
        if sys.platform == "linux":
            return self._send_keypress_x11(key)
        if sys.platform == "darwin":
            return self._send_keypress_macos(key)
        logger.warning(f"OS keypress not supported on platform: {sys.platform}")
        return False

    def _send_keypress_win32(self, key: str) -> bool:
        if sys.platform != "win32":
            return False

        vk = self._VK_WIN.get(key)
        if vk is None:
            logger.warning(f"No Win32 VK mapping for key {key!r}")
            return False

        KEYEVENTF_KEYUP = 0x0002
        INPUT_KEYBOARD = 1

        try:
            user32 = ctypes.windll.user32
            down = _INPUT(type=INPUT_KEYBOARD, i=_INPUT_I(ki=_KEYBDINPUT(vk, 0, 0, 0, None)))
            up = _INPUT(type=INPUT_KEYBOARD, i=_INPUT_I(ki=_KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, None)))
            n_down = user32.SendInput(1, ctypes.byref(down), ctypes.sizeof(_INPUT))
            time.sleep(0.05)
            n_up = user32.SendInput(1, ctypes.byref(up), ctypes.sizeof(_INPUT))
            ok: bool = bool(n_down == 1) and bool(n_up == 1)
            if ok:
                logger.info(f"Sent synthetic Win32 keypress: {key!r} (VK=0x{vk:02X})")
            else:
                logger.warning(f"Win32 SendInput partial: down={n_down}, up={n_up}")
            return ok
        except Exception as e:
            logger.warning(f"Win32 SendInput failed: {e}")
            return False

    def _send_keypress_x11(self, key: str) -> bool:
        try:
            from Xlib import XK
            from Xlib import display as xlib_display
            from Xlib.ext import xtest
        except ImportError as e:
            logger.warning(f"X11 keypress unavailable (Wayland-pure session?): {e}")
            return False

        keysym_str = self._XKEYSYM.get(key)
        if keysym_str is None:
            logger.warning(f"No X11 keysym mapping for key {key!r}")
            return False

        try:
            d = xlib_display.Display()
            keysym = XK.string_to_keysym(keysym_str)
            keycode = d.keysym_to_keycode(keysym)
            if not keycode:
                logger.warning(f"X11 could not resolve keycode for {key!r}")
                return False
            xtest.fake_input(d, 2, keycode)  # KeyPress
            d.sync()
            time.sleep(0.05)
            xtest.fake_input(d, 3, keycode)  # KeyRelease
            d.sync()
            logger.info(f"Sent synthetic X11 keypress: {key!r} (kc={keycode})")
            return True
        except Exception as e:
            logger.warning(f"X11 fake_input failed (likely Wayland): {e}")
            return False

    def _send_keypress_macos(self, key: str) -> bool:
        try:
            from Quartz import (
                CGEventCreateKeyboardEvent,
                CGEventPost,
                kCGHIDEventTap,
            )
        except ImportError as e:
            logger.warning(f"Quartz keypress unavailable: {e}")
            return False

        vk = self._VK_MAC.get(key)
        if vk is None:
            logger.warning(f"No macOS VK mapping for key {key!r}")
            return False

        try:
            down = CGEventCreateKeyboardEvent(None, vk, True)
            up = CGEventCreateKeyboardEvent(None, vk, False)
            CGEventPost(kCGHIDEventTap, down)
            time.sleep(0.05)
            CGEventPost(kCGHIDEventTap, up)
            logger.info(
                f"Sent synthetic macOS keypress: {key!r} (VK=0x{vk:02X}). "
                "If the task did not receive it, grant Accessibility "
                "permission to this binary in System Settings."
            )
            return True
        except Exception as e:
            logger.warning(f"Quartz CGEventPost failed: {e}")
            return False

    def _copy_volume_file(self, src_file: Path) -> Path:
        """
        Copy a volume file (and its JSON sidecar for NIfTI) to target directory.

        Copies directly to the final filename using shutil.copyfile (content
        only, no metadata) for compatibility with network filesystems (SMB).
        The file watcher's _wait_for_stable_file handles partially-written files.

        Parameters
        ----------
        src_file : Path
            Source volume file to copy

        Returns
        -------
        Path
            Path to copied volume file in target directory
        """
        dst_file = self.target_dir / src_file.name
        shutil.copyfile(src_file, dst_file)

        # For NIfTI files, also copy the JSON sidecar if it exists
        if is_nifti_file(src_file):
            json_src = get_json_sidecar_path(src_file)
            if json_src.exists():
                json_dst = get_json_sidecar_path(dst_file)
                shutil.copyfile(json_src, json_dst)

        return dst_file

    def copy_volumes_iter(self, n_volumes: int | None = None, stop_event=None):
        """
        Generator that copies volumes one at a time, yielding after each.

        This method is designed for integration with the feedback coordinator,
        allowing the simulator to run in the same thread/process while yielding
        control between volume copies.

        Uses deadline-based timing: each volume has an expected arrival time,
        and we sleep until that deadline rather than sleeping a fixed duration.
        This ensures consistent TR intervals regardless of copy operation duration.

        For NIfTI files, also copies the JSON sidecar file if present.

        Parameters
        ----------
        n_volumes : int, optional
            Number of volumes to simulate. If None, simulates all available volumes.
        stop_event : threading.Event, optional
            If provided, the simulator will stop when this event is set.

        Yields
        ------
        int
            Volume number (1-indexed) after each copy

        Examples
        --------
        >>> sim = VolumeSimulator(source_dir, target_dir, tr=1.5, speed=2.0)
        >>> for vol_num in sim.copy_volumes_iter(n_volumes=10):
        ...     print(f"Copied volume {vol_num}")
        """
        self.target_dir.mkdir(parents=True, exist_ok=True)

        if self.clear_target:
            self._clear_target_directory()

        files_to_send = self.volume_files[:n_volumes] if n_volumes else self.volume_files
        n_total = len(files_to_send)

        logger.info(f"Starting {self.source_type} simulator: {n_total} volumes at {self.speed}x speed")
        logger.debug(f"TR: {self.tr}s | Delay: {self.delay:.3f}s")

        # Issue or wait for the trigger before copying any volumes. The
        # start_time is reset afterward so volume 0's deadline is measured
        # from the trigger instant, not from simulator-thread launch.
        # When trigger_event is None (collection-only contexts with no task),
        # there is nothing to coordinate with, so trigger handling is skipped.
        if self.trigger_event is not None:
            logger.info(f"Simulator trigger_mode={self.trigger_mode}")
            if self.trigger_mode == "manual":
                logger.info("Simulator: waiting for manual trigger keypress")
                while not self.trigger_event.is_set():
                    if stop_event is not None and stop_event.is_set():
                        logger.info("Simulator stopped while waiting for trigger")
                        return
                    self.trigger_event.wait(timeout=0.1)
            elif self.trigger_mode == "os_key":
                ok = self._send_os_keypress(self.trigger_key)
                if not ok:
                    logger.warning(
                        "Simulator: OS keypress failed; falling back to manual mode. "
                        "On Wayland, synthetic input is blocked by design. "
                        "On macOS, grant Accessibility permission and retry."
                    )
                    while not self.trigger_event.is_set():
                        if stop_event is not None and stop_event.is_set():
                            return
                        self.trigger_event.wait(timeout=0.1)

        self.start_time = time.monotonic()

        volumes_copied = 0
        for i, src_file in enumerate(files_to_send):
            # Check for stop signal
            if stop_event is not None and stop_event.is_set():
                logger.info(f"Simulator stopped by user after {volumes_copied}/{n_total} volumes")
                return

            # Compute deadline for this volume (0-indexed, so vol 0 arrives at t=delay)
            # Volume arrives after one TR (simulating acquisition time)
            expected_time = (i + 1) * self.delay
            deadline = self.start_time + expected_time

            # Sleep until deadline (interruptible if stop_event provided)
            self._sleep_until_deadline(deadline, stop_event)

            # Check again after sleep in case we were interrupted
            if stop_event is not None and stop_event.is_set():
                logger.info(f"Simulator stopped by user after {volumes_copied}/{n_total} volumes")
                return

            # Copy the file (and JSON sidecar for NIfTI)
            copy_start = time.monotonic()
            self._copy_volume_file(src_file)
            copy_duration = time.monotonic() - copy_start
            volumes_copied += 1

            actual_elapsed = time.monotonic() - self.start_time
            logger.debug(
                f"Copied volume {i + 1}/{n_total}: {src_file.name} "
                f"(expected={expected_time:.3f}s, actual={actual_elapsed:.3f}s, "
                f"copy={copy_duration * 1000:.1f}ms)"
            )

            # Invoke callback if set
            if self.volume_copied_callback is not None:
                try:
                    self.volume_copied_callback(i + 1, actual_elapsed, expected_time)
                except Exception as e:
                    logger.error(f"Volume callback error: {e}")

            # Yield control back to caller
            yield i + 1

        logger.info(f"Simulator complete: {volumes_copied} volumes copied")

        if self.cleanup_on_complete:
            self.cleanup()

    def run(self, n_volumes: int | None = None) -> None:
        """Run the volume simulation with CLI progress output.

        Uses deadline-based timing to ensure consistent TR intervals.

        This method uses print() instead of logging for user-facing CLI output
        with formatted banners and real-time progress. For programmatic use,
        see copy_volumes_iter() which uses logging instead.

        For NIfTI files, also copies the JSON sidecar file if present.

        Parameters
        ----------
        n_volumes : int, optional
            Number of volumes to simulate. If None, simulates all available volumes.

        """
        self.target_dir.mkdir(parents=True, exist_ok=True)

        if self.clear_target:
            self._clear_target_directory()

        files_to_send = self.volume_files[:n_volumes] if n_volumes else self.volume_files
        n_total = len(files_to_send)

        source_type_label = self.source_type.upper()
        print(f"\n{'=' * 60}")
        print(f"{source_type_label} Simulator Started")
        print(f"{'=' * 60}")
        print(f"Volumes: {n_total}")
        print(f"TR: {self.tr}s | Speed: {self.speed}x")
        print(f"Target: {self.target_dir}")
        print(f"{'=' * 60}\n")

        # Set start time
        self.start_time = time.monotonic()

        for i, src_file in enumerate(files_to_send):
            # Compute deadline for this volume
            expected_time = (i + 1) * self.delay
            deadline = self.start_time + expected_time

            # Sleep until deadline
            self._sleep_until_deadline(deadline)

            # Copy the file (and JSON sidecar for NIfTI)
            self._copy_volume_file(src_file)

            actual_elapsed = time.monotonic() - self.start_time
            drift = actual_elapsed - expected_time
            drift_str = f"+{drift:.3f}" if drift >= 0 else f"{drift:.3f}"
            print(f"[{actual_elapsed:7.2f}s] Volume {i + 1:4d}/{n_total}: {src_file.name} (drift: {drift_str}s)")

        total_time = time.monotonic() - self.start_time
        expected_total = n_total * self.delay
        print(f"\n{'=' * 60}")
        print(f"Simulation complete. Total time: {total_time:.1f}s (expected: {expected_total:.1f}s)")
        print(f"{'=' * 60}\n")

        if self.cleanup_on_complete:
            self.cleanup()


def main():
    """Command-line entry point for volume simulator."""
    parser = argparse.ArgumentParser(description="Simulate real-time volume acquisition (DICOM or NIfTI)")
    parser.add_argument("--source", "-s", required=True, help="Source volume directory (DICOM or NIfTI)")
    parser.add_argument("--target", "-t", required=True, help="Target directory")
    parser.add_argument("--tr", type=float, default=None, help="TR in seconds")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed multiplier")
    parser.add_argument("--n", type=int, default=None, dest="n_volumes", help="Number of volumes")
    parser.add_argument("--no-clear", action="store_true", help="Keep existing files")
    parser.add_argument("--cleanup", action="store_true", help="Delete target directory after completion")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    sim = VolumeSimulator(
        source_dir=args.source,
        target_dir=args.target,
        tr=args.tr,
        speed=args.speed,
        clear_target=not args.no_clear,
        cleanup_on_complete=args.cleanup,
    )

    try:
        sim.run(n_volumes=args.n_volumes)
    except KeyboardInterrupt:
        print("\nSimulation interrupted")


if __name__ == "__main__":
    main()
