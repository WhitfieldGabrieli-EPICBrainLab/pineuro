"""Tests for pineuro.simulator — VolumeSimulator deadline-based volume copier."""

import json
import sys
import threading
import time

import nibabel as nib
import numpy as np
import pytest

from pineuro.simulator import VolumeSimulator

# =============================================================================
# Construction and validation
# =============================================================================


class TestVolumeSimulatorConstruction:
    def test_nifti_source(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        # Create 3 NIfTI volumes (no sidecars — get_volume_files returns ALL files)
        for i in range(3):
            vol = source / f"vol_{i:03d}.nii.gz"
            data = np.ones((4, 4, 4), dtype=np.float32)
            nib.save(nib.Nifti1Image(data, np.eye(4)), vol)

        sim = VolumeSimulator(source, tmp_path / "target", tr=1.5)
        assert sim.source_type == "nifti"
        assert len(sim.volume_files) == 3
        assert sim.tr == 1.5

    def test_auto_detect_tr_from_sidecar(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        vol = source / "vol_000.nii.gz"
        data = np.ones((4, 4, 4), dtype=np.float32)
        nib.save(nib.Nifti1Image(data, np.eye(4)), vol)
        sidecar = source / "vol_000.json"
        sidecar.write_text(json.dumps({"RepetitionTime": 2.0}))

        sim = VolumeSimulator(source, tmp_path / "target")
        assert sim.tr == 2.0

    def test_source_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Source directory not found"):
            VolumeSimulator(tmp_path / "nonexistent", tmp_path / "target", tr=1.0)

    def test_source_not_directory(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="not a directory"):
            VolumeSimulator(f, tmp_path / "target", tr=1.0)

    def test_empty_source(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        with pytest.raises(ValueError, match="No volume files"):
            VolumeSimulator(source, tmp_path / "target", tr=1.0)

    def test_speed_affects_delay(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        vol = source / "vol_000.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), vol)

        sim = VolumeSimulator(source, tmp_path / "target", tr=2.0, speed=4.0)
        assert sim.delay == pytest.approx(0.5)


# =============================================================================
# copy_volumes_iter
# =============================================================================


class TestCopyVolumesIter:
    def _make_nifti_source(self, tmp_path, n_volumes=5, tr=0.1, with_sidecars=False):
        """Create a source directory with n NIfTI volumes."""
        source = tmp_path / "source"
        source.mkdir()
        for i in range(n_volumes):
            vol = source / f"vol_{i:03d}.nii.gz"
            data = np.ones((4, 4, 4), dtype=np.float32) * (i + 1)
            nib.save(nib.Nifti1Image(data, np.eye(4)), vol)
            if with_sidecars:
                sidecar = source / f"vol_{i:03d}.json"
                sidecar.write_text(json.dumps({"RepetitionTime": tr}))
        return source

    def test_copies_all_volumes(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=3)
        target = tmp_path / "target"

        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0)
        copied = list(sim.copy_volumes_iter())

        assert copied == [1, 2, 3]
        nifti_files = sorted(target.glob("*.nii.gz"))
        assert len(nifti_files) == 3

    def test_copies_json_sidecars(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=2, with_sidecars=True)
        target = tmp_path / "target"

        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0)
        list(sim.copy_volumes_iter())

        json_files = sorted(target.glob("*.json"))
        assert len(json_files) == 2

    def test_n_volumes_limits_output(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=5)
        target = tmp_path / "target"

        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0)
        copied = list(sim.copy_volumes_iter(n_volumes=3))

        assert len(copied) == 3
        assert copied == [1, 2, 3]

    def test_stop_event_interrupts(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=10)
        target = tmp_path / "target"

        sim = VolumeSimulator(source, target, tr=0.5, speed=1.0)
        stop = threading.Event()

        # Stop after brief delay
        def stop_later():
            time.sleep(0.3)
            stop.set()

        threading.Thread(target=stop_later).start()

        copied = list(sim.copy_volumes_iter(stop_event=stop))
        # Should have stopped early (not all 10 volumes)
        assert len(copied) < 10

    def test_volume_copied_callback(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=3)
        target = tmp_path / "target"

        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0)
        callback_args = []
        sim.volume_copied_callback = lambda vol, elapsed, expected: callback_args.append((vol, elapsed, expected))

        list(sim.copy_volumes_iter())

        assert len(callback_args) == 3
        assert callback_args[0][0] == 1
        assert callback_args[1][0] == 2
        assert callback_args[2][0] == 3

    def test_manual_trigger_waits_for_event(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=2)
        target = tmp_path / "target"

        trigger_event = threading.Event()
        sim = VolumeSimulator(
            source,
            target,
            tr=0.05,
            speed=100.0,
            trigger_mode="manual",
            trigger_event=trigger_event,
        )

        # Start consumer in a thread; it should block on trigger_event.
        copied: list[int] = []

        def consume():
            for vol in sim.copy_volumes_iter():
                copied.append(vol)

        thread = threading.Thread(target=consume)
        thread.start()
        # Without the trigger, simulator must not have copied anything.
        time.sleep(0.05)
        assert copied == []
        trigger_event.set()
        thread.join(timeout=2.0)
        assert copied == [1, 2]

    def test_collection_only_no_trigger_event(self, tmp_path):
        # When trigger_event is None (collection-only contexts), the simulator
        # must not block waiting for a trigger.
        source = self._make_nifti_source(tmp_path, n_volumes=2)
        target = tmp_path / "target"

        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0, trigger_mode="manual")
        copied = list(sim.copy_volumes_iter())
        assert copied == [1, 2]

    def test_clear_target_removes_old_files(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=2)
        target = tmp_path / "target"
        target.mkdir()
        (target / "old_file.txt").write_text("stale")

        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0, clear_target=True)
        list(sim.copy_volumes_iter())

        assert not (target / "old_file.txt").exists()

    def test_creates_target_dir(self, tmp_path):
        source = self._make_nifti_source(tmp_path, n_volumes=1)
        target = tmp_path / "new_target"

        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0)
        list(sim.copy_volumes_iter())

        assert target.is_dir()


# =============================================================================
# cleanup
# =============================================================================


class TestCleanup:
    def test_cleanup_removes_target(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        vol = source / "vol_000.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), vol)

        target = tmp_path / "target"
        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0)
        list(sim.copy_volumes_iter())
        assert target.exists()

        sim.cleanup()
        assert not target.exists()

    def test_cleanup_on_complete(self, tmp_path):
        source = tmp_path / "source"
        source.mkdir()
        vol = source / "vol_000.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), vol)

        target = tmp_path / "target"
        sim = VolumeSimulator(source, target, tr=0.05, speed=100.0, cleanup_on_complete=True)
        list(sim.copy_volumes_iter())

        assert not target.exists()


# =============================================================================
# Win32 SendInput struct layout (regression for os_key trigger mode)
# =============================================================================


@pytest.mark.skipif(sys.platform != "win32", reason="Win32-only SendInput INPUT struct")
def test_win32_input_struct_size_matches_sendinput():
    import ctypes

    from pineuro.simulator import _INPUT

    # Win32 INPUT is 40 bytes on 64-bit Windows, 28 on 32-bit. If sizeof(INPUT)
    # doesn't match what SendInput expects, every call returns 0 and the
    # synthetic-keypress trigger path silently breaks.
    expected = 40 if ctypes.sizeof(ctypes.c_void_p) == 8 else 28
    assert ctypes.sizeof(_INPUT) == expected
