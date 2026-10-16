"""Tests for pineuro.gui.components.collection_progress — pure helpers only.

The CollectionProgressPanel itself is a Flet UI builder whose body is hard to
unit-test without extensive Flet mocking. We cover (1) the dicom_to_base64_png
graceful-failure path, (2) _estimate_time_remaining math (no Flet involved), and
(3) the slice preview's gapless playback (construction only; no page attached).
"""

import pytest


class TestDicomToBase64PngFailure:
    def test_returns_none_on_missing_file(self, tmp_path):
        from pineuro.gui.components.collection_progress import dicom_to_base64_png

        missing = tmp_path / "no_such.dcm"
        # Either pydicom raises (caught + logged → None) or PYDICOM unavailable → None.
        assert dicom_to_base64_png(missing) is None

    def test_returns_none_on_invalid_file(self, tmp_path):
        from pineuro.gui.components.collection_progress import dicom_to_base64_png

        bogus = tmp_path / "not_a_dicom.dcm"
        bogus.write_bytes(b"this is not a dicom file")
        assert dicom_to_base64_png(bogus) is None


class TestEstimateTimeRemaining:
    @staticmethod
    def _panel(*, n_volumes=200, tr=None, volumes_collected=0, elapsed=0.0):
        from pineuro.gui.components.collection_progress import CollectionProgressPanel

        panel = CollectionProgressPanel.__new__(CollectionProgressPanel)
        panel.n_volumes = n_volumes
        panel.tr = tr
        panel._volumes_collected = volumes_collected
        panel._elapsed_time = elapsed
        return panel

    def test_zero_when_complete(self):
        panel = self._panel(n_volumes=100, volumes_collected=100)
        assert panel._estimate_time_remaining() == 0.0

    def test_negative_remaining_treated_as_done(self):
        # Overshoot collection (defensive)
        panel = self._panel(n_volumes=100, volumes_collected=150)
        assert panel._estimate_time_remaining() == 0.0

    def test_uses_tr_when_available(self):
        # 50 volumes remaining × 2.0s TR = 100.0s
        panel = self._panel(n_volumes=100, tr=2.0, volumes_collected=50)
        assert panel._estimate_time_remaining() == 100.0

    def test_falls_back_to_observed_rate(self):
        # 50 collected in 25.0s → rate=2 vol/s; 50 remaining → 25.0s.
        panel = self._panel(n_volumes=100, tr=None, volumes_collected=50, elapsed=25.0)
        assert panel._estimate_time_remaining() == pytest.approx(25.0)

    def test_returns_none_without_data(self):
        # No TR and no progress yet
        panel = self._panel(n_volumes=100, tr=None, volumes_collected=0, elapsed=0.0)
        assert panel._estimate_time_remaining() is None

    def test_zero_or_negative_tr_falls_back(self):
        # Bad TR should not be used; rate fallback should kick in.
        panel = self._panel(n_volumes=100, tr=0.0, volumes_collected=10, elapsed=5.0)
        # rate = 10/5 = 2/s; remaining = 90 → 45s
        assert panel._estimate_time_remaining() == pytest.approx(45.0)


class TestBuild:
    def test_slice_preview_keeps_last_frame_between_volumes(self):
        """Regression guard: Flet's default (``gapless_playback=False``) blanks the image
        on every new ``src``, so each volume's slice flashed against the white page.
        """
        from pineuro.gui.components.collection_progress import CollectionProgressPanel

        panel = CollectionProgressPanel(n_volumes=10)
        panel.build()
        assert panel._image.gapless_playback is True
