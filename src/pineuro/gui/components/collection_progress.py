"""Collection progress component for DICOM acquisition monitoring.

This module provides a progress display with DICOM slice preview for
collection steps, showing the middle axial slice along with volume count,
progress bar, and estimated time remaining.
"""

import base64
import contextlib
import io
import logging
from pathlib import Path

import flet as ft
import numpy as np

from pineuro.gui import styles as st

logger = logging.getLogger(__name__)

# Optional imports for DICOM display
try:
    import pydicom

    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def dicom_to_base64_png(dicom_path: Path) -> str | None:
    """Convert DICOM middle slice to base64 PNG for Flet display.

    Parameters
    ----------
    dicom_path : Path
        Path to a DICOM file.

    Returns
    -------
    str or None
        Base64 data URI for the PNG image, or None if conversion fails.
    """
    if not PYDICOM_AVAILABLE or not PIL_AVAILABLE:
        return None

    try:
        dcm = pydicom.dcmread(dicom_path)
        pixel_array = dcm.pixel_array

        # Handle 3D volumes - get middle slice
        if pixel_array.ndim == 3:
            mid_idx = pixel_array.shape[0] // 2
            slice_data = pixel_array[mid_idx]
        else:
            slice_data = pixel_array

        # Normalize to 0-255 for display
        slice_data = slice_data.astype(np.float32)
        min_val = slice_data.min()
        max_val = slice_data.max()
        if max_val > min_val:
            slice_data = (slice_data - min_val) / (max_val - min_val)
        else:
            slice_data = np.zeros_like(slice_data)
        slice_data = (slice_data * 255).astype(np.uint8)

        # Convert to PNG bytes
        img = Image.fromarray(slice_data, mode="L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")

        # Return as base64 data URI
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    except Exception as e:
        logger.warning(f"Failed to convert DICOM to PNG: {e}")
        return None


class CollectionProgressPanel:
    """Progress display for DICOM collection with slice preview.

    Shows:
    - DICOM slice preview (middle axial slice)
    - Volume count (e.g., "45 / 200 volumes")
    - Progress bar
    - Estimated time remaining
    """

    def __init__(
        self,
        n_volumes: int,
        tr: float | None = None,
    ):
        """Initialize the collection progress panel.

        Parameters
        ----------
        n_volumes : int
            Expected total number of volumes.
        tr : float, optional
            Repetition time in seconds (for time estimation).
        """
        self.n_volumes = n_volumes
        self.tr = tr

        # Track state
        self._volumes_collected = 0
        self._elapsed_time = 0.0

        # UI elements (created in build())
        self._image: ft.Image | None = None
        self._volume_text: ft.Text | None = None
        self._progress_bar: ft.ProgressBar | None = None
        self._time_text: ft.Text | None = None
        self._placeholder: ft.Container | None = None

    def build(self) -> ft.Container:
        """Build the progress panel control.

        Returns
        -------
        ft.Container
            Container with progress display elements.
        """
        # Placeholder for DICOM preview
        self._image = ft.Image(
            src="",
            width=st.DICOM_PREVIEW,
            height=st.DICOM_PREVIEW,
            fit=ft.BoxFit.CONTAIN,
            visible=False,  # Hidden until first image
        )

        # Placeholder shown before first volume
        self._placeholder = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.IMAGE_OUTLINED, size=st.ICON_XXL, color=st.NEUTRAL),
                    ft.Text(
                        "Waiting for first volume...",
                        size=st.BODY,
                        color=st.MUTED,
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=st.GAP_SM,
            ),
            width=st.DICOM_PREVIEW,
            height=st.DICOM_PREVIEW,
            alignment=ft.Alignment(0, 0),
            border=ft.border.all(1, st.PANEL_BORDER),
            border_radius=8,
        )

        self._volume_text = ft.Text(
            f"0 / {self.n_volumes} volumes",
            size=st.SECTION,
            weight=ft.FontWeight.W_500,
        )

        self._progress_bar = ft.ProgressBar(
            value=0,
            width=st.PROGRESS_BAR_W,
            color=st.ACCENT,
            bgcolor=st.SELECTED_BG,
        )

        self._time_text = ft.Text(
            "",
            size=st.BODY,
            color=st.MUTED,
        )

        return ft.Container(
            content=ft.Column(
                [
                    # Image preview area
                    ft.Stack(
                        [
                            self._placeholder,
                            self._image,
                        ],
                    ),
                    # Spacing
                    ft.Container(height=st.GAP),
                    # Volume count
                    self._volume_text,
                    # Progress bar
                    ft.Container(height=st.GAP_XS),
                    self._progress_bar,
                    # Time estimate
                    ft.Container(height=st.GAP_XS),
                    self._time_text,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=0,
            ),
            padding=st.PAD,
            alignment=ft.Alignment(0, 0),
        )

    def update(
        self,
        volumes_collected: int,
        elapsed_time: float,
        dicom_path: Path | None = None,
    ):
        """Update the progress display.

        Parameters
        ----------
        volumes_collected : int
            Number of volumes collected so far.
        elapsed_time : float
            Time elapsed since collection started (seconds).
        dicom_path : Path, optional
            Path to the latest collected DICOM file for preview.
        """
        self._volumes_collected = volumes_collected
        self._elapsed_time = elapsed_time

        # Update volume text
        if self._volume_text is not None:
            self._volume_text.value = f"{volumes_collected} / {self.n_volumes} volumes"
            with contextlib.suppress(Exception):
                self._volume_text.update()

        # Update progress bar
        if self._progress_bar is not None:
            progress = volumes_collected / self.n_volumes if self.n_volumes > 0 else 0
            self._progress_bar.value = min(1.0, progress)
            with contextlib.suppress(Exception):
                self._progress_bar.update()

        # Update time estimate
        if self._time_text is not None:
            time_remaining = self._estimate_time_remaining()
            if time_remaining is not None and time_remaining > 0:
                mins, secs = divmod(int(time_remaining), 60)
                if mins > 0:
                    self._time_text.value = f"~{mins}m {secs}s remaining"
                else:
                    self._time_text.value = f"~{secs}s remaining"
            elif volumes_collected >= self.n_volumes:
                self._time_text.value = "Complete"
            else:
                self._time_text.value = ""
            with contextlib.suppress(Exception):
                self._time_text.update()

        # Update DICOM preview
        if dicom_path is not None and self._image is not None:
            try:
                b64_png = dicom_to_base64_png(dicom_path)
                if b64_png is not None:
                    self._image.src = b64_png
                    self._image.visible = True
                    self._image.update()

                    # Hide placeholder
                    if self._placeholder is not None:
                        self._placeholder.visible = False
                        self._placeholder.update()
            except Exception as e:
                logger.warning(f"Failed to display DICOM: {e}")

    def _estimate_time_remaining(self) -> float | None:
        """Estimate time remaining in seconds.

        Returns
        -------
        float or None
            Estimated seconds remaining, or None if cannot estimate.
        """
        remaining = self.n_volumes - self._volumes_collected
        if remaining <= 0:
            return 0.0

        # Use TR if available
        if self.tr is not None and self.tr > 0:
            return remaining * self.tr

        # Estimate from observed rate
        if self._volumes_collected > 0 and self._elapsed_time > 0:
            rate = self._volumes_collected / self._elapsed_time
            if rate > 0:
                return remaining / rate

        return None

    def clear(self):
        """Reset the progress display."""
        self._volumes_collected = 0
        self._elapsed_time = 0.0
        self.update(0, 0.0)

        # Reset image visibility
        if self._image is not None:
            self._image.visible = False
            with contextlib.suppress(Exception):
                self._image.update()

        if self._placeholder is not None:
            self._placeholder.visible = True
            with contextlib.suppress(Exception):
                self._placeholder.update()
