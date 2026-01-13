"""
DICOM simulator for offline testing of real-time neurofeedback.

This module simulates real-time scanner acquisition by copying DICOM files from
a source directory to a target directory at TR intervals, enabling offline testing
of real-time fMRI neurofeedback pipelines.

Key Components
--------------
DicomSimulator : Simulate real-time DICOM acquisition by copying files at TR intervals

Dependencies
------------
- pydicom
- pathlib
- shutil

Notes
-----
Files are sorted by InstanceNumber or AcquisitionTime to ensure proper temporal ordering.
TR can be specified or automatically read from DICOM headers. The speed parameter enables
faster-than-realtime simulation (e.g., speed=2.0 for 2x speed).

Command-line usage:
    python -m pyrtnf.simulator --source /path/to/dicoms --target /path/to/incoming --tr 1.5 --speed 1.0

References
----------
[1] DICOM Standard: https://www.dicomstandard.org/

"""

# Standard library imports
from typing import Union, Optional, List
from pathlib import Path
import shutil
import time
import argparse

# Third-party imports
import pydicom

# Configure module-level logger
import logging
logger = logging.getLogger(__name__)

class DicomSimulator:
    """
    Simulates real-time DICOM acquisition by copying files at TR intervals.

    Parameters
    ----------
    source_dir : str or Path
        Directory containing source DICOM files
    target_dir : str or Path
        Directory to copy DICOMs to
    tr : float, optional
        Repetition time in seconds. If None, reads from DICOM headers.
    speed : float, default=1.0
        Speed multiplier for simulation
    clear_target : bool, default=True
        Whether to clear target directory before starting

    Raises
    ------
    FileNotFoundError
        If source_dir does not exist
    ValueError
        If source_dir is not a directory or contains no DICOM files

    """

    def __init__(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        tr: Optional[float] = None,
        speed: float = 1.0,
        clear_target: bool = True
    ):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.tr = tr
        self.speed = speed
        self.clear_target = clear_target

        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        if not self.source_dir.is_dir():
            raise ValueError(f"Source is not a directory: {self.source_dir}")

        self.dicom_files = self._get_sorted_dicoms()
        if len(self.dicom_files) == 0:
            raise ValueError(f"No DICOM files found in: {self.source_dir}")

        logger.info(f"Found {len(self.dicom_files)} DICOM files in source")

        if self.tr is None:
            self.tr = self._get_tr_from_dicom(self.dicom_files[0])
            logger.info(f"Read TR from DICOM: {self.tr}s")

        self.delay = self.tr / self.speed
        logger.info(f"Effective delay: {self.delay:.3f}s (speed={self.speed}x)")

    def _get_sorted_dicoms(self) -> List[Path]:
        """
        Get DICOM files sorted by filename.

        Returns
        -------
        List[Path]
            Sorted list of DICOM file paths

        """
        dicom_files = []
        for ext in ['*.dcm', '*.DCM', '*.ima', '*.IMA']:
            dicom_files.extend(self.source_dir.glob(ext))

        # Filter to only files and sort by name
        valid_dicoms = [f for f in dicom_files if f.is_file()]
        valid_dicoms.sort()
        return valid_dicoms

    def _get_tr_from_dicom(self, dicom_path: Path) -> float:
        """
        Extract repetition time (TR) from DICOM header.

        Parameters
        ----------
        dicom_path : Path
            Path to DICOM file

        Returns
        -------
        float
            TR in seconds. Returns 2.0s if TR cannot be read.

        """
        try:
            dcm = pydicom.dcmread(dicom_path, stop_before_pixels=True)
            if hasattr(dcm, 'RepetitionTime'):
                return float(dcm.RepetitionTime) / 1000.0
            logger.warning("Could not read TR, using default 2.0s")
            return 2.0
        except Exception as e:
            logger.warning(f"Error reading TR: {e}, using default 2.0s")
            return 2.0

    def _clear_target_directory(self) -> None:
        """Remove all files from target directory."""
        if self.target_dir.exists():
            for f in self.target_dir.iterdir():
                if f.is_file():
                    f.unlink()
            logger.info(f"Cleared target directory: {self.target_dir}")

    def run(self, n_volumes: Optional[int] = None) -> None:
        """
        Run the DICOM simulation.

        Parameters
        ----------
        n_volumes : int, optional
            Number of volumes to simulate. If None, simulates all available volumes.

        """
        self.target_dir.mkdir(parents=True, exist_ok=True)

        if self.clear_target:
            self._clear_target_directory()

        files_to_send = self.dicom_files[:n_volumes] if n_volumes else self.dicom_files
        n_total = len(files_to_send)

        print(f"\n{'='*60}")
        print("DICOM Simulator Started")
        print(f"{'='*60}")
        print(f"Volumes: {n_total}")
        print(f"TR: {self.tr}s | Speed: {self.speed}x")
        print(f"Target: {self.target_dir}")
        print(f"{'='*60}\n")

        start_time = time.time()
        for i, src_file in enumerate(files_to_send):
            dst_file = self.target_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            elapsed = time.time() - start_time
            print(f"[{elapsed:7.2f}s] Volume {i+1:4d}/{n_total}: {src_file.name}")
            if i < n_total - 1:
                time.sleep(self.delay)

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Simulation complete. Total time: {total_time:.1f}s")
        print(f"{'='*60}\n")


def main():
    """Command-line entry point for DICOM simulator."""
    parser = argparse.ArgumentParser(description='Simulate real-time DICOM acquisition')
    parser.add_argument('--source', '-s', required=True, help='Source DICOM directory')
    parser.add_argument('--target', '-t', required=True, help='Target directory')
    parser.add_argument('--tr', type=float, default=None, help='TR in seconds')
    parser.add_argument('--speed', type=float, default=1.0, help='Speed multiplier')
    parser.add_argument('--n', type=int, default=None, dest='n_volumes', help='Number of volumes')
    parser.add_argument('--no-clear', action='store_true', help='Keep existing files')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    sim = DicomSimulator(
        source_dir=args.source,
        target_dir=args.target,
        tr=args.tr,
        speed=args.speed,
        clear_target=not args.no_clear
    )

    try:
        sim.run(n_volumes=args.n_volumes)
    except KeyboardInterrupt:
        print("\nSimulation interrupted")


if __name__ == '__main__':
    main()