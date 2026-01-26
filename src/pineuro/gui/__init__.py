"""
pineuro GUI - Flet-based graphical interface for neurofeedback research.

This package provides a two-tier GUI system:
1. Study Configuration Manager - Create/edit/load reusable study protocols
2. Participant Workflow Dashboard - Execute and monitor pipeline steps per participant
"""

from pathlib import Path

# Version info
__version__ = "0.1.0"

# Default study storage location
STUDIES_DIR = Path.home() / ".pineuro" / "studies"

__all__ = ["STUDIES_DIR"]
