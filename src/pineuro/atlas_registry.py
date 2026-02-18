"""Atlas source registry — lightweight metadata for GUI dropdowns.

This module is intentionally free of heavy dependencies (no nibabel, scipy,
nilearn, etc.) so that the GUI can build atlas-related dropdowns without
triggering multi-second import chains.

The actual atlas *fetching* and *label resolution* live in mask_extraction.py,
which imports ATLAS_SOURCES from here.
"""

from typing import Any, Dict

# ────────────────────────────────────────────────────────────────────
# Atlas source registry
# ────────────────────────────────────────────────────────────────────
ATLAS_SOURCES: Dict[str, Dict[str, Any]] = {
    # Network atlases (resting-state ICA pathway)
    "yeo_7":              {"display_name": "Yeo 7-network",              "type": "network", "n_networks": 7},
    "yeo_17":             {"display_name": "Yeo 17-network",             "type": "network", "n_networks": 17},
    "schaefer_100_7net":  {"display_name": "Schaefer 100 (7-network)",   "type": "network", "n_networks": 7,  "n_rois": 100},
    "schaefer_100_17net": {"display_name": "Schaefer 100 (17-network)",  "type": "network", "n_networks": 17, "n_rois": 100},
    "schaefer_400_7net":  {"display_name": "Schaefer 400 (7-network)",   "type": "network", "n_networks": 7,  "n_rois": 400},
    "schaefer_400_17net": {"display_name": "Schaefer 400 (17-network)",  "type": "network", "n_networks": 17, "n_rois": 400},
    # Anatomical/parcel atlases (cluster thresholding pathway)
    "harvard_oxford":     {"display_name": "Harvard-Oxford Cortical",    "type": "anatomical"},
    "aal":                {"display_name": "AAL (SPM12)",                "type": "anatomical"},
}
