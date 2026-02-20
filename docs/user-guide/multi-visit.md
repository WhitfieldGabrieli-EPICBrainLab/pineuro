# Multi-Visit Studies

Neurofeedback protocols typically span multiple sessions over days or weeks. pineuro supports multi-visit studies with mask reuse and per-session coregistration.

## Enabling Multi-Visit Mode

In the Study Editor, check **"Enable Multi-Visit"** in the study settings. When starting a participant workflow, you'll be prompted to select or create a visit.

## Visit Workflow

### First Visit (Compute Masks)

The first visit runs the full pipeline:
1. Localizer collection (resting-state or task)
2. Preprocessing
3. Mask extraction (ICA or GLM)
4. 2-volume reference collection
5. Mask coregistration to 2vol space
6. Feedback runs

### Follow-Up Visits (Reuse Masks)

Subsequent visits can reuse masks from a previous visit:
1. 2-volume reference collection (new head position)
2. Mask coregistration (align previous masks to current 2vol)
3. Feedback runs

This skips the localizer, preprocessing, and mask extraction steps entirely — saving significant time.

## Directory Structure

Multi-visit data is organized by visit:

```
base_dir/sub-001/
├── ses-01/
│   ├── collection/
│   ├── preprocessing/
│   ├── mask_extraction/
│   └── feedback/
├── ses-02/
│   ├── collection/         # Only 2vol + feedback DICOMs
│   ├── preprocessing/      # Only 2vol preprocessing
│   └── feedback/           # Feedback with reused masks
└── ses-03/
    └── ...
```

## Visit Configuration

When creating a new visit, you specify:

| Parameter | Description |
|-----------|-------------|
| Visit ID | Auto-incremented (ses-01, ses-02, ...) or custom |
| Mask Source | `compute` (run full extraction) or `reuse` |
| Mask Source Visit | Which visit to reuse masks from (e.g., ses-01) |
| Localizer Enabled | Whether to run the localizer in this visit |

## Coregistration Between Visits

Head position changes between sessions. The 2-volume reference scan at the start of each visit captures the new head position. pineuro computes a rigid registration between the original mask reference and the new 2-volume reference, then applies the transform to all masks.

This ensures the same functional regions are targeted across visits, even with different head positions.

## Migrating from Single-Visit

If you started with a single-visit participant and later enable multi-visit mode, pineuro can detect the existing single-visit data layout and migrate it to the multi-visit directory structure (moving data into a `ses-01/` subdirectory).

## Detecting Existing Visits

The GUI automatically detects completed visits for each participant by scanning for `ses-XX/` directories. The visit selection dialog shows:
- Existing visits with their completion status
- Option to create a new visit
- Option to reuse masks from a completed visit
