# Multi-Session Studies

Neurofeedback protocols typically span multiple sessions over days or weeks. pineuro supports multi-session studies with mask reuse and per-session coregistration.

## Enabling Multi-Session Mode

In the Study Editor, check **"Enable Multi-Session"** in the study settings. When starting a participant workflow, you'll be prompted to select or create a session.

## Session Workflow

### First Session (Compute Masks)

The first session runs the full pipeline:
1. Localizer collection (resting-state or task)
2. Preprocessing
3. Mask extraction (ICA or GLM)
4. Reference scan collection
5. Mask coregistration to reference space
6. Feedback runs

### Follow-Up Sessions (Reuse Masks)

Subsequent sessions can reuse masks from a previous session:
1. Reference scan collection (new head position)
2. Mask coregistration (align previous masks to current reference scan)
3. Feedback runs

This skips the localizer, preprocessing, and mask extraction steps entirely — saving significant time.

## Directory Structure

Multi-session data is organized by session:

```
base_dir/sub-001/
├── ses-01/
│   ├── collection/
│   ├── preprocessing/
│   ├── mask_extraction/
│   └── feedback/
├── ses-02/
│   ├── collection/         # Only reference + feedback DICOMs
│   ├── preprocessing/      # Only reference preprocessing
│   └── feedback/           # Feedback with reused masks
└── ses-03/
    └── ...
```

## Session Configuration

When creating a new session, you specify:

| Parameter | Description |
|-----------|-------------|
| Session ID | Auto-incremented (ses-01, ses-02, ...) or custom |
| Mask Source | `compute` (run full extraction) or `reuse` |
| Mask Source Session | Which session to reuse masks from (e.g., ses-01) |
| Localizer Enabled | Whether to run the localizer in this session |

## Coregistration Between Sessions

Head position changes between sessions. The reference scan at the start of each session captures the new head position. pineuro computes a rigid registration between the original mask reference and the new reference scan, then applies the transform to all masks.

This ensures the same functional regions are targeted across sessions, even with different head positions.

## Migrating from Single-Session

If you started with a single-session participant and later enable multi-session mode, pineuro can detect the existing single-session data layout and migrate it to the multi-session directory structure (moving data into a `ses-01/` subdirectory).

## Detecting Existing Sessions

The GUI automatically detects completed sessions for each participant by scanning for `ses-XX/` directories. The session selection dialog shows:
- Existing sessions with their completion status
- Option to create a new session
- Option to reuse masks from a completed session
