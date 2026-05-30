# Multi-Session Studies

Neurofeedback protocols typically span multiple sessions over days or weeks. In
pineuro a study declares an ordered list of **sessions**, and each session has its
own localizer and feedback tasks. A session may also **reuse masks** from an earlier
session — per localizer — and per-session coregistration aligns those masks to each
session's head position.

## Defining Sessions

In the Study Editor, the **Study** tab has a **Sessions** strip (`Session 1`,
`Session 2`, … plus *Add Session* and *Duplicate Session*). The Experiment Design,
Localizer, and Feedback editors below it always apply to the **selected** session, so
each session can run completely different localizer/feedback tasks. *Duplicate Session*
copies the current session as a starting template.

The study's session **count** is the number of sessions you define. When you start a
participant, the session-selection dialog only lets you create up to that many
sessions, and creates them in order (`ses-01`, `ses-02`, …).

## Per-Localizer Mask Reuse

Each localizer in a session has a **Masks** dropdown:

- **Compute fresh** — run this localizer's collection, preprocessing, and mask
  extraction this session (the default).
- **Reuse ses-NN** — skip this localizer this session and pull its masks from that
  earlier session, coregistering them to the current session's reference scan.

Because reuse is per-localizer, a session can **mix** the two: e.g. reuse the
resting-state network mask from `ses-01` while freshly localizing a new task ROI this
session. A session in which *every* localizer is reused runs no localizer at all — it
only collects a reference scan and coregisters the reused masks (the old
"whole-session reuse" behavior).

Feedback tasks still reference a localizer by its `mask_source` (the localizer's `id`),
so a feedback run consumes whichever masks that localizer provides — fresh or reused.

## Session Workflow

### A session with fresh localizers

1. Localizer collection (resting-state or task) for each fresh localizer
2. Preprocessing + mask extraction (ICA or GLM) per fresh localizer
3. Initial reference scan + coregistration
4. Real-time reference scan + coregistration
5. Feedback runs

### A session that reuses an earlier session's masks

1. Real-time reference scan collection (new head position)
2. Coregistration — align the reused (and any fresh) masks to the current reference
3. Feedback runs

Reused localizers skip their collection/preprocessing/extraction steps; fresh
localizers in the same session still run theirs.

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
│   ├── collection/         # reference (+ any fresh-localizer) DICOMs
│   ├── preprocessing/      # any fresh-localizer preprocessing
│   ├── mask_extraction/    # any freshly-extracted masks
│   └── feedback/           # feedback; 01_reference/masks holds the coregistered set
└── ses-03/
    └── ...
```

## Coregistration Between Sessions

Head position changes between sessions. The reference scan at the start of each session
captures the new head position. pineuro computes a rigid registration from each
localizer's BOLD reference (the current session's for fresh localizers, the source
session's for reused ones) to the current reference scan, then applies the transform to
that localizer's masks. All masks for a session therefore end up in the same reference
space, so the same functional regions are targeted across sessions.

## Validation

The Study Editor refuses to save a study where a localizer reuses a session that is not
*earlier* in the order. The session-selection dialog additionally refuses to start a
session whose reused-mask source session has not yet been run (its masks must exist on
disk first).

## Detecting Existing Sessions

The GUI automatically detects completed sessions for each participant by scanning for
`ses-XX/` directories. The session-selection dialog shows existing sessions with their
completion status and offers to create the next declared session.
