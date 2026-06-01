# Multi-Session Studies

Neurofeedback protocols typically span multiple sessions over days or weeks. In
pineuro a study declares an ordered list of **sessions**, and each session has its
own localizers and feedback runs. A feedback run can **reuse masks** from an earlier
session, and per-session coregistration aligns those masks to each session's head
position.

## Defining Sessions

In the Study Editor, the **Study** tab has a **Sessions** strip (`Session 1`,
`Session 2`, … plus *Add Session* and *Duplicate Session*). The Experiment Design,
Localizer, and Feedback editors below it always apply to the **selected** session, so
each session can run completely different localizer/feedback tasks. *Duplicate Session*
copies the current session as a starting template.

The study's session **count** is the number of sessions you define. When you start a
participant, the session-selection dialog only lets you create up to that many
sessions, and creates them in order (`ses-01`, `ses-02`, …).

## Mask Reuse Across Sessions

Each session computes its own localizers. To reuse an earlier session's masks instead of
re-localizing, a **feedback run selects them directly**: the feedback editor's mask-source
dropdown lists every mask produced this session *and in earlier sessions*. Pick an earlier
session's mask and that run consumes it — no new localizer is needed in the current session
(the runtime skips re-localizing and simply coregisters the reused masks).

Because reuse is chosen per feedback run, a session can **mix** sources: one run on a network
localized fresh this session, another on a network reused from `ses-01`. A session whose
feedback runs all reuse earlier masks runs no localizer at all — it only collects a reference
scan and coregisters the reused masks.

## Session Workflow

### A session with fresh localizers

1. Localizer collection (resting-state or task) for each fresh localizer
2. Preprocessing + mask extraction (ICA or GLM) per fresh localizer
3. Initial reference scan + coregistration
4. Real-time reference scan + coregistration
5. Feedback runs

### A session that reuses earlier masks

1. Real-time reference scan collection (new head position)
2. Coregistration — align the reused (and any fresh) masks to the current reference
3. Feedback runs

Reused masks skip collection/preprocessing/extraction; any localizers defined in the
session still run theirs.

## Directory Structure

Multi-session data is organized by session:

```
base_dir/sub-01/
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

A feedback run can only reuse masks from an *earlier* session. The session-selection dialog
refuses to start a session whose reused-mask source session has not yet been run (its masks
must exist on disk first).

## Detecting Existing Sessions

The GUI automatically detects completed sessions for each participant by scanning for
`ses-XX/` directories. The session-selection dialog shows existing sessions with their
completion status and offers to create the next declared session.
