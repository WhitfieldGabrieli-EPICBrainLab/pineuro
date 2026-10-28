# Sham & Blinding

pineuro supports sham/control conditions and operator blinding for controlled neurofeedback studies. Real activations are always computed and logged, regardless of whether the participant receives real or sham feedback.

## Enabling Sham

In the Study Editor's **Sham** tab, toggle sham on; blind mode and the randomization file are set there too. *How* a sham signal is built for a given task — the delivery method and alternative-mask mode below — is configured per feedback run on the **Feedback** tab.

## Delivery Methods

### Alternative Masks

Sham participants receive feedback derived from a different brain network than the one being studied. For example, if the target is DMN, sham participants might receive feedback from motor cortex.

**Hemisphere Split Mode:** Uses left or right hemisphere of a specified network (e.g., right SomMotA). This produces feedback that looks statistically similar to real feedback but is not contingent on the target process.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sham_mask_mode` | `hemisphere_split` | `hemisphere_split` or `two_networks` |
| `hemisphere_network` | `SomMotA` | Which network to use for hemisphere split |
| `target_hemisphere` | `right` | Which hemisphere (`left` or `right`) |
| `alternative_mask_targets` | — | Network names to use for `two_networks` mode |

### Replay

Sham participants receive a replay of another participant's activation timeseries. The feedback display looks identical to real feedback but is not driven by the current participant's brain activity.

Replay requires specifying a matched participant (whose data to replay) in the randomization file. A generated replay list pairs each sham participant with a different real participant scanned earlier, the earliest one not yet paired, so at most half of the list can be sham.

## Blind Operator Mode

When `blind_mode` is enabled, the operator GUI hides network names and shows generic labels:
- Network names replaced with "Target" / "Non-Target"
- The operator cannot tell whether the participant is in the real or sham condition
- Logging still records full details for post-hoc analysis

## Randomization File

A TSV file that assigns participants to conditions. Generate one using the Study Editor's "Generate Randomization" button, or create manually.

**Format:**

```
subject_id	condition	matched_participant	matched_session	notes
sub-01	real
sub-02	sham	sub-01		Matched for age/sex
sub-03	real
sub-04	sham	sub-03
```

| Column | Required | Description |
|--------|----------|-------------|
| `subject_id` | Yes | Participant identifier |
| `condition` | Yes | `real` or `sham` |
| `matched_participant` | For replay | Whose data to replay |
| `matched_session` | No | Which session (defaults to same session number) |
| `notes` | No | Free-text notes |
| `slot` | No | Makes the file a slot list (see below) |
| `matched_slot` | No | In a slot list, the slot whose participant is replayed |
| `assigned_at` | No | Written by pineuro when it assigns a slot |

The randomization file is configured in the Study Editor sham section.

When sham is enabled, membership in this file is enforced: the session start is hard-blocked if the randomization file is missing/unconfigured, or if the participant's ID is not listed in it. This prevents an unlisted participant from silently defaulting the whole study to real feedback. (A runtime guard backs up the dialog-time check.) A slot list is the exception, as described below.

### Slot Lists

A study can pre-allocate its randomization as numbered slots and assign participants to them in the order they are scanned. Give the file a `slot` column. A new list needs no `subject_id` column; pineuro adds it at the first assignment, and a blank `subject_id` marks an unused slot. Study IDs can then have gaps, for example from participants who withdraw before scanning, without unbalancing the list.

```
slot	subject_id	condition	matched_slot	assigned_at
001	sub-charms255	real		2026-10-30T09:12:04
002	sub-charms270	sham	001	2026-11-02T14:40:51
003		real
```

When a participant with no data starts their first session and is not yet listed, the Start dialog names the slot they will receive, and **Assign Slot and Start** writes their ID and an `assigned_at` time into the first unused slot in file order. The dialog never shows the condition. Slots are assigned only at a participant's first session, so an unlisted ID at a later session is still blocked as a likely typo, and a new participant is blocked once every slot is assigned. pineuro does not undo an assignment; correcting one means editing the file.

For replay, `matched_slot` names the earlier slot whose participant is replayed, and pineuro resolves it to the participant assigned to that slot. The **Generate Randomization** dialog creates a slot list when **Participant IDs** is set to **Slots**.

The file lists every slot's condition, so in blind mode keep it where operators do not open it.

## How Logging Works

Regardless of condition, pineuro always computes and logs both:
- **Real activations** — What the participant's brain is actually doing
- **Delivered activations** — What the task display shows (same as real for real condition, sham-derived for sham condition)

The feedbacklog TSV contains separate columns for real and delivered values, enabling post-hoc analysis of both the actual neural signal and what was shown.

## Configuration Summary

| Parameter | GUI Location | Default |
|-----------|-------------|---------|
| Enable sham | Sham tab | Off |
| Blind mode | Sham tab | Off |
| Delivery method | Feedback tab (per run) | `alternative_masks` |
| Sham mask mode | Feedback tab (per run) | `hemisphere_split` |
| Hemisphere network | Feedback tab (per run) | `SomMotA` |
| Target hemisphere | Feedback tab (per run) | `right` |
| Randomization file | Sham tab | — |
