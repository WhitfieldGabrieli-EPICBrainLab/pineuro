# Sham & Blinding

pineuro supports sham/control conditions and operator blinding for controlled neurofeedback studies. Real activations are always computed and logged, regardless of whether the participant receives real or sham feedback.

## Enabling Sham

In the Study Editor, enable sham in the Task tab's sham section. This adds sham-related controls and integrates with the randomization system.

## Delivery Methods

### Alternative Masks

Sham participants receive feedback derived from a different brain network than the one being studied. For example, if the target is DMN, sham participants might receive feedback from motor cortex.

**Hemisphere Split Mode:** Uses left or right hemisphere of a specified network (e.g., right SomMotA). This produces feedback that looks statistically similar to real feedback but is not contingent on the target process.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sham_mask_mode` | `hemisphere_split` | `hemisphere_split` or `random_network` |
| `hemisphere_network` | `SomMotA` | Which network to use for hemisphere split |
| `target_hemisphere` | `right` | Which hemisphere (`left` or `right`) |
| `alternative_mask_targets` | — | Alternative network names for random_network mode |

### Replay

Sham participants receive a replay of another participant's activation timeseries. The feedback display looks identical to real feedback but is not driven by the current participant's brain activity.

Replay requires specifying a matched participant (whose data to replay) in the randomization file.

## Blind Operator Mode

When `blind_mode` is enabled, the operator GUI hides network names and shows generic labels:
- Network names replaced with "Target" / "Non-Target"
- The operator cannot tell whether the participant is in the real or sham condition
- Logging still records full details for post-hoc analysis

## Randomization File

A TSV file that assigns participants to conditions. Generate one using the Study Editor's "Generate Randomization" button, or create manually.

**Format:**

```
subject_id	condition	matched_participant	matched_visit	notes
sub-001	real
sub-002	sham	sub-001		Matched for age/sex
sub-003	real
sub-004	sham	sub-003
```

| Column | Required | Description |
|--------|----------|-------------|
| `subject_id` | Yes | Participant identifier |
| `condition` | Yes | `real` or `sham` |
| `matched_participant` | For replay | Whose data to replay |
| `matched_visit` | No | Which visit (defaults to same visit number) |
| `notes` | No | Free-text notes |

The randomization file is configured in the Study Editor sham section.

## How Logging Works

Regardless of condition, pineuro always computes and logs both:
- **Real activations** — What the participant's brain is actually doing
- **Delivered activations** — What the task display shows (same as real for real condition, sham-derived for sham condition)

The feedbacklog TSV contains separate columns for real and delivered values, enabling post-hoc analysis of both the actual neural signal and what was shown.

## Configuration Summary

| Parameter | GUI Location | Default |
|-----------|-------------|---------|
| Enable sham | Task tab → Sham | Off |
| Blind mode | Task tab → Sham | Off |
| Delivery method | Task tab → Sham | `alternative_masks` |
| Sham mask mode | Task tab → Sham | `hemisphere_split` |
| Hemisphere network | Task tab → Sham | `SomMotA` |
| Target hemisphere | Task tab → Sham | `right` |
| Randomization file | Task tab → Sham | — |
