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

The randomization file is configured in the Study Editor sham section.

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
