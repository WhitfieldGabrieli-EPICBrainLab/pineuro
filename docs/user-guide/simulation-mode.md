# Simulation Mode

Simulation mode replays pre-collected DICOMs at configurable speed, allowing you to test the complete pipeline without a scanner. The simulator uses deadline-based timing to maintain accurate TR intervals regardless of copy duration.

## Enabling Simulation

In the Study Editor Scan tab:
1. Check **"Enable Simulator"**
2. Set **Collection Speed** — Speed multiplier for localizer/preprocessing (e.g., `10.0` = 10x faster)
3. Set **Feedback Simulator Speed** — Speed multiplier for feedback runs (e.g., `1.0` = real-time)
4. Provide DICOM source paths (see below)

## Source Mapping File

The recommended way to specify DICOM sources for multiple participants. Create a TSV or CSV file with one row per participant:

```
subject	rest1	rest2	reference_initial	reference_realtime	feedback_run-01	feedback_run-02	transfer_run-01	transfer_run-02
sub-001	/data/dicoms/sub-001/rest1	/data/dicoms/sub-001/rest2	/data/dicoms/sub-001/reference_init	/data/dicoms/sub-001/reference_rt	/data/dicoms/sub-001/feedback1	/data/dicoms/sub-001/feedback2	/data/dicoms/sub-001/transfer1	/data/dicoms/sub-001/transfer2
sub-002	/data/dicoms/sub-002/rest1	...
```

### Column Names

| Column | Maps To |
|--------|---------|
| `subject` | Participant ID entered in the GUI |
| `rest1`, `rest2` | Localizer runs 1 and 2 |
| `reference_initial` | Initial reference scan (typically 2-4 volumes) |
| `reference_realtime` | Real-time reference scan (typically 2-4 volumes) |
| `feedback_run-01`, `feedback_run-02` | Feedback run DICOMs |
| `transfer_run-01`, `transfer_run-02` | Transfer run DICOMs |

Only include columns for the scans your study uses. For example, if you have 1 localizer run and 1 feedback run with no transfer runs:

```
subject	rest1	reference_realtime	feedback_run-01
sub-001	/data/rest	/data/reference	/data/feedback
```

Set the source mapping path in the Study Editor Scan tab → **Source Mapping**.

## Per-Run DICOM Sources

Alternatively, specify DICOM directories directly in the Study Editor for each run type:
- **Localizer Source DICOMs** — One path per localizer run
- **Feedback Source DICOMs** — One path per feedback run
- **Transfer Source DICOMs** — One path per transfer run
- **Reference Initial/Realtime DICOMs** — One path each

This approach doesn't support per-participant mapping, so it's best for single-participant testing.

## Timing Model

The simulator uses deadline-based timing rather than fixed delays:

```
Effective interval = TR / speed

Volume 0: copied immediately
Volume 1: deadline = start + 1 × (TR / speed)
Volume 2: deadline = start + 2 × (TR / speed)
...
```

If copying a volume takes longer than expected, the next volume's deadline is still computed from the session start time. This prevents cumulative drift and maintains accurate timing over long runs.

**Example:** TR = 2.0s, speed = 10x → effective interval = 0.2s per volume. A 412-volume run completes in ~82 seconds.

## Speed Recommendations

| Use Case | Collection Speed | Feedback Speed |
|----------|-----------------|----------------|
| Quick pipeline test | 10-20x | 10-20x |
| Realistic feedback test | 10x | 1x (real-time) |
| Timing validation | 1x | 1x |

Set collection speed high (10x+) for preprocessing and mask extraction where timing doesn't matter. Set feedback speed to 1x when you want to verify the task display at realistic timing.

## Using Without the GUI

The simulator can also be used programmatically:

```python
from pineuro import VolumeSimulator

sim = VolumeSimulator(
    source_dir="/path/to/source_dicoms",
    target_dir="/tmp/incoming_volumes",
    tr=2.0,
    speed=10.0,
    clear_target=True,
)
sim.run(n_volumes=412)
```
