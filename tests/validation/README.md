# pineuro Validation Suite

**Scientific-correctness tests.** The regular suite in `tests/` is a *regression*
suite — it mocks ANTsPy / nilearn / subprocess and checks that functions run and
that refactors preserve behavior. It does **not** check that pineuro produces
*correct* output.

This suite does. It constructs synthetic data with **known ground truth**, runs
it through the **real** pipeline (real ANTsPy, real nilearn — no mocking of the
science), and asserts that pineuro recovers the ground truth **exactly**.

## Design principle

> **Test pineuro's deterministic workflow exactly; never assert on a third-party
> optimizer's recovery accuracy.**

Every metric recorded on pineuro-owned computation is exact: an equality (`==`),
a machine-precision bound (`< 1e-9` / `1e-12` for floating-point accumulation),
an integer-zero count, or an exact deterministic boolean invariant. There are
**no soft floors on pineuro's own math**. Exactly two metrics — both correlation
floors on ANTs' motion output, checking pineuro's `.par` column layout rather
than ANTs' accuracy — are not exact; they are named and justified under
[Tolerances](#tolerances).

Concretely:

- For deterministic operations (FD, DVARS, voxel-efficiency combine, ROI compute,
  task design regressors, GLM betas, connectivity, state persistence, mask
  threshold helpers): pick inputs where the expected output is computable by
  hand or by an orthogonal procedure that does not share pineuro's
  implementation, write it as a *named constant* in the test, and assert exact
  equality.
- For third-party optimizer wraps (ANTs registration, nilearn CanICA): the
  optimizer's recovery *accuracy* (how precisely ANTs converges, how well
  CanICA separates sources) is **not pineuro's** and is **never asserted**.
  Test only the deterministic, pineuro-owned behavior — output structure,
  geometry, `.par` column layout and sign, transform application — exactly.
  A phantom is designed so that pineuro's part of the result is deterministic
  (e.g. a 2-level brain box makes Otsu unambiguous; a monotonic motion ramp
  makes the `.par` column monotonic).
- Phantom defaults are **noise-free** (`background_noise_sd=0.0`,
  `ROISpec.noise_sd=0.0`). Tests that need noise (e.g.
  `voxel_efficiency_matches_batch_reference` and the pipeline test need non-degenerate
  per-voxel variance for the voxel-efficiency weighting to be well-conditioned) pass
  it explicitly and document the reason in their docstring.

## Running it

```bash
pytest -m validation                    # full suite (real ANTs + nilearn; ~40 s)
pytest -m "validation and not slow"      # fast numeric core only (~2 s)
pytest -m validation -k feedback         # one stage
pytest                                   # plain pytest SKIPS validation (opt-in)
```

`pyproject.toml` sets `addopts = "-m 'not validation'"`, so a plain `pytest`
excludes this suite. It is **opt-in locally** but runs on **every push/PR** via
the `validation` job in `.github/workflows/ci.yml` — a hard correctness gate.

## The report

Every run writes `tests/validation/_report/` (git-ignored):

- `validation_metrics.json` — every recorded metric (the durable CI artifact).
- `validation_report.md` / `.html` — the human-readable report, grouped
  **stage → test → metric**. Each test is a titled block: its purpose (the
  test's docstring, captured automatically) followed by every metric it
  measured, stated in plain English — what was measured, what a pass requires,
  and a per-metric explanation (the `description=` passed to `record`).
- `figures/summary.png` — per-stage pass/fail chart.

At the end of a `pytest -m validation` run the terminal also prints a one-line
pass/fail summary and the report path. CI uploads `_report/` as the
`validation-report` artifact (even on failure). Regenerate the report from the
JSON with `report.generate_from_json(...)`.

## Layout

```
tests/validation/
├── phantom/                          synthetic-data library (see below)
├── metrics.py                        recovery metrics + ResultRecorder
├── report.py                         metrics JSON -> markdown/HTML/figure
├── conftest.py                       the `record` fixture, environment pinning
│
│   Core math (exact / independent-reference)
├── test_numeric_core.py              HRF, task designs, smoother/normalizer, IncrementalGLM, voxel-efficiency combine
├── test_feedback_estimator.py        FD/DVARS/tSNR/voxel-efficiency hand-calculated; SVD reference; multi-mask recovery
├── test_connectivity_estimator.py    sliding-window Pearson on hand-calculated signals
├── test_qc_helpers.py                summarize_mask (voxels, volume_ml, centroid), _pick_slices
├── test_task_state_persistence.py    TaskState / RunInfo JSON round-trip
├── test_sham_randomization.py        randomization generation, load, condition lookup
├── test_simulator_roundtrip.py       file copy contract (data + sidecars)
├── test_tasks_lifecycle.py           plugin registration, parameter validation, error aggregation
├── test_cluster_threshold.py         iterative cluster_threshold_stat_map
├── test_script_generator.py          generated script parses and contains parameters
├── test_sham_replay.py               sham activation replay verbatim
├── test_mask_extraction_task.py      GLM top-N task mask: Dice == 1.0 exact
├── test_phantom.py                   self-tests for the phantom library
│
│   ANTs / CanICA wrappers — exact checks on pineuro's deterministic glue only
├── test_masking_utils.py             compute_mean / apply_mask exact; apply_xfm identity & integer-shift exact; compute_xfm structure
├── test_motion_correction.py         .par column layout + sign monotonicity; geometry exact; variance strictly reduced
├── test_brain_extraction.py          ANTs Otsu mask sandwiched exactly inside the brain box; scipy erosion exact
├── test_ica.py                       spatial_correlation exact (1e-9); run_ica output structure exact
└── test_full_pipeline.py             stages compose (exact geometry handoff) + feedback GLM == SVD reference (1e-9)
```

## The phantom library (`phantom/`)

A self-contained digital-phantom generator (numpy / scipy / nibabel only — it
does **not** import pineuro, so its ground truth is independent of the system
under test). Generalizes `analysis/weights_bug/murfi_synthetic/generate_synthetic.py`.

- `build_phantom(PhantomSpec)` → `(data4d, GroundTruth)` — a structural brain
  with embedded ROIs carrying a known HRF-convolved signal plus structured noise.
- `GroundTruth` / `ROIGroundTruth` — records exactly what was injected; save/load
  via `ground_truth.npz`.
- `signals.py` — HRF-convolved events / boxcar / sine generators.
- `geometry.py` — tissue compartments, ROIs, Gaussian blobs.
- `motion.py` — apply a known rigid-motion sequence; emit dcm2niix-style sidecars.
- `io.py` — write 4D / per-volume NIfTIs, JSON sidecars, BIDS events.

## Tolerances

Every metric on pineuro-owned computation is **exact**. Three patterns, the third
covering the suite's only two non-exact criteria:

- **Independent reference** (e.g. `glm_vs_lstsq_max_error < 1e-9`,
  `voxel_efficiency_vs_batch_reference_max_err < 1e-9`, `hrf_max_abs_error < 1e-9`,
  `end_to_end_glm_vs_reference_max_err < 1e-9`): the test reimplements the
  algorithm via an independent code path (LAPACK SVD vs Gentleman's QR,
  `scipy.stats.gamma` vs pineuro's own gamma), and pineuro must match to
  floating-point precision. The expected value is computed from first
  principles, not from pineuro's output.
- **Exact invariant** (e.g. `motion_par_reference_row_max == 0`,
  `erode_voxel_count_error == 0`, `task_mask_dice == 1`,
  `brain_extraction_background_leak == 0`, `apply_xfm_identity_error == 0`,
  `motion_par_translation_dominant_axis == 0`, `motion_variance_decreased == 1`):
  a structural fact — an integer-zero count, an exact equality, or a
  deterministic boolean — that a correct implementation must satisfy exactly.
- **Optimizer-robustness floor** (`motion_par_translation_ramp_corr >= 0.5`,
  `motion_par_rotation_ramp_corr >= 0.5`): the **only two** non-exact criteria in
  the suite, both in [test_motion_correction.py](test_motion_correction.py). Each
  checks that pineuro places a recovered motion parameter in the correct `.par`
  column with a consistent sign, by correlating that column against a monotonic
  injected ramp. An exact bound is not available here: ANTs' per-volume recovery
  is noisy and intermittently nondeterministic even seeded and single-threaded,
  and a strict per-bin monotonicity check flaked on it. A correlation over all
  volumes absorbs that jitter. The floor sits far below what a correct
  implementation produces — translation tracks its ramp near-perfectly
  (|r| ~ 1.0), and rotation, the noisier of the two, runs |r| ~ 0.66-0.83 across
  runs — and far above every failure mode it must catch: a column swap or a
  sign-scrambling bug leaves the column as near-zero noise, |r| ~ 0.
  Consistent with the design principle above, this asserts pineuro's column
  layout and sign, never ANTs' recovery accuracy: no threshold here tightens if
  ANTs converges better, and none loosens if it converges worse.

The numerical environment: `conftest.py` forces single-threaded ITK and BLAS
(so float-reduction order is stable for the 1e-9 reference checks), and
`antspyx`/`nilearn` are exact-pinned in `pyproject.toml`. The suite does **not**
require ANTs registration to be deterministic and makes **no test-only changes
to pineuro's `src/`**: every ANTs/CanICA test is designed to be robust to the
optimizer's run-to-run non-determinism — binned-mean monotonicity averages out
per-volume jitter, references are computed on the optimizer's *actual* output,
brain extraction relies on Otsu (a 2-level phantom makes it unambiguous). This
was verified by running the ANTs-dependent tests repeatedly with unseeded ANTs.

**Third-party optimizer accuracy is never asserted.** Where a test exercises
ANTs registration or CanICA, it checks only the deterministic, pineuro-owned
part — output structure, geometry, `.par` column layout and sign, transform
application — exactly. How precisely ANTs converges or how well CanICA
separates sources is the optimizer's concern, not pineuro's, and validating it
is explicitly out of scope. Phantoms are designed so that pineuro's part of the
result *is* deterministic (a 2-level brain box makes Otsu unambiguous; a
monotonic motion ramp makes the recovered `.par` column monotonic; a feedback
GLM is checked against an SVD reference on the *same* data the estimator
received, so motion-correction accuracy never enters the comparison).

## Out of scope

- **PsychoPy task rendering** — visual, not amenable to ground-truth checks. The
  computational task helpers (designs/HRF, smoothing, normalization, task
  plugin registration) *are* covered.
- **`mask_extraction_rest` end-to-end** — the full ICA→MNI-registration→atlas
  pipeline needs real MNI registration, which is not meaningful on a synthetic
  phantom. Its pineuro-owned core — `run_ica`'s CanICA invocation and output
  handling, and the `spatial_correlation` math — *is* validated by `test_ica.py`.
- **Third-party optimizer recovery accuracy** — how precisely ANTs registration
  converges or how well CanICA separates sources. These are properties of the
  third-party packages, not of pineuro; the suite tests pineuro's invocation of
  and post-processing on them, never their convergence quality.
- **Synthetic DICOM** — dcm2niix is too strict for robust synthetic-DICOM
  generation; the suite starts from NIfTI (every stage downstream of DICOM
  conversion operates on NIfTI anyway).
- **Real-time orchestration threading** (`run_feedback_task` coordinator) —
  the feedback *computation* is validated directly via
  `FeedbackEstimator.process_volume`; the simulator's *data round-trip*
  contract IS covered by `test_simulator_roundtrip.py`. The coordinator
  threading is covered by the regression suite's `test_simulator.py`.

## Adding an exact test (worked example)

For deterministic operations, the FD test in
[test_feedback_estimator.py](test_feedback_estimator.py) is the worked example:

```python
# Hand-calculated values derived from the FD definition (sum of |Δ| with
# rotations at HEAD_RADIUS_MM=50 mm arc length per degree), not from
# re-running pineuro's algorithm.
_FD_CASES = [
    ([0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], 1.0),                       # 1 mm along x
    ([0, 0, 0, 0, 0, 0], [3, -2, 1, 0, 0, 0], 6.0),                      # |3|+|2|+|1|
    ([0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], 50.0 * np.pi / 180.0),      # 1 deg of arc at 50 mm
    ...
]

@pytest.mark.parametrize("prev, curr, expected_fd", _FD_CASES)
def test_fd_hand_calculated(record, prev, curr, expected_fd):
    produced = pineuro_compute_fd(curr, prev_motion=np.asarray(prev))
    err = abs(produced - expected_fd)
    assert record(...) < 1e-12
```

The expected values are constants you can verify by reading the docstring.
A bug in pineuro (wrong head radius, dropped abs/sum, wrong axis indexing)
moves the produced value off the literal — the case fails. Discriminative
cases (e.g. `[10.0, 0.0], [1.0, 100.0] -> 10/1.1` for voxel-efficiency) yield
different numbers under common buggy variants (`1/rss` vs `1/sqrt(rss)`),
making the discrimination explicit.

## Adding a validation test

1. Build a phantom with known ground truth (`build_phantom`, or the helpers in
   `phantom/`).
2. Run the real pineuro function on it.
3. Compute a recovery metric (`metrics.py`).
4. `record(...)` the metric **and** `assert` it against a tolerance. Pass a
   `description=` explaining, in plain English, what the metric measures and why
   its threshold is the right bar — it is what makes the report self-explanatory
   (the test's docstring is captured automatically for test-level context).
5. Mark the test `@pytest.mark.validation`; add `@pytest.mark.slow` if it uses
   ANTs/nilearn or runs > ~10 s.
