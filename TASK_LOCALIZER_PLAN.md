# Plan: Task-Based Mask Extraction for pineuro

## Context

pineuro currently supports only **ICA-based mask extraction from resting-state data** (`mask_extraction_rest()`). This works by decomposing rest fMRI into spatial components via CanICA, correlating them with Yeo atlas networks, and thresholding the best-matching component per network.

For publication, we need a **task-based alternative**: run a GLM on task localizer data, compute contrast maps (e.g., "nback - rest"), and directly threshold those maps into binary masks. This is a well-established approach in the neurofeedback literature and is more targeted than ICA for specific task-relevant networks.

**Design decisions (per user):**
- **Direct thresholding** — no Yeo atlas matching; each contrast map becomes a mask directly
- **nilearn FirstLevelModel** — established GLM engine; handles HRF convolution, drift, AR(1) noise
- **Flexible contrasts** — full linear combination support (e.g., `"2*nback - rest - fixation"`)

## Implementation

### 1. Core function: `mask_extraction_task()` in `src/pineuro/mask_extraction.py`

Add after `mask_extraction_rest()` (~line 1504). Signature:

```python
def mask_extraction_task(
    participant_id: str,
    root_dir: Union[str, Path],
    input_file: Union[str, Path],         # Preprocessed 4D NIfTI
    mask_file: Union[str, Path],          # Brain mask
    events: Union[str, Path, pd.DataFrame],  # BIDS events (onset, duration, trial_type)
    contrasts: dict[str, str],            # {mask_name: contrast_expression}
    tr: float,
    threshold: int = 2000,
    stat_type: str = "t",                 # "t" or "z"
    smoothing_fwhm: Optional[float] = 5.0,
    high_pass: float = 0.01,
    drift_model: str = "cosine",
    hrf_model: str = "spm",
    overwrite: bool = False,
    qc_enabled: bool = True,
    qc_dir: Optional[Union[str, Path]] = None,
    progress_callback: Optional[Callable] = None,
) -> list[Path]:
```

**Two-step pipeline:**

**Step 1 — GLM Fit (`01_glm/`):**
- Load/validate events (DataFrame or .tsv path with `onset`, `duration`, `trial_type` columns)
- Validate that all condition names in contrast expressions appear in `events['trial_type']`
- Fit `nilearn.glm.first_level.FirstLevelModel` with `t_r`, `mask_img`, `smoothing_fwhm`, `high_pass`, `drift_model`, `hrf_model`, `noise_model='ar1'`
- For each `(mask_name, contrast_expr)`: call `model.compute_contrast()` → save stat map as `{participant}_{mask_name}_{stat_type}map.nii.gz`
- Save metadata JSON (`glm_info.json`) and design matrix TSV

**Step 2 — Threshold to Masks (`02_masks/`):**
- For each stat map: keep only positive voxels, threshold to top N (default 2000) using `np.argpartition`, save binary mask as `{participant}_{mask_name}_thr.nii.gz`
- Save unthresholded map as `{participant}_{mask_name}.nii.gz`
- Save `mask_selections.json` with metadata (contrast definition, peak stat, voxel count)

**Returns** list of `_thr.nii.gz` paths (same format as `mask_extraction_rest()`).

**Edge cases:**
- Contrast references unknown condition → `ValueError` listing available conditions
- Fewer than `threshold` positive voxels → keep all positive, log warning
- No positive voxels → log warning, save empty mask

**Metadata dataclass** `TaskMaskSelection` (analogous to existing `MaskSelection`):
- `mask_name`, `source_type="glm_contrast"`, `contrast_definition`, `stat_type`, `peak_stat`, `n_voxels_above_zero`, `n_voxels_thresholded`, `threshold_method="top_n"`

### 2. Config changes in `src/pineuro/session.py`

**Extend `MaskConfig`** (line 596) with new fields:
```python
localizer_type: str = "rest"          # "rest" or "task"
contrasts: dict[str, str] = field(default_factory=dict)
events_file: Optional[Path] = None
stat_type: str = "t"
hrf_model: str = "spm"
drift_model: str = "cosine"
threshold: int = 2000
```
Add `__post_init__` validation: `localizer_type` must be "rest" or "task"; contrasts required when "task".

**Update `run_mask_extraction()`** (line 1340): dispatch on `config.mask.localizer_type`:
- `"rest"` → existing `mask_extraction_rest()` call (unchanged)
- `"task"` → new `mask_extraction_task()` call

**Update `resolve_mask_source()`** (line 1012): currently hardcodes `"03_masks"`. Change to check both `"02_masks"` (task) and `"03_masks"` (rest):
```python
mask_dir = mask_extraction_dir / "03_masks"
if not mask_dir.exists():
    mask_dir = mask_extraction_dir / "02_masks"
```
Same for the source-visit branch and all `"03_masks"` references in `detect_existing_visits()` and error messages.

**Update `ref_file_path()`** callers: when `localizer_type="task"`, pass `task=config.scan.localizer_task_type` instead of defaulting to `"rest"`.

### 3. Public API in `src/pineuro/__init__.py`

- Add `mask_extraction_task` to import (line 62) and `__all__` list

### 4. QC report in `src/pineuro/qc.py`

Extend `generate_mask_extraction_report()` to auto-detect pipeline type:
- If `01_glm/` exists → GLM report sections:
  1. Design matrix heatmap (from saved TSV)
  2. Unthresholded contrast maps (continuous colormap overlay on BOLD ref)
  3. Thresholded mask overlays (reuse existing `plot_mask_overlay_tripanel()`)
  4. Summary table (mask name, contrast, peak stat, voxel count)
- If `01_ica/` exists → existing ICA report (unchanged)

### 5. GUI updates

**`src/pineuro/gui/screens/study_editor.py`:**
- Remove "coming soon" text at lines 945, 1481, 1538
- When `localizer_task_type != "rest"`: show contrast definition UI (table with mask name + contrast expression columns) and optional events file picker
- Update `_build_save_config()` to save new MaskConfig fields

**`src/pineuro/gui/components/pipeline_runner.py`:**
- Update `_run_mask_extraction()` to dispatch based on `localizer_task_type`

**`src/pineuro/gui/screens/participant_workflow.py`:**
- Update `_get_mask_extraction_step_info()` to show GLM-specific details when task-based

### 6. Script generator in `src/pineuro/script_generator.py`

Add `generate_task_mask_extraction_script()` following the existing pattern for reproducible standalone scripts.

## Output Directory Structure

```
{root}/
├── 01_glm/
│   ├── {participant}_glm_info.json
│   ├── {participant}_design_matrix.tsv
│   └── {participant}_{mask_name}_tmap.nii.gz   (per contrast)
├── 02_masks/
│   ├── {participant}_{mask_name}.nii.gz         (unthresholded)
│   ├── {participant}_{mask_name}_thr.nii.gz     (binary mask)
│   └── mask_selections.json
└── qc/
    └── mask_extraction_qc.html
```

Downstream `coregister_2vol()` works unchanged — it just takes `_thr.nii.gz` files.

## Files to Modify (in order)

| # | File | Change |
|---|------|--------|
| 1 | `src/pineuro/mask_extraction.py` | Add `TaskMaskSelection` dataclass + `mask_extraction_task()` function |
| 2 | `src/pineuro/session.py` | Extend `MaskConfig`, update `run_mask_extraction()`, fix `resolve_mask_source()` |
| 3 | `src/pineuro/__init__.py` | Export `mask_extraction_task` |
| 4 | `src/pineuro/qc.py` | Extend report for GLM-based sections |
| 5 | `src/pineuro/gui/screens/study_editor.py` | Remove "coming soon", add contrast config UI |
| 6 | `src/pineuro/gui/components/pipeline_runner.py` | Dispatch mask extraction by localizer type |
| 7 | `src/pineuro/gui/screens/participant_workflow.py` | Update step info for task-based |
| 8 | `src/pineuro/script_generator.py` | Add task-based script generation |

## Verification

1. **Unit test**: Call `mask_extraction_task()` with a small synthetic 4D NIfTI, brain mask, events DataFrame, and a simple contrast. Verify output masks exist, are binary, have correct shape, and contain ~2000 voxels.
2. **Integration test**: Full pipeline `preprocessing() → mask_extraction_task() → coregister_2vol()` with simulated task data. Verify masks end up in feedback-ready space.
3. **Edge cases**: Empty events (should error), invalid contrast (should error with helpful message), fewer positive voxels than threshold (should keep all positive).
4. **Regression**: Ensure `mask_extraction_rest()` is untouched and existing pipeline still works.
5. **GUI**: Verify study editor saves/loads task-based config correctly, pipeline runner dispatches correctly.
