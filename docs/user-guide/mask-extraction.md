# Mask Extraction

Mask extraction identifies participant-specific brain regions for neurofeedback. pineuro supports three approaches: ICA-based extraction from resting-state data, GLM-based extraction from task localizers, and manual ROI import.

## ICA-Based Extraction (Resting-State)

The default approach for resting-state localizers. Uses Independent Component Analysis to find participant-specific functional networks, then matches them to standard atlas labels.

### How It Works

1. **ICA Decomposition** — nilearn's CanICA extracts spatially independent components from the preprocessed resting-state data
2. **MNI Registration** — ANTs SyN registers the participant's BOLD reference to the MNI152 template, bringing atlas labels into subject space
3. **Network Matching** — Each ICA component is spatially correlated with each atlas network. The best-matching component is selected per target network.
4. **Thresholding** — The selected component is thresholded to create a binary mask with a target number of voxels

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 30 | Number of ICA components to extract |
| `smooth_fwhm` | 5.0 | Spatial smoothing (mm) before ICA |
| `highpass` | 0.01 | High-pass filter cutoff (Hz) |
| `lowpass` | None | Low-pass filter cutoff (Hz, disabled by default) |
| `threshold` | 2000 | Target voxels per mask |
| `threshold_method` | `top_n` | `top_n` or `cluster` (see below) |

### Output Structure

```
mask_extraction/
├── 01_ica/
│   └── sub-01_task-rest_preproc_components.nii.gz   # 4D ICA components
├── 02_mnireg/
│   ├── sub-01_task-rest_boldref_to_MNI_boldref.nii.gz   # Forward transform
│   └── Yeo_to_sub-01_task-rest_boldref.nii.gz            # Atlas in subject space
├── 03_masks/
│   ├── sub-01_Cont.nii.gz         # Full component map
│   ├── sub-01_Cont_thr.nii.gz     # Thresholded binary mask
│   ├── sub-01_Default.nii.gz
│   ├── sub-01_Default_thr.nii.gz
│   └── mask_selections.json         # Component → network mapping metadata
└── 05_tissue/                        # WM and CSF masks (for nuisance regression)
    ├── sub-01_tissue_wm_mask.nii.gz
    └── sub-01_tissue_csf_mask.nii.gz
```

## GLM-Based Extraction (Task Localizer)

When using a task-based localizer (e.g., SelfOther voice localizer), masks are derived from statistical contrasts rather than ICA.

### How It Works

1. **GLM Fit** — A design matrix with task conditions is fit to the localizer data
2. **Contrast Map** — A t-statistic map is computed for the specified contrast (e.g., self > other)
3. **Thresholding** — The contrast map is thresholded to extract the target ROI

### Threshold Methods

**`top_n`** — Select the N voxels with the highest t-statistics. Simple and predictable.

**`cluster`** — Atlas-guided iterative thresholding:
1. Start at a low t-threshold (default: 2.0)
2. Restrict to an anatomical ROI from the atlas
3. Find the largest connected cluster
4. If cluster > target size, increase threshold by `cluster_step` (default: 0.1)
5. Repeat until cluster approximates target size

Cluster thresholding produces more anatomically coherent masks compared to top_n.

## Manual ROI Import

For pre-defined ROIs (e.g., published coordinates, hand-drawn masks, or masks from another pipeline).

### Supported Input Spaces

| Space | Description | What pineuro does |
|-------|-------------|-------------------|
| `native` | Already in functional/EPI space | Validate, threshold, save directly |
| `anatomical` | In T1 structural space | ANTs rigid registration to EPI → resample |
| `mni` | In MNI152 space | ANTs SyN registration to EPI → resample |

### Configuration

Set the localizer type to `manual` in the Study Editor. Then in the Participant Workflow, you'll be prompted to provide:
- Mask NIfTI file(s)
- BOLD reference (for native space)
- T1 structural (for anatomical space)
- Mask space selection

The preprocessing and localizer steps are automatically skipped when using manual masks.

## Atlas Options

pineuro supports multiple atlases for network identification:

| Atlas | Key | Networks/Regions |
|-------|-----|------------------|
| Yeo 7-network | `yeo_7` | Vis, SomMot, DorsAttn, SalVentAttn, Limbic, Cont, Default |
| Yeo 17-network | `yeo_17` | ContA/B, DefaultA/B, DorsAttnA/B, SalVentAttnA/B, LimbicA/B, SomMotA/B, TempPar, VisA/B/C |
| Schaefer 100 (7-net) | `schaefer_100_7net` | 100 parcels grouped by 7 networks |
| Schaefer 100 (17-net) | `schaefer_100_17net` | 100 parcels grouped by 17 networks |
| Schaefer 400 (7-net) | `schaefer_400_7net` | 400 parcels grouped by 7 networks |
| Schaefer 400 (17-net) | `schaefer_400_17net` | 400 parcels grouped by 17 networks |
| Harvard-Oxford | `harvard_oxford` | Anatomical cortical regions |
| AAL (SPM12) | `aal` | Automated Anatomical Labeling regions |
| Gray Matter | `gray_matter` | Whole-brain gray matter (no region selection) |

Atlas labels are fetched on demand and cached locally.

## QC Review

After mask extraction, review the QC report (`qc/sub-01_mask_extraction_qc.html`) which shows:
- ICA component spatial maps
- Correlation heatmap (components x atlas networks)
- Mask overlays on the participant's anatomy
- Alternative component selections (if the auto-selected component isn't ideal, you can override)

The Participant Workflow GUI also shows overlay images for each mask, allowing visual verification before proceeding to feedback.

## Tissue Masks

When WM and/or CSF nuisance regressors are enabled in the feedback configuration, tissue masks are created automatically during mask extraction. They come from **template probability priors**, not from a segmentation of the participant's own anatomy:

1. **Fetch priors** — the `MNI152NLin2009cAsym` res-2 WM and CSF probability maps are pulled from templateflow.
2. **Warp to subject space** — each prior is resampled from MNI into the participant's functional space using the inverse ANTs SyN transform. The transform computed during mask extraction is reused when available; otherwise a fresh MNI registration is run.
3. **Threshold** — voxels with prior probability >= 0.9 are kept, giving conservative masks well inside each tissue compartment.
4. **Erode** — 2 voxels, falling back to 1, then to no erosion, whichever first leaves a non-empty mask. CSF is thin at BOLD resolution and does not always survive aggressive erosion; the fallback is logged when it is used.

An empty mask after thresholding raises an error, since it usually indicates poor MNI registration of the BOLD reference.
