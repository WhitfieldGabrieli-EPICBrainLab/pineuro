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
| `threshold` | 2000 | Target voxels per mask (the top N voxels of the selected component) |

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

**`cluster_extent`** — Classic cluster-extent thresholding:
1. Keep voxels with t at or above a fixed cluster-forming threshold (`cluster_start_threshold`, default 2.0), optionally only within grey matter ("Restrict to grey matter")
2. Group them into clusters of face-connected voxels
3. Drop clusters smaller than `min_cluster_size` (default 50 voxels)
4. If `max_total_voxels` is set and the surviving clusters are larger, keep only their highest-t voxels up to that cap

The threshold is not lowered automatically. If no cluster reaches the minimum size, the mask is empty; the review below shows this and offers a re-threshold or a fallback mask.

### Reviewing Task Masks

The **Review & Approve Masks** window shows each task mask's overlay, voxel count (read from the mask file) and cluster table. Each mask also offers:

- **Adjust threshold** — edit the localizer's own thresholding parameters for this mask (cluster-forming threshold, minimum cluster size and cap for `cluster_extent`; voxel count for `top_n`; target voxels and start threshold for `cluster`). **Preview** recomputes the mask from the saved GLM map without refitting the GLM and shows it in the overlay with the voxel and cluster counts; **Apply** replaces the mask. While a preview is showing but not applied, Approve is disabled; **Discard preview** (or closing **Adjust threshold**) drops it and shows the current mask again. **Reset to study parameters** undoes it, along with any excluded clusters.
- **Exclude clusters** — leave whole clusters out of the mask, for example one in a venous sinus. Tick clusters in the **Clusters** table; each row's view button shows that cluster alone in the overlay, which is the way to check what a row covers, since the Region column only names the cortical label the cluster overlaps most. **Preview without selected** shows the remaining mask without changing it, and **Exclude selected** applies it. The mask becomes the reviewed mask minus those clusters: no voxels are added back to reach a cap or target, and at least one cluster must stay. Further exclusions add to earlier ones and **Restore excluded clusters** undoes them, keeping any adjusted threshold. Applying a new threshold brings excluded clusters back, because a lower threshold can merge a cluster with its neighbours, so re-exclude from the new table. The table lists up to the 50 largest clusters, and only those can be excluded; a last row gives the voxels in any smaller clusters not listed.
- **Use fallback** — when the study configures a fallback mask for this contrast (below), replace the mask with it.
- **Replace…** — replace the mask, for this participant only, with an atlas network, an MNI-space mask file, or a mask file in this participant's space (on the localizer's BOLD reference grid, such as a copy of the mask edited by hand in a viewer).

An atlas network or MNI-space file is warped to the participant's space with the boldref's SyN transform; a participant-space file is used as is, and a file on any other grid is refused. A participant-space file is copied into the localizer's `mask_edits/` folder and the record points at the copy, so it stays valid if the original is edited again or moved. The replacement is saved under the contrast's mask name, so feedback uses it without any manual file changes. A mask with 0 voxels is flagged and cannot be approved, since feedback cannot run with it. Any change revokes approval and deletes the coregistered masks, so coregistration runs again. Masks cannot be changed once a feedback run of the session has been completed.

### Fallback Masks

On the Localizer tab of a task localizer, **Fallback Masks (optional)** sets, per contrast, an MNI-space mask the operator can apply at review:

```yaml
extraction:
  fallback_masks:
    rumination_network: {atlas: yeo_17, network: DefaultA}   # an atlas network
    # or: rumination_network: {file: /path/to/rumination_mni.nii.gz}
```

The whole network (or the binarized file) is used, as with the ICA pipeline's Yeo fallback. The session-wide **Use MNI Atlas Masks** buttons are shown only when every localizer is a rest localizer, because they produce masks named after atlas networks rather than contrasts.

### Records and Reproduction

`03_masks/mask_selections.json` records, per mask, how it was made: the thresholding parameters used (`threshold_params`), the clusters found before the size filter (`n_clusters_input`, `largest_cluster_input`), and for a changed mask its `selection_reason` (`operator_rethreshold`, `operator_exclude_clusters`, `fallback_configured`, `operator_replace` or `atlas_roi`), its `source_type` (`atlas_network`, `mni_file`, `native_file` or `atlas_roi`) and `source` (e.g. `yeo_17:DefaultA` or the file used). Excluded clusters are listed in `excluded_clusters` (size, peak value and peak voxel) and in the QC report, and they are saved by peak voxel; if a re-run refits the GLM and a saved peak is no longer in the mask, it is ignored with a warning and the cluster shows in the review again. The operator's changes are also saved in `mask_overrides.json` next to `reproduce_masks.py`, and the script's `PARAMS` carry them as `mask_overrides`, so replaying the script on a clean tree recomputes the same masks. A re-run of mask extraction for the participant re-applies the saved changes.

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
