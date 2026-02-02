# MURFI vs pineuro Discrepancy Analysis

## Current Status

**Latest comparison results:**
- ContA vs cen: Correlation 0.990, **Slope 0.929** (7% lower than MURFI)
- DefaultA vs dmn: Correlation 0.993, **Slope 0.971** (3% lower than MURFI)

## Issues Fixed

### 1. RSS Weight Factor (FIXED - Major Impact)
**File:** `src/pineuro/feedback.py:640`

**Problem:** pineuro accumulated RSS with the Gentleman algorithm weight factor:
```python
self._rss += weight * (transformed_y ** 2)
```

**Root cause:** MURFI passes `weight=1.0` to `include()` and the d[] array accumulates, making effective weight ≈1.0. pineuro was using a decaying weight.

**Fix:**
```python
self._rss += transformed_y ** 2
```

**Impact:** Slopes improved from 1.22/1.39 → 0.93/0.97

### 2. Task Regressor Offset (FIXED - Previously)
**File:** `src/pineuro/utils.py:586`

**Problem:** 1-volume offset in HRF convolution onset.

**Fix:** `onset_hires = n_baseline * murfi_downsample_factor` (was `n_baseline - 1`)

## Ruled Out

### Design Matrix Differences
**Status:** ✅ Ruled out as cause

Comparison of design matrices shows:
- Intercept: **Exact match**
- Drift: **Exact match**
- Motion parameters (all 6): **Exact match**
- Task regressor: Max difference 0.00015 (4th decimal place)

The tiny HRF numerical differences cannot explain 3-7% slope discrepancy.

### Voxel Indexing
**Status:** ✅ Resolved

MURFI uses Fortran-order (column-major) voxel indexing. Once corrected, voxel matching works correctly.

### Per-Voxel RSS Values
**Status:** ✅ Ruled out as cause (matches exactly after RSS fix)

After fixing the RSS weight factor, we compared per-voxel RSS values between implementations using algo dumps:

| Volume | pineuro RSS delta | MURFI RSS delta | Ratio |
|--------|------------------|-----------------|-------|
| 9      | 1017545.67       | 1017545.67      | 1.000 |
| 10     | 3076193.00       | 3076193.00      | 1.000 |
| 11     | 755913.75        | 755913.75       | 1.000 |
| 12     | 728618.50        | 728618.50       | 1.000 |

**Conclusion:** The core incremental GLM algorithm (Givens rotations, beta estimation, RSS accumulation) now produces **identical results** at the per-voxel level. The remaining 3-7% discrepancy must occur **after** the GLM computation, in the ROI averaging/normalization step.

### Core GLM Algorithm
**Status:** ✅ Ruled out as cause

Since per-voxel RSS values match exactly, the following are confirmed working correctly:
- Gentleman's algorithm (AS 75.1) implementation
- Givens rotation computation
- Beta coefficient estimation
- RSS accumulation
- Design matrix row processing

### Per-Voxel Activation Formula
**Status:** ✅ Ruled out as cause (formulas match)

Analyzed MURFI source code (`RtCurrentActivation.cpp`):
```cpp
// MURFI: Remove nuisance, normalize by deviation
double err = y;
for (each nuisance regressor j) {
    err -= beta[j] * X[j];
}
double dev = sqrt(residual_mse / timepoint);
activation = err / dev;
```

pineuro (`feedback.py:562-587`):
```python
# Remove nuisance regressors
error = y - (nuisance_X @ nuisance_beta)
# Normalize
deviation = sqrt(rss / timepoint)
activation = error / deviation
```

**Conclusion:** Identical formulas.

### ROI Weighting Method
**Status:** ✅ Ruled out as cause (methods match)

MURFI `roi-combine` with `weighted-ave` method (`RtRoiCombine.cpp`):
```cpp
weightedSum += weight * pix;  // weight = residual_mse per voxel
summedWeights += weight;
result = weightedSum / summedWeights;
```

pineuro (`feedback.py:1212-1224`):
```python
weights = mse / sum(mse)  # Normalize to sum=1
weighted_mean = sum(weights * activation)
```

Both compute: `sum(activation_i * mse_i) / sum(mse_i)`

**Conclusion:** Mathematically equivalent.

### Steady-State Freezing
**Status:** ✅ Ruled out as cause (timing matches)

MURFI (`RtCurrentActivation.cpp`):
- Uses `if (timepoint < numDataPointsForErrEst)` to determine freeze
- At timepoint=25 with n_baseline=25, uses residual from timepoint=24

pineuro (`feedback.py:1339`):
- Freezes when `n_volumes == n_baseline - 1` (i.e., at volume 24)
- Both use timepoint 24 in denominator after freeze

**Conclusion:** Identical freeze timing.

### Single-Voxel Test
**Status:** ✅ PERFECT MATCH confirms per-voxel activation is identical

With single-voxel masks:
```
ContA vs cen:
  Correlation: 1.000000
  Slope:       0.999994
  Intercept:   0.000051

DefaultA vs dmn:
  Correlation: 1.000000
  Slope:       1.000002
  Intercept:   0.000030
```

**Conclusion:** Per-voxel activation is identical. The 3-7% discrepancy with 5-voxel masks occurs in the **ROI weighting/averaging step**.

## Root Cause Analysis

Since per-voxel activation matches perfectly but multi-voxel ROI averaging differs by 3-7%, the issue must be in how weights are computed or applied:

### CONFIRMED: Dynamic vs Frozen Weights

**MURFI** (`RtRoiCombine.cpp`):
```cpp
// Retrieved FRESH at each timepoint via msg->getData()
weightedAve = (RtActivation*) msg->getData(weightsModuleID,
                                           weightsDataName,
                                           inputRoiID);
```
- Gets weights from `incremental-glm`'s `residual_mse-img` at **each timepoint**
- Weights change as RSS accumulates throughout the session

**pineuro** (`feedback.py:1339-1345`):
- Freezes weights at `n_volumes == n_baseline - 1` (volume 24)
- Uses **frozen** weights for all subsequent volumes

**Impact:**
- MURFI: `weight_i = current_rss_i` (changes every volume)
- pineuro: `weight_i = frozen_rss_i` (fixed after baseline)

As RSS grows throughout the session, the relative weights between voxels shift, causing the weighted average to diverge by 3-7%.

## Fix Applied

**Changed pineuro to use dynamic weights** (`feedback.py:1347-1367`):

```python
# Get current RSS for dynamic weighting (matches MURFI behavior)
current_rss = self._glm.get_rss()

# Dynamic weights using current RSS (matches MURFI roi-combine)
mask_rss = current_rss[idx]
weighted_mean = float(np.sum(mask_rss * mask_values) / np.sum(mask_rss))
```

Added `IncrementalGLM.get_rss()` method to return current (not frozen) RSS values.

**Expected result:** Slope should now be ~1.0 with multi-voxel masks.
