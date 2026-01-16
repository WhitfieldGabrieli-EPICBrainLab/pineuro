# Performance Evaluation & Optimization Plan for pyrtnf Pipeline

## Implementation Status

**✅ Phase 1: Benchmarking (COMPLETED)**
**✅ Phase 2: Parallelization (COMPLETED)**
**✅ Phase 3: Algorithm Optimization (COMPLETED)**
**⏸️ Phase 4: Documentation & Testing (Ready for user testing)**

### Summary of Changes

**Timing Infrastructure:**
- Added `Timer` context manager for measuring execution times
- Integrated timing into all major operations (preprocessing, mask_extraction, registration)
- Created JSON/CSV timing report generation
- Added `--benchmark` flag to test.py for performance profiling

**Parallelization:**
- Created `parallel_run_command()` helper for multiprocessing
- Parallelized DICOM conversion across runs
- Parallelized motion correction (mcflirt) across runs
- Added `n_jobs` parameter to `preprocessing()` function (default=-1 uses all CPUs)

**Algorithm Optimizations:**
- Vectorized spatial correlation computation (2-3x speedup expected)
- Confirmed compression control available in dcm2niix wrapper

**Expected Performance Improvements:**
- DICOM conversion: ~2x speedup for 2-run workflow
- Motion correction: ~2x speedup for 2-run workflow
- Spatial correlation: ~2-3x speedup
- **Total pipeline: 25-35% reduction in preprocessing time**

---

## Overview
Evaluate computational time for each preprocessing and mask extraction step, identify bottlenecks, and implement optimizations to reduce total pipeline runtime.

## User Requirements
- **Goal**: Minimize total pipeline time (complete preprocessing + mask extraction)
- **Resources**: 8-16 CPU cores available for parallel processing
- **Typical data**: 2 resting-state runs per participant
- **Reporting**: Detailed timing reports (JSON/CSV) for analysis and debugging

## 1. Performance Benchmarking Implementation

### 1.1 Add Timing Infrastructure
**Goal**: Instrument existing code to measure execution time for each pipeline step

**Implementation**:
- Create a `Timer` context manager in [utils.py](../src/pyrtnf/utils.py)
- Add timing decorators/wrappers that log execution times
- Integrate timing into existing `run_command()` function (lines 193-288)
- Store timing results in structured format (JSON/CSV) for analysis

**What to measure**:
- DICOM collection (for reference only - constrained by scanner TR, not optimizable)
- DICOM to NIfTI conversion (per run)
- Motion correction with mcflirt (per run)
- Brain extraction with bet (one-time)
- Multi-run coregistration (per run):
  - ANTs registration (`compute_xfm()`)
  - Transform application (`apply_xfm()`)
  - Concatenation (`concat()`)
- ICA decomposition (`run_ica()`)
- MNI registration (`compute_xfm_to_mni()`)
- Spatial correlation (`spatial_correlation()`)
- Mask thresholding

### 1.2 Create Benchmark Script
**Goal**: Run complete workflow from [tests/test.py](../tests/test.py) with timing enabled

**Implementation**:
- Extend test.py to enable/disable timing via CLI flag: `--benchmark`
- Run on example data with timing enabled
- Generate timing report showing:
  - Per-step execution times
  - Percentage of total runtime
  - Cumulative times
- Save report to `benchmarks/timing_report.json` and readable summary

**Critical files**:
- [tests/test.py](../tests/test.py) - Main test workflow
- [src/pyrtnf/preprocessing.py](../src/pyrtnf/preprocessing.py) - Preprocessing pipeline
- [src/pyrtnf/mask_extraction.py](../src/pyrtnf/mask_extraction.py) - Mask extraction

---

## 2. Bottleneck Analysis

Based on codebase exploration, expected bottlenecks (ranked by impact):

**Important**: DICOM collection is constrained by scanner TR and is not included as an optimization target. The following analysis focuses on post-collection processing.

### Critical (60-80% of runtime):
1. **CanICA decomposition** - 60-70% of total time
   - Location: [mask_extraction.py:75-209](../src/pyrtnf/mask_extraction.py#L75-L209) `run_ica()`
   - Already optimized with `n_jobs=-1` (line 171)
   - Limited by FastICA algorithm convergence

2. **ANTs registration** - 20-25% of total time
   - Location: [utils.py:484-555](../src/pyrtnf/utils.py#L484-L555) `compute_xfm()`
   - Used for multi-run coregistration (preprocessing.py:438-453)
   - ~30-120s per registration, scales with number of runs

3. **FSL mcflirt** - 10-15% of total time
   - Location: [preprocessing.py:73-106](../src/pyrtnf/preprocessing.py#L73-L106) `motion_correction()`
   - Sequential processing in loop (lines 397-401)
   - ~30-60s per run

### Moderate (5-10% each):
4. **Spatial correlation** - Location: [mask_extraction.py:211-325](../src/pyrtnf/mask_extraction.py#L211-L325)
5. **DICOM conversion** - Sequential dcm2niix calls (preprocessing.py:382-390)
6. **Data I/O** - Loading/saving large NIfTI files

---

## 3. Optimization Strategy

### 3.1 High-Impact: Parallelize Sequential Loops

**Target 1: Motion Correction**
- **Current**: Sequential mcflirt calls (preprocessing.py:397-401)
- **Solution**: Use `multiprocessing.Pool` to run mcflirt in parallel
- **Expected speedup**: ~Nx for N runs (e.g., 2x for 2 runs)
- **Risk**: Low - runs are independent
- **Files**: [preprocessing.py](../src/pyrtnf/preprocessing.py)

**Target 2: Multi-Run Coregistration**
- **Current**: Sequential ANTs registration (preprocessing.py:438-453)
- **Reality for 2 runs**: Only 1 registration (run-02 → run-01), so NO parallelization benefit
- **Note**: This would only help with 3+ runs where multiple registrations could run in parallel
- **For 2-run workflow**: This is not an optimization target

**Target 3: DICOM Conversion**
- **Current**: Sequential dcm2niix calls (preprocessing.py:382-390)
- **Solution**: Parallelize dicom_to_nifti() across runs
- **Expected speedup**: ~Nx for N runs
- **Risk**: Very low - completely independent operations
- **Note**: DICOM collection itself cannot be sped up (constrained by scanner TR)
- **Files**: [preprocessing.py](../src/pyrtnf/preprocessing.py)

### 3.2 Medium-Impact: Algorithm Optimizations

**Target 4: Spatial Correlation Vectorization**
- **Current**: Loop over networks (mask_extraction.py:300-318)
- **Solution**: Vectorize to compute all network correlations at once
- **Expected speedup**: 2-3x for correlation step
- **Risk**: Low - mathematical equivalence maintained
- **Files**: [mask_extraction.py](../src/pyrtnf/mask_extraction.py)

**Target 5: Reduce DICOM Compression Overhead**
- **Current**: dcm2niix with compression `-z y` (utils.py:365)
- **Solution**: Add option for uncompressed output in real-time mode
- **Expected speedup**: 50-100ms per volume
- **Risk**: Low - optional parameter
- **Files**: [utils.py](../src/pyrtnf/utils.py)

### 3.3 Configuration Options

**Add parallelization controls**:
- `n_jobs` parameter for preprocessing() and mask_extraction()
- Default: `n_jobs=-1` (use all cores)
- Allow user to limit parallel jobs if needed

**Note on scalability**:
- With 2 runs: Parallelization benefits are limited (only motion correction + DICOM conversion)
- With 3+ runs: Additional benefit from parallelizing multiple coregistrations
- Current implementation will support both cases

---

## 4. Implementation Plan

### Phase 1: Benchmarking (Establish Baseline) ✅ **COMPLETED**
1. ✅ **COMPLETED** - Add `Timer` context manager to [utils.py](../src/pyrtnf/utils.py)
2. ✅ **COMPLETED** - Integrate timing into `run_command()` and key functions
3. ✅ **COMPLETED** - Add `--benchmark` flag to [test.py](../tests/test.py)
4. ✅ **COMPLETED** - Added timing to all major functions (CanICA, spatial correlation, ANTs registration, FSL commands)
5. ✅ **COMPLETED** - Implemented JSON/CSV timing report generation

**Implementation Notes:**
- Created `Timer` context manager with enable/disable functionality
- Integrated timing into `run_command()` with `timing_category` parameter
- Added timing to preprocessing functions (mcflirt, bet, dicom_to_nifti)
- Added timing to mask_extraction functions (CanICA, spatial_correlation)
- Added timing to ANTs registration (compute_xfm)
- Exported timing functions in `__init__.py` for easy access
- Test script now supports `--benchmark` flag to enable timing and generate reports

### Phase 2: Parallelization (High-Impact) ✅ **COMPLETED**
1. ✅ **COMPLETED** - Create helper function `parallel_run_command()` in [utils.py](../src/pyrtnf/utils.py)
2. ✅ **COMPLETED** - Parallelize motion correction in [preprocessing.py](../src/pyrtnf/preprocessing.py)
3. ✅ **COMPLETED** - Parallelize DICOM conversion in [preprocessing.py](../src/pyrtnf/preprocessing.py)
4. ✅ **COMPLETED** - Add `n_jobs` parameter to `preprocessing()` function
5. (Testing) - Run benchmark to measure improvement
6. (Deferred) - Add coregistration parallelization for users with 3+ runs (not needed for 2-run workflow)

**Implementation Notes:**
- Created `parallel_run_command()` helper function using multiprocessing.Pool
- Updated DICOM conversion loop to build job specifications and run in parallel
- Updated motion correction loop to build job specifications and run in parallel
- Added `n_jobs` parameter to `preprocessing()` function (default=-1 uses all CPUs)
- Parallelization automatically skips for single-run workflows (n_jobs=1 or only 1 run)
- Both parallel paths include fallback to sequential processing for compatibility

### Phase 3: Algorithm Optimization (Medium-Impact) ✅ **COMPLETED**
1. ✅ **COMPLETED** - Vectorize spatial correlation in [mask_extraction.py](../src/pyrtnf/mask_extraction.py)
2. ✅ **COMPLETED** - Add compression control to dcm2niix wrapper in [utils.py](../src/pyrtnf/utils.py)
3. (Testing) - Run benchmark to measure improvement

**Implementation Notes:**
- Replaced loop-based spatial correlation with vectorized numpy operations
- Create all network mask matrices at once using np.stack()
- Demean all networks simultaneously with broadcasting
- Compute all correlations in single matrix multiply: (n, V) @ (V, m) = (n, m)
- Expected speedup: 2-3x for correlation step (17 networks, 20 components typical case)
- Compression control already implemented: `compress` parameter in `dicom_to_nifti()`
- Documented usage: Set compress=False for real-time processing (saves ~50-100ms per volume)

### Phase 4: Documentation & Testing
1. Update CLAUDE.md with performance characteristics
2. Add benchmark results to documentation
3. Test with multi-run data to verify parallelization works correctly
4. Verify output consistency (parallel results match sequential)

---

## 5. Technical Details

### Parallelization Approach
- Use `multiprocessing.Pool` for CPU-bound tasks (mcflirt, ANTs, dcm2niix)
- Avoid `threading` due to Python GIL
- Handle subprocess output collection carefully
- Maintain logging integrity with process-safe logging

### Spatial Correlation Vectorization
**Current approach** (lines 300-318):
```python
for j, net_id in enumerate(labels):
    y = (data_parcellation == j+1).astype(float)
    y_dm = y - y.mean()
    num = X_dm.T @ y_dm  # O(V×n) per network
    corr = num / (x_norm * y_norm)
```

**Optimized approach**:
```python
# Create matrix of all network masks at once
Y = np.stack([(data_parcellation == j+1).astype(float) for j in labels])  # (m, V)
Y_dm = Y - Y.mean(axis=1, keepdims=True)  # Demean all networks
Y_norm = np.sqrt((Y_dm**2).sum(axis=1))  # All norms at once
num = X_dm.T @ Y_dm.T  # Single matrix multiply: (n, V) @ (V, m) = (n, m)
corr = num / (x_norm[:, None] * Y_norm[None, :])  # Broadcast division
```

Expected speedup: ~2-3x for typical case (17 networks, 20 components)

---

## 6. Verification Strategy

### Correctness Testing
- Compare parallelized outputs with sequential baseline
- Check that NIfTI files are bit-identical (excluding metadata)
- Verify timing improvements match expectations
- Test with different numbers of runs (1, 2, 4)

### Performance Testing
- Run benchmarks before and after each optimization
- Measure speedup for each component
- Calculate overall pipeline speedup
- Test on different data sizes

### End-to-End Test
- Run complete workflow from [test.py](../tests/test.py) with optimizations enabled
- Verify all outputs match expected structure
- Confirm mask extraction produces valid masks
- Run feedback session to verify downstream compatibility

---

## 7. Expected Results

### For 2-Run Workflow (User's Configuration):
**Parallelizable operations:**
- Motion correction: 2x speedup (2 runs in parallel)
- DICOM conversion: 2x speedup (2 conversions in parallel)

**NOT parallelizable (single operation):**
- Coregistration: No speedup (only 1 registration: run-02 → run-01)

**Algorithm optimizations:**
- Spatial correlation: 2-3x speedup (vectorization)

**Realistic estimates:**
- Overall preprocessing: ~30-35% faster (motion correction + DICOM conversion parallelized)
- Overall mask extraction: ~5-10% faster (spatial correlation vectorization, but ICA still dominates)

### Total Pipeline Speedup (2 runs, 8-16 cores):
- Baseline (post-DICOM collection): ~20-30 minutes
- Optimized (post-DICOM collection): ~15-20 minutes (~25-35% reduction)
- With detailed timing reports: +1-2% overhead (negligible)

**Note**: DICOM collection time is excluded from optimization as it is hardware-constrained by the scanner TR and cannot be sped up through software optimization.

---

## 8. Critical Files to Modify

1. [src/pyrtnf/utils.py](../src/pyrtnf/utils.py) - Add Timer, parallel helpers
2. [src/pyrtnf/preprocessing.py](../src/pyrtnf/preprocessing.py) - Parallelize loops
3. [src/pyrtnf/mask_extraction.py](../src/pyrtnf/mask_extraction.py) - Vectorize correlation
4. [tests/test.py](../tests/test.py) - Add benchmark mode
5. [CLAUDE.md](../CLAUDE.md) - Document performance characteristics

---

## 9. Future Optimization Opportunities (Out of Scope)

- GPU acceleration for CanICA (CuPy integration)
- Incremental ICA (process volumes as they arrive)
- Alternative motion correction (ANTs-based instead of mcflirt)
- Smart caching of intermediate results
- Dimensionality reduction before correlation computation
