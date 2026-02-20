# Paper Publication Plan: pineuro

## Target Journal
**NeuroImage (Software article type)** - premier neuroimaging journal with dedicated software track

## Current Status
- **Data:** 10+ participants available (sufficient for publication)
- **MURFI:** Container available for gold-standard comparison
- **Ready to proceed:** Full validation pipeline can be executed

---

## Paper Title
**"pineuro: A Python Library for Real-Time fMRI Neurofeedback with Incremental GLM Estimation"**

---

## Paper Structure

### Abstract (250 words)
- Problem: Real-time fMRI neurofeedback requires sub-TR latency
- Solution: pineuro - open-source Python library with Gentleman's incremental GLM
- Key innovations: O(p²) updates, continuous normalization, ICA-based masks
- Validation: r > 0.98 vs MURFI gold standard
- Impact: Complete DICOM-to-display pipeline, extensible task framework

### 1. Introduction
- Motivation for real-time neurofeedback (clinical/research applications)
- Computational challenge: sub-second latency requirements
- Existing tools (MURFI, OpenNFT, turbo-brainvoyager) and limitations
- Gap: need for open-source, Python-native, well-documented solution
- Contribution summary

### 2. Methods

#### 2.1 System Architecture
- Modular pipeline overview (Figure 1)
- Threading model: coordinator (background) + display (main)
- Volume collection via watchdog + priority queue

#### 2.2 Preprocessing Pipeline
- 4-step workflow: DICOM→NIfTI, motion correction, brain extraction, coregistration
- Optimization: 1-stage mcflirt (1.66x faster, r=0.9999)
- ANTsPy bindings (2x faster than shell ANTs)

#### 2.3 ICA-Based Network Mask Extraction
- CanICA decomposition (nilearn)
- SyN registration to MNI152
- Vectorized spatial correlation with Yeo atlas
- Auto-selection of best-matching component

#### 2.4 Incremental GLM (Core Algorithm)
- Problem: Traditional GLM is O(np²) per update
- Solution: Gentleman's algorithm achieves O(p²) via Givens rotations
- Design matrix: [task_regressors | intercept | drift | motion_x6]
- Continuous normalization (Hinds et al., 2011)
- Steady-state freezing after baseline
- References: Gentleman (1974), Hinds et al. (2011)

#### 2.5 Task Framework
- BaseTask abstract class with thread-safe polling
- BallTask: dual-network, adaptive difficulty
- Session state persistence

#### 2.6 Session Management
- Multi-run orchestration
- Multi-visit support with mask reuse
- Sham/control condition handling

### 3. Validation

#### 3.1 MURFI Comparison
- Same inputs, masks, parameters
- Correlation analysis across networks

#### 3.2 Gold-Standard Offline Comparison
- fMRIPrep + MELODIC pipeline comparison
- Quantify speed/accuracy tradeoff

#### 3.3 Timing Benchmarks
- Per-volume latency breakdown
- Scalability analysis

### 4. Results
- MURFI validation statistics
- Offline pipeline comparison
- Timing performance
- QC output examples

### 5. Discussion
- Design decisions and tradeoffs
- Limitations
- Comparison with other tools

### 6. Conclusion

---

## Analyses to Run

### Analysis 1: MURFI Comparison (PRIMARY VALIDATION)
**Script:** `tests/murfi_comparison/compare_murfi_pineuro.py`

**Requirements:**
- N ≥ 10 participants (ideally 15-20)
- Complete neurofeedback sessions with rest + feedback runs
- MURFI apptainer image

**Run:**
```bash
python compare_murfi_pineuro.py \
    --csv test_subjects.csv \
    --output-dir outputs \
    --murfi-sif /path/to/murfi.sif \
    --tr 1.2 --n-volumes 150 --n-baseline 25
```

**Metrics to report:**
| Metric | Target | Network |
|--------|--------|---------|
| Pearson r | > 0.98 | ContA, DefaultA, Difference |
| Slope | 0.9-1.1 | All |
| Intercept | ~0 | All |
| RMSE | < 0.1 | All |

**Presentation:** Table 1 (summary stats), Figure 5 (timeseries + scatter)

---

### Analysis 2: Gold-Standard Offline Pipeline Comparison
**Purpose:** Quantify accuracy tradeoff for real-time speed

**Comparison pipeline:**
1. Process same raw data through fMRIPrep (full offline preprocessing)
2. Run FSL MELODIC ICA on fMRIPrep output
3. Run nilearn FirstLevelModel with fMRIPrep confounds
4. Compare to pineuro results

**Three comparison levels:**

#### A. Preprocessing Comparison (pineuro vs fMRIPrep)
| Metric | Method |
|--------|--------|
| Motion parameters | Correlation of 6-DOF estimates |
| Brain masks | Dice coefficient overlap |
| Spatial normalization | Overlay inspection in MNI space |
| Confound quality | Compare FD, DVARS if available |

#### B. ICA Mask Comparison (CanICA vs MELODIC)
| Metric | Method |
|--------|--------|
| Component similarity | Spatial correlation matrix |
| Network identification | Both correlated with Yeo atlas |
| Mask overlap | Dice coefficient for same network |
| Voxel-wise agreement | % overlap in thresholded masks |

#### C. Activation Timeseries Comparison
| Metric | Method |
|--------|--------|
| Timeseries correlation | Pearson r per run |
| Systematic bias | Slope/intercept of regression |
| Noise characteristics | Compare residual variance |

**Scripts to create:**
```bash
# 1. Run fMRIPrep on all participants
fmriprep /data /out participant --participant-label sub-XX

# 2. Run MELODIC on fMRIPrep output
melodic -i fmriprep_func.nii.gz -o melodic_out --dim=20

# 3. Create comparison script
tests/compare_fmriprep_melodic.py
```

**Expected outcomes:**
- Preprocessing: High correlation (r > 0.95) for motion, moderate mask overlap
- ICA: Moderate-high correlation (r > 0.7) for matched components
- Timeseries: High correlation (r > 0.9) - validates real-time approximation

**Key insight for paper:** Quantify the speed/accuracy tradeoff
- "pineuro achieves X% of offline accuracy at Y% of the processing time"

**Presentation:**
- Figure: Side-by-side mask overlays (pineuro vs MELODIC)
- Figure: Timeseries comparison plots
- Table: Summary metrics across all participants

---

### Analysis 3: Timing Benchmarks
**Purpose:** Characterize real-time feasibility

**Method:**
1. Enable timing: `from pineuro import enable_timing`
2. Run feedback sessions with varying parameters
3. Collect per-stage latencies

**Variables to test:**
- Mask sizes: 500, 1000, 2000, 5000 voxels
- Number of masks: 1, 2, 3, 4
- TRs: 1.0, 1.2, 1.5, 2.0s

**Metrics:**
| Stage | Expected (ms) |
|-------|---------------|
| DICOM→NIfTI | 50-150 |
| Motion extraction | 5-20 (JSON) or 400-500 (ANTsPy) |
| GLM update | 5-15 |
| **Total** | **< 200** (must be < TR) |

**Presentation:** Figure (boxplot + scaling curves), Table

---

### Analysis 4: ICA Mask Quality Assessment
**Purpose:** Evaluate network identification accuracy

**Metrics:**
- Spatial correlation with target Yeo network
- Visual alignment with expected anatomy
- Consistency across participants

**Presentation:** Figure (ICA grid + heatmap + overlays), Table

---

### Analysis 5: Feedback Signal Quality
**Purpose:** Characterize activation signal properties

**Metrics:**
- SNR (task vs baseline)
- Effect of efficiency weighting
- Temporal characteristics

**Presentation:** Figure (example timeseries), Table

---

## Figures

### Main Text (10 figures)
1. **System Architecture** - Pipeline flowchart with threading model
2. **Preprocessing Pipeline** - 4-panel showing each step
3. **ICA Mask Extraction** - Components, heatmap, overlays
4. **Incremental GLM** - Algorithm diagram, complexity comparison
5. **MURFI Validation** - Timeseries comparison, scatter, correlation distribution
6. **fMRIPrep Comparison** - Preprocessing quality (motion params, masks)
7. **MELODIC vs CanICA** - Side-by-side network masks, Dice overlap
8. **Offline vs Online Timeseries** - Activation comparison with fMRIPrep pipeline
9. **Timing Performance** - Stage breakdown, scaling curves
10. **Task Framework & GUI** - BallTask + application screenshots

### Supplementary (3+ figures)
- S1: Multi-visit mask reuse example
- S2: QC report examples
- S3: Sham condition implementation

---

## Data Requirements

### Minimum for Publication
- **N = 10 participants** with complete sessions:
  - 2x resting-state runs (200 volumes each)
  - Reference scan (typically 2-4 volumes)
  - 3+ feedback runs (150 volumes each)
- Varied data quality (some high-motion subjects)

### Ideal
- N = 20+ participants
- Multiple scanner sites (generalizability)
- Multiple manufacturers (Siemens, GE, Philips)

### Public Data Options
- OpenNFT example data (for direct comparison)
- OpenNeuro neurofeedback datasets
- HCP task data (preprocessing validation only)

---

## Supplementary Materials

### Methods
- Gentleman's algorithm mathematical derivation
- HRF parameters (double-gamma)
- Motion extraction from Siemens MoCo JSON

### Tables
- S1: Dependencies and versions
- S2: FSL command parameters
- S3: ANTs registration parameters
- S4: Complete timing breakdown

### Data/Code
- GitHub repository link
- Example configuration files
- Docker/Singularity container
- Sample QC reports

---

## Critical Files

| File | Purpose |
|------|---------|
| `src/pineuro/feedback.py` | IncrementalGLM algorithm (lines 328-651) |
| `tests/murfi_comparison/compare_murfi_pineuro.py` | MURFI validation framework |
| `src/pineuro/mask_extraction.py` | ICA pipeline |
| `src/pineuro/preprocessing.py` | Preprocessing with mcflirt optimization |
| `src/pineuro/qc.py` | QC report generation |

---

## Implementation Order

1. **Data organization** - Ensure all 10+ participants have complete data
2. **MURFI validation** - Run comparison on all participants (existing script)
3. **fMRIPrep processing** - Run on all participants (computationally intensive)
4. **MELODIC ICA** - Run on fMRIPrep outputs
5. **Create comparison scripts:**
   - `tests/compare_fmriprep_melodic.py` - offline pipeline comparison
   - `tests/benchmark_timing.py` - systematic timing analysis
6. **Run all comparisons** - Generate statistics for all participants
7. **Generate figures** - All 10 main + supplementary
8. **Write manuscript** - Following NeuroImage Software format
9. **Prepare supplementary** - Code, data, documentation

---

## Verification

After implementation:
- [ ] MURFI comparison shows r > 0.98 for N≥10 participants
- [ ] fMRIPrep preprocessing comparison completed (motion params, masks)
- [ ] MELODIC vs CanICA mask comparison (Dice coefficients, spatial correlation)
- [ ] Offline timeseries comparison shows r > 0.9 vs fMRIPrep pipeline
- [ ] Timing benchmarks show total latency < TR
- [ ] All 10 main figures generated and publication-ready
- [ ] QC reports demonstrate comprehensive validation
- [ ] GitHub repo clean with documentation
- [ ] Speed/accuracy tradeoff quantified for Discussion section
