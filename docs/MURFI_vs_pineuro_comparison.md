# MURFI vs pineuro Comparison

| Aspect | MURFI | pineuro |
|--------|-------|--------|
| **Language** | C++ | Python |
| **Deployment** | Singularity container | pip install |
| **Motion Correction** | MCFLIRT | MCFLIRT (same) |
| **Brain Extraction** | BET | BET (same) |
| **Registration** | FSL FLIRT | ANTs |
| **ICA** | FSL MELODIC (manual) | nilearn CanICA (automated) |
| **Network Identification** | Visual inspection + manual correlation | Automated spatial correlation with Yeo atlas |
| **Real-Time GLM** | Full refit each TR | Incremental (Gentleman's algorithm) |
| **Data Monitoring** | Socket connections | Filesystem watchdog |
| **Task Communication** | Socket-based communicator | Direct function calls |
| **Offline Testing** | Limited/none | DICOM simulator at realistic TR |
| **QC** | Manual inspection | Automated HTML reports |
| **Sham Support** | Manual file swapping | Built-in with dual logging |
| **Overall Workflow** | Resting → ICA → Masks → Feedback | Resting → ICA → Masks → Feedback (same) |

## Key Takeaways

**What's the same:**
- General pipeline structure (collect resting data, extract masks, run feedback)
- Core FSL preprocessing tools (MCFLIRT, BET)
- Ball task design and mechanics

**What's different:**
- **Registration**: ANTs instead of FLIRT (better quality)
- **ICA workflow**: Automated CanICA + Yeo correlation vs manual MELODIC inspection
- **Real-time processing**: Incremental GLM vs full refit
- **Integration**: Python API vs socket communication
- **Testing**: Built-in simulator vs hoping it works on scan day
- **QC**: Automated reports vs manual checking
