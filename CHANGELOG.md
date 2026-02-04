# Changelog

## [Unreleased] - 2026-02-04

### Added - Analysis Package (6 commits)

#### Commit 5a4193d: Infrastructure
- **Added `.gitattributes`** to disable Git LFS for derivative files
  - Prevents accidental conversion of local .mat files to LFS pointers
  - Ensures 6.8GB of EEG data remains as actual files

#### Commit 5a21afc: Core Analysis Modules
- **`analysis/metrics_aggregator.py`**: Aggregate LENet metrics across 25 subjects
- **`analysis/visualization.py`**: Generate classification performance plots
  - Accuracy distribution histograms
  - Subject ranking bar charts
  - Class accuracy box plots (Rest/Elbow/Hand)
- **`analysis/__init__.py`**: Package exports for 51 analysis functions

#### Commit 02c1e09: Statistical Analysis
- **`analysis/statistical_tests.py`**: Comprehensive statistical toolkit
  - Paired and independent t-tests with Cohen's d effect sizes
  - One-way and repeated measures ANOVA
  - Multiple comparison corrections (Bonferroni, FDR/Benjamini-Hochberg)
  - Correlation analysis with confidence intervals
  - Automated statistical report generation
- **Validated on 25 subjects**: ANOVA F(2,72)=5.83, p=0.0045

#### Commit 6619cab: Brain Mapping
- **`analysis/topography.py`**: Topographical visualization module
  - Standard 10-20 electrode montage creation via MNE
  - Channel subset selection for motor cortex (C3/Cz/C4)
  - Multi-class topomap comparisons
  - Alpha (8-13 Hz) and beta (13-30 Hz) band mapping
  - Publication-quality figure generation (300 DPI)

#### Commit 4df59c9: Time-Frequency Analysis
- **`analysis/time_frequency.py`**: Memory-optimized TFR computation
  - **Batch processing**: 150 epochs/batch reduces peak memory 6x (26GB → 4-5GB)
  - **Morlet wavelet TFR**: 4-40 Hz, 37 frequencies
  - **ERD/ERS calculation**: Event-Related Desynchronization/Synchronization
  - **Band power extraction**: Delta, theta, alpha, beta, gamma
  - **Joblib caching**: Stores computed TFR in `~/.cache/mi3_eeg/tfr/`
  - **Progress tracking**: tqdm progress bars for 25-subject processing
  - **Float32 optimization**: 50% memory reduction
- **Fixes**:
  - Added missing return statement in `compute_morlet_tfr()`
  - Fixed memory exhaustion during concatenation (immediate float32 conversion)
  - Prevents OOM errors on 32GB RAM systems

#### Commit b1c23c9: Pipeline Orchestration
- **`analysis/group_analysis.py`**: 4-step analysis pipeline
  1. Load classification metrics from all subjects
  2. Create performance summary plots
  3. Perform statistical analysis (ANOVA + pairwise comparisons)
  4. Compute time-frequency and topographical maps
- **Features**:
  - 25-subject group averaging with streaming accumulation
  - ERD/ERS relative to Rest baseline
  - Separate output folders: `figures/`, `tfr_analysis/`, `statistics/`
  - JSON and text statistical reports

### Results Summary

**25-Subject Group Analysis (Complete)**:
- **Mean accuracy**: 51.47% ± 8.19% (range 37.22%-75.00%)
- **ANOVA**: F(2,72) = 5.83, p = 0.0045 (highly significant)
- **Pairwise comparisons (FDR-corrected)**:
  - Rest vs Elbow: p = 0.0005**, d = 0.81 (strong effect)
  - Rest vs Hand: p = 0.029*, d = 0.47 (moderate effect)
  - Elbow vs Hand: p = 0.27 (not significant)
- **Generated outputs**: 6 figures (11.8 MB total)
  - 3 classification performance plots (0.40 MB)
  - 3 TFR/topographical maps (11.4 MB)

### Performance Optimizations

**Memory Management**:
- Batch TFR processing: 900 epochs → 6 batches × 150 epochs
- Peak memory reduced from 26GB to 4-5GB (6x improvement)
- Float32 conversion: Additional 50% memory savings
- Mathematical proof: TFR is linear; batch concatenation ≡ full processing

**Computational Efficiency**:
- Joblib Memory caching: Avoids recomputing TFR (5+ min per subject)
- Adaptive batch sizing: Target 2-3 GB per batch based on available RAM
- Progress tracking: Real-time tqdm bars show time remaining

### Test Coverage

**Existing Tests (72 total)**:
- ✓ 60 passed: config, CUDA, dataset, logger, model, training
- ✗ 10 failed: 9 due to external drive space issue, 1 assertion mismatch
- ⊘ 2 skipped: Require real dataset files

**Missing Test Coverage** (Future Work):
- evaluation.py
- visualization.py
- metrics_aggregator.py
- analysis/group_analysis.py
- analysis/statistical_tests.py
- analysis/time_frequency.py
- analysis/topography.py

### Known Issues

1. **TFR visualization module not persisted**: `tfr_visualization.py` and `regenerate_tfr_visualizations.py` were designed but not saved to disk. Visualization code remains inline in `group_analysis.py`.
2. **Dataformatter tests failing**: 9 tests fail due to "No space left on device" on G: drive (Google Drive mount).

### Dependencies Added
- `mne>=1.11.0`: EEG analysis, topography, TFR
- `pywavelets>=1.4.1`: Wavelet transforms
- `seaborn>=0.13.0`: Statistical visualizations
- `joblib>=1.3.0`: Caching and parallel processing
- `tqdm>=4.67.2`: Progress bars

### File Structure
```
src/mi3_eeg/analysis/
├── __init__.py                  # 51 exported functions
├── group_analysis.py            # 4-step pipeline orchestrator
├── metrics_aggregator.py        # Cross-subject metric aggregation
├── statistical_tests.py         # ANOVA, t-tests, FDR correction
├── time_frequency.py            # Batch TFR with memory optimization
├── topography.py                # Brain mapping visualization
└── visualization.py             # Classification performance plots

reports/group_analysis/
├── figures/                     # Classification plots (3 files)
├── tfr_analysis/                # TFR maps and topomaps (3 files)
└── statistics/                  # JSON + TXT results (2 files)
```

### Migration Notes
- Old metrics aggregation moved from root to `analysis/` package
- All analysis functions now imported from `mi3_eeg.analysis`
- Cache directory: `~/.cache/mi3_eeg/tfr/` for TFR results

### Next Steps
1. Create separate `tfr_visualization.py` module (designed but not implemented)
2. Add `regenerate_tfr_visualizations.py` script for plot regeneration
3. Add test coverage for analysis modules
4. Fix dataformatter test environment (G: drive space issue)
5. Push commits to origin/batch
