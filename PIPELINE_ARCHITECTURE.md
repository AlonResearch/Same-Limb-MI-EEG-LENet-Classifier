# Pipeline Architecture & Parallel Execution

## 🔀 The Four Independent Pipelines

### Key Insight
**Pipeline A (Training) and Pipeline C (TFR Analysis) are completely independent** and can run in parallel!

---

## Pipeline Dependencies

```
┌─────────────────────────────────────────────────────────┐
│ Pipeline A: Subject-Level Training                      │
│ Input: Raw EEG (.mat files)                             │
│ Output: 25 trained models + per-subject metrics        │
│ Duration: 30-60 minutes (depending on GPU)             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Pipeline B: Classification Analysis                     │
│ Input: Trained models + per-subject metrics            │
│ Output: Cross-subject stats, performance plots         │
│ Duration: 2-5 minutes                                   │
│ Dependency: REQUIRES Pipeline A                         │
└─────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────┐
│ Pipeline C: TFR & Topographical Analysis                │
│ Input: Raw EEG (.mat files)                             │
│ Output: TFR plots, topomaps, cached data                │
│ Duration: 15-30 minutes                                 │
│ Dependency: NONE - completely independent!             │
│ Can run in parallel: YES - with both A & B             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ Pipeline D: Visualization Regeneration (Optional)       │
│ Input: Cached TFR data                                  │
│ Output: Updated TFR plots                               │
│ Duration: 10-30 seconds                                 │
│ Dependency: Requires Pipeline C (for cache)            │
└─────────────────────────────────────────────────────────┘
```

---

## Why Pipeline C is Independent

Pipeline C (TFR Analysis) **only** uses:

1. **Raw EEG data** - Loaded directly from `.mat` files
2. **Data preprocessing** - Resampling, filtering
3. **Wavelet analysis** - Morlet wavelets, frequency decomposition
4. **Visualization** - Plots and brain maps

Pipeline C **does NOT** use:
- ❌ Trained neural network models
- ❌ Model predictions
- ❌ Classification accuracies
- ❌ Any output from training pipeline

**Code Evidence:**

```python
# In group_analysis.py, Step 4 (TFR analysis):
mat_files = _get_subject_mat_files(paths, subjects_subset)  # ← Raw .mat files only
_compute_group_time_frequency_and_topography(mat_files, config, output_dir)

# Inside time_frequency.py:
# - Loads raw EEG from .mat
# - Applies Morlet wavelets
# - Computes ERD/ERS
# - No model predictions needed!
```

---

## Execution Scenarios

### Scenario 1: Training Only
```bash
# Terminal: Sequential
python -m mi3_eeg.run_all_subjects
# Result: 25 trained models, per-subject metrics
```

### Scenario 2: Analysis Only (No Training)
```bash
# Terminal: Sequential - can run without ever training!
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr
# Result: TFR plots, topomaps - requires only raw EEG data
```

### Scenario 3: Complete Analysis (Optimal - Uses Parallelization)
```bash
# Terminal 1: Training (Pipeline A) - 30-60 min
python -m mi3_eeg.run_all_subjects

# Terminal 2: TFR Analysis (Pipeline C) - 15-30 min
# Start immediately - doesn't need training to finish!
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr

# Terminal 3: After Pipeline A finishes, Classification (Pipeline B) - 2-5 min
python -m mi3_eeg.analysis.group_analysis --analysis-type classification

# ┌─────────────────┐
# │ Terminal 1 (A)  │  ████████████████████████████ (50 min)
# ├─────────────────┤
# │ Terminal 2 (C)  │  ─────████████████████████ (20 min, starts immediately)
# ├─────────────────┤
# │ Terminal 3 (B)  │  ─────────────────────────────███ (3 min, after A)
# └─────────────────┘
# Total time: ~50 min (not 50+15+2=67 min)
```

### Scenario 4: Tweak Visualizations
```bash
# Only run Pipeline D after Pipeline C completes
python -m mi3_eeg.analysis.regenerate_visualizations  # 10-30 seconds
```

---

## Use Cases

### Use Case 1: Researcher wants to classify motor imagery
```bash
# Just need trained models
python -m mi3_eeg.run_all_subjects
# → Get: models/, reports/metrics/, reports/figures/
# → Done! Analysis pipelines optional
```

### Use Case 2: Neuroscientist wants to understand EEG oscillations
```bash
# Just need TFR/topography analysis
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr
# → Get: TFR plots, topomaps
# → Doesn't need training!
```

### Use Case 3: Full pipeline paper - classification + neurophysiology
```bash
# Run both in parallel (Example 6 in README)
# Terminal 1: Training
python -m mi3_eeg.run_all_subjects

# Terminal 2: TFR analysis (immediately, no wait)
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr

# Terminal 3: Classification stats (after Training)
python -m mi3_eeg.analysis.group_analysis --analysis-type classification

# Result: All analyses complete in ~50 min instead of ~67 min
```

### Use Case 4: Iterate on visualization styling
```bash
# Fast regeneration without recomputation
python -m mi3_eeg.analysis.regenerate_visualizations
# Takes 10-30 seconds instead of 15+ minutes
```

---

## Performance Impact

### Sequential Execution (Old Understanding)
```
Training (50 min) → Classification (3 min) → TFR (20 min) → Done
Total: 73 minutes
```

### Parallel Execution (Correct Architecture)
```
Training (50 min)     ─────┐
                             ├→ Classification (3 min) → Done
TFR (20 min, parallel) ─┘

Total: 50 minutes (23 minutes faster!)
```

### Why It Matters
- **Development:** Faster iteration when testing both models and neuroscience hypotheses
- **Cluster Computing:** Better resource utilization
- **User Experience:** Get results faster
- **Flexibility:** Users choose what analysis they need

---

## Technical Details

### What Pipeline C (TFR) Needs
```python
# Raw EEG data structure (loaded from .mat)
eeg_data.shape = (num_trials, num_channels, num_timepoints)
# e.g., (900, 62, 800) for 900 trials, 62 channels, 800 timepoints

# No model weights needed
# No predictions needed
# No accuracies needed
```

### What Pipeline B (Classification) Needs
```python
# Per-subject metrics from Pipeline A
{
    "subject": "sub-001",
    "overall_accuracy": 0.52,
    "class_accuracies": {"Rest": 0.60, "Elbow": 0.45, "Hand": 0.50},
    ...
}
```

---

## Implementation in Code

### group_analysis.py Entry Point
```python
def run_group_analysis(
    analysis_type: str = "all",  # ← NEW parameter!
    ...
):
    """
    analysis_type options:
    - "classification": Steps 1-3 only (REQUIRES training)
    - "tfr": Step 4 only (independent)
    - "all": All steps (Pipeline B waits for A, C is parallel)
    """
```

### CLI Support
```bash
python -m mi3_eeg.analysis.group_analysis --analysis-type tfr
python -m mi3_eeg.analysis.group_analysis --analysis-type classification
python -m mi3_eeg.analysis.group_analysis  # All analyses
```

---

## Key Takeaways

| Aspect | Before | After |
|--------|--------|-------|
| Pipeline C independence | Not recognized | ✅ Can run in parallel with A |
| Recommended execution | Sequential | Parallel (A & C together) |
| Total time for all analyses | 73 min | **50 min** |
| User choice | All or nothing | Independent (`--analysis-type`) |
| Use case: EEG only | Not possible | ✅ Possible |
| Use case: Classification only | ✅ Possible | ✅ Still possible |

---

## Next Steps (Implementation)

If you want to fully implement this in `group_analysis.py`:

1. **Add `--analysis-type` parameter** to CLI
2. **Conditionally run steps** based on type:
   ```python
   if analysis_type in ["classification", "all"]:
       # Steps 1-3: Classification analysis
   if analysis_type in ["tfr", "all"]:
       # Step 4: TFR analysis
   ```
3. **Update documentation** (already done in README!)
4. **Add to Python API**:
   ```python
   group_analysis.run_group_analysis(analysis_type="tfr")
   ```

This architecture maximizes flexibility while maintaining the ability to run all analyses together.
