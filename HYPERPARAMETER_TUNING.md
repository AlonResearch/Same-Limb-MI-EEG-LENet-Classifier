# Hyperparameter Tuning Guide

## Overview

The hyperparameter tuning feature uses **Optuna** with Bayesian optimization (TPE sampler) to automatically find the best hyperparameters for the LENet model on a per-subject basis.

## What Gets Tuned

The following hyperparameters are optimized:

| Parameter | Search Range | Type | Impact |
|-----------|--------------|------|--------|
| `learning_rate` | [1e-5, 1e-1] | log-uniform | High |
| `dropout` | [0.1, 0.7] | uniform | High |
| `batch_size` | {16, 32, 64, 128} | categorical | Medium |
| `early_stopping_patience` | [50, 300] | integer | Medium |
| `early_stopping_min_delta` | [1e-5, 1e-3] | log-uniform | Low |

**Optimization Metric:** Validation F1-score (macro-averaged)

## Quick Start

### 1. Tune a Single Subject (Recommended for Testing)

```bash
# Activate environment
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
# or
source .venv/bin/activate  # Linux/Mac

# Tune sub-001 with 50 trials (~3-4 hours on GPU)
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 --n-trials 50
```

This will:
- Run 50 optimization trials for sub-001
- Save best hyperparameters to `models/Hyperparameters/best_configs/sub-001-200hz.json`
- Use Bayesian optimization to efficiently search the space
- Apply median pruning to stop unpromising trials early

### 2. Train with Tuned Hyperparameters

```bash
# Run training for all subjects
# sub-001 will use tuned hyperparameters, others use defaults
python -m mi3_eeg.run_all_subjects --epochs 600
```

The script automatically:
- Checks for tuned hyperparameters in `models/Hyperparameters/best_configs/`
- Loads them if available
- Falls back to `config.py` defaults if not found

### 3. Analyze Tuning Results

```bash
# View tuning summary for a subject
python -m mi3_eeg.tuning.analysis --subject sub-001

# Compare hyperparameters across all tuned subjects
python -m mi3_eeg.tuning.analysis --compare
```

## Advanced Usage

### Tune Multiple Subjects

```bash
# Tune sub-001, sub-005, and sub-010
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 sub-005 sub-010 --n-trials 50
```

### Quick Tuning (Fewer Trials)

```bash
# Fast exploration with 25 trials (~1.5-2 hours)
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 --n-trials 25
```

### Thorough Tuning (More Trials)

```bash
# Extensive search with 100 trials (~6-8 hours)
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 --n-trials 100
```

### Manual Hyperparameter Testing

You can manually test specific hyperparameters without tuning:

```bash
python -m mi3_eeg.main \
  --subject-file sub-001_eeg200hz.mat \
  --learning-rate 0.005 \
  --dropout 0.35 \
  --batch-size 32 \
  --epochs 600
```

## Configuration Files

### Storage Structure

```
models/Hyperparameters/
└── best_configs/
    ├── sub-001-200hz.json
    ├── sub-002-200hz.json
    └── ...
```

### JSON Format

```json
{
  "subject_id": "sub-001",
  "sampling_rate": 200,
  "tuned_date": "2026-02-08T15:30:00",
  "best_trial_number": 42,
  "best_val_f1": 0.683,
  "n_trials": 50,
  "hyperparameters": {
    "learning_rate": 0.00234,
    "dropout": 0.35,
    "batch_size": 32,
    "early_stopping_patience": 125,
    "early_stopping_min_delta": 0.0002
  }
}
```

## Expected Performance

### Time Estimates (GPU)

- **Per trial**: ~2-5 minutes (depends on early stopping)
- **25 trials**: ~1.5-2 hours
- **50 trials**: ~3-4 hours
- **100 trials**: ~6-8 hours

### Typical Improvements

Based on MI-EEG literature:
- **Learning rate**: ±5-10% accuracy impact
- **Dropout**: ±3-8% accuracy impact
- **Batch size**: ±2-5% accuracy impact

Expected overall improvement: **5-15% validation F1 score** compared to default hyperparameters.

## Workflow Recommendations

### Strategy 1: Single Representative (Fastest)

1. Tune one "average" subject (50 trials)
2. Apply to all subjects
3. Fine-tune outliers if needed

```bash
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 --n-trials 50
python -m mi3_eeg.run_all_subjects --epochs 600
```

**Time:** ~3-4 hours tuning + ~30-60 minutes training all

### Strategy 2: Representative Sample (Balanced)

1. Tune 3-5 diverse subjects
2. Apply average or best config to all

```bash
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 sub-010 sub-020 --n-trials 50
python -m mi3_eeg.run_all_subjects --epochs 600
```

**Time:** ~10-15 hours tuning + ~30-60 minutes training all

### Strategy 3: Per-Subject (Most Accurate)

1. Tune all 25 subjects individually
2. Each uses its own optimal hyperparameters

```bash
# This would require ~75-100 hours - not recommended unless you have time
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size range in search space (edit `src/mi3_eeg/tuning/optimizer.py`):

```python
"batch_size": trial.suggest_categorical("batch_size", [16, 32])  # Exclude 64, 128
```

### Very Slow Trials

Reduce max epochs:

```bash
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 --n-trials 50 --epochs 300
```

### Inconsistent Results

Increase number of trials for more robust optimization:

```bash
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 --n-trials 75
```

### Re-tune a Subject

Delete the old config and run tuning again:

```bash
rm models/Hyperparameters/best_configs/sub-001-200hz.json
python -m mi3_eeg.run_all_subjects --tune --tune-subjects sub-001 --n-trials 50
```

## Technical Details

### Optimization Algorithm

- **Sampler**: TPE (Tree-structured Parzen Estimator) - Bayesian optimization
- **Pruner**: MedianPruner - stops unpromising trials early
- **Startup trials**: 5 (random exploration before TPE kicks in)
- **Warmup steps**: 50 epochs before pruning can occur

### Why TPE?

TPE is the state-of-the-art for hyperparameter optimization:
- More efficient than grid/random search
- Adaptively balances exploration vs exploitation
- Used in EEGNet, LENet, and other MI-EEG papers
- Typically finds good hyperparameters in 50-100 trials

### Search Space Design

Based on MI-EEG classification literature:
- Learning rate: log-scale (orders of magnitude matter)
- Dropout: linear (small changes matter)
- Batch size: categorical (discrete hardware-friendly values)
- Patience: linear (integer, affects training time)

## References

- Optuna Documentation: https://optuna.readthedocs.io/
- TPE Paper: Bergstra et al. (2011) "Algorithms for Hyper-Parameter Optimization"
- EEGNet Paper: Lawhern et al. (2018) "EEGNet: A Compact CNN for EEG-based BCI"
- LENet for EEG: Various MI-EEG classification papers using similar architectures

## Next Steps

After tuning and training:

1. **Compare performance:**
   ```bash
   python -m mi3_eeg.tuning.analysis --compare
   ```

2. **Analyze results:**
   Check `reports/metrics/` for per-subject performance

3. **Run group analysis:**
   ```bash
   python -m mi3_eeg.analysis.group_analysis --analysis-type classification
   ```

4. **Iterate if needed:**
   - Tune more subjects
   - Adjust search space
   - Try different trial budgets
