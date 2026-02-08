# Project Timeline & Reports Folder Confusion - Resolution

## 📊 Your Target Results (Found!)

The results you're looking for are from commit **`cfdd96b`** at **19:15:47 on Feb 8, 2026** (~3 hours ago).

**Overall Accuracy**: 59.21% (mean), n=25 subjects
- Rest: 72.74%
- Elbow: 56.44%
- Hand: 48.44%

**Statistical Significance**: 
- F=24.47, p<0.001
- All pairwise comparisons significant (FDR-corrected)

### ✅ Results Restored
Your complete 25-subject analysis has been saved to:
```
reports/group_analysis/statistics/lenet_classification_statistics_ALL_25_SUBJECTS.json
```

---

## 📋 Timeline of Last 10 Hours

### 1:30 PM - 1:41 PM: Individual Subject Training Completed
- All 25 subjects trained and saved to `reports/metrics/sub-*_lenet_results.json`
- Models saved to `models/sub-*_lenet_best.pth` and `models/sub-*_lenet_final.pth`

### ~7:03 PM - 7:15 PM: First Group Analysis Run (ALL 25 SUBJECTS)
- **Pipeline B (Classification)**: Loaded 25 subjects
- Computed statistical tests (ANOVA, pairwise t-tests)
- Generated performance plots
- **Key stats from log**:
  - F=24.4694, p=0.000000
  - Rest vs Elbow: t=5.6336, p=0.000008, d=1.1267
  - Rest vs Hand: t=9.9242, p=0.000000, d=1.9848
  - Elbow vs Hand: t=2.7165, p=0.012043, d=0.5433

### 7:15 PM: **Commit `cfdd96b`** - "fix: Separate Pipeline B and Pipeline C"
**This commit contains your target results!**
- Separated classification analysis from TFR analysis
- Saved statistics for all 25 subjects
- Updated models and hyperparameters

### 8:10 PM: **Commit `c94f064`** - OVERWROTE with 2 subjects
- Updated documentation about group_analysis entry point
- **Side effect**: Re-ran group_analysis with only 2 subjects in metrics folder
- Overwrote `lenet_classification_statistics.json` with n=2 results

---

## 🔍 What Happened to the Reports Folder

The confusion stems from **two separate analysis pipelines**:

### **Pipeline B: Classification Statistics** (What you want)
- Input: `reports/metrics/sub-*_lenet_results.json` (25 files)
- Output: `reports/group_analysis/statistics/lenet_classification_statistics.json`
- **Issue**: Gets overwritten every time `group_analysis.py` runs
- **Current state**: Only 2 subjects (accidentally)
- **Desired state**: 25 subjects (in commit `cfdd96b`)

### **Pipeline C: Time-Frequency Analysis**
- Input: Raw `.mat` files from `Datasets/MI3/derivatives/`
- Output: `reports/group_analysis/tfr_analysis/` and `/figures/`
- Takes much longer (TFR computation is expensive)
- Uses caching in `~/.cache/mi3_eeg/tfr/`

---

## 🎯 How the Analysis Works

### The `group_analysis.py` script performs 4 steps:

1. **Load Classification Metrics** from `reports/metrics/*.json`
2. **Create Performance Summary Plots** (boxplots, confusion matrices)
3. **Perform Statistical Analysis** (ANOVA, t-tests) → saves to `statistics/`
4. **Time-Frequency & Topographical Analysis** (optional, takes hours)

### Why results changed:
The script aggregates **whatever JSON files it finds** in `reports/metrics/`. If you:
- Delete some subject files → next run will have fewer subjects
- Add subject files → next run will include them
- Re-run the script → overwrites previous statistics

---

## 🔧 How to Reproduce Your Results

Your n=25 analysis came from running classification on all subjects. The models exist and were trained around 1:30-1:41 PM today.

### Option 1: Use the restored file (recommended)
```powershell
# Copy the saved version back
Copy-Item reports/group_analysis/statistics/lenet_classification_statistics_ALL_25_SUBJECTS.json `
          reports/group_analysis/statistics/lenet_classification_statistics.json
```

### Option 2: Re-run group analysis (if all 25 JSON files exist)
```powershell
python -m mi3_eeg.analysis --model lenet --skip-tfr
```

This will:
- Load all 25 subject results from `reports/metrics/`
- Recompute statistics
- Skip the expensive TFR computation

---

## 📁 Current Project State

### Training Complete
- ✅ 25 subject-specific models trained
- ✅ Individual results saved in `reports/metrics/`
- ✅ Models saved in `models/sub-*_lenet_*.pth`

### Group Analysis
- ⚠️ Statistics file overwritten (n=2 instead of n=25)
- ✅ **Recovered**: Full results saved to `*_ALL_25_SUBJECTS.json`
- ❓ TFR analysis: May have run partially (check cache)

### Recent Code Changes (Past 10 hours)
1. Separated classification and TFR pipelines
2. Fixed RuntimeWarning in group_analysis module
3. Updated documentation for entry points
4. Minor hyperparameter tuning for subjects 12 and 14

---

## 🎓 Key Insights from Your Results

### Strong Findings:
1. **Rest state is easiest to classify** (72.74% accuracy)
2. **Hand is hardest** (48.44% accuracy)
3. **All motor tasks differ significantly** from each other and from rest
4. **Effect size is large** (η²=0.40, Cohen's d range: 0.54-1.98)

### Model Performance:
- Overall: 59.21% (chance level = 33.3%)
- Substantial inter-subject variability (SD=9.6%)
- Best subject: 79.01%, Worst: 39.51%

---

## 📌 Recommendations

1. **Preserve the ALL_25_SUBJECTS file** - This is your complete analysis
2. **Don't re-run group_analysis** unless you want to update it
3. **The reports folder structure is correct** - just got overwritten accidentally
4. **TFR analysis is separate** - can be run independently with `--analysis-type tfr`

---

## 🔗 Related Files

- Training results: `reports/metrics/sub-*_lenet_results.json`
- Aggregated CSV: `reports/metrics/lenet_all_subjects_metrics.csv`
- Log file: `reports/logs/group_analysis_lenet.log`
- Analysis script: `src/mi3_eeg/analysis/group_analysis.py`
- Commit with full results: `cfdd96b`
