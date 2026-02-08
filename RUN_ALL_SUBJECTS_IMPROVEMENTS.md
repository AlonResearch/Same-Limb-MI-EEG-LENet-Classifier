# run_all_subjects.py - Validation & User Interaction Improvements

## Changes Made

### 1. Added File Validation Functions

**`validate_mat_file(mat_path) -> tuple[bool, str | None]`**
- Attempts to load each .mat file independently
- Returns: (is_valid, error_message)
- Catches both ValueError and general exceptions
- Provides detailed error information (format errors, etc.)

**`validate_all_files(mat_files: list) -> tuple[list, list, dict]`**
- Validates all files upfront before starting training
- Logs validation results in real-time with ✓/✗ indicators
- Returns three items:
  1. `valid_files`: List of loadable files
  2. `invalid_files`: List of unloadable files
  3. `validation_results`: Dict with detailed error messages

**`ask_user_proceed(valid_count: int, invalid_count: int) -> bool`**
- Shows validation summary with counts
- Returns False if no valid files found
- Asks user for confirmation if there are invalid files
- Loops until user provides valid yes/no response
- Proceeds automatically if all files are valid

### 2. Validation Flow Integration

Before starting training, the script now:

1. **Finds all .mat files** in derivatives folder
2. **Deduplicates by subject ID** (prefers standardized format)
3. **Validates each file** with proper error messages
4. **Shows summary** with counts of valid/invalid
5. **Prompts user** to proceed or cancel with just valid files
6. **Only processes valid files** (skips invalid ones)
7. **Reports skipped files** at the end with reasons

### 3. Output Example

```
Found 25 .mat file(s):
  - sub-001_eeg200hz.mat
  - sub-002_eeg200hz.mat
  ... (23 more files)

After deduplication: 25 unique subject(s):
  - sub-001_eeg200hz.mat
  ... (24 more files)

Validating all .mat files...
  ✓ sub-001_eeg200hz.mat
  ✗ sub-002_eeg200hz.mat - Format error: Unknown mat file type, version 51, 53
  ✗ sub-003_eeg200hz.mat - Format error: Unknown mat file type, version 51, 53
  ... (results for all 25 files)

================================================================================
VALIDATION SUMMARY: 0 valid, 25 invalid
================================================================================
❌ No valid files found! Cannot proceed.
   Please check the dataset or download from the original source.
```

**Or if some files are valid:**

```
================================================================================
VALIDATION SUMMARY: 15 valid, 10 invalid
================================================================================

⚠️  10 file(s) could not be loaded (format errors).
   Would you like to proceed with the 15 valid file(s)?

Proceed with training? (yes/no): yes

Starting training runs with 600 epochs each (15 total)...
[1/15] Processing: sub-001_eeg200hz.mat
... (training continues with only valid files)

Training completed! Results saved in: models/, reports/metrics/, reports/figures/

Note: 10 file(s) were skipped due to format errors:
  - sub-002_eeg200hz.mat: Format error: Unknown mat file type, version 51, 53
  - sub-003_eeg200hz.mat: Format error: Unknown mat file type, version 51, 53
  ... (all skipped files listed with reasons)
```

## Key Improvements

✅ **Upfront validation:** No more 25 identical error messages  
✅ **User control:** Option to proceed with just working files  
✅ **Clear feedback:** Shows which files work and which don't  
✅ **Better error info:** Detailed error messages for each file  
✅ **Graceful handling:** Skips invalid files, continues with valid ones  
✅ **Summary reporting:** Lists skipped files with reasons at the end

## Technical Details

### Imports Added
```python
import scipy.io as scio
```

### Function Signatures
```python
def validate_mat_file(mat_path) -> tuple[bool, str | None]
def validate_all_files(mat_files: list) -> tuple[list, list, dict]
def ask_user_proceed(valid_count: int, invalid_count: int) -> bool
def main(models, epochs, device)  # Main function unchanged
```

### Error Handling
- Catches `ValueError`: MATLAB format errors
- Catches `Exception`: Any other loading issues
- Provides human-readable error messages
- Doesn't crash on first error

## Usage

No changes to command-line usage:
```bash
python -m mi3_eeg.run_all_subjects                    # Validate all files first
python -m mi3_eeg.run_all_subjects --epochs 50        # With custom epochs
python -m mi3_eeg.run_all_subjects --device cpu       # Force CPU
```

The script will automatically:
1. Find all .mat files
2. Validate them
3. Show summary
4. Ask for user confirmation (if needed)
5. Train with only valid files

## Backwards Compatibility

✅ Fully backwards compatible  
✅ No changes to main() function signature  
✅ No changes to command-line arguments  
✅ Existing scripts/calls still work
