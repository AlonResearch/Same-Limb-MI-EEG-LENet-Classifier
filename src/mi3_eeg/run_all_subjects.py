"""Run training on all subjects in the derivatives folder."""

import argparse
import subprocess
import sys

import scipy.io as scio

from mi3_eeg.config import Paths, TrainingConfig
from mi3_eeg.logger import logger
from mi3_eeg.metrics_aggregator import generate_metrics_report
from mi3_eeg.tuning import load_hyperparameters, tune_subject


def validate_mat_file(mat_path) -> tuple[bool, str | None]:
    """Check if a .mat file can be loaded.
    
    Args:
        mat_path: Path to .mat file
        
    Returns:
        Tuple (is_valid, error_message)
        - is_valid: True if file can be loaded
        - error_message: None if valid, error string if invalid
    """
    try:
        scio.loadmat(str(mat_path), simplify_cells=True)
        return True, None
    except ValueError as e:
        return False, f"Format error: {str(e)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def validate_all_files(mat_files: list) -> tuple[list, list, dict]:
    """Validate all .mat files and return valid/invalid lists.
    
    Args:
        mat_files: List of Path objects
        
    Returns:
        Tuple (valid_files, invalid_files, validation_results)
        - valid_files: List of files that can be loaded
        - invalid_files: List of files that cannot be loaded
        - validation_results: Dict mapping filename to (is_valid, error_msg)
    """
    logger.info("Validating all .mat files...")
    valid_files = []
    invalid_files = []
    validation_results = {}
    
    for mat_file in mat_files:
        is_valid, error_msg = validate_mat_file(mat_file)
        validation_results[mat_file.name] = (is_valid, error_msg)
        
        if is_valid:
            valid_files.append(mat_file)
            logger.info(f"  ✓ {mat_file.name}")
        else:
            invalid_files.append(mat_file)
            logger.info(f"  ✗ {mat_file.name} - {error_msg}")
    
    return valid_files, invalid_files, validation_results


def ask_user_proceed(valid_count: int, invalid_count: int) -> bool:
    """Ask user if they want to proceed with valid files only.
    
    Args:
        valid_count: Number of valid files
        invalid_count: Number of invalid files
        
    Returns:
        True if user wants to proceed, False otherwise
    """
    print()
    print("=" * 80)
    print(f"VALIDATION SUMMARY: {valid_count} valid, {invalid_count} invalid")
    print("=" * 80)
    
    if valid_count == 0:
        print("❌ No valid files found! Cannot proceed.")
        print("   Please check the dataset or download from the original source.")
        return False
    
    if invalid_count > 0:
        print(
            f"\n⚠️  {invalid_count} file(s) could not be loaded (format errors).\n"
            f"   Would you like to proceed with the {valid_count} valid file(s)?"
        )
        while True:
            response = input("\nProceed with training? (yes/no): ").strip().lower()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print("Please enter 'yes' or 'no'")
    
    return True


def main(
    models: list[str] | None = None,
    epochs: int | None = None,
    device: str | None = None,
    tune: bool = False,
    n_trials: int = 50,
    tune_subjects: list[str] | None = None,
):
    """Run training on all .mat files in derivatives folder.
    
    Args:
        models: List of model types to train.
        epochs: Number of training epochs.
        device: Device to use for training.
        tune: If True, run hyperparameter tuning before training.
        n_trials: Number of Optuna trials for hyperparameter tuning.
        tune_subjects: List of subject IDs to tune (e.g., ["sub-001"]). If None, no tuning.
    """
    paths = Paths.from_here()
    derivatives_path = paths.dataset_derivatives
    metrics_path = paths.reports_metrics
    
    # Create training config with overridden values if provided
    training_config = TrainingConfig()
    if epochs is not None:
        # Create new config with custom epochs (frozen dataclass pattern)
        training_config = TrainingConfig(
            epochs=epochs,
            batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            dropout=training_config.dropout,
            early_stopping_patience=training_config.early_stopping_patience,
            early_stopping_min_delta=training_config.early_stopping_min_delta,
            device=device if device is not None else training_config.device,
        )
    elif device is not None:
        # Create new config with custom device only
        training_config = TrainingConfig(device=device)
    
    # Find all .mat files (both raw and standardized formats)
    all_files = sorted(derivatives_path.glob("*.mat"))
    
    if not all_files:
        logger.error(
            "No .mat files found in derivatives folder!\n"
            "See the Dataset section in README.md for download instructions."
        )
        return
    
    logger.info(f"Found {len(all_files)} .mat file(s):")
    for f in all_files:
        logger.info(f"  - {f.name}")
    
    # Deduplicate by subject ID: prefer standardized format over raw format
    subject_files = {}
    for f in all_files:
        # Extract subject ID (e.g., "sub-001" from "sub-001_eeg200hz.mat" or "sub-001_task-motorimagery_eeg.mat")
        subject_id = f.name.split('_')[0]
        
        # Check if this is a standardized file (contains "eegXXXhz" pattern)
        is_standardized = 'eeg200hz' in f.name.lower() or 'eeg90hz' in f.name.lower()
        
        if subject_id not in subject_files:
            # First file for this subject
            subject_files[subject_id] = f
        elif is_standardized:
            # Prefer standardized format over raw format
            subject_files[subject_id] = f
        # else: keep existing (either both are raw, or we already have standardized)
    
    mat_files = list(subject_files.values())
    
    logger.info(f"\nAfter deduplication: {len(mat_files)} unique subject(s):")
    for f in mat_files:
        logger.info(f"  - {f.name}")
    
    # Validate all files before starting
    valid_files, invalid_files, validation_results = validate_all_files(mat_files)
    
    # Ask user if they want to proceed with just the valid files
    if not ask_user_proceed(len(valid_files), len(invalid_files)):
        logger.info("Training cancelled by user.")
        return
    
    if not valid_files:
        logger.error("No valid files to process. Exiting.")
        return
    
    # === HYPERPARAMETER TUNING PHASE (Optional) ===
    if tune and tune_subjects:
        logger.info(
            f"\n{'='*80}\n"
            f"HYPERPARAMETER TUNING PHASE\n"
            f"{'='*80}\n"
            f"Subjects to tune: {tune_subjects}\n"
            f"Trials per subject: {n_trials}\n"
            f"Device: {device or 'auto'}\n"
        )
        
        # Find the files for subjects to tune
        for subject_id in tune_subjects:
            # Find the corresponding file
            subject_file = None
            for mat_file in valid_files:
                if mat_file.name.startswith(subject_id):
                    subject_file = mat_file.name
                    break
            
            if subject_file is None:
                logger.warning(f"Subject {subject_id} not found in valid files. Skipping tuning.")
                continue
            
            logger.info(f"\nTuning hyperparameters for {subject_id}...")
            try:
                tune_subject(
                    subject_file=subject_file,
                    n_trials=n_trials,
                    device=device or "cuda",
                    max_epochs=epochs or 600,
                )
                logger.info(f"✓ Tuning completed for {subject_id}")
            except Exception as e:
                logger.error(f"✗ Tuning failed for {subject_id}: {e}", exc_info=True)
                logger.info("Continuing with default hyperparameters...")
        
        logger.info(
            f"\n{'='*80}\n"
            f"HYPERPARAMETER TUNING COMPLETE\n"
            f"{'='*80}\n"
        )
    
    # === TRAINING PHASE ===
    logger.info(f"\nStarting training runs with {training_config.epochs} epochs each ({len(valid_files)} total)...")
    
    # Run training on each valid file
    for i, mat_file in enumerate(valid_files, 1):
        logger.info(f"[{i}/{len(valid_files)}] Processing: {mat_file.name}")
        
        # Extract subject info
        subject_id = mat_file.name.split("_")[0]
        sampling_rate = 200  # default
        if "200hz" in mat_file.name.lower():
            sampling_rate = 200
        elif "90hz" in mat_file.name.lower():
            sampling_rate = 90
        
        # Try to load tuned hyperparameters
        hyperparams = load_hyperparameters(subject_id, sampling_rate)
        
        # Build the command with all arguments
        cmd = [
            sys.executable,
            "-m",
            "mi3_eeg.main",
            "--subject-file",
            mat_file.name,
            "--epochs",
            str(training_config.epochs),
        ]
        
        # Add device argument
        if training_config.device:
            cmd.extend(["--device", training_config.device])
        
        # Add models argument if specified
        if models:
            cmd.extend(["--models"] + models)
        
        # Add tuned hyperparameters if available
        if hyperparams:
            cmd.extend(["--learning-rate", str(hyperparams["learning_rate"])])
            cmd.extend(["--dropout", str(hyperparams["dropout"])])
            cmd.extend(["--batch-size", str(hyperparams["batch_size"])])
            cmd.extend(["--early-stopping-patience", str(hyperparams["early_stopping_patience"])])
            cmd.extend(["--early-stopping-min-delta", str(hyperparams["early_stopping_min_delta"])])
            logger.info(f"  Using tuned hyperparameters for {subject_id}")
        else:
            logger.info(f"  Using default hyperparameters for {subject_id}")
        
        try:
            result = subprocess.run(cmd, check=True)
            logger.info(f"✓ Successfully completed: {mat_file.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed on: {mat_file.name}", exc_info=True)
            # Continue with next file
            continue
    
    logger.info(
        f"\nTraining completed! Results saved in: "
        "models/, reports/metrics/, reports/figures/"
    )
    
    if invalid_files:
        logger.info(f"Note: {len(invalid_files)} file(s) were skipped due to format errors:")
        for f in invalid_files:
            is_valid, error_msg = validation_results[f.name]
            logger.info(f"  - {f.name}: {error_msg}")
    
    # Generate comprehensive metrics report
    logger.info("Generating comprehensive metrics report...")
    
    try:
        generate_metrics_report(metrics_path)
        logger.info("✓ Metrics report generated successfully!")
    except Exception as e:
        logger.error(f"✗ Failed to generate metrics report: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run training on all subjects in derivatives folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lenet"],
        default=None,
        help="Model type(s) to train (default: lenet)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from TrainingConfig)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for training (default: auto-detect)",
    )
    
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run hyperparameter tuning before training (requires --tune-subjects)",
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials for hyperparameter tuning (default: 50)",
    )
    
    parser.add_argument(
        "--tune-subjects",
        nargs="+",
        default=None,
        help="Subject IDs to tune (e.g., sub-001 sub-002). Use with --tune flag.",
    )
    
    args = parser.parse_args()
    main(
        models=args.models,
        epochs=args.epochs,
        device=args.device,
        tune=args.tune,
        n_trials=args.n_trials,
        tune_subjects=args.tune_subjects,
    )
