"""Run training on all subjects in the derivatives folder."""

import argparse
import subprocess
import sys

from mi3_eeg.config import Paths, TrainingConfig
from mi3_eeg.logger import logger
from mi3_eeg.metrics_aggregator import generate_metrics_report

def main(
    models: list[str] | None = None,
    epochs: int | None = None,
    device: str | None = None,
):
    """Run training on all .mat files in derivatives folder."""
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
    
    logger.info(f"\nStarting training runs with {training_config.epochs} epochs each ({len(mat_files)} total)...")
    
    # Run training on each file
    for i, mat_file in enumerate(mat_files, 1):
        logger.info(f"[{i}/{len(mat_files)}] Processing: {mat_file.name}")
        
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
        
        try:
            result = subprocess.run(cmd, check=True)
            logger.info(f"✓ Successfully completed: {mat_file.name}")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed on: {mat_file.name}", exc_info=True)
            # Continue with next file
            continue
    
    logger.info(
        "All training runs completed! Results saved in: "
        "models/, reports/metrics/, reports/figures/"
    )
    
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
    
    args = parser.parse_args()
    main(models=args.models, epochs=args.epochs, device=args.device)
