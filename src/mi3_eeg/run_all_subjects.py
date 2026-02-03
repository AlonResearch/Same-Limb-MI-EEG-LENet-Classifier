"""Run training on all subjects in the derivatives folder."""

import subprocess
import sys

from mi3_eeg.config import Paths, TrainingConfig
from mi3_eeg.logger import logger
from mi3_eeg.metrics_aggregator import generate_metrics_report

def main():
    """Run training on all .mat files in derivatives folder."""
    paths = Paths.from_here()
    training_config = TrainingConfig()
    derivatives_path = paths.dataset_derivatives
    metrics_path = paths.reports_metrics
    
    # Find all *_eeg200hz.mat files (skip the original sub-011_eeg.mat for now)
    mat_files = sorted(derivatives_path.glob("*_eeg200hz.mat"))
    
    if not mat_files:
        logger.error("No *_eeg200hz.mat files found in derivatives folder!")
        return
    
    logger.info(f"Found {len(mat_files)} subject files:")
    for f in mat_files:
        logger.info(f"  - {f.name}")
    
    logger.info(f"Starting training runs with {training_config.epochs} epochs each ({len(mat_files)} total)...")
    
    # Run training on each file
    for i, mat_file in enumerate(mat_files, 1):
        logger.info(f"[{i}/{len(mat_files)}] Processing: {mat_file.name}")
        
        # Run the training
        cmd = [
            sys.executable,
            "-m",
            "mi3_eeg.main",
            "--subject-file",
            mat_file.name,
            "--epochs",
            str(training_config.epochs)
        ]
        
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
    main()
