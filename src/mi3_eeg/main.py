"""Main entry point for the MI3 EEG classification project.

This script orchestrates the full pipeline: data loading, training,
evaluation, and visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mi3_eeg.config import DataConfig, ModelConfig, Paths, TrainingConfig
from mi3_eeg.dataset import load_dataset_from_config, prepare_data_loaders
from mi3_eeg.evaluation import (
    compare_models,
    evaluate_model,
    print_evaluation_summary,
    save_evaluation_results,
)
from mi3_eeg.logger import logger, setup_logger
from mi3_eeg.model import create_model, save_model
from mi3_eeg.train import train_model
from mi3_eeg.visualization import create_all_visualizations


def main(
    model_types: list[str] | None = None,
    epochs: int | None = None,
    device: str | None = None,
) -> None:
    """Run the full ML pipeline.
    
    Args:
        model_types: List of model types to train (only 'lenet' supported).
                If None, defaults to 'lenet'.
        epochs: Number of training epochs. If None, uses config default.
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
    """
    logger.info("=" * 80)
    logger.info("MI3 EEG Motor Imagery Classification Pipeline")
    logger.info("=" * 80)
    
    # Initialize paths
    paths = Paths.from_here()
    paths.create_directories()
    
    # Setup logger with file output
    log_file = paths.reports_logs / "training_run.log"
    setup_logger(log_file=log_file)
    
    # Detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Default to training both models
    if model_types is None:
        model_types = ["lenet"]
    
    # === STAGE 1: Data Loading ===
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 1: Loading and Preprocessing Data")
    logger.info("=" * 80)
    
    data_config = DataConfig()
    logger.info(f"Dataset: {data_config.mat_filename}")
    logger.info(f"Subject: {data_config.subject_id}")
    logger.info(f"Test split: {data_config.test_size * 100}%")
    
    # Load dataset
    data_bundle = load_dataset_from_config(config=data_config, paths=paths)
    logger.info(f"Data shape: {data_bundle.data.shape}")
    logger.info(f"Class distribution: {data_bundle.class_distribution}")
    
    # Prepare data loaders
    train_loader, test_loader = prepare_data_loaders(
        data_bundle, data_config, device=device
    )
    
    # === STAGE 2: Model Training ===
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2: Training Models")
    logger.info("=" * 80)
    
    model_config = ModelConfig(
        channel_count=data_bundle.channel_count,
        classes_num=data_bundle.num_classes,
    )
    
    training_config = TrainingConfig(
        device=device,
    )
    if epochs is not None:
        training_config = TrainingConfig(
            epochs=epochs,
            batch_size=training_config.batch_size,
            learning_rate=training_config.learning_rate,
            dropout=training_config.dropout,
            early_stopping_patience=training_config.early_stopping_patience,
            early_stopping_min_delta=training_config.early_stopping_min_delta,
            device=device,
        )
    
    trained_models = {}
    training_histories = {}
    
    for model_type in model_types:
        logger.info(f"\nTraining {model_type.upper()} model...")
        
        # Create model
        model = create_model(model_type, model_config, device)
        
        # Train
        history = train_model(
            model,
            train_loader,
            test_loader,
            training_config,
            save_path=paths.models / f"{model_type}_best.pth",
        )
        
        # Save final model
        save_model(model, paths.models / f"{model_type}_final.pth")
        
        trained_models[model_type] = model
        training_histories[model_type] = history
        
        logger.info(
            f"{model_type.upper()} - Final Train Acc: {history.train_acc[-1] * 100:.2f}%, "
            f"Best Val Acc: {history.best_val_acc * 100:.2f}%"
        )
    
    # === STAGE 3: Evaluation ===
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 3: Model Evaluation")
    logger.info("=" * 80)
    
    evaluation_results = compare_models(trained_models, test_loader, device)
    
    # Print summary
    print_evaluation_summary(evaluation_results)
    
    # Save results
    for model_name, results in evaluation_results.items():
        save_evaluation_results(
            results,
            paths.reports_metrics / f"{model_name}_results.json",
            model_name,
        )
    
    # === STAGE 4: Visualization ===
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 4: Creating Visualizations")
    logger.info("=" * 80)
    
    create_all_visualizations(
        training_histories,
        evaluation_results,
        paths.reports_figures,
    )
    
    # === COMPLETION ===
    logger.info("\n" + "=" * 80)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 80)
    logger.info(f"Models saved to: {paths.models}")
    logger.info(f"Results saved to: {paths.reports_metrics}")
    logger.info(f"Figures saved to: {paths.reports_figures}")
    logger.info(f"Logs saved to: {log_file}")


def cli() -> None:
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(
        description="MI3 EEG Motor Imagery Classification Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["lenet"],
        default=None,
        help="Model type to train (default: lenet)",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (default: auto-detect)",
    )
    
    args = parser.parse_args()
    
    main(
        model_types=args.models,
        epochs=args.epochs,
        device=args.device,
    )


if __name__ == "__main__":
    cli()
