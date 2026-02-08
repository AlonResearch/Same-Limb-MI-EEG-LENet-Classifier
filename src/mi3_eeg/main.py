"""Main entry point for the MI3 EEG classification project.

This script orchestrates the full pipeline: data loading, training,
evaluation, and visualization.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mi3_eeg.config import DataConfig, ModelConfig, Paths, TuningConfig
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


def _create_tuning_config_from_args(
    epochs: int | None = None,
    learning_rate: float | None = None,
    dropout: float | None = None,
    batch_size: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
    device: str = "cuda",
) -> TuningConfig:
    """Create TuningConfig with optional hyperparameter overrides.
    
    Args:
        epochs: Number of training epochs.
        learning_rate: Learning rate for optimizer.
        dropout: Dropout probability.
        batch_size: Batch size for training.
        early_stopping_patience: Epochs to wait before early stopping.
        early_stopping_min_delta: Minimum improvement for early stopping.
        device: Device to use ("cuda" or "cpu").
    
    Returns:
        TuningConfig with specified or default values.
    """
    # Start with defaults
    default_config = TuningConfig(device=device)
    
    # Override with provided values
    return TuningConfig(
        epochs=epochs if epochs is not None else default_config.epochs,
        batch_size=batch_size if batch_size is not None else default_config.batch_size,
        learning_rate=learning_rate if learning_rate is not None else default_config.learning_rate,
        dropout=dropout if dropout is not None else default_config.dropout,
        early_stopping_patience=early_stopping_patience if early_stopping_patience is not None else default_config.early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta if early_stopping_min_delta is not None else default_config.early_stopping_min_delta,
        device=device,
    )


def main(
    model_types: list[str] | None = None,
    epochs: int | None = None,
    device: str | None = None,
    subject_file: str | None = None,
    subject_id: str | None = None,
    learning_rate: float | None = None,
    dropout: float | None = None,
    batch_size: int | None = None,
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float | None = None,
) -> None:
    """Run the full ML pipeline.
    
    Args:
        model_types: List of model types to train (only 'lenet' supported).
                If None, defaults to 'lenet'.
        epochs: Number of training epochs. If None, uses config default.
        device: Device to use ('cuda' or 'cpu'). If None, auto-detects.
        subject_file: Subject .mat file to use. If None, auto-selects first available subject.
        subject_id: Subject ID. If None, inferred from subject_file.
        learning_rate: Learning rate for optimizer. If None, uses config default.
        dropout: Dropout probability. If None, uses config default.
        batch_size: Batch size for training. If None, uses config default.
        early_stopping_patience: Epochs to wait before early stopping. If None, uses config default.
        early_stopping_min_delta: Minimum improvement for early stopping. If None, uses config default.
    """
    # Initialize paths and setup logger with file output FIRST
    paths = Paths.from_here()
    paths.create_directories()
    log_file = paths.reports_logs / "training_run.log"
    setup_logger(log_file=log_file)
    
    logger.info("MI3 EEG Motor Imagery Classification Pipeline")
    
    # Detect device if not specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Default to training both models
    if model_types is None:
        model_types = ["lenet"]
    
    # === STAGE 1: Data Loading ===
    logger.info("STAGE 1: Loading and Preprocessing Data")
    
    # Auto-select first subject if none specified
    if subject_file is None or subject_file == "":
        paths = Paths.from_here()
        available_files = sorted(paths.dataset_derivatives.glob("*.mat"))
        
        if not available_files:
            logger.error(
                f"No .mat files found in {paths.dataset_derivatives}\n"
                "Please download the MI3 dataset and place it in the derivatives folder.\n"
                "See the Dataset section in README.md for download instructions."
            )
            return
        
        # Auto-select first subject
        subject_file = available_files[0].name
        logger.info(
            f"No subject specified. Auto-selecting first available subject: {subject_file}"
        )
    
    # Infer subject_id from filename if not provided
    if subject_id is None:
        subject_id = subject_file.split('_')[0]
    # Infer sampling rate from filename
    sampling_rate = 90  # default
    if '200hz' in subject_file.lower():
        sampling_rate = 200
    elif '90hz' in subject_file.lower():
        sampling_rate = 90
    data_config = DataConfig(
        mat_filename=subject_file,
        subject_id=subject_id,
        sampling_rate=sampling_rate,
    )
    
    logger.info(
        f"Dataset: {data_config.mat_filename}, Subject: {data_config.subject_id}, "
        f"Sampling: {data_config.sampling_rate}Hz, "
        f"Val split: {data_config.val_size * 100}%, "
        f"Test split: {data_config.test_size * 100}% of remaining"
    )
    
    # Load dataset with error handling
    try:
        data_bundle = load_dataset_from_config(config=data_config, paths=paths)
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        raise SystemExit(1) from e
    except Exception as e:
        logger.error(
            f"\n{'='*80}\n"
            f"ERROR: Unexpected error during data loading\n"
            f"{'='*80}\n"
            f"{type(e).__name__}: {str(e)}\n"
            f"See above for details.\n"
            f"{'='*80}\n"
        )
        raise SystemExit(1) from e
    
    logger.info(
        f"Data shape: {data_bundle.data.shape}, "
        f"Class distribution: {data_bundle.class_distribution}"
    )
    
    # Create tuning config with hyperparameter overrides
    tuning_config = _create_tuning_config_from_args(
        epochs=epochs,
        learning_rate=learning_rate,
        dropout=dropout,
        batch_size=batch_size,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        device=device,
    )
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_bundle, data_config, batch_size=tuning_config.batch_size, device=device
    )
    
    # === STAGE 2: Model Training ===
    logger.info("STAGE 2: Training Models")
    
    model_config = ModelConfig(
        channel_count=data_bundle.channel_count,
        classes_num=data_bundle.num_classes,
        drop_out=tuning_config.dropout,
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
            val_loader,
            tuning_config,
            save_path=paths.models / f"{data_config.subject_id}_{model_type}_best.pth",
        )
        
        # Save final model
        save_model(model, paths.models / f"{data_config.subject_id}_{model_type}_final.pth")
        
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
        filename = f"{data_config.subject_id}_{model_name}_results.json"
        save_evaluation_results(
            results,
            paths.reports_metrics / filename,
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
        subject_id=data_config.subject_id,
    )
    
    # === COMPLETION ===
    logger.info(
        f"Pipeline completed successfully! "
        f"Models: {paths.models}, Results: {paths.reports_metrics}, "
        f"Figures: {paths.reports_figures}, Logs: {log_file}"
    )


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
    
    parser.add_argument(
        "--subject-file",
        type=str,
        default=None,
        help="Subject .mat file to use (e.g., sub-008_eeg200hz.mat)",
    )
    
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject ID (default: inferred from filename)",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate for optimizer (default: from config)",
    )
    
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Dropout probability (default: from config)",
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (default: from config)",
    )
    
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Epochs to wait before early stopping (default: from config)",
    )
    
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Minimum improvement for early stopping (default: from config)",
    )
    
    args = parser.parse_args()
    
    main(
        model_types=args.models,
        epochs=args.epochs,
        device=args.device,
        subject_file=args.subject_file,
        subject_id=args.subject_id,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        batch_size=args.batch_size,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


if __name__ == "__main__":
    cli()
