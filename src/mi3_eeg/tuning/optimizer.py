"""Hyperparameter optimization using Optuna with Bayesian search.

This module implements hyperparameter tuning for the LENet model using
Optuna's Tree-structured Parzen Estimator (TPE) sampler for Bayesian optimization.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from mi3_eeg.config import DataConfig, ModelConfig, Paths, TrainingConfig
from mi3_eeg.dataset import load_dataset_from_config, prepare_data_loaders
from mi3_eeg.logger import logger
from mi3_eeg.model import create_model
from mi3_eeg.train import train_model

if TYPE_CHECKING:
    from optuna import Trial


def create_search_space(trial: Trial) -> dict[str, float | int]:
    """Define hyperparameter search space for Optuna trial.
    
    Args:
        trial: Optuna trial object.
    
    Returns:
        Dictionary of hyperparameters to test.
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.7),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "early_stopping_patience": trial.suggest_int("early_stopping_patience", 50, 300),
        "early_stopping_min_delta": trial.suggest_float(
            "early_stopping_min_delta", 1e-5, 1e-3, log=True
        ),
    }


def run_single_trial(
    trial: Trial,
    subject_file: str,
    paths: Paths,
    device: str = "cuda",
    max_epochs: int = 600,
) -> float:
    """Run a single training trial with specific hyperparameters.
    
    Args:
        trial: Optuna trial object with suggested hyperparameters.
        subject_file: Name of the subject's .mat file.
        paths: Paths configuration object.
        device: Device to use for training ("cuda" or "cpu").
        max_epochs: Maximum number of training epochs.
    
    Returns:
        Validation F1 score (macro-averaged) for this trial.
    """
    # Get hyperparameters from trial
    hyperparams = create_search_space(trial)
    
    # Extract subject info from filename
    subject_id = subject_file.split("_")[0]
    sampling_rate = 200  # default
    if "200hz" in subject_file.lower():
        sampling_rate = 200
    elif "90hz" in subject_file.lower():
        sampling_rate = 90
    
    logger.info(
        f"Trial {trial.number}: lr={hyperparams['learning_rate']:.6f}, "
        f"dropout={hyperparams['dropout']:.2f}, batch_size={hyperparams['batch_size']}, "
        f"patience={hyperparams['early_stopping_patience']}"
    )
    
    # Create configs
    data_config = DataConfig(
        mat_filename=subject_file,
        subject_id=subject_id,
        sampling_rate=sampling_rate,
    )
    
    # Load data
    try:
        data_bundle = load_dataset_from_config(config=data_config, paths=paths)
    except Exception as e:
        logger.error(f"Trial {trial.number} failed during data loading: {e}")
        raise optuna.TrialPruned()
    
    # Prepare data loaders with trial's batch size
    train_loader, val_loader, test_loader = prepare_data_loaders(
        data_bundle,
        data_config,
        batch_size=int(hyperparams["batch_size"]),
        device=device,
    )
    
    # Create model
    model_config = ModelConfig(
        channel_count=data_bundle.channel_count,
        classes_num=data_bundle.num_classes,
        drop_out=hyperparams["dropout"],
    )
    
    model = create_model("lenet", model_config, device)
    
    # Create training config with trial hyperparameters
    training_config = TrainingConfig(
        epochs=max_epochs,
        batch_size=int(hyperparams["batch_size"]),
        learning_rate=hyperparams["learning_rate"],
        dropout=hyperparams["dropout"],
        early_stopping_patience=int(hyperparams["early_stopping_patience"]),
        early_stopping_min_delta=hyperparams["early_stopping_min_delta"],
        device=device,
    )
    
    # Train model (saving to temporary location)
    temp_save_path = paths.models / f"_trial_{trial.number}_temp.pth"
    
    try:
        history = train_model(
            model,
            train_loader,
            val_loader,
            training_config,
            save_path=temp_save_path,
        )
        
        # Clean up temporary model file
        if temp_save_path.exists():
            temp_save_path.unlink()
        
        # Return best validation F1 score as objective
        best_val_f1 = history.best_val_f1
        logger.info(f"Trial {trial.number} completed: val_f1={best_val_f1:.4f}")
        
        return best_val_f1
        
    except Exception as e:
        logger.error(f"Trial {trial.number} failed during training: {e}")
        # Clean up on failure
        if temp_save_path.exists():
            temp_save_path.unlink()
        raise optuna.TrialPruned()


def tune_subject(
    subject_file: str,
    n_trials: int = 50,
    device: str = "cuda",
    max_epochs: int = 600,
    study_name: str | None = None,
) -> optuna.Study:
    """Run hyperparameter optimization for a single subject.
    
    Args:
        subject_file: Name of the subject's .mat file (e.g., "sub-001_eeg200hz.mat").
        n_trials: Number of optimization trials to run.
        device: Device to use for training ("cuda" or "cpu").
        max_epochs: Maximum epochs per trial.
        study_name: Optional name for the Optuna study. If None, uses subject ID.
    
    Returns:
        Completed Optuna study object.
    """
    paths = Paths.from_here()
    
    # Extract subject info
    subject_id = subject_file.split("_")[0]
    sampling_rate = 200
    if "200hz" in subject_file.lower():
        sampling_rate = 200
    elif "90hz" in subject_file.lower():
        sampling_rate = 90
    
    if study_name is None:
        study_name = f"{subject_id}_{sampling_rate}hz"
    
    # Setup persistent storage
    studies_dir = paths.models / "Hyperparameters" / "studies"
    studies_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{studies_dir}/{study_name}.db"
    
    # Check if study exists
    try:
        existing_study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
        )
        n_completed = len([t for t in existing_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(
            f"Resuming existing study for {subject_id} ({sampling_rate}Hz)\n"
            f"  Found {n_completed} completed trials\n"
            f"  Will run {n_trials} additional trials\n"
            f"  Device: {device}\n"
            f"  Max epochs per trial: {max_epochs}"
        )
    except KeyError:
        logger.info(
            f"Starting new hyperparameter tuning for {subject_id} ({sampling_rate}Hz)\n"
            f"  Trials: {n_trials}\n"
            f"  Device: {device}\n"
            f"  Max epochs per trial: {max_epochs}\n"
            f"  Storage: {storage_url}"
        )
    
    # Create or load Optuna study with persistent SQLite storage
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize validation F1 score
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=50),
        storage=storage_url,
        load_if_exists=True,  # Resume if study exists
    )
    
    # Define objective function with fixed parameters
    def objective(trial: Trial) -> float:
        return run_single_trial(
            trial,
            subject_file=subject_file,
            paths=paths,
            device=device,
            max_epochs=max_epochs,
        )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Count completed trials
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    # Log best results
    best_trial = study.best_trial
    logger.info(
        f"\nOptimization complete for {subject_id}!\n"
        f"  Total completed trials: {n_completed}\n"
        f"  Best trial: {best_trial.number}\n"
        f"  Best val_f1: {best_trial.value:.4f}\n"
        f"  Best hyperparameters:\n"
        f"    learning_rate: {best_trial.params['learning_rate']:.6f}\n"
        f"    dropout: {best_trial.params['dropout']:.3f}\n"
        f"    batch_size: {best_trial.params['batch_size']}\n"
        f"    early_stopping_patience: {best_trial.params['early_stopping_patience']}\n"
        f"    early_stopping_min_delta: {best_trial.params['early_stopping_min_delta']:.6f}"
    )
    
    # Save best hyperparameters
    save_hyperparameters(subject_id, sampling_rate, study)
    
    return study


def save_hyperparameters(
    subject_id: str,
    sampling_rate: int,
    study: optuna.Study,
) -> Path:
    """Save best hyperparameters from Optuna study to JSON file.
    
    Args:
        subject_id: Subject ID (e.g., "sub-001").
        sampling_rate: Sampling rate in Hz (e.g., 200).
        study: Completed Optuna study.
    
    Returns:
        Path to saved JSON file.
    """
    paths = Paths.from_here()
    config_dir = paths.models / "Hyperparameters" / "best_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{subject_id}-{sampling_rate}hz.json"
    filepath = config_dir / filename
    
    best_trial = study.best_trial
    
    config_data = {
        "subject_id": subject_id,
        "sampling_rate": sampling_rate,
        "tuned_date": datetime.now().isoformat(),
        "best_trial_number": best_trial.number,
        "best_val_f1": best_trial.value,
        "n_trials": len(study.trials),
        "hyperparameters": {
            "learning_rate": best_trial.params["learning_rate"],
            "dropout": best_trial.params["dropout"],
            "batch_size": best_trial.params["batch_size"],
            "early_stopping_patience": best_trial.params["early_stopping_patience"],
            "early_stopping_min_delta": best_trial.params["early_stopping_min_delta"],
        },
    }
    
    with open(filepath, "w") as f:
        json.dump(config_data, f, indent=2)
    
    logger.info(f"Saved best hyperparameters to {filepath}")
    
    return filepath


def load_hyperparameters(
    subject_id: str,
    sampling_rate: int,
) -> dict[str, float | int] | None:
    """Load hyperparameters from JSON file.
    
    Args:
        subject_id: Subject ID (e.g., "sub-001").
        sampling_rate: Sampling rate in Hz (e.g., 200).
    
    Returns:
        Dictionary of hyperparameters, or None if file doesn't exist.
    """
    paths = Paths.from_here()
    config_dir = paths.models / "Hyperparameters" / "best_configs"
    filename = f"{subject_id}-{sampling_rate}hz.json"
    filepath = config_dir / filename
    
    if not filepath.exists():
        logger.debug(f"No tuned hyperparameters found for {subject_id} at {filepath}")
        return None
    
    with open(filepath) as f:
        config_data = json.load(f)
    
    logger.info(
        f"Loaded tuned hyperparameters for {subject_id} from {filename}\n"
        f"  Tuned on: {config_data['tuned_date']}\n"
        f"  Best val_f1: {config_data['best_val_f1']:.4f}\n"
        f"  Trials run: {config_data['n_trials']}"
    )
    
    return config_data["hyperparameters"]
