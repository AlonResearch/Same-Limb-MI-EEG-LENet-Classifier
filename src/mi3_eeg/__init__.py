"""MI3 EEG Motor Imagery Classification Package.

A modular PyTorch-based package for classifying motor imagery EEG signals
from the MI3 dataset using deep learning models.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "MI3-EEG Team"

# Configuration
from mi3_eeg.config import (
    DataConfig,
    ModelConfig,
    Paths,
    TrainingConfig,
)

# Logging
from mi3_eeg.logger import logger, setup_logger

# Dataset
from mi3_eeg.dataset import (
    EEGDataBundle,
    create_data_loader,
    load_dataset_from_config,
    load_mat_from_derivatives,
    prepare_data_loaders,
)

# Model
from mi3_eeg.model import (
    LENet,
    LENet_FCL,
    create_model,
    initialize_weights,
    load_model,
    save_model,
)

# Training
from mi3_eeg.train import (
    EarlyStopper,
    TrainingHistory,
    quick_train,
    train_model,
)

# Evaluation
from mi3_eeg.evaluation import (
    EvaluationResults,
    compare_models,
    evaluate_model,
    get_predictions,
)

# Visualization
from mi3_eeg.visualization import (
    plot_confusion_matrix,
    plot_training_curves,
)

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Paths",
    "DataConfig",
    "ModelConfig",
    "TrainingConfig",
    # Logging
    "logger",
    "setup_logger",
    # Dataset
    "EEGDataBundle",
    "load_mat_from_derivatives",
    "load_dataset_from_config",
    "create_data_loader",
    "prepare_data_loaders",
    # Models
    "LENet",
    "LENet_FCL",
    "create_model",
    "load_model",
    "save_model",
    "initialize_weights",
    # Training
    "EarlyStopper",
    "TrainingHistory",
    "train_model",
    "quick_train",
    # Evaluation
    "EvaluationResults",
    "evaluate_model",
    "compare_models",
    "get_predictions",
    # Visualization
    "plot_training_curves",
    "plot_confusion_matrix",
]
