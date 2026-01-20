"""Project configuration module.

This module contains all configuration settings, paths, and constants
for the MI3 EEG motor imagery classification project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """Project paths respecting BIDS dataset structure.
    
    Attributes:
        project_root: Root directory of the project.
        dataset_root: BIDS-formatted MI3 dataset root.
        dataset_sourcedata: Raw .cnt files (immutable).
        dataset_derivatives: Processed .mat files from MATLAB.
        dataset_code: MATLAB preprocessing scripts.
        data_tensors: Cached PyTorch tensor files.
        data_splits: Train/test split indices.
        models: Saved model weights.
        reports_figures: Generated plots and visualizations.
        reports_metrics: Metrics and evaluation results.
        reports_logs: Training and execution logs.
    """

    project_root: Path
    dataset_root: Path
    dataset_sourcedata: Path
    dataset_derivatives: Path
    dataset_code: Path
    data_tensors: Path
    data_splits: Path
    models: Path
    reports_figures: Path
    reports_metrics: Path
    reports_logs: Path

    @staticmethod
    def from_here() -> Paths:
        """Create Paths instance from current file location.
        
        Returns:
            Paths instance with all project directories.
        """
        root = Path(__file__).resolve().parents[2]
        dataset_root = root / "Datasets" / "MI3"
        return Paths(
            project_root=root,
            dataset_root=dataset_root,
            dataset_sourcedata=dataset_root / "sourcedata",
            dataset_derivatives=dataset_root / "derivatives",
            dataset_code=dataset_root / "code",
            data_tensors=root / "data" / "tensors",
            data_splits=root / "data" / "splits",
            models=root / "models",
            reports_figures=root / "reports" / "figures",
            reports_metrics=root / "reports" / "metrics",
            reports_logs=root / "reports" / "logs",
        )

    def create_directories(self) -> None:
        """Create all necessary project directories if they don't exist."""
        for path_attr in [
            "data_tensors",
            "data_splits",
            "models",
            "reports_figures",
            "reports_metrics",
            "reports_logs",
        ]:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    """Dataset-specific configuration.
    
    Attributes:
        mat_filename: Name of .mat file in derivatives folder.
        subject_id: BIDS subject identifier.
        sampling_rate: EEG sampling rate in Hz.
        bandpass_filter: Tuple of (low_freq, high_freq) for bandpass.
        num_channels: Number of EEG channels.
        num_classes: Number of motor imagery classes.
        class_names: Tuple of class labels.
        reduce_rest_ratio: Ratio of Rest samples to keep (1.0 = all).
        test_size: Proportion of data for testing.
        random_seed: Random seed for reproducibility.
    """

    mat_filename: str = "sub-011_eeg90hz.mat"
    subject_id: str = "sub-011"
    sampling_rate: int = 90
    bandpass_filter: tuple[int, int] = (7, 35)
    num_channels: int = 62
    num_classes: int = 3
    class_names: tuple[str, ...] = ("Rest", "Elbow", "Hand")
    reduce_rest_ratio: float = 0.6
    test_size: float = 0.2
    random_seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters.
    
    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training and validation.
        learning_rate: Initial learning rate for optimizer.
        dropout: Dropout probability for regularization.
        early_stopping_patience: Epochs to wait before early stopping.
        early_stopping_min_delta: Minimum improvement for early stopping.
        device: Device to use for training ('cuda' or 'cpu').
    """

    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 0.01
    dropout: float = 0.35
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 5e-4
    device: str = "cuda"


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture configuration.
    
    Attributes:
        channel_count: Number of EEG channels (input dimension).
        classes_num: Number of output classes.
        drop_out: Dropout rate for model layers.
    """

    channel_count: int = 62
    classes_num: int = 3
    drop_out: float = 0.35


# Global constants
CLASS_LABELS = {0: "Rest", 1: "Elbow", 2: "Hand"}
CLASS_COLORS = {"Rest": "#2ecc71", "Elbow": "#3498db", "Hand": "#e74c3c"}
