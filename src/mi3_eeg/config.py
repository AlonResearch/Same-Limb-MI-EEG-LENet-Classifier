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
    models: Path
    reports_figures: Path
    reports_metrics: Path
    reports_logs: Path
    reports_group_analysis: Path

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
            models=root / "models",
            reports_figures=root / "reports" / "figures",
            reports_metrics=root / "reports" / "metrics",
            reports_logs=root / "reports" / "logs",
            reports_group_analysis=root / "reports" / "group_analysis",
        )

    def create_directories(self) -> None:
        """Create all necessary project directories if they don't exist."""
        for path_attr in [
            "models",
            "reports_figures",
            "reports_metrics",
            "reports_logs",
            "reports_group_analysis",
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
        val_size: Proportion of data for validation (first split).
        test_size: Proportion of remaining data for testing (second split).
        random_seed: Random seed for reproducibility.
        expected_sampling_rate: Expected sampling rate for validation. If None, no validation.
        validate_timepoints: Whether to validate timepoints against expected_sampling_rate.
    """

    mat_filename: str = ""
    subject_id: str = ""
    sampling_rate: int = 200
    bandpass_filter: tuple[int, int] = (7, 35)
    num_channels: int = 62
    num_classes: int = 3
    class_names: tuple[str, ...] = ("Rest", "Elbow", "Hand")
    val_size: float = 0.1
    test_size: float = 0.1
    random_seed: int = 42
    expected_sampling_rate: int | None = None
    validate_timepoints: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Training hyperparameters configuration.
    
    Attributes:
        epochs: Maximum number of training epochs.
        batch_size: Batch size for training and validation.
        learning_rate: Initial learning rate for optimizer.
        dropout: Dropout probability for regularization.
        early_stopping_patience: Epochs to wait before early stopping.
        early_stopping_min_delta: Minimum improvement for early stopping.
        device: Device to use for training ('cuda' or 'cpu').
    """

    epochs: int = 600
    batch_size: int = 64
    learning_rate: float = 0.01
    dropout: float = 0.4
    early_stopping_patience: int = 200
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
    drop_out: float = 0.4


@dataclass(frozen=True)
class GroupAnalysisConfig:
    """Group-level analysis configuration.
    
    Attributes:
        sampling_rate: EEG sampling rate in Hz.
        tfr_freqs: Tuple of (f_min, f_max, f_step) for time-frequency decomposition.
            Note: f_min should be ≥ 4 Hz to avoid wavelets longer than signal windows (800 samples @ 200 Hz = 4 sec).
        electrodes_of_interest: List of electrode names for topographical analysis.
        baseline_method: Method for baseline correction ('ratio' or 'percent').
        use_cache: Whether to cache computed TFR data.
    """

    sampling_rate: int = 200
    tfr_freqs: tuple[int, int, int] = (4, 40, 1)
    electrodes_of_interest: tuple[str, ...] = (
        "Cz", "C3", "C4", "CPz", "CP3", "CP4",
        "Pz", "P3", "P4", "POz", "PO3", "PO4",
    )
    baseline_method: str = "ratio"
    use_cache: bool = True


# Global constants
CLASS_LABELS = {0: "Rest", 1: "Elbow", 2: "Hand"}
CLASS_COLORS = {"Rest": "#2ecc71", "Elbow": "#3498db", "Hand": "#e74c3c"}
