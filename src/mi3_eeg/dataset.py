"""Dataset loading and preprocessing for MI3 EEG data.

This module handles loading EEG data from BIDS-formatted MI3 dataset,
preprocessing, and creating PyTorch DataLoaders. Supports both standardized
format (all_data/all_label) and raw format (task_data/task_label/rest_data).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io as scio
import torch
import torch.utils.data as data_utils

from mi3_eeg.config import DataConfig, Paths
from mi3_eeg.logger import logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EEGDataBundle:
    """Container for EEG data and metadata.
    
    Attributes:
        data: EEG data array, shape (samples, channels, timepoints).
        labels: Class labels, shape (samples, 1).
        channel_count: Number of EEG channels.
        num_classes: Number of motor imagery classes.
        sample_rate: Sampling rate in Hz.
        class_distribution: Dictionary mapping class names to counts.
    """

    data: np.ndarray
    labels: np.ndarray
    channel_count: int
    num_classes: int
    sample_rate: int
    class_distribution: dict[str, int]


def load_mat_from_derivatives(
    mat_path: Path,
    reduce_rest_ratio: float = 1.0,
    random_seed: int | None = None,
    expected_sampling_rate: int | None = None,
    validate_timepoints: bool = True,
) -> EEGDataBundle:
    """Load preprocessed EEG data from BIDS derivatives folder.
    
    Supports two formats:
    1. Standardized format: 'all_data' and 'all_label' keys
    2. Raw format: 'task_data', 'task_label', and 'rest_data' keys
       (automatically converted to standardized format)
    
    Args:
        mat_path: Path to .mat file in Datasets/MI3/derivatives/.
        reduce_rest_ratio: Fraction of 'Rest' samples to keep (1.0 = all).
        random_seed: Random seed for reproducibility. If None, uses random.
        expected_sampling_rate: Expected sampling rate for validation. If None, no validation.
        validate_timepoints: Whether to validate timepoints against expected_sampling_rate.
    
    Returns:
        EEGDataBundle with loaded and balanced data.
    
    Raises:
        FileNotFoundError: If .mat file doesn't exist.
        KeyError: If required keys missing from .mat file.
        ValueError: If timepoint validation fails.
    """
    if not mat_path.exists():
        msg = (
            f"Dataset not found at: {mat_path}\n"
            f"Expected in BIDS derivatives folder."
        )
        raise FileNotFoundError(msg)
    
    logger.info(f"Loading dataset from: {mat_path}")
    mat_data = scio.loadmat(str(mat_path))
    
    # Detect format and load data accordingly
    from mi3_eeg.dataformatter import detect_format, convert_raw_format, format_and_save
    
    data_format = detect_format(mat_data)
    logger.info(f"Detected format: {data_format}")
    
    if data_format == 'raw':
        # Convert and save raw format to standardized format in derivatives folder
        logger.info("Converting raw format to standardized format...")
        
        # Get the derivatives path for saving
        from mi3_eeg.config import Paths
        paths = Paths.from_here()
        
        # Format and save to derivatives folder
        try:
            formatted_path = format_and_save(
                input_path=mat_path,
                output_dir=paths.dataset_derivatives
            )
            logger.info(f"Saved formatted data to: {formatted_path}")
            # Load from the newly saved formatted file
            mat_path = formatted_path
            mat_data = scio.loadmat(str(mat_path))
            all_data = mat_data["all_data"]
            all_label = mat_data["all_label"]
            inferred_sampling_rate = int(mat_data.get("sampling_rate", [90])[0])
        except Exception as e:
            logger.warning(f"Could not save formatted file: {e}, using in-memory conversion")
            # Fallback: use in-memory conversion
            task_data = mat_data['task_data']
            task_label = mat_data['task_label']
            rest_data = mat_data['rest_data']
            
            all_data, all_label, inferred_sampling_rate = convert_raw_format(
                task_data, task_label, rest_data
            )
        
        # Update expected sampling rate if not provided
        if expected_sampling_rate is None:
            expected_sampling_rate = inferred_sampling_rate
            logger.info(f"Using inferred sampling rate: {expected_sampling_rate}Hz")
    else:
        # Extract data and labels from standardized format
        try:
            all_data = mat_data["all_data"]
            all_label = mat_data["all_label"]
        except KeyError as e:
            msg = f"Required key {e} not found in .mat file"
            raise KeyError(msg) from e
    
    # Validate timepoints if requested
    if validate_timepoints and expected_sampling_rate is not None:
        timepoints = all_data.shape[2] if all_data.ndim == 3 else all_data.shape[1]
        expected_timepoints = expected_sampling_rate * 4  # 4-second trials
        tolerance = expected_timepoints * 0.1  # 10% tolerance
        
        if abs(timepoints - expected_timepoints) > tolerance:
            logger.warning(
                f"Timepoint mismatch: expected ~{expected_timepoints} "
                f"({expected_sampling_rate}Hz × 4s), got {timepoints}. "
                f"Tolerance: ±{tolerance:.0f}"
            )
    
    # Log original distribution
    logger.info(f"Original data shape: {all_data.shape}")
    logger.info(f"Original label shape: {all_label.shape}")
    
    # Calculate original class distribution
    original_dist = _calculate_class_distribution(all_label)
    logger.info(f"Original class distribution: {original_dist}")
    
    # Verify labels are correct (0, 1, 2)
    unique_labels = np.unique(all_label)
    expected_labels = np.array([0, 1, 2])
    if not np.array_equal(unique_labels, expected_labels):
        logger.warning(
            f"Label values mismatch! Expected {expected_labels}, got {unique_labels}"
        )
    
    # Balance dataset if needed
    if reduce_rest_ratio < 1.0:
        logger.info(f"Reducing Rest class by factor {reduce_rest_ratio}")
        all_data, all_label = _balance_rest_class(
            all_data, all_label, reduce_rest_ratio, random_seed
        )
        balanced_dist = _calculate_class_distribution(all_label)
        logger.info(f"Balanced class distribution: {balanced_dist}")
    
    # Calculate final statistics
    channel_count = all_data.shape[1]
    num_classes = len(np.unique(all_label))
    class_dist = _calculate_class_distribution(all_label)
    
    # Determine actual sampling rate
    if data_format == 'raw' and 'inferred_sampling_rate' in locals():
        actual_sampling_rate = inferred_sampling_rate
    elif expected_sampling_rate is not None:
        actual_sampling_rate = expected_sampling_rate
    else:
        actual_sampling_rate = 90  # Default from MI3 dataset specification
    
    return EEGDataBundle(
        data=all_data,
        labels=all_label,
        channel_count=channel_count,
        num_classes=num_classes,
        sample_rate=actual_sampling_rate,
        class_distribution=class_dist,
    )


def _calculate_class_distribution(labels: np.ndarray) -> dict[str, int]:
    """Calculate class distribution from labels.
    
    Args:
        labels: Label array.
    
    Returns:
        Dictionary mapping class names to counts.
    """
    flat_labels = labels.flatten()
    return {
        "Rest": int(np.sum(flat_labels == 0)),
        "Elbow": int(np.sum(flat_labels == 1)),
        "Hand": int(np.sum(flat_labels == 2)),
    }


def _balance_rest_class(
    data: np.ndarray,
    labels: np.ndarray,
    keep_ratio: float,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce Rest class samples to balance dataset.
    
    Args:
        data: EEG data array.
        labels: Label array.
        keep_ratio: Fraction of Rest samples to keep.
        random_seed: Random seed for reproducibility.
    
    Returns:
        Tuple of (balanced_data, balanced_labels).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Find indices for each class
    label_flat = labels.flatten()
    rest_indices = np.where(label_flat == 0)[0]
    other_indices = np.where(label_flat != 0)[0]
    
    # Randomly select a subset of Rest samples
    num_rest_to_keep = int(len(rest_indices) * keep_ratio)
    selected_rest_indices = np.random.choice(
        rest_indices, size=num_rest_to_keep, replace=False
    )
    
    # Combine selected Rest indices with all other class indices
    balanced_indices = np.concatenate((selected_rest_indices, other_indices))
    np.random.shuffle(balanced_indices)
    
    # Apply indexing
    balanced_data = data[balanced_indices]
    balanced_labels = labels[balanced_indices]
    
    return balanced_data, balanced_labels


def create_data_loader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    device: str = "cuda",
) -> DataLoader:
    """Create PyTorch DataLoader from EEG data.
    
    This function preprocesses the data to fit the model input format:
    - Converts to PyTorch tensors
    - Adds channel dimension for Conv2D: (batch, 1, channels, timepoints)
    - Moves data to specified device
    
    Args:
        data: EEG data, shape (samples, channels, timepoints).
        labels: Class labels, shape (samples, 1) or (samples,).
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle data.
        drop_last: Whether to drop last incomplete batch.
        device: Device to place tensors on ('cuda' or 'cpu').
    
    Returns:
        PyTorch DataLoader with preprocessed data.
    """
    # Convert labels to flat LongTensor
    label_tensor = torch.LongTensor(labels.flatten()).to(device)
    
    # Ensure data is in shape (samples, channels, timepoints)
    if data.shape[1] >= data.shape[2]:
        logger.debug("Swapping axes to get (samples, channels, timepoints)")
        data = data.swapaxes(1, 2)
    
    # Convert to tensor and add channel dimension for Conv2D
    # Shape: (samples, 1, channels, timepoints)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    data_tensor = torch.unsqueeze(data_tensor, dim=1).to(device)
    
    logger.debug(f"Data tensor shape: {data_tensor.shape}")
    logger.debug(f"Label tensor shape: {label_tensor.shape}")
    
    # Create TensorDataset and DataLoader
    dataset = data_utils.TensorDataset(data_tensor, label_tensor)
    loader = data_utils.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    
    logger.info(
        f"Created DataLoader: {len(dataset)} samples, "
        f"batch_size={batch_size}, shuffle={shuffle}"
    )
    
    return loader


def load_dataset_from_config(
    config: DataConfig | None = None,
    paths: Paths | None = None,
) -> EEGDataBundle:
    """Load dataset using configuration objects.
    
    Args:
        config: DataConfig instance. If None, uses default.
        paths: Paths instance. If None, creates from current location.
    
    Returns:
        EEGDataBundle with loaded data.
    """
    if config is None:
        config = DataConfig()
    if paths is None:
        paths = Paths.from_here()
    
    mat_path = paths.dataset_derivatives / config.mat_filename
    
    return load_mat_from_derivatives(
        mat_path=mat_path,
        reduce_rest_ratio=config.reduce_rest_ratio,
        random_seed=config.random_seed,
        expected_sampling_rate=config.expected_sampling_rate,
        validate_timepoints=config.validate_timepoints,
    )


def prepare_data_loaders(
    data_bundle: EEGDataBundle,
    config: DataConfig,
    device: str = "cuda",
) -> tuple[DataLoader, DataLoader]:
    """Prepare train and test DataLoaders from data bundle.
    
    Args:
        data_bundle: EEGDataBundle with data and labels.
        config: DataConfig with split parameters.
        device: Device to place tensors on.
    
    Returns:
        Tuple of (train_loader, test_loader).
    """
    from sklearn.model_selection import train_test_split
    
    # Split data
    train_data, test_data, train_labels, test_labels = train_test_split(
        data_bundle.data,
        data_bundle.labels,
        test_size=config.test_size,
        shuffle=True,
        random_state=config.random_seed,
    )
    
    logger.info(
        f"Split: {len(train_data)} train samples, {len(test_data)} test samples"
    )
    
    # Create DataLoaders
    train_loader = create_data_loader(
        train_data,
        train_labels,
        batch_size=64,  # Could be configurable
        shuffle=True,
        drop_last=False,
        device=device,
    )
    
    test_loader = create_data_loader(
        test_data,
        test_labels,
        batch_size=64,
        shuffle=False,  # Don't shuffle test data
        drop_last=False,
        device=device,
    )
    
    return train_loader, test_loader
