"""Dataset loading and preprocessing for MI3 EEG data.

This module handles loading EEG data from BIDS-formatted MI3 dataset,
preprocessing, and creating PyTorch DataLoaders.
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
) -> EEGDataBundle:
    """Load preprocessed EEG data from BIDS derivatives folder.
    
    Args:
        mat_path: Path to .mat file in Datasets/MI3/derivatives/.
        reduce_rest_ratio: Fraction of 'Rest' samples to keep (1.0 = all).
        random_seed: Random seed for reproducibility. If None, uses random.
    
    Returns:
        EEGDataBundle with loaded and balanced data.
    
    Raises:
        FileNotFoundError: If .mat file doesn't exist.
        KeyError: If required keys missing from .mat file.
    """
    if not mat_path.exists():
        msg = (
            f"Dataset not found at: {mat_path}\n"
            f"Expected in BIDS derivatives folder."
        )
        raise FileNotFoundError(msg)
    
    logger.info(f"Loading dataset from: {mat_path}")
    mat_data = scio.loadmat(str(mat_path))
    
    # Extract data and labels
    try:
        all_data = mat_data["all_data"]
        all_label = mat_data["all_label"]
    except KeyError as e:
        msg = f"Required key {e} not found in .mat file"
        raise KeyError(msg) from e
    
    # Log original distribution
    logger.info(f"Original data shape: {all_data.shape}")
    logger.info(f"Original label shape: {all_label.shape}")
    
    # Calculate original class distribution
    original_dist = _calculate_class_distribution(all_label)
    logger.info(f"Original class distribution: {original_dist}")
    
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
    
    return EEGDataBundle(
        data=all_data,
        labels=all_label,
        channel_count=channel_count,
        num_classes=num_classes,
        sample_rate=90,  # From MI3 dataset specification
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
