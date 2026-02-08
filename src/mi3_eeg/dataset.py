"""Load MI3 EEG datasets and build PyTorch DataLoaders.

Supported .mat formats:
- standardized: all_data, all_label
- raw: task_data, task_label, rest_data (auto-converted)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io as scio
import torch
import torch.utils.data as data_utils
from scipy.signal import butter, sosfiltfilt

from mi3_eeg.config import DataConfig, Paths
from mi3_eeg.logger import logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EEGDataBundle:
    """Container for EEG data and metadata.

    Attributes:
        data: EEG array, shape (samples, channels, timepoints).
        labels: Label array, shape (samples, 1) or (samples,).
        channel_count: Number of EEG channels.
        num_classes: Number of classes.
        sample_rate: Sampling rate in Hz.
        class_distribution: Dict with counts for Rest/Elbow/Hand.
    """

    data: np.ndarray
    labels: np.ndarray
    channel_count: int
    num_classes: int
    sample_rate: int
    class_distribution: dict[str, int]


def load_mat_from_derivatives(
    mat_path: Path,
    expected_sampling_rate: int | None = None,
    validate_timepoints: bool = True,
) -> EEGDataBundle:
    """Load a .mat file and return a normalized EEGDataBundle.

    Args:
        mat_path: Path to .mat file (raw or standardized).
        expected_sampling_rate: Expected Hz for 4-second trials.
        validate_timepoints: If True, warn when timepoints != 4s * sampling_rate.

    Returns:
        EEGDataBundle with data/labels ready for training.

    Raises:
        FileNotFoundError: If mat_path does not exist.
        KeyError: If required keys are missing.
    """
    if not mat_path.exists():
        # Check if there are raw format files in the derivatives folder
        derivatives_folder = mat_path.parent
        if derivatives_folder.exists():
            raw_files = list(derivatives_folder.glob('*.mat'))
            if raw_files:
                # Auto-detect and convert raw format files
                from mi3_eeg.data_formatting import detect_format, format_and_save
                
                logger.info(f"Detected {len(raw_files)} .mat files in derivatives folder")
                logger.info("Scanning for raw format files (task_data/task_label/rest_data)...")
                
                converted_count = 0
                for f in raw_files:
                    try:
                        mat_test = scio.loadmat(str(f), simplify_cells=True)
                        if all(k in mat_test for k in ['task_data', 'task_label', 'rest_data']):
                            logger.info(f"Converting raw format: {f.name}")
                            # Convert this raw file
                            converted_path = format_and_save(
                                input_path=f,
                                output_dir=derivatives_folder
                            )
                            converted_count += 1
                            logger.info(f"[OK] Converted: {converted_path.name}")
                    except Exception as e:
                        # Not a raw format file or cannot convert, skip it
                        pass
                
                if converted_count > 0:
                    logger.info(f"\nSuccessfully auto-converted {converted_count} raw format file(s)")
                    # Try loading again now that files are converted
                    if mat_path.exists():
                        logger.info("Retrying to load converted standardized dataset...")
                    else:
                        msg = (
                            f"\n{'='*80}\n"
                            f"ERROR: Dataset file not found after conversion\n"
                            f"{'='*80}\n"
                            f"Expected: {mat_path}\n"
                            f"Converted {converted_count} raw file(s) but target file not found.\n"
                            f"\n"
                            f"This may happen if the filename pattern doesn't match.\n"
                            f"Your converted files are saved in: {derivatives_folder}/\n"
                            f"Please check the filenames and ensure they follow the pattern:\n"
                            f"  sub-XXX_eegSSSHz.mat\n"
                            f"where XXX is subject number and SSS is sampling rate.\n"
                            f"{'='*80}\n"
                        )
                        raise FileNotFoundError(msg)
        
        if not mat_path.exists():
            msg = (
                f"\n{'='*80}\n"
                f"ERROR: Dataset file not found\n"
                f"{'='*80}\n"
                f"Expected location: {mat_path}\n"
                f"\n"
                f"The MI3 dataset is not included in this repository.\n"
                f"Please download it from the original source and place it in:\n"
                f"  {mat_path.parent}/\n"
                f"\n"
                f"For download instructions and dataset details, please refer to the README.md\n"
                f"{'='*80}\n"
            )
            raise FileNotFoundError(msg)
    
    logger.info(f"Loading dataset from: {mat_path}")
    
    # Try to load the .mat file with error handling
    try:
        mat_data = scio.loadmat(str(mat_path))
    except ValueError as e:
        msg = (
            f"\n{'='*80}\n"
            f"ERROR: Failed to load .mat file (corrupted or unsupported format)\n"
            f"{'='*80}\n"
            f"File: {mat_path}\n"
            f"Error: {str(e)}\n"
            f"\n"
            f"This usually means:\n"
            f"  1. The file is corrupted or incomplete\n"
            f"  2. The file is in an unsupported MATLAB format\n"
            f"  3. The file was not properly downloaded\n"
            f"\n"
            f"Solutions:\n"
            f"  • Re-download the dataset from the original source\n"
            f"  • Verify the download is complete (check file size)\n"
            f"  • Try a different subject file to test\n"
            f"  • Check README.md for dataset download instructions\n"
            f"{'='*80}\n"
        )
        logger.error(msg)
        raise FileNotFoundError(msg) from e
    except Exception as e:
        msg = (
            f"\n{'='*80}\n"
            f"ERROR: Unexpected error while loading .mat file\n"
            f"{'='*80}\n"
            f"File: {mat_path}\n"
            f"Error: {type(e).__name__}: {str(e)}\n"
            f"\n"
            f"Please check that:\n"
            f"  • File exists and is readable\n"
            f"  • File is a valid MATLAB .mat file\n"
            f"  • No other processes are using the file\n"
            f"{'='*80}\n"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e
    
    # Detect format and load data accordingly
    from mi3_eeg.data_formatting import detect_format, convert_raw_format, format_and_save
    
    inferred_sampling_rate: int | None = None
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
            
            # Ensure consistent label shape
            all_label = np.atleast_1d(all_label).flatten()
            if all_label.ndim == 1:
                all_label = all_label.reshape(-1, 1)
        except Exception as e:
            logger.warning(f"Could not save formatted file: {e}, using in-memory conversion")
            logger.warning(
                "WARNING: Using in-memory conversion. This is inefficient for repeated use.\n"
                "For better performance, run: python -m mi3_eeg.data_formatting.convert_subject <file> [subject_id]\n"
                "This will save the converted file permanently to the derivatives folder."
            )
            # Fallback: use in-memory conversion
            task_data = mat_data['task_data']
            task_label = mat_data['task_label']
            rest_data = mat_data['rest_data']
            
            all_data, all_label, inferred_sampling_rate = convert_raw_format(
                task_data, task_label, rest_data
            )
            
            # Ensure consistent label shape after in-memory conversion
            all_label = np.atleast_1d(all_label).flatten()
            if all_label.ndim == 1:
                all_label = all_label.reshape(-1, 1)
        
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
        
        # Ensure labels are in consistent shape (samples, 1) or (samples,)
        all_label = np.atleast_1d(all_label).flatten()
        if all_label.ndim == 1:
            all_label = all_label.reshape(-1, 1)
    
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
    
    # Determine actual sampling rate
    if data_format == 'raw' and inferred_sampling_rate is not None:
        actual_sampling_rate = inferred_sampling_rate
    elif expected_sampling_rate is not None:
        actual_sampling_rate = expected_sampling_rate
    else:
        actual_sampling_rate = 200 # Default from MI3 dataset specification
    
    # Apply 0.5 Hz high-pass filter with padding to reduce edge artifacts.
    logger.info(
        f"Applying 0.5 Hz high-pass filter with padding (sfreq={actual_sampling_rate}Hz)..."
    )
    all_data = _apply_high_pass_filter_to_epochs(
        all_data, sfreq=actual_sampling_rate, l_freq=0.5, pad_length=100
    )
    logger.info("High-pass filter applied successfully")
    
    # Calculate final statistics
    channel_count = all_data.shape[1]
    num_classes = len(np.unique(all_label))
    class_dist = _calculate_class_distribution(all_label)
    
    return EEGDataBundle(
        data=all_data,
        labels=all_label,
        channel_count=channel_count,
        num_classes=num_classes,
        sample_rate=actual_sampling_rate,
        class_distribution=class_dist,
    )


def _calculate_class_distribution(labels: np.ndarray) -> dict[str, int]:
    """Return counts for Rest/Elbow/Hand from a label array.

    Args:
        labels: Label array, shape (samples,) or (samples, 1).

    Returns:
        Dict with keys: Rest, Elbow, Hand.
    """
    flat_labels = labels.flatten()
    return {
        "Rest": int(np.sum(flat_labels == 0)),
        "Elbow": int(np.sum(flat_labels == 1)),
        "Hand": int(np.sum(flat_labels == 2)),
    }


def _apply_high_pass_filter_to_epochs(
    data: np.ndarray,
    sfreq: int,
    l_freq: float,
    pad_length: int = 100,
) -> np.ndarray:
    """Apply a high-pass filter to epoched EEG data with reflected padding.

    Args:
        data: EEG array, shape (epochs, channels, timepoints).
        sfreq: Sampling rate in Hz.
        l_freq: High-pass cutoff in Hz.
        pad_length: Number of samples to reflect-pad on each side.

    Returns:
        Filtered EEG array with the same shape as input.
    """
    if data.ndim != 3:
        msg = f"Expected data with 3 dims (epochs, channels, timepoints), got {data.ndim}"
        raise ValueError(msg)

    nyquist = sfreq / 2
    normalized_freq = l_freq / nyquist
    sos = butter(4, normalized_freq, btype="high", output="sos")

    epochs, channels, timepoints = data.shape
    safe_pad = min(pad_length, max(1, timepoints // 4))
    filtered = np.empty_like(data, dtype=np.float32)

    for epoch_idx in range(epochs):
        for channel_idx in range(channels):
            trial = data[epoch_idx, channel_idx, :]
            padded = np.concatenate(
                [trial[:safe_pad][::-1], trial, trial[-safe_pad:][::-1]]
            )
            filtered_padded = sosfiltfilt(sos, padded)
            filtered[epoch_idx, channel_idx, :] = filtered_padded[
                safe_pad : safe_pad + timepoints
            ]

    return filtered


def create_data_loader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    device: str = "cuda",
) -> DataLoader:
    """Create a PyTorch DataLoader from numpy EEG arrays.

    Args:
        data: EEG array, shape (samples, channels, timepoints).
        labels: Label array, shape (samples,) or (samples, 1).
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        drop_last: Whether to drop last incomplete batch.
        device: Target device for tensors ("cuda" or "cpu").

    Returns:
        DataLoader yielding (batch, 1, channels, timepoints) and labels.
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
    """Load a dataset using DataConfig and Paths.

    Args:
        config: DataConfig instance (uses defaults if None).
        paths: Paths instance (auto-derived if None).

    Returns:
        EEGDataBundle with normalized data/labels.
    """
    if config is None:
        config = DataConfig()
    if paths is None:
        paths = Paths.from_here()
    
    mat_path = paths.dataset_derivatives / config.mat_filename
    
    return load_mat_from_derivatives(
        mat_path=mat_path,
        expected_sampling_rate=config.expected_sampling_rate,
        validate_timepoints=config.validate_timepoints,
    )


def prepare_data_loaders(
    data_bundle: EEGDataBundle,
    config: DataConfig,
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split EEGDataBundle and return train/val/test DataLoaders.

    Args:
        data_bundle: Loaded EEGDataBundle.
        config: DataConfig containing val_size, test_size, and random_seed.
        batch_size: Batch size for all DataLoaders.
        device: Device for tensors ("cuda" or "cpu").

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    labels_flat = data_bundle.labels.flatten()
    
    # First split: train+test vs validation
    train_test_data, val_data, train_test_labels, val_labels = train_test_split(
        data_bundle.data,
        data_bundle.labels,
        test_size=config.val_size,
        shuffle=True,
        stratify=labels_flat,
        random_state=config.random_seed,
    )
    
    # Second split: train vs test from remaining data
    train_data, test_data, train_labels, test_labels = train_test_split(
        train_test_data,
        train_test_labels,
        test_size=config.test_size,
        shuffle=True,
        stratify=train_test_labels.flatten(),
        random_state=config.random_seed,
    )
    
    logger.info(
        f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples"
    )
    
    # Create DataLoaders
    train_loader = create_data_loader(
        train_data,
        train_labels,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        device=device,
    )
    
    val_loader = create_data_loader(
        val_data,
        val_labels,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        device=device,
    )
    
    test_loader = create_data_loader(
        test_data,
        test_labels,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        drop_last=False,
        device=device,
    )
    
    return train_loader, val_loader, test_loader
