"""Data formatter module for converting raw MI3 data to model-ready format.

This module handles conversion of raw .mat files from the Nature paper format
(task_data, task_label, rest_data) to the standardized format expected by the model
(all_data, all_label).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io as scio

from mi3_eeg.logger import logger

if TYPE_CHECKING:
    from numpy.typing import NDArray


def convert_raw_format(
    task_data: NDArray,
    task_label: NDArray,
    rest_data: NDArray,
) -> tuple[NDArray, NDArray, int]:
    """Convert raw MI3 format to standardized all_data/all_label format.
    
    Converts from Nature paper format:
    - task_data: (sessions, trials_per_session, channels, timepoints) = (15, 40, 62, 800)
    - task_label: (sessions, trials_per_session) = (15, 40) with values [1, 2]
    - rest_data: (trials, channels, timepoints) = (300, 62, 800)
    
    To standardized format:
    - all_data: (total_trials, channels, timepoints) = (900, 62, 800)
    - all_label: (total_trials,) with values [0, 1, 2]
    
    Label mapping:
    - Rest: 0 (created for rest_data)
    - Elbow: 1 (from task_label)
    - Hand: 2 (from task_label)
    
    Args:
        task_data: Task data with shape (sessions, trials_per_session, channels, timepoints).
        task_label: Task labels with shape (sessions, trials_per_session).
        rest_data: Rest data with shape (trials, channels, timepoints).
    
    Returns:
        Tuple of (all_data, all_label, sampling_rate):
        - all_data: Combined data array (900, 62, timepoints)
        - all_label: Combined labels (900, 1) with values [0, 1, 2]
        - sampling_rate: Inferred sampling rate in Hz
    """
    logger.info("Converting raw format to standardized format...")
    
    # Validate input shapes
    if task_data.ndim != 4:
        msg = f"Expected task_data to have 4 dimensions, got {task_data.ndim}"
        raise ValueError(msg)
    if task_label.ndim != 2:
        msg = f"Expected task_label to have 2 dimensions, got {task_label.ndim}"
        raise ValueError(msg)
    if rest_data.ndim != 3:
        msg = f"Expected rest_data to have 3 dimensions, got {rest_data.ndim}"
        raise ValueError(msg)
    
    sessions, trials_per_session, channels, timepoints = task_data.shape
    logger.info(
        f"Task data: {sessions} sessions × {trials_per_session} trials × "
        f"{channels} channels × {timepoints} timepoints"
    )
    logger.info(
        f"Rest data: {rest_data.shape[0]} trials × {rest_data.shape[1]} channels × "
        f"{rest_data.shape[2]} timepoints"
    )
    
    # Infer sampling rate from timepoints
    # Nature paper: 200Hz, 4-second trials = 800 timepoints
    # Older data: 90Hz, 4-second trials = 360 timepoints
    if timepoints == 800:
        sampling_rate = 200
        logger.info(f"Inferred sampling rate: {sampling_rate}Hz (800 timepoints / 4 seconds)")
    elif timepoints == 360:
        sampling_rate = 90
        logger.info(f"Inferred sampling rate: {sampling_rate}Hz (360 timepoints / 4 seconds)")
    else:
        # Assume 4-second trials
        sampling_rate = int(timepoints / 4)
        logger.warning(
            f"Unknown timepoint count ({timepoints}), assuming 4-second trials → {sampling_rate}Hz"
        )
    
    # Reshape task_data: (15, 40, 62, 800) → (600, 62, 800)
    task_data_flat = task_data.reshape(-1, channels, timepoints)
    logger.info(f"Reshaped task_data: {task_data.shape} → {task_data_flat.shape}")
    
    # Flatten task_label: (15, 40) → (600,)
    task_label_flat = task_label.flatten()
    logger.info(f"Flattened task_label: {task_label.shape} → {task_label_flat.shape}")
    
    # Verify task label values (should be 1 and 2 for Elbow and Hand)
    unique_task_labels = np.unique(task_label_flat)
    logger.info(f"Unique task label values: {unique_task_labels}")
    if not all(label in [1, 2] for label in unique_task_labels):
        logger.warning(f"Expected task labels to be [1, 2], got {unique_task_labels}")
    
    # Create rest labels: (300,) with all zeros
    rest_label = np.zeros(rest_data.shape[0], dtype=task_label.dtype)
    logger.info(f"Created rest labels: {rest_label.shape} with value 0")
    
    # Concatenate data and labels
    # Rest first, then tasks (to match potential existing data ordering)
    all_data = np.concatenate([rest_data, task_data_flat], axis=0)
    all_label = np.concatenate([rest_label, task_label_flat], axis=0)
    
    logger.info(f"Combined all_data shape: {all_data.shape}")
    logger.info(f"Combined all_label shape: {all_label.shape}")
    
    # Verify final class distribution
    class_counts = {
        "Rest (0)": int(np.sum(all_label == 0)),
        "Elbow (1)": int(np.sum(all_label == 1)),
        "Hand (2)": int(np.sum(all_label == 2)),
    }
    logger.info(f"Final class distribution: {class_counts}")
    
    # Verify labels match expected: 0, 1, 2
    expected_labels = {0, 1, 2}
    actual_labels = set(np.unique(all_label).tolist())
    if actual_labels != expected_labels:
        logger.warning(
            f"Label mismatch! Expected {expected_labels}, got {actual_labels}"
        )
    
    # Reshape all_label to (trials, 1) to match expected format
    all_label = all_label.reshape(-1, 1)
    
    return all_data, all_label, sampling_rate


def format_and_save(
    input_path: Path,
    output_path: Path | None = None,
    subject_id: str | None = None,
) -> Path:
    """Format raw MI3 .mat file and save in standardized format.
    
    Args:
        input_path: Path to raw .mat file with task_data/task_label/rest_data.
        output_path: Optional output path. If None, generates filename with sampling rate.
        subject_id: Optional subject ID for output filename (e.g., 'sub-017').
    
    Returns:
        Path to the saved formatted .mat file.
    
    Raises:
        FileNotFoundError: If input file doesn't exist.
        KeyError: If required keys missing from input file.
        ValueError: If data format is invalid.
    """
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise FileNotFoundError(msg)
    
    logger.info(f"Loading raw data from: {input_path}")
    mat_data = scio.loadmat(str(input_path))
    
    # Check for required keys
    required_keys = ['task_data', 'task_label', 'rest_data']
    missing_keys = [key for key in required_keys if key not in mat_data]
    if missing_keys:
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        msg = (
            f"Missing required keys: {missing_keys}\n"
            f"Available keys: {available_keys}"
        )
        raise KeyError(msg)
    
    # Extract raw data
    task_data = mat_data['task_data']
    task_label = mat_data['task_label']
    rest_data = mat_data['rest_data']
    
    # Convert to standardized format
    all_data, all_label, sampling_rate = convert_raw_format(
        task_data, task_label, rest_data
    )
    
    # Generate output path if not provided
    if output_path is None:
        if subject_id is None:
            # Extract subject ID from input filename
            filename = input_path.stem
            if 'sub-' in filename:
                # Extract sub-XXX from filename
                parts = filename.split('sub-')
                if len(parts) > 1:
                    subject_num = parts[1].split('_')[0].split('-')[0]
                    subject_id = f"sub-{subject_num}"
                else:
                    subject_id = "sub-unknown"
            else:
                subject_id = "sub-unknown"
        
        # Include sampling rate in filename: sub-017_eeg200hz.mat
        output_filename = f"{subject_id}_eeg{sampling_rate}hz.mat"
        output_path = input_path.parent / output_filename
    
    # Save formatted data
    logger.info(f"Saving formatted data to: {output_path}")
    save_dict = {
        'all_data': all_data,
        'all_label': all_label,
        'sampling_rate': sampling_rate,
        'source_file': str(input_path),
    }
    scio.savemat(str(output_path), save_dict)
    
    logger.info(f"✓ Formatting complete: {output_path}")
    logger.info(f"  Data shape: {all_data.shape}")
    logger.info(f"  Label shape: {all_label.shape}")
    logger.info(f"  Sampling rate: {sampling_rate}Hz")
    
    return output_path


def detect_format(mat_data: dict) -> str:
    """Detect the format of a .mat file.
    
    Args:
        mat_data: Dictionary loaded from .mat file.
    
    Returns:
        'standardized' if has all_data/all_label, 'raw' if has task_data/etc.
    
    Raises:
        ValueError: If format cannot be determined.
    """
    has_standardized = 'all_data' in mat_data and 'all_label' in mat_data
    has_raw = all(key in mat_data for key in ['task_data', 'task_label', 'rest_data'])
    
    if has_standardized:
        return 'standardized'
    elif has_raw:
        return 'raw'
    else:
        available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        msg = (
            f"Cannot determine format. Available keys: {available_keys}\n"
            f"Expected either ['all_data', 'all_label'] or "
            f"['task_data', 'task_label', 'rest_data']"
        )
        raise ValueError(msg)
