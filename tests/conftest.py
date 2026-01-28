"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def sample_eeg_data() -> tuple[np.ndarray, np.ndarray]:
    """Create sample EEG data for testing.
    
    Returns:
        Tuple of (data, labels) where:
            - data shape: (samples, channels, timepoints)
            - labels shape: (samples, 1)
    """
    np.random.seed(42)
    samples = 30
    channels = 62
    timepoints = 360
    num_classes = 3
    
    # Create synthetic EEG data
    data = np.random.randn(samples, channels, timepoints).astype(np.float32)
    
    # Create balanced labels
    labels = np.repeat(np.arange(num_classes), samples // num_classes)
    labels = labels.reshape(-1, 1)
    
    return data, labels


@pytest.fixture
def sample_tensor_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample tensor data for testing PyTorch models.
    
    Returns:
        Tuple of (data, labels) as PyTorch tensors.
    """
    torch.manual_seed(42)
    batch_size = 8
    channels = 62
    timepoints = 360
    num_classes = 3
    
    # Create synthetic tensor data
    data = torch.randn(batch_size, 1, channels, timepoints)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    return data, labels


@pytest.fixture
def temp_mat_file(tmp_path: Path, sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> Path:
    """Create a temporary .mat file with sample EEG data.
    
    Args:
        tmp_path: Pytest temporary directory fixture.
        sample_eeg_data: Sample EEG data fixture.
    
    Returns:
        Path to the created .mat file.
    """
    import scipy.io as scio
    
    data, labels = sample_eeg_data
    mat_file = tmp_path / "test_eeg.mat"
    
    # Save as .mat file
    scio.savemat(str(mat_file), {"all_data": data, "all_label": labels})
    
    return mat_file


@pytest.fixture
def mock_paths(tmp_path: Path) -> Path:
    """Create a mock project directory structure for testing.
    
    Args:
        tmp_path: Pytest temporary directory fixture.
    
    Returns:
        Path to the temporary project root.
    """
    # Create directory structure
    (tmp_path / "Datasets" / "MI3" / "derivatives").mkdir(parents=True)
    (tmp_path / "Datasets" / "MI3" / "sourcedata").mkdir(parents=True)
    (tmp_path / "Datasets" / "MI3" / "code").mkdir(parents=True)
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    (tmp_path / "reports" / "metrics").mkdir(parents=True)
    (tmp_path / "reports" / "logs").mkdir(parents=True)
    
    return tmp_path
