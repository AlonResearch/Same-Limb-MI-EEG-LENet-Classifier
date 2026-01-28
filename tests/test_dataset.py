"""Tests for dataset module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io as scio
import torch

from mi3_eeg.config import DataConfig, Paths
from mi3_eeg.dataset import (
    EEGDataBundle,
    _balance_rest_class,
    _calculate_class_distribution,
    create_data_loader,
    load_dataset_from_config,
    load_mat_from_derivatives,
    prepare_data_loaders,
)


def test_calculate_class_distribution() -> None:
    """Test class distribution calculation."""
    labels = np.array([[0], [0], [1], [1], [2], [2]])
    
    dist = _calculate_class_distribution(labels)
    
    assert dist["Rest"] == 2
    assert dist["Elbow"] == 2
    assert dist["Hand"] == 2


def test_balance_rest_class() -> None:
    """Test balancing of Rest class."""
    # Create imbalanced data: 10 Rest, 3 Elbow, 3 Hand
    data = np.random.randn(16, 62, 360)
    labels = np.array([0] * 10 + [1] * 3 + [2] * 3).reshape(-1, 1)
    
    # Keep only 30% of Rest samples
    balanced_data, balanced_labels = _balance_rest_class(
        data, labels, keep_ratio=0.3, random_seed=42
    )
    
    # Check that we have 3 Rest samples (30% of 10)
    rest_count = np.sum(balanced_labels == 0)
    assert rest_count == 3
    
    # Check that other classes are unchanged
    elbow_count = np.sum(balanced_labels == 1)
    hand_count = np.sum(balanced_labels == 2)
    assert elbow_count == 3
    assert hand_count == 3
    
    # Total should be 9 samples
    assert len(balanced_labels) == 9


def test_eeg_data_bundle_immutable() -> None:
    """Test that EEGDataBundle is immutable."""
    data = np.random.randn(10, 62, 360)
    labels = np.zeros((10, 1))
    
    bundle = EEGDataBundle(
        data=data,
        labels=labels,
        channel_count=62,
        num_classes=3,
        sample_rate=90,
        class_distribution={"Rest": 10, "Elbow": 0, "Hand": 0},
    )
    
    with pytest.raises(AttributeError):
        bundle.channel_count = 64  # type: ignore[misc]


def test_load_mat_from_derivatives(temp_mat_file: Path) -> None:
    """Test loading .mat file."""
    bundle = load_mat_from_derivatives(
        mat_path=temp_mat_file,
        reduce_rest_ratio=1.0,
        random_seed=42,
    )
    
    assert isinstance(bundle, EEGDataBundle)
    assert bundle.data.shape[0] == 30  # Number of samples
    assert bundle.data.shape[1] == 62  # Number of channels
    assert bundle.data.shape[2] == 360  # Number of timepoints
    assert bundle.channel_count == 62
    assert bundle.num_classes == 3
    assert bundle.sample_rate == 90


def test_load_mat_file_not_found() -> None:
    """Test error when .mat file doesn't exist."""
    fake_path = Path("/nonexistent/file.mat")
    
    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_mat_from_derivatives(fake_path)


def test_load_mat_missing_keys(tmp_path: Path) -> None:
    """Test error when .mat file is missing required keys."""
    mat_file = tmp_path / "bad_data.mat"
    scio.savemat(str(mat_file), {"wrong_key": np.array([1, 2, 3])})
    
    with pytest.raises(KeyError, match="Required key"):
        load_mat_from_derivatives(mat_file)


def test_create_data_loader_basic(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test basic DataLoader creation."""
    data, labels = sample_eeg_data
    
    loader = create_data_loader(
        data,
        labels,
        batch_size=8,
        shuffle=True,
        device="cpu",
    )
    
    assert len(loader.dataset) == 30  # type: ignore[arg-type]
    
    # Check one batch
    batch_data, batch_labels = next(iter(loader))
    assert batch_data.shape == (8, 1, 62, 360)  # (batch, 1, channels, timepoints)
    assert batch_labels.shape == (8,)
    assert batch_data.device.type == "cpu"


def test_create_data_loader_shapes(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test DataLoader handles different input shapes."""
    data, labels = sample_eeg_data
    
    # Test with data in shape (samples, channels, timepoints)
    loader = create_data_loader(data, labels, batch_size=4, device="cpu")
    batch_data, _ = next(iter(loader))
    
    assert batch_data.shape == (4, 1, 62, 360)


def test_create_data_loader_no_shuffle(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test DataLoader with shuffle=False."""
    data, labels = sample_eeg_data
    
    loader = create_data_loader(
        data,
        labels,
        batch_size=10,
        shuffle=False,
        device="cpu",
    )
    
    # Get first batch labels
    _, batch_labels = next(iter(loader))
    
    # Should match first 10 labels in order
    expected_labels = torch.LongTensor(labels[:10].flatten())
    assert torch.equal(batch_labels, expected_labels)


def test_prepare_data_loaders(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test preparing train/test DataLoaders."""
    data, labels = sample_eeg_data
    
    bundle = EEGDataBundle(
        data=data,
        labels=labels,
        channel_count=62,
        num_classes=3,
        sample_rate=90,
        class_distribution={"Rest": 10, "Elbow": 10, "Hand": 10},
    )
    
    config = DataConfig(test_size=0.2, random_seed=42)
    
    train_loader, test_loader = prepare_data_loaders(
        bundle, config, device="cpu"
    )
    
    # Check sizes (80/20 split of 30 samples)
    assert len(train_loader.dataset) == 24  # type: ignore[arg-type]
    assert len(test_loader.dataset) == 6  # type: ignore[arg-type]


def test_load_dataset_from_config_default() -> None:
    """Test loading dataset with default config."""
    # This test checks that the function works with the real dataset
    # Skip if dataset not available
    paths = Paths.from_here()
    mat_path = paths.dataset_derivatives / "sub-011_eeg90hz.mat"
    
    if not mat_path.exists():
        pytest.skip("Real dataset not available")
    
    bundle = load_dataset_from_config()
    
    assert isinstance(bundle, EEGDataBundle)
    assert bundle.channel_count > 0
    assert bundle.num_classes == 3


def test_load_dataset_with_custom_config(temp_mat_file: Path, mock_paths: Path) -> None:
    """Test loading dataset with custom config and paths."""
    # Create custom paths pointing to temp directory
    custom_paths = Paths(
        project_root=mock_paths,
        dataset_root=mock_paths / "Datasets" / "MI3",
        dataset_sourcedata=mock_paths / "Datasets" / "MI3" / "sourcedata",
        dataset_derivatives=mock_paths / "Datasets" / "MI3" / "derivatives",
        dataset_code=mock_paths / "Datasets" / "MI3" / "code",
        models=mock_paths / "models",
        reports_figures=mock_paths / "reports" / "figures",
        reports_metrics=mock_paths / "reports" / "metrics",
        reports_logs=mock_paths / "reports" / "logs",
    )
    
    # Copy temp mat file to mock derivatives folder
    import shutil
    dest_file = custom_paths.dataset_derivatives / "test_eeg.mat"
    shutil.copy(temp_mat_file, dest_file)
    
    config = DataConfig(mat_filename="test_eeg.mat", reduce_rest_ratio=0.5)
    
    bundle = load_dataset_from_config(config=config, paths=custom_paths)
    
    assert isinstance(bundle, EEGDataBundle)
    # Should have reduced Rest samples
    total_samples = sum(bundle.class_distribution.values())
    assert total_samples < 30  # Less than original due to Rest reduction
