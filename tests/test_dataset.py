"""Tests for dataset module."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import pytest
import scipy.io as scio
import torch

from mi3_eeg.config import DataConfig, Paths
from mi3_eeg.dataset import (
    EEGDataBundle,
    _calculate_class_distribution,
    create_data_loader,
    load_dataset_from_config,
    load_mat_from_derivatives,
    prepare_data_loaders,
)


def _write_standardized_mat(mat_file: Path, data: np.ndarray, labels: np.ndarray) -> None:
    scio.savemat(str(mat_file), {"all_data": data, "all_label": labels})


def test_calculate_class_distribution() -> None:
    """Test class distribution calculation."""
    labels = np.array([[0], [0], [1], [1], [2], [2]])
    
    dist = _calculate_class_distribution(labels)
    
    assert dist["Rest"] == 2
    assert dist["Elbow"] == 2
    assert dist["Hand"] == 2


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
        expected_sampling_rate=90,
        validate_timepoints=False,
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
    
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_mat_from_derivatives(fake_path)


def test_load_mat_missing_keys(tmp_path: Path) -> None:
    """Test error when .mat file is missing required keys."""
    mat_file = tmp_path / "bad_data.mat"
    scio.savemat(str(mat_file), {"wrong_key": np.array([1, 2, 3])})
    
    with pytest.raises((KeyError, ValueError)):
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


def test_load_mat_with_short_timepoints(tmp_path: Path) -> None:
    """Test high-pass filter with short timepoint arrays."""
    data = np.random.randn(5, 62, 50).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1]).reshape(-1, 1)
    mat_file = tmp_path / "short_timepoints.mat"
    _write_standardized_mat(mat_file, data, labels)

    bundle = load_mat_from_derivatives(
        mat_path=mat_file,
        expected_sampling_rate=12,
        validate_timepoints=False,
    )

    assert bundle.data.shape == data.shape
    assert bundle.labels.shape == labels.shape


def test_load_mat_with_singleton_classes(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test stratified split fails when a class has one sample."""
    data, _ = sample_eeg_data
    labels = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(-1, 1)

    bundle = EEGDataBundle(
        data=data,
        labels=labels,
        channel_count=62,
        num_classes=2,
        sample_rate=90,
        class_distribution={"Rest": 1, "Elbow": 29, "Hand": 0},
    )

    config = DataConfig(val_size=0.2, test_size=0.2, random_seed=42)

    with pytest.raises(ValueError):
        prepare_data_loaders(bundle, config, device="cpu")


def test_load_mat_with_wrong_label_values(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test that label mismatch warnings are logged for unexpected labels."""
    data = np.random.randn(6, 62, 360).astype(np.float32)
    labels = np.array([0, 1, 3, 0, 1, 3]).reshape(-1, 1)
    mat_file = tmp_path / "wrong_labels.mat"
    _write_standardized_mat(mat_file, data, labels)

    logger = logging.getLogger("mi3_eeg")
    original_propagate = logger.propagate
    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING):
            load_mat_from_derivatives(
                mat_path=mat_file,
                expected_sampling_rate=90,
                validate_timepoints=False,
            )
    finally:
        logger.propagate = original_propagate

    assert any(
        "Label values mismatch" in record.message for record in caplog.records
    )


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


def test_create_data_loader_with_large_batch(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test DataLoader when batch_size exceeds dataset size."""
    data, labels = sample_eeg_data

    loader = create_data_loader(
        data,
        labels,
        batch_size=100,
        shuffle=False,
        device="cpu",
    )

    assert len(loader) == 1
    batch_data, batch_labels = next(iter(loader))
    assert batch_data.shape[0] == data.shape[0]
    assert batch_labels.shape[0] == labels.shape[0]


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
    
    config = DataConfig(val_size=0.1, test_size=0.1, random_seed=42)
    
    train_loader, val_loader, test_loader = prepare_data_loaders(
        bundle, config, device="cpu"
    )
    
    # Check sizes (30 samples -> 3 val, 3 test, 24 train)
    assert len(train_loader.dataset) == 24  # type: ignore[arg-type]
    assert len(val_loader.dataset) == 3  # type: ignore[arg-type]
    assert len(test_loader.dataset) == 3  # type: ignore[arg-type]


def test_prepare_data_loaders_with_zero_val_size(
    sample_eeg_data: tuple[np.ndarray, np.ndarray]
) -> None:
    """Test splitting behavior when val_size is zero."""
    data, labels = sample_eeg_data

    bundle = EEGDataBundle(
        data=data,
        labels=labels,
        channel_count=62,
        num_classes=3,
        sample_rate=90,
        class_distribution={"Rest": 10, "Elbow": 10, "Hand": 10},
    )

    config = DataConfig(val_size=0.0, test_size=0.1, random_seed=42)

    with pytest.raises(ValueError):
        prepare_data_loaders(bundle, config, device="cpu")


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


def test_load_mat_with_1d_labels(tmp_path: Path) -> None:
    """Test that 1D labels are reshaped to (samples, 1)."""
    data = np.random.randn(8, 62, 360).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    mat_file = tmp_path / "labels_1d.mat"
    _write_standardized_mat(mat_file, data, labels)

    bundle = load_mat_from_derivatives(
        mat_path=mat_file,
        expected_sampling_rate=90,
        validate_timepoints=False,
    )

    assert bundle.labels.shape == (8, 1)
    assert np.array_equal(bundle.labels.flatten(), labels)


def test_axis_swapping_edge_cases() -> None:
    """Test DataLoader axis swapping on ambiguous shapes."""
    data = np.arange(2 * 100 * 100).reshape(2, 100, 100)
    labels = np.array([[0], [1]])

    loader = create_data_loader(
        data,
        labels,
        batch_size=2,
        shuffle=False,
        device="cpu",
    )

    batch_data, _ = next(iter(loader))
    expected = data.swapaxes(1, 2)
    assert batch_data.shape == (2, 1, 100, 100)
    assert np.array_equal(batch_data[0, 0].cpu().numpy(), expected[0])


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
        reports_group_analysis=mock_paths / "reports" / "group_analysis",
    )
    
    # Copy temp mat file to mock derivatives folder
    import shutil
    dest_file = custom_paths.dataset_derivatives / "test_eeg.mat"
    shutil.copy(temp_mat_file, dest_file)
    
    config = DataConfig(mat_filename="test_eeg.mat")
    
    bundle = load_dataset_from_config(config=config, paths=custom_paths)
    
    assert isinstance(bundle, EEGDataBundle)
    # Check that all 30 samples are present (no downsampling)
    total_samples = sum(bundle.class_distribution.values())
    assert total_samples == 30
