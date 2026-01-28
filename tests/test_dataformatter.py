"""Tests for data formatting and dataloader functionality.

Tests the conversion of raw MI3 data format to standardized format,
and verifies the dataloader works correctly with the formatted data.
"""

from pathlib import Path

import numpy as np
import pytest

from mi3_eeg.dataset import create_data_loader, load_mat_from_derivatives


class TestFormattedDataLoading:
    """Test suite for loading and processing formatted EEG data."""

    @pytest.fixture
    def formatted_data_path(self):
        """Path to the formatted sub-017 file (200Hz)."""
        path = Path("G:/My Drive/ML/dataset/sub-017_eeg200hz.mat")
        if not path.exists():
            pytest.skip(f"Test data not available at {path}")
        return path

    def test_load_formatted_file(self, formatted_data_path):
        """Test loading a formatted .mat file."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        assert data_bundle is not None
        assert data_bundle.data.shape == (900, 62, 800)
        assert data_bundle.labels.shape == (900, 1)

    def test_data_bundle_attributes(self, formatted_data_path):
        """Test that DataBundle has correct attributes."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        assert data_bundle.channel_count == 62
        assert data_bundle.num_classes == 3
        assert data_bundle.sample_rate == 200

    def test_label_values(self, formatted_data_path):
        """Test that labels are correct (0=Rest, 1=Elbow, 2=Hand)."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        unique_labels = np.unique(data_bundle.labels)
        expected_labels = np.array([0, 1, 2])

        assert np.array_equal(unique_labels, expected_labels), \
            f"Labels mismatch! Expected {expected_labels}, got {unique_labels}"

    def test_class_distribution(self, formatted_data_path):
        """Test that class distribution is balanced (300 each)."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        assert data_bundle.class_distribution["Rest"] == 300
        assert data_bundle.class_distribution["Elbow"] == 300
        assert data_bundle.class_distribution["Hand"] == 300

    def test_create_dataloader(self, formatted_data_path):
        """Test creating a PyTorch DataLoader from formatted data."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        dataloader = create_data_loader(
            data=data_bundle.data,
            labels=data_bundle.labels,
            batch_size=32,
            shuffle=True,
            device="cpu",
        )

        assert dataloader is not None
        assert len(dataloader) == 29  # 900 samples / 32 batch size = ~28 batches

    def test_dataloader_batch_shape(self, formatted_data_path):
        """Test that dataloader produces batches with correct shapes."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        dataloader = create_data_loader(
            data=data_bundle.data,
            labels=data_bundle.labels,
            batch_size=32,
            shuffle=True,
            device="cpu",
        )

        # Get first batch
        batch_data, batch_labels = next(iter(dataloader))

        # Check shapes
        assert batch_data.shape[0] == 32  # batch size
        assert batch_data.shape[1] == 1   # channel dimension
        assert batch_data.shape[2] == 62  # num channels
        assert batch_data.shape[3] == 800  # timepoints
        assert batch_labels.shape == (32,)

    def test_dataloader_contains_all_labels(self, formatted_data_path):
        """Test that dataloader batches contain all three label types."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        dataloader = create_data_loader(
            data=data_bundle.data,
            labels=data_bundle.labels,
            batch_size=32,
            shuffle=False,  # Don't shuffle for consistent testing
            device="cpu",
        )

        # Collect labels from multiple batches to ensure we get all classes
        all_batch_labels = []
        for i, (_, batch_labels) in enumerate(dataloader):
            all_batch_labels.extend(batch_labels.cpu().numpy())
            if i >= 15:  # Get enough batches to cover all classes
                break

        unique_in_batches = np.unique(all_batch_labels)
        expected_labels = np.array([0, 1, 2])

        assert np.array_equal(unique_in_batches, expected_labels), \
            f"Not all labels present in batches: {unique_in_batches}"

    def test_sampling_rate_200hz(self, formatted_data_path):
        """Test that sampling rate is correctly identified as 200Hz."""
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        assert data_bundle.sample_rate == 200
        # Verify: 200Hz * 4 seconds = 800 timepoints
        assert data_bundle.data.shape[2] == 800

    def test_timepoint_validation(self, formatted_data_path):
        """Test that timepoint validation works for 200Hz data."""
        # Should not raise error for correct sampling rate
        data_bundle = load_mat_from_derivatives(
            mat_path=formatted_data_path,
            reduce_rest_ratio=1.0,
            expected_sampling_rate=200,
            validate_timepoints=True,
        )

        assert data_bundle.data.shape[2] == 800  # 200Hz * 4s
