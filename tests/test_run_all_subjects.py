"""Tests for batch processing utilities in run_all_subjects."""

from pathlib import Path

import numpy as np
import scipy.io as scio

from mi3_eeg.run_all_subjects import (
    ask_user_proceed,
    extract_subject_id,
    validate_all_files,
    validate_mat_file,
)


def test_validate_corrupted_mat_file(tmp_path: Path) -> None:
    """Test that corrupted .mat files are detected."""
    bad_file = tmp_path / "corrupted.mat"
    bad_file.write_bytes(b"not a real mat file")

    is_valid, error_msg = validate_mat_file(bad_file)

    assert is_valid is False
    assert error_msg is not None


def test_validate_all_files_with_mixed_validity(tmp_path: Path) -> None:
    """Test validation with some valid, some invalid files."""
    valid1 = tmp_path / "valid1.mat"
    valid2 = tmp_path / "valid2.mat"
    scio.savemat(str(valid1), {"all_data": np.zeros((10, 62, 360)), "all_label": np.zeros((10, 1))})
    scio.savemat(str(valid2), {"all_data": np.zeros((10, 62, 360)), "all_label": np.zeros((10, 1))})

    invalid1 = tmp_path / "invalid1.mat"
    invalid1.write_bytes(b"corrupted")

    all_files = [valid1, valid2, invalid1]
    valid_files, invalid_files, results = validate_all_files(all_files)

    assert len(valid_files) == 2
    assert len(invalid_files) == 1
    assert valid1 in valid_files
    assert valid2 in valid_files
    assert invalid1 in invalid_files
    assert results[invalid1.name][0] is False


def test_ask_user_proceed_no_valid_files() -> None:
    """Test that processing stops when no valid files exist."""
    should_proceed = ask_user_proceed(valid_count=0, invalid_count=2)

    assert should_proceed is False


def test_extract_subject_id() -> None:
    """Test subject ID parsing from various filename formats."""
    assert extract_subject_id("sub-001_eeg200hz.mat") == "sub-001"
    assert extract_subject_id("sub-001.mat") == "sub-001"
    assert extract_subject_id("subject-001.mat") == "subject-001"
