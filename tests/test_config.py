"""Tests for configuration module."""

from __future__ import annotations

from pathlib import Path

import pytest

from mi3_eeg.config import (
    CLASS_LABELS,
    DataConfig,
    ModelConfig,
    Paths,
    TrainingConfig,
)


def test_paths_from_here() -> None:
    """Test that Paths.from_here() creates valid paths."""
    paths = Paths.from_here()
    
    # Check that all paths are Path objects
    assert isinstance(paths.project_root, Path)
    assert isinstance(paths.dataset_root, Path)
    assert isinstance(paths.models, Path)
    
    # Check that paths are absolute
    assert paths.project_root.is_absolute()
    assert paths.dataset_root.is_absolute()
    
    # Check expected directory names
    assert paths.dataset_root.name == "MI3"
    assert "Datasets" in str(paths.dataset_root)


def test_paths_structure() -> None:
    """Test that Paths contains all necessary directories."""
    paths = Paths.from_here()
    
    # Dataset paths (BIDS structure)
    assert paths.dataset_sourcedata == paths.dataset_root / "sourcedata"
    assert paths.dataset_derivatives == paths.dataset_root / "derivatives"
    assert paths.dataset_code == paths.dataset_root / "code"
    
    # Output paths
    assert paths.models == paths.project_root / "models"
    assert paths.reports_figures == paths.project_root / "reports" / "figures"
    assert paths.reports_metrics == paths.project_root / "reports" / "metrics"
    assert paths.reports_logs == paths.project_root / "reports" / "logs"


def test_paths_create_directories(tmp_path: Path) -> None:
    """Test directory creation method."""
    # Create a test paths instance with tmp_path
    test_paths = Paths(
        project_root=tmp_path,
        dataset_root=tmp_path / "Datasets" / "MI3",
        dataset_sourcedata=tmp_path / "Datasets" / "MI3" / "sourcedata",
        dataset_derivatives=tmp_path / "Datasets" / "MI3" / "derivatives",
        dataset_code=tmp_path / "Datasets" / "MI3" / "code",
        models=tmp_path / "models",
        reports_figures=tmp_path / "reports" / "figures",
        reports_metrics=tmp_path / "reports" / "metrics",
        reports_logs=tmp_path / "reports" / "logs",
    )
    
    # Create directories
    test_paths.create_directories()
    
    # Verify directories exist
    assert test_paths.models.exists()
    assert test_paths.reports_figures.exists()
    assert test_paths.reports_metrics.exists()
    assert test_paths.reports_logs.exists()


def test_data_config_immutable() -> None:
    """Test that DataConfig is immutable."""
    config = DataConfig()
    
    with pytest.raises(AttributeError):
        config.test_size = 0.3  # type: ignore[misc]


def test_model_config_immutable() -> None:
    """Test that ModelConfig is immutable."""
    config = ModelConfig()
    
    with pytest.raises(AttributeError):
        config.drop_out = 0.5  # type: ignore[misc]


def test_class_labels_constant() -> None:
    """Test CLASS_LABELS constant."""
    assert CLASS_LABELS == {0: "Rest", 1: "Elbow", 2: "Hand"}
    assert len(CLASS_LABELS) == 3
    assert all(isinstance(k, int) for k in CLASS_LABELS.keys())
    assert all(isinstance(v, str) for v in CLASS_LABELS.values())
