"""Integration tests for the full pipeline."""

from __future__ import annotations

import pytest

from mi3_eeg.config import DataConfig, ModelConfig, Paths, TrainingConfig
from mi3_eeg.dataset import load_dataset_from_config, prepare_data_loaders
from mi3_eeg.evaluation import evaluate_model
from mi3_eeg.model import create_model
from mi3_eeg.train import train_model


def test_full_pipeline_with_real_data() -> None:
    """Test the complete pipeline with real MI3 data (if available)."""
    # Check if dataset exists
    paths = Paths.from_here()
    mat_path = paths.dataset_derivatives / "sub-011_eeg90hz.mat"
    
    if not mat_path.exists():
        pytest.skip("Real dataset not available for integration test")
    
    # Load data
    data_config = DataConfig()
    data_bundle = load_dataset_from_config(config=data_config, paths=paths)
    
    assert data_bundle.channel_count > 0
    assert data_bundle.num_classes == 3
    
    # Prepare loaders
    train_loader, test_loader = prepare_data_loaders(
        data_bundle, data_config, device="cpu"
    )
    
    assert len(train_loader.dataset) > 0  # type: ignore[arg-type]
    assert len(test_loader.dataset) > 0  # type: ignore[arg-type]
    
    # Create model
    model_config = ModelConfig(
        channel_count=data_bundle.channel_count,
        classes_num=data_bundle.num_classes,
        drop_out=0.5,
    )
    model = create_model("lenet", model_config, device="cpu")
    
    # Train for just a few epochs
    training_config = TrainingConfig(
        epochs=3,
        batch_size=64,
        device="cpu",
        early_stopping_patience=50,  # Don't trigger early stopping
    )
    
    history = train_model(model, train_loader, test_loader, training_config)
    
    # Check history
    assert len(history.train_acc) <= 3
    assert len(history.test_acc) <= 3
    assert 0 <= history.best_val_acc <= 1
    
    # Evaluate
    results = evaluate_model(model, test_loader, device="cpu")
    
    assert 0 <= results.overall_accuracy <= 1
    assert len(results.class_accuracies) == 3
    assert results.confusion_matrix.shape == (3, 3)


def test_package_imports() -> None:
    """Test that all main package components can be imported."""
    from mi3_eeg import (
        DataConfig,
        EEGDataBundle,
        EarlyStopper,
        EvaluationResults,
        LENet,
        ModelConfig,
        Paths,
        TrainingConfig,
        TrainingHistory,
        compare_models,
        create_data_loader,
        create_model,
        evaluate_model,
        get_predictions,
        initialize_weights,
        load_dataset_from_config,
        load_mat_from_derivatives,
        load_model,
        logger,
        plot_confusion_matrix,
        plot_training_curves,
        prepare_data_loaders,
        quick_train,
        save_model,
        setup_logger,
        train_model,
    )
    
    # Basic smoke tests
    assert Paths is not None
    assert logger is not None
    assert DataConfig is not None
