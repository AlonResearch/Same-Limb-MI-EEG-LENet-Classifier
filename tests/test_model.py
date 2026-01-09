"""Tests for model module."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from mi3_eeg.config import ModelConfig
from mi3_eeg.model import (
    LENet,
    LENet_FCL,
    create_model,
    initialize_weights,
    load_model,
    save_model,
)


def test_lenet_initialization() -> None:
    """Test LENet model initialization."""
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    
    assert isinstance(model, nn.Module)
    assert model.classes_num == 3
    assert model.channel_count == 62
    assert model.drop_out == 0.5


def test_lenet_forward_pass(sample_tensor_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test LENet forward pass with correct output shape."""
    data, _ = sample_tensor_data
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    model.eval()
    
    with torch.no_grad():
        output = model(data)
    
    # Output should be (batch_size, classes_num)
    assert output.shape == (8, 3)
    
    # Output should be logits (unbounded values)
    assert torch.all(torch.isfinite(output))


def test_lenet_fcl_initialization() -> None:
    """Test LENet_FCL model initialization."""
    model = LENet_FCL(classes_num=3, channel_count=62, drop_out=0.5)
    
    assert isinstance(model, nn.Module)
    assert model.classes_num == 3
    assert model.channel_count == 62
    assert model.drop_out == 0.5
    assert model.fc is None  # FC layer created on first forward pass


def test_lenet_fcl_forward_pass(sample_tensor_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test LENet_FCL forward pass."""
    data, _ = sample_tensor_data
    model = LENet_FCL(classes_num=3, channel_count=62, drop_out=0.5)
    model.eval()
    
    with torch.no_grad():
        output = model(data)
    
    # Output should be (batch_size, classes_num)
    assert output.shape == (8, 3)
    
    # FC layer should be created after first forward pass
    assert model.fc is not None
    assert isinstance(model.fc, nn.Linear)


def test_lenet_fcl_fc_layer_creation(sample_tensor_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test that FC layer is created correctly on first forward pass."""
    data, _ = sample_tensor_data
    model = LENet_FCL(classes_num=3, channel_count=62, drop_out=0.5)
    
    # Before forward pass
    assert model.fc is None
    
    # After forward pass
    with torch.no_grad():
        _ = model(data)
    
    assert model.fc is not None
    assert model.fc.out_features == 3


def test_initialize_weights() -> None:
    """Test weight initialization function."""
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    
    # Get initial weights
    conv_layer = model.block_TCB_1[1]  # First conv layer
    initial_weight = conv_layer.weight.clone()
    
    # Initialize
    initialize_weights(model)
    
    # Weights should be initialized (different from initial random values)
    assert conv_layer.weight is not None
    
    # Check that weights are not all zeros
    assert not torch.all(conv_layer.weight == 0)


def test_create_model_lenet() -> None:
    """Test model creation with factory function - LENet."""
    config = ModelConfig(classes_num=3, channel_count=62, drop_out=0.35)
    model = create_model("lenet", config, device="cpu")
    
    assert isinstance(model, LENet)
    assert model.classes_num == 3
    assert model.channel_count == 62


def test_create_model_lenet_fcl() -> None:
    """Test model creation with factory function - LENet_FCL."""
    config = ModelConfig(classes_num=3, channel_count=62, drop_out=0.35)
    model = create_model("lenet_fcl", config, device="cpu")
    
    assert isinstance(model, LENet_FCL)
    assert model.classes_num == 3


def test_create_model_invalid_type() -> None:
    """Test error on invalid model type."""
    with pytest.raises(ValueError, match="Unknown model type"):
        create_model("invalid_model", device="cpu")


def test_create_model_case_insensitive() -> None:
    """Test that model type is case-insensitive."""
    model1 = create_model("LENET", device="cpu")
    model2 = create_model("LeNet", device="cpu")
    model3 = create_model("lenet", device="cpu")
    
    assert all(isinstance(m, LENet) for m in [model1, model2, model3])


def test_save_model(tmp_path: Path) -> None:
    """Test saving model weights."""
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    save_path = tmp_path / "test_model.pth"
    
    save_model(model, save_path)
    
    assert save_path.exists()
    
    # Check that we can load the saved weights
    loaded_state = torch.load(save_path, map_location="cpu", weights_only=True)
    assert isinstance(loaded_state, dict)
    assert len(loaded_state) > 0


def test_save_model_creates_directory(tmp_path: Path) -> None:
    """Test that save_model creates parent directories if needed."""
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    save_path = tmp_path / "nested" / "dir" / "model.pth"
    
    save_model(model, save_path)
    
    assert save_path.exists()
    assert save_path.parent.exists()


def test_load_model(tmp_path: Path) -> None:
    """Test loading model weights."""
    # Create and save a model
    original_model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    save_path = tmp_path / "test_model.pth"
    save_model(original_model, save_path)
    
    # Load the model
    loaded_model = load_model("lenet", save_path, device="cpu")
    
    assert isinstance(loaded_model, LENet)
    
    # Check that weights match
    for (name1, param1), (name2, param2) in zip(
        original_model.named_parameters(),
        loaded_model.named_parameters(),
        strict=False,
    ):
        assert name1 == name2
        assert torch.allclose(param1, param2)


def test_load_model_file_not_found() -> None:
    """Test error when loading non-existent model file."""
    fake_path = Path("/nonexistent/model.pth")
    
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model("lenet", fake_path, device="cpu")


def test_load_model_with_config(tmp_path: Path) -> None:
    """Test loading model with custom config."""
    config = ModelConfig(classes_num=3, channel_count=62, drop_out=0.4)
    original_model = LENet(
        classes_num=config.classes_num,
        channel_count=config.channel_count,
        drop_out=config.drop_out,
    )
    
    save_path = tmp_path / "model.pth"
    save_model(original_model, save_path)
    
    loaded_model = load_model("lenet", save_path, config=config, device="cpu")
    
    assert loaded_model.classes_num == 3
    assert loaded_model.channel_count == 62
    assert loaded_model.drop_out == 0.4


def test_model_training_mode() -> None:
    """Test that model can switch between train and eval modes."""
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    
    # Default should be training mode
    assert model.training
    
    # Switch to eval
    model.eval()
    assert not model.training
    
    # Switch back to train
    model.train()
    assert model.training


def test_model_device_placement() -> None:
    """Test that model can be placed on correct device."""
    model = create_model("lenet", device="cpu")
    
    # Check that model parameters are on CPU
    for param in model.parameters():
        assert param.device.type == "cpu"


def test_lenet_different_channel_counts() -> None:
    """Test LENet with different channel counts."""
    for channel_count in [22, 62, 128]:
        model = LENet(classes_num=3, channel_count=channel_count, drop_out=0.5)
        assert model.channel_count == channel_count
        
        # Test forward pass
        batch_size = 4
        timepoints = 360
        data = torch.randn(batch_size, 1, channel_count, timepoints)
        
        with torch.no_grad():
            output = model(data)
        
        assert output.shape == (batch_size, 3)


def test_model_gradient_flow(sample_tensor_data: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Test that gradients flow through the model correctly."""
    data, labels = sample_tensor_data
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    
    # Check that gradients exist and are non-zero
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            # At least some gradients should be non-zero
            if "weight" in name:
                assert torch.any(param.grad != 0), f"All gradients zero for {name}"
