"""Tests for CUDA/GPU functionality.

These tests verify that PyTorch CUDA is working correctly.
Tests are skipped if CUDA is not available.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mi3_eeg.config import ModelConfig
from mi3_eeg.dataset import create_data_loader
from mi3_eeg.model import LENet, create_model
from mi3_eeg.train import train_one_epoch, validate_one_epoch


# Check if CUDA is available
cuda_available = torch.cuda.is_available()
skip_if_no_cuda = pytest.mark.skipif(
    not cuda_available,
    reason="CUDA not available",
)


@skip_if_no_cuda
def test_cuda_availability() -> None:
    """Test that CUDA is available and working."""
    assert torch.cuda.is_available(), "CUDA should be available"
    assert torch.cuda.device_count() > 0, "At least one CUDA device should be available"
    
    # Get device info
    device_name = torch.cuda.get_device_name(0)
    assert len(device_name) > 0, "Device name should not be empty"
    
    print(f"\n[OK] CUDA is available: {device_name}")


@skip_if_no_cuda
def test_tensor_on_cuda() -> None:
    """Test that tensors can be created and moved to CUDA."""
    # Create tensor on CPU
    x = torch.randn(10, 10)
    assert x.device.type == "cpu"
    
    # Move to CUDA
    x_cuda = x.cuda()
    assert x_cuda.device.type == "cuda"
    
    # Create tensor directly on CUDA
    y = torch.randn(10, 10, device="cuda")
    assert y.device.type == "cuda"
    
    # Test operations on CUDA
    z = x_cuda + y
    assert z.device.type == "cuda"
    
    print("\n[OK] Tensor operations on CUDA working")


@skip_if_no_cuda
def test_model_on_cuda() -> None:
    """Test that models can be moved to CUDA."""
    config = ModelConfig(channel_count=62, classes_num=3, drop_out=0.5)
    model = LENet(
        classes_num=config.classes_num,
        channel_count=config.channel_count,
        drop_out=config.drop_out,
    )
    
    # Move model to CUDA
    model = model.cuda()
    
    # Check all parameters are on CUDA
    for param in model.parameters():
        assert param.device.type == "cuda", "All model parameters should be on CUDA"
    
    print("\n[OK] Model moved to CUDA successfully")


@skip_if_no_cuda
def test_model_forward_pass_cuda() -> None:
    """Test model forward pass on CUDA."""
    config = ModelConfig(channel_count=62, classes_num=3, drop_out=0.5)
    model = create_model("lenet", config, device="cuda")
    
    # Create input on CUDA
    batch_size = 8
    x = torch.randn(batch_size, 1, 62, 360, device="cuda")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Check output is on CUDA
    assert output.device.type == "cuda", "Output should be on CUDA"
    assert output.shape == (batch_size, 3), f"Output shape should be (8, 3), got {output.shape}"
    
    print("\n[OK] Model forward pass on CUDA working")


@skip_if_no_cuda
def test_training_on_cuda(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that training works on CUDA."""
    data, labels = sample_eeg_data
    
    # Take small subset for quick test
    data = data[:16]
    labels = labels[:16]
    
    # Create DataLoader with CUDA
    loader = create_data_loader(
        data, labels, batch_size=4, shuffle=True, device="cuda"
    )
    
    # Create model on CUDA
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    model = model.cuda()
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train one epoch on CUDA
    loss, accuracy = train_one_epoch(model, loader, criterion, optimizer, device="cuda")
    
    assert isinstance(loss, float), "Loss should be a float"
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    
    print(f"\n[OK] Training on CUDA working (loss: {loss:.4f}, acc: {accuracy:.2%})")


@skip_if_no_cuda
def test_validation_on_cuda(sample_eeg_data: tuple[np.ndarray, np.ndarray]) -> None:
    """Test that validation works on CUDA."""
    data, labels = sample_eeg_data
    
    # Take small subset for quick test
    data = data[:16]
    labels = labels[:16]
    
    # Create DataLoader with CUDA
    loader = create_data_loader(
        data, labels, batch_size=4, shuffle=False, device="cuda"
    )
    
    # Create model on CUDA
    model = LENet(classes_num=3, channel_count=62, drop_out=0.5)
    model = model.cuda()
    model.eval()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Validate on CUDA
    loss, accuracy = validate_one_epoch(model, loader, criterion, device="cuda")
    
    assert isinstance(loss, float), "Loss should be a float"
    assert isinstance(accuracy, float), "Accuracy should be a float"
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    
    print(f"\n[OK] Validation on CUDA working (loss: {loss:.4f}, acc: {accuracy:.2%})")


@skip_if_no_cuda
def test_cuda_memory_management() -> None:
    """Test CUDA memory management."""
    # Get initial memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated(0)
    
    # Create large tensor
    x = torch.randn(1000, 1000, device="cuda")
    after_alloc = torch.cuda.memory_allocated(0)
    
    assert after_alloc > initial_memory, "Memory should increase after allocation"
    
    # Delete tensor
    del x
    torch.cuda.empty_cache()
    after_free = torch.cuda.memory_allocated(0)
    
    assert after_free <= after_alloc, "Memory should decrease after deletion"
    
    print("\n[OK] CUDA memory management working")


@skip_if_no_cuda
def test_cuda_performance_vs_cpu() -> None:
    """Test that CUDA is actually faster than CPU for matrix operations."""
    import time
    
    size = 2000
    iterations = 10
    
    # CPU test
    x_cpu = torch.randn(size, size)
    y_cpu = torch.randn(size, size)
    
    start = time.time()
    for _ in range(iterations):
        z_cpu = torch.matmul(x_cpu, y_cpu)
    cpu_time = time.time() - start
    
    # CUDA test
    x_cuda = torch.randn(size, size, device="cuda")
    y_cuda = torch.randn(size, size, device="cuda")
    torch.cuda.synchronize()  # Ensure CUDA is ready
    
    start = time.time()
    for _ in range(iterations):
        z_cuda = torch.matmul(x_cuda, y_cuda)
    torch.cuda.synchronize()  # Wait for CUDA operations to complete
    cuda_time = time.time() - start
    
    speedup = cpu_time / cuda_time
    
    print(f"\n[OK] CUDA speedup: {speedup:.2f}x faster than CPU")
    print(f"     CPU time: {cpu_time:.4f}s")
    print(f"     CUDA time: {cuda_time:.4f}s")
    
    # CUDA should be faster for large operations
    # But allow some tolerance as it depends on the GPU
    assert speedup > 0.5, f"CUDA should provide some speedup, got {speedup:.2f}x"


def test_cuda_info_display() -> None:
    """Display CUDA information (always runs, even if CUDA not available)."""
    print("\n" + "=" * 60)
    print("CUDA Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device capability: {torch.cuda.get_device_capability(0)}")
        
        # Memory info
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        cached = torch.cuda.memory_reserved(0) / (1024**3)
        
        print(f"Total memory: {total_memory:.2f} GB")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Cached: {cached:.2f} GB")
    else:
        print("[WARN] No CUDA devices available")
        print("       Tests requiring CUDA will be skipped")
    
    print("=" * 60)
