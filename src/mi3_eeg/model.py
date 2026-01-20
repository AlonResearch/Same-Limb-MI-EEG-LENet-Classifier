"""Neural network models for EEG motor imagery classification.

This module contains the LENet-based architectures for classifying
EEG motor imagery signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from mi3_eeg.config import ModelConfig
from mi3_eeg.logger import logger

if TYPE_CHECKING:
    from pathlib import Path


class LENet(nn.Module):
    """LENet Model with Classification Convolution Block (CCB).
    
    This model uses multi-scale temporal convolutions followed by
    spatial convolutions and feature fusion for EEG classification.
    
    Architecture:
        1. Three parallel temporal convolution blocks (64, 32, 16 kernels)
        2. Fusion layer
        3. Spatial convolution block
        4. Feature fusion convolution block
        5. Classification convolution block with adaptive pooling
    
    Input shape:x
        (batch_size, 1, channels, timepoints)
        Example: (64, 1, 62, 360) for MI3 dataset
    
    Output:
        (batch_size, classes_num) logits for each class
    
    Attributes:
        classes_num: Number of output classes.
        channel_count: Number of EEG channels.
        drop_out: Dropout probability.
    """

    def __init__(
        self,
        classes_num: int = 3,
        channel_count: int = 62,
        drop_out: float = 0.5,
    ) -> None:
        """Initialize LENet model.
        
        Args:
            classes_num: Number of output classes.
            channel_count: Number of input EEG channels.
            drop_out: Dropout probability for regularization.
        """
        super().__init__()
        self.classes_num = classes_num
        self.channel_count = channel_count
        self.drop_out = drop_out

        # Temporal Convolution Block 1: kernel_size (1, 64)
        self.block_TCB_1 = nn.Sequential(
            nn.ZeroPad2d((32, 31, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 64),
                bias=False,
            ),
            nn.BatchNorm2d(8),
        )

        # Temporal Convolution Block 2: kernel_size (1, 32)
        self.block_TCB_2 = nn.Sequential(
            nn.ZeroPad2d((16, 15, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 32),
                bias=False,
            ),
            nn.BatchNorm2d(8),
        )

        # Temporal Convolution Block 3: kernel_size (1, 16)
        self.block_TCB_3 = nn.Sequential(
            nn.ZeroPad2d((8, 7, 0, 0)),
            nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(1, 16),
                bias=False,
            ),
            nn.BatchNorm2d(8),
        )

        # Temporal Convolution Block Fusion: kernel_size (1, 1)
        self.TCB_fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=24,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(24),
        )

        # Spatial Convolution Block: kernel_size (channels, 1)
        self.SCB = nn.Sequential(
            nn.Conv2d(
                in_channels=24,
                out_channels=16,
                kernel_size=(channel_count, 1),
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out),
        )

        # Feature Fusion Convolution Block: kernel_size (1, 16) and (1, 1)
        self.FFCB = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 16),
                groups=16,
                bias=False,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_out),
        )

        # Classification Convolution Block: kernel_size (1, 1)
        self.CCB = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=classes_num,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor, shape (batch, 1, channels, timepoints).
        
        Returns:
            Output logits, shape (batch, classes_num).
        """
        # Multi-scale temporal convolutions
        x1 = self.block_TCB_1(x)
        x2 = self.block_TCB_2(x)
        x3 = self.block_TCB_3(x)
        
        # Concatenate and fuse
        x4 = torch.cat([x1, x2, x3], dim=1)
        x = self.TCB_fusion(x4)
        
        # Spatial and feature fusion
        x = self.SCB(x)
        x = self.FFCB(x)
        
        # Classification
        x = self.CCB(x)
        
        return x


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights using Kaiming initialization.
    
    Args:
        model: PyTorch model to initialize.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    logger.info("Model weights initialized with Kaiming initialization")


def create_model(
    model_type: str = "lenet",
    config: ModelConfig | None = None,
    device: str = "cuda",
) -> nn.Module:
    """Factory function to create a model.
    
    Args:
        model_type: Type of model (only 'lenet' supported). Default 'lenet'.
        config: ModelConfig instance. If None, uses defaults.
        device: Device to place model on ('cuda' or 'cpu').
    
    Returns:
        Initialized model on specified device.
    
    Raises:
        ValueError: If model_type is not 'lenet'.
    """
    if config is None:
        config = ModelConfig()
    
    model_type_lower = model_type.lower()
    
    if model_type_lower != "lenet":
        msg = f"Unknown model type: {model_type}. Only 'lenet' is supported."
        raise ValueError(msg)
    
    model = LENet(
        classes_num=config.classes_num,
        channel_count=config.channel_count,
        drop_out=config.drop_out,
    )
    
    model = model.to(device)
    initialize_weights(model)
    
    logger.info(f"Created {model_type} model on {device}")
    
    return model


def save_model(model: nn.Module, save_path: Path) -> None:
    """Save model state dict to file.
    
    Args:
        model: PyTorch model to save.
        save_path: Path to save the model weights.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to: {save_path}")


def load_model(
    model_type: str,
    load_path: Path,
    config: ModelConfig | None = None,
    device: str = "cuda",
) -> nn.Module:
    """Load a trained model from file.
    
    Args:
        model_type: Type of model ('lenet' or 'lenet_fcl').
        load_path: Path to saved model weights.
        config: ModelConfig instance. If None, uses defaults.
        device: Device to place model on.
    
    Returns:
        Loaded model on specified device.
    
    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    if not load_path.exists():
        msg = f"Model file not found: {load_path}"
        raise FileNotFoundError(msg)
    
    # Create model architecture
    model = create_model(model_type, config, device)
    
    # Load weights
    state_dict = torch.load(load_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    
    logger.info(f"Model loaded from: {load_path}")
    
    return model
