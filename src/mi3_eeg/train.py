"""Training orchestration for EEG classification models.

This module handles model training, validation, and early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn, optim

from mi3_eeg.config import TuningConfig
from mi3_eeg.logger import logger

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


@dataclass
class EarlyStopper:
    """Early stopping to prevent overfitting.
    
    Monitors a metric and stops training if it doesn't improve
    for a specified number of epochs.
    
    Attributes:
        patience: Number of epochs to wait for improvement.
        min_delta: Minimum change to qualify as an improvement.
        best_metric: Best metric value seen so far.
        counter: Number of epochs without improvement.
    """

    patience: int = 40
    min_delta: float = 1e-4
    best_metric: float = -np.inf
    counter: int = 0

    def step(self, metric: float) -> bool:
        """Check if training should stop.
        
        Args:
            metric: Current epoch metric value (higher is better).
        
        Returns:
            True if training should stop, False otherwise.
        """
        if metric - self.best_metric > self.min_delta:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
        
        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset the early stopper state."""
        self.best_metric = -np.inf
        self.counter = 0


@dataclass(frozen=True)
class TrainingHistory:
    """Container for training history.
    
    Attributes:
        train_acc: List of training accuracies per epoch.
        test_acc: List of validation accuracies per epoch.
        train_loss: List of training losses per epoch.
        test_loss: List of validation losses per epoch.
        best_epoch: Epoch with best validation F1 score.
        best_val_acc: Best validation accuracy achieved.
        best_val_f1: Best validation F1 score achieved.
    """

    train_acc: list[float]
    test_acc: list[float]
    train_loss: list[float]
    test_loss: list[float]
    best_epoch: int
    best_val_acc: float
    best_val_f1: float


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cuda",
) -> tuple[float, float, float]:
    """Train model for one epoch.
    
    Args:
        model: Neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use for training.
    
    Returns:
        Tuple of (average_loss, accuracy, macro_f1).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        # Track statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_predictions.extend(predicted.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    macro_f1 = f1_score(all_targets, all_predictions, average="macro")

    return avg_loss, accuracy, macro_f1


def validate_one_epoch(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda",
) -> tuple[float, float, float]:
    """Validate model for one epoch.
    
    Args:
        model: Neural network model.
        test_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device to use for validation.
    
    Returns:
        Tuple of (average_loss, accuracy, macro_f1).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            all_predictions.extend(predicted.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    macro_f1 = f1_score(all_targets, all_predictions, average="macro")

    return avg_loss, accuracy, macro_f1


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TuningConfig,
    save_path: Path | None = None,
) -> TrainingHistory:
    """Train a model with the full training loop.
    
    Args:
        model: Neural network model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Tuning configuration with hyperparameters.
        save_path: Optional path to save best model weights.
    
    Returns:
        TrainingHistory with training metrics.
    """
    logger.info("Starting training...")
    logger.info(
        f"Config: epochs={config.epochs}, batch_size={config.batch_size}, "
        f"lr={config.learning_rate}, device={config.device}"
    )

    # Setup training components
    device = config.device
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    
    early_stopper = EarlyStopper(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
    )

    # Training history
    train_acc_history: list[float] = []
    test_acc_history: list[float] = []
    train_loss_history: list[float] = []
    test_loss_history: list[float] = []
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None

    # Training loop
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_acc_history.append(train_acc)
        train_loss_history.append(train_loss)

        # Validate
        val_loss, val_acc, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device
        )
        test_acc_history.append(val_acc)
        test_loss_history.append(val_loss)

        # Update learning rate
        lr_scheduler.step()

        # Log progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.epochs} | "
                f"Train Acc: {train_acc * 100:.2f}%, Train F1: {train_f1 * 100:.2f}% | "
                f"Val Acc: {val_acc * 100:.2f}%, Val F1: {val_f1 * 100:.2f}%"
            )

        # Track best model
        if val_f1 > best_val_f1 or best_model_state is None:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            
            if save_path is not None:
                torch.save(best_model_state, save_path)
                logger.debug(f"Best model saved at epoch {epoch + 1}")

        # Early stopping check
        if early_stopper.step(val_f1):
            logger.info(
                f"Early stopping triggered at epoch {epoch + 1}. "
                f"Best val F1: {best_val_f1 * 100:.2f}% at epoch {best_epoch + 1}"
            )
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model from epoch {best_epoch + 1}")

    logger.info(
        f"Training completed. Best validation F1: {best_val_f1 * 100:.2f}%"
    )

    return TrainingHistory(
        train_acc=train_acc_history,
        test_acc=test_acc_history,
        train_loss=train_loss_history,
        test_loss=test_loss_history,
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_f1=best_val_f1,
    )


def quick_train(
    model: nn.Module,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 500,
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[float, float]:
    """Quick training function (simplified interface).
    
    This function provides a simpler interface for quick training,
    similar to the original notebook function.
    
    Args:
        model: Neural network model.
        train_data: Training data array.
        train_labels: Training labels array.
        val_data: Validation data array.
        val_labels: Validation labels array.
        epochs: Number of training epochs.
        batch_size: Batch size.
        device: Device to use.
    
    Returns:
        Tuple of (final_train_acc, final_val_acc).
    """
    from mi3_eeg.dataset import create_data_loader

    # Create DataLoaders
    train_loader = create_data_loader(
        train_data, train_labels, batch_size=batch_size, device=device
    )
    val_loader = create_data_loader(
        val_data, val_labels, batch_size=batch_size, device=device
    )

    # Create config
    config = TuningConfig(
        epochs=epochs,
        batch_size=batch_size,
        device=device,
    )

    # Train
    history = train_model(model, train_loader, val_loader, config)

    return history.train_acc[-1], history.test_acc[-1]
