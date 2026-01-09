"""Visualization functions for training and evaluation results.

This module provides functions for creating plots and visualizations
of training progress, confusion matrices, and model comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from mi3_eeg.config import CLASS_COLORS
from mi3_eeg.logger import logger

if TYPE_CHECKING:
    from mi3_eeg.evaluation import EvaluationResults
    from mi3_eeg.train import TrainingHistory


def plot_training_curves(
    history: TrainingHistory,
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot training and validation curves.
    
    Args:
        history: TrainingHistory object with metrics.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
    
    Returns:
        Matplotlib figure object.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history.train_acc) + 1)
    
    # Accuracy plot
    ax1.plot(epochs, history.train_acc, "b-", label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, history.test_acc, "r-", label="Validation Accuracy", linewidth=2)
    ax1.axvline(
        x=history.best_epoch + 1,
        color="g",
        linestyle="--",
        label=f"Best Epoch ({history.best_epoch + 1})",
        alpha=0.7,
    )
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(epochs, history.train_loss, "b-", label="Train Loss", linewidth=2)
    ax2.plot(epochs, history.test_loss, "r-", label="Validation Loss", linewidth=2)
    ax2.axvline(
        x=history.best_epoch + 1,
        color="g",
        linestyle="--",
        label=f"Best Epoch ({history.best_epoch + 1})",
        alpha=0.7,
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Training curves saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_confusion_matrix(
    results: EvaluationResults,
    save_path: Path | None = None,
    show: bool = False,
    normalize: bool = False,
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        results: EvaluationResults with confusion matrix.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        normalize: Whether to normalize the confusion matrix.
    
    Returns:
        Matplotlib figure object.
    """
    cm = results.confusion_matrix
    
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
    else:
        fmt = "d"
    
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=results.class_names,
    )
    
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", values_format=fmt)
    
    ax.set_title(
        f"Confusion Matrix (Accuracy: {results.overall_accuracy * 100:.2f}%)",
        fontsize=14,
        fontweight="bold",
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_confusion_matrix_custom_colors(
    results: EvaluationResults,
    save_path: Path | None = None,
    show: bool = False,
    good_threshold: float = 0.5,
    bad_threshold: float = 0.2,
) -> plt.Figure:
    """Plot confusion matrix with custom coloring (green=good, red=bad).
    
    Similar to the notebook's custom visualization where:
    - Diagonal elements: green if high, red if low
    - Off-diagonal elements: red if high, green if low
    
    Args:
        results: EvaluationResults with confusion matrix.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
        good_threshold: Threshold for diagonal elements to be considered good.
        bad_threshold: Threshold for off-diagonal elements to be considered bad.
    
    Returns:
        Matplotlib figure object.
    """
    cm = results.confusion_matrix
    
    # Normalize by row
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    # Create custom color matrix
    num_classes = cm.shape[0]
    color_matrix = np.zeros((num_classes, num_classes, 3))
    
    cmap_greens = plt.cm.get_cmap("Greens")
    cmap_reds = plt.cm.get_cmap("Reds")
    
    for i in range(num_classes):
        for j in range(num_classes):
            norm_value = cm_normalized[i, j]
            
            if i == j:  # Diagonal (correct predictions)
                if norm_value > good_threshold:
                    color_val = norm_value
                    color_matrix[i, j, :] = cmap_greens(color_val)[:3]
                else:
                    color_val = 1.0 - norm_value
                    color_matrix[i, j, :] = cmap_reds(color_val)[:3]
            else:  # Off-diagonal (incorrect predictions)
                if norm_value > bad_threshold:
                    color_val = norm_value
                    color_matrix[i, j, :] = cmap_reds(color_val)[:3]
                else:
                    color_val = 1.0 - norm_value
                    color_matrix[i, j, :] = cmap_greens(color_val)[:3]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(color_matrix, aspect="auto")
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text_color = "white" if cm_normalized[i, j] > 0.5 else "black"
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
                fontweight="bold",
            )
    
    # Formatting
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(results.class_names)
    ax.set_yticklabels(results.class_names)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix - Custom Colors (Accuracy: {results.overall_accuracy * 100:.2f}%)",
        fontsize=14,
        fontweight="bold",
    )
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Custom confusion matrix saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_class_accuracies(
    results: dict[str, EvaluationResults],
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot comparison of per-class accuracies across models.
    
    Args:
        results: Dictionary mapping model names to EvaluationResults.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
    
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    model_names = list(results.keys())
    class_names = next(iter(results.values())).class_names
    
    x = np.arange(len(class_names))
    width = 0.8 / len(model_names)
    
    for idx, model_name in enumerate(model_names):
        accs = [
            results[model_name].class_accuracies[cn] * 100
            if not np.isnan(results[model_name].class_accuracies[cn])
            else 0
            for cn in class_names
        ]
        offset = (idx - len(model_names) / 2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Per-Class Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Class accuracies plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_overall_comparison(
    results: dict[str, EvaluationResults],
    save_path: Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot overall accuracy comparison across models.
    
    Args:
        results: Dictionary mapping model names to EvaluationResults.
        save_path: Optional path to save the figure.
        show: Whether to display the plot.
    
    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = list(results.keys())
    accuracies = [results[mn].overall_accuracy * 100 for mn in model_names]
    
    bars = ax.bar(model_names, accuracies, color=["steelblue", "coral", "lightgreen"][: len(model_names)])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Overall Model Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 100])
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Overall comparison plot saved to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def create_all_visualizations(
    training_history: dict[str, TrainingHistory],
    evaluation_results: dict[str, EvaluationResults],
    output_dir: Path,
) -> None:
    """Create and save all visualization plots.
    
    Args:
        training_history: Dictionary mapping model names to TrainingHistory.
        evaluation_results: Dictionary mapping model names to EvaluationResults.
        output_dir: Directory to save all plots.
    """
    logger.info("Creating all visualizations...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training curves for each model
    for model_name, history in training_history.items():
        plot_training_curves(
            history,
            save_path=output_dir / f"{model_name}_training_curves.png",
        )
    
    # Confusion matrices for each model
    for model_name, results in evaluation_results.items():
        plot_confusion_matrix(
            results,
            save_path=output_dir / f"{model_name}_confusion_matrix.png",
        )
        plot_confusion_matrix_custom_colors(
            results,
            save_path=output_dir / f"{model_name}_confusion_matrix_custom.png",
        )
    
    # Comparisons
    if len(evaluation_results) > 1:
        plot_class_accuracies(
            evaluation_results,
            save_path=output_dir / "class_accuracies_comparison.png",
        )
        plot_overall_comparison(
            evaluation_results,
            save_path=output_dir / "overall_accuracy_comparison.png",
        )
    
    plt.close("all")  # Close all figures to free memory
    logger.info(f"All visualizations saved to: {output_dir}")
