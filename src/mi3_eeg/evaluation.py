"""Evaluation and metrics computation for trained models.

This module handles model evaluation, predictions, confusion matrices,
and metric computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from mi3_eeg.config import CLASS_LABELS
from mi3_eeg.logger import logger

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EvaluationResults:
    """Container for evaluation results.
    
    Attributes:
        predictions: Model predictions for each sample.
        true_labels: Ground truth labels.
        overall_accuracy: Overall classification accuracy.
        class_accuracies: Per-class accuracy dictionary.
        confusion_matrix: Confusion matrix array.
        class_names: List of class names.
    """

    predictions: np.ndarray
    true_labels: np.ndarray
    overall_accuracy: float
    class_accuracies: dict[str, float]
    confusion_matrix: np.ndarray
    class_names: list[str]


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Get model predictions for a dataset.
    
    Args:
        model: Trained model.
        data_loader: DataLoader with data to predict on.
        device: Device to use for inference.
    
    Returns:
        Tuple of (predictions, true_labels) as numpy arrays.
    """
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)


def compute_class_accuracies(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-class accuracies.
    
    Args:
        predictions: Model predictions.
        true_labels: Ground truth labels.
        class_names: List of class names. If None, uses default CLASS_LABELS.
    
    Returns:
        Dictionary mapping class names to their accuracies.
    """
    if class_names is None:
        class_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    
    class_accuracies = {}
    
    for class_idx, class_name in enumerate(class_names):
        # Get indices for this class
        class_indices = np.where(true_labels == class_idx)[0]
        
        if len(class_indices) > 0:
            class_predictions = predictions[class_indices]
            class_true = true_labels[class_indices]
            class_acc = accuracy_score(class_true, class_predictions)
            class_accuracies[class_name] = class_acc
        else:
            class_accuracies[class_name] = np.nan
    
    return class_accuracies


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = "cuda",
    class_names: list[str] | None = None,
) -> EvaluationResults:
    """Evaluate a trained model and compute all metrics.
    
    Args:
        model: Trained model to evaluate.
        data_loader: DataLoader with evaluation data.
        device: Device to use for inference.
        class_names: List of class names. If None, uses default.
    
    Returns:
        EvaluationResults with all metrics.
    """
    logger.info("Evaluating model...")
    
    if class_names is None:
        class_names = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    
    # Get predictions
    predictions, true_labels = get_predictions(model, data_loader, device)
    
    # Compute overall accuracy
    overall_acc = accuracy_score(true_labels, predictions)
    
    # Compute per-class accuracies
    class_accs = compute_class_accuracies(predictions, true_labels, class_names)
    
    # Compute confusion matrix
    cm = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(len(class_names))),
    )
    
    logger.info(f"Overall Accuracy: {overall_acc * 100:.2f}%")
    for class_name, acc in class_accs.items():
        if not np.isnan(acc):
            logger.info(f"{class_name} Accuracy: {acc * 100:.2f}%")
    
    return EvaluationResults(
        predictions=predictions,
        true_labels=true_labels,
        overall_accuracy=overall_acc,
        class_accuracies=class_accs,
        confusion_matrix=cm,
        class_names=class_names,
    )


def compare_models(
    models: dict[str, nn.Module],
    data_loader: DataLoader,
    device: str = "cuda",
) -> dict[str, EvaluationResults]:
    """Evaluate and compare multiple models.
    
    Args:
        models: Dictionary mapping model names to model instances.
        data_loader: DataLoader with evaluation data.
        device: Device to use for inference.
    
    Returns:
        Dictionary mapping model names to their EvaluationResults.
    """
    logger.info(f"Comparing {len(models)} models...")
    
    results = {}
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}...")
        results[model_name] = evaluate_model(model, data_loader, device)
    
    return results


def save_evaluation_results(
    results: EvaluationResults,
    save_path: Path,
    model_name: str = "model",
) -> None:
    """Save evaluation results to a JSON file.
    
    Args:
        results: EvaluationResults to save.
        save_path: Path to save the results.
        model_name: Name of the model for labeling.
    """
    import json
    
    # Convert to serializable format
    results_dict = {
        "model_name": model_name,
        "overall_accuracy": float(results.overall_accuracy),
        "class_accuracies": {
            k: float(v) if not np.isnan(v) else None
            for k, v in results.class_accuracies.items()
        },
        "confusion_matrix": results.confusion_matrix.tolist(),
        "class_names": results.class_names,
        "num_samples": len(results.true_labels),
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"Evaluation results saved to: {save_path}")


def print_evaluation_summary(results: dict[str, EvaluationResults]) -> None:
    """Print a formatted summary table of evaluation results.
    
    Args:
        results: Dictionary mapping model names to EvaluationResults.
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)
    
    # Header
    class_names = next(iter(results.values())).class_names
    header = f"{'Model':<20} | {'Overall Acc.':<12}"
    for class_name in class_names:
        header += f" | {class_name + ' Acc.':<12}"
    print(header)
    print("-" * 80)
    
    # Results for each model
    for model_name, result in results.items():
        row = f"{model_name:<20} | {result.overall_accuracy * 100:>11.2f}%"
        for class_name in class_names:
            acc = result.class_accuracies[class_name]
            if np.isnan(acc):
                row += f" | {'N/A':>12}"
            else:
                row += f" | {acc * 100:>11.2f}%"
        print(row)
    
    print("=" * 80 + "\n")
