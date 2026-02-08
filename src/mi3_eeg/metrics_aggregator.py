"""Aggregate evaluation metrics from all subjects into a comprehensive table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mi3_eeg.logger import logger


def load_subject_results(metrics_dir: Path, subject_id: str, model_name: str = "lenet") -> dict[str, Any] | None:
    """Load evaluation results for a single subject.
    
    Args:
        metrics_dir: Path to the metrics directory.
        subject_id: Subject identifier (e.g., "sub-001").
        model_name: Name of the model.
    
    Returns:
        Dictionary with results or None if file doesn't exist.
    """
    results_file = metrics_dir / f"{subject_id}_{model_name}_results.json"
    
    if not results_file.exists():
        return None
    
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load {results_file}: {e}")
        return None


def compute_metrics_from_confusion_matrix(
    cm: list[list[int]], class_names: list[str]
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, and F1 from confusion matrix.
    
    Args:
        cm: Confusion matrix as list of lists.
        class_names: List of class names.
    
    Returns:
        Dictionary with precision, recall, and F1 scores per class.
    """
    cm_array = np.array(cm)
    metrics = {}
    
    for class_idx, class_name in enumerate(class_names):
        # True Positives: diagonal element
        tp = cm_array[class_idx, class_idx]
        
        # False Positives: sum of column minus TP
        fp = cm_array[:, class_idx].sum() - tp
        
        # False Negatives: sum of row minus TP
        fn = cm_array[class_idx, :].sum() - tp
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1: 2 * (precision * recall) / (precision + recall)
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    
    return metrics


def aggregate_all_metrics(metrics_dir: Path, model_name: str = "lenet") -> pd.DataFrame:
    """Aggregate metrics from all subjects into a single DataFrame.
    
    Args:
        metrics_dir: Path to the metrics directory.
        model_name: Name of the model.
    
    Returns:
        DataFrame with all subjects' metrics.
    """
    # Find all subject result files
    result_files = sorted(metrics_dir.glob(f"*_{model_name}_results.json"))
    
    if not result_files:
        logger.warning(f"No result files found for model '{model_name}' in {metrics_dir}")
        return pd.DataFrame()
    
    all_data = []
    
    for result_file in result_files:
        # Extract subject ID from filename (e.g., "sub-001_lenet_results.json" -> "sub-001")
        subject_id = result_file.name.split("_")[0]
        
        # Load results
        with open(result_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Extract base data
        row_data = {
            "Subject": subject_id,
            "Overall_Accuracy": results.get("overall_accuracy", np.nan),
            "Overall_F1": results.get("overall_f1", np.nan),
        }
        
        # Add per-class accuracies
        class_accs = results.get("class_accuracies", {})
        for class_name, acc in class_accs.items():
            row_data[f"{class_name}_Accuracy"] = acc if acc is not None else np.nan
        
        # Compute and add precision, recall, F1 from confusion matrix
        cm = results.get("confusion_matrix", [])
        class_names = results.get("class_names", [])
        
        if cm and class_names:
            class_metrics = compute_metrics_from_confusion_matrix(cm, class_names)
            
            for class_name, metrics in class_metrics.items():
                row_data[f"{class_name}_Precision"] = metrics["precision"]
                row_data[f"{class_name}_Recall"] = metrics["recall"]
                row_data[f"{class_name}_F1"] = metrics["f1"]
        
        all_data.append(row_data)
    
    df = pd.DataFrame(all_data)
    return df


def create_summary_table(df: pd.DataFrame) -> str:
    """Create a formatted summary table as a string.
    
    Args:
        df: DataFrame with aggregated metrics.
    
    Returns:
        Formatted table as string.
    """
    if df.empty:
        return "No data available"
    
    # Round numeric columns to 4 decimal places
    df_display = df.copy()
    numeric_cols = df_display.select_dtypes(include=np.number).columns
    df_display[numeric_cols] = df_display[numeric_cols].round(4)
    
    # Convert to string with formatting
    table_str = df_display.to_string(index=False)
    return table_str


def save_metrics_table(
    df: pd.DataFrame, output_dir: Path, model_name: str = "lenet"
) -> Path:
    """Save metrics table to CSV and TXT files.
    
    Args:
        df: DataFrame with aggregated metrics.
        output_dir: Directory to save files.
        model_name: Name of the model.
    
    Returns:
        Path to the saved CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_file = output_dir / f"{model_name}_all_subjects_metrics.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Metrics table saved to: {csv_file}")
    
    # Save as formatted TXT
    txt_file = output_dir / f"{model_name}_all_subjects_metrics.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("="*120 + "\n")
        f.write(f"COMPREHENSIVE METRICS TABLE - {model_name.upper()}\n")
        f.write("="*120 + "\n\n")
        f.write(create_summary_table(df))
        f.write(f"\n\n{'='*120}\n")
        f.write(f"Total Subjects: {len(df)}\n")
        f.write("="*120 + "\n")
    
    logger.info(f"Formatted metrics table saved to: {txt_file}")
    
    return csv_file


def generate_metrics_report(
    metrics_dir: Path, output_dir: Path | None = None, model_name: str = "lenet"
) -> pd.DataFrame:
    """Generate comprehensive metrics report for all subjects.
    
    Args:
        metrics_dir: Path to the metrics directory.
        output_dir: Optional output directory. If None, uses metrics_dir.
        model_name: Name of the model.
    
    Returns:
        DataFrame with all aggregated metrics.
    """
    if output_dir is None:
        output_dir = metrics_dir
    
    logger.info(f"Generating metrics report for model '{model_name}'...")
    
    # Aggregate all metrics
    df = aggregate_all_metrics(metrics_dir, model_name)
    
    if df.empty:
        logger.warning(f"No metrics found for model '{model_name}'")
        return df
    
    # Save tables
    save_metrics_table(df, output_dir, model_name)
    
    # Log summary
    logger.debug("="*120)
    logger.info(f"COMPREHENSIVE METRICS REPORT - {model_name.upper()}")
    logger.debug("="*120)
    logger.info(create_summary_table(df))
    logger.debug("="*120)
    logger.info(f"Total Subjects: {len(df)}")
    logger.debug("="*120)
    
    return df


def main():
    """Generate metrics report from command line."""
    from mi3_eeg.config import Paths
    
    paths = Paths()
    metrics_dir = paths.reports_metrics
    
    generate_metrics_report(metrics_dir)


if __name__ == "__main__":
    main()
