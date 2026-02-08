"""Analysis utilities for hyperparameter tuning results.

This module provides functions to visualize and analyze Optuna tuning results,
including optimization history, hyperparameter importance, and comparisons.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import optuna

from mi3_eeg.config import Paths
from mi3_eeg.logger import logger


def plot_optimization_history(
    study: optuna.Study,
    save_path: Path | None = None,
) -> None:
    """Plot optimization history showing trial progression.
    
    Args:
        study: Optuna study object.
        save_path: Optional path to save the figure.
    """
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    fig.suptitle(f"Optimization History: {study.study_name}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved optimization history plot to {save_path}")
    
    plt.close()


def plot_param_importances(
    study: optuna.Study,
    save_path: Path | None = None,
) -> None:
    """Plot hyperparameter importance ranking.
    
    Args:
        study: Optuna study object.
        save_path: Optional path to save the figure.
    """
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.suptitle(f"Hyperparameter Importance: {study.study_name}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved parameter importance plot to {save_path}")
    
    plt.close()


def plot_parallel_coordinate(
    study: optuna.Study,
    save_path: Path | None = None,
) -> None:
    """Plot parallel coordinate plot showing relationship between parameters and objective.
    
    Args:
        study: Optuna study object.
        save_path: Optional path to save the figure.
    """
    fig = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    fig.suptitle(f"Parallel Coordinate Plot: {study.study_name}")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved parallel coordinate plot to {save_path}")
    
    plt.close()


def analyze_tuning_results(
    subject_id: str,
    sampling_rate: int = 200,
    save_figures: bool = True,
) -> None:
    """Analyze and visualize tuning results for a subject.
    
    Args:
        subject_id: Subject ID (e.g., "sub-001").
        sampling_rate: Sampling rate in Hz.
        save_figures: If True, save visualization figures.
    """
    paths = Paths.from_here()
    config_dir = paths.models / "Hyperparameters" / "best_configs"
    filename = f"{subject_id}-{sampling_rate}hz.json"
    filepath = config_dir / filename
    
    if not filepath.exists():
        logger.error(f"No tuning results found for {subject_id} at {filepath}")
        return
    
    # Load configuration
    with open(filepath) as f:
        config_data = json.load(f)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Tuning Results for {subject_id} ({sampling_rate}Hz)")
    print(f"{'='*80}")
    print(f"Tuned on: {config_data['tuned_date']}")
    print(f"Best trial: {config_data['best_trial_number']} / {config_data['n_trials']}")
    print(f"Best val F1: {config_data['best_val_f1']:.4f}")
    print(f"\nBest Hyperparameters:")
    for param, value in config_data['hyperparameters'].items():
        print(f"  {param:30s}: {value}")
    print(f"{'='*80}\n")


def compare_subjects(
    subject_ids: list[str] | None = None,
) -> dict[str, dict]:
    """Compare tuned hyperparameters across subjects.
    
    Args:
        subject_ids: List of subject IDs to compare. If None, compares all available.
    
    Returns:
        Dictionary mapping subject_id to their hyperparameters.
    """
    paths = Paths.from_here()
    config_dir = paths.models / "Hyperparameters" / "best_configs"
    
    if not config_dir.exists():
        logger.error(f"No tuning results directory found at {config_dir}")
        return {}
    
    # Find all tuned subject configs
    if subject_ids is None:
        config_files = sorted(config_dir.glob("*.json"))
    else:
        config_files = []
        for subject_id in subject_ids:
            # Try both 200hz and 90hz
            for hz in [200, 90]:
                filepath = config_dir / f"{subject_id}-{hz}hz.json"
                if filepath.exists():
                    config_files.append(filepath)
    
    if not config_files:
        logger.warning("No tuned configurations found.")
        return {}
    
    # Load all configs
    results = {}
    for filepath in config_files:
        with open(filepath) as f:
            config_data = json.load(f)
        
        subject_id = config_data['subject_id']
        results[subject_id] = {
            'sampling_rate': config_data['sampling_rate'],
            'best_val_f1': config_data['best_val_f1'],
            'n_trials': config_data['n_trials'],
            'hyperparameters': config_data['hyperparameters'],
        }
    
    # Print comparison
    print(f"\n{'='*80}")
    print(f"Hyperparameter Comparison Across Subjects")
    print(f"{'='*80}\n")
    print(f"{'Subject':<12} {'Val F1':<10} {'LR':<12} {'Dropout':<10} {'Batch':<8} {'Patience':<10}")
    print(f"{'-'*80}")
    
    for subject_id, data in sorted(results.items()):
        hp = data['hyperparameters']
        print(
            f"{subject_id:<12} "
            f"{data['best_val_f1']:<10.4f} "
            f"{hp['learning_rate']:<12.6f} "
            f"{hp['dropout']:<10.3f} "
            f"{hp['batch_size']:<8} "
            f"{hp['early_stopping_patience']:<10}"
        )
    
    print(f"{'-'*80}\n")
    
    return results


def main():
    """CLI for analyzing tuning results."""
    parser = argparse.ArgumentParser(
        description="Analyze hyperparameter tuning results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Subject ID to analyze (e.g., sub-001)",
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare hyperparameters across all tuned subjects",
    )
    
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=200,
        help="Sampling rate for the subject",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_subjects()
    elif args.subject:
        analyze_tuning_results(args.subject, args.sampling_rate)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python -m mi3_eeg.tuning.analysis --subject sub-001")
        print("  python -m mi3_eeg.tuning.analysis --compare")


if __name__ == "__main__":
    main()
