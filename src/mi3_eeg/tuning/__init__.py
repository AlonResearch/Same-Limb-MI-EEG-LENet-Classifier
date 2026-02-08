"""Hyperparameter tuning module for MI3 EEG classification.

This module provides Bayesian optimization for hyperparameter search using Optuna.
"""

from __future__ import annotations

from mi3_eeg.tuning.optimizer import (
    load_hyperparameters,
    save_hyperparameters,
    tune_subject,
)

__all__ = [
    "tune_subject",
    "load_hyperparameters",
    "save_hyperparameters",
]
