"""Statistical tests for group-level EEG analysis.

This module provides functions for performing statistical comparisons
across subjects, classes, and conditions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from mi3_eeg.logger import logger


def compute_cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
) -> float:
    """Compute Cohen's d effect size for two groups.
    
    Cohen's d interpretation:
    - Small effect: |d| ~ 0.2
    - Medium effect: |d| ~ 0.5
    - Large effect: |d| ~ 0.8
    
    Args:
        group1: First group data.
        group2: Second group data.
    
    Returns:
        Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return float(d)


def paired_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alternative: str = 'two-sided',
) -> dict[str, float]:
    """Perform paired t-test between two groups.
    
    Args:
        group1: First group data (matched pairs with group2).
        group2: Second group data (matched pairs with group1).
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater').
    
    Returns:
        Dictionary with t-statistic, p-value, and effect size.
    """
    if len(group1) != len(group2):
        raise ValueError(f"Groups must have same length: {len(group1)} != {len(group2)}")
    
    t_stat, p_val = stats.ttest_rel(group1, group2, alternative=alternative)
    
    # Compute effect size (Cohen's d for paired samples)
    diff = group1 - group2
    d = np.mean(diff) / np.std(diff, ddof=1)
    
    logger.debug(f"Paired t-test: t={t_stat:.4f}, p={p_val:.6f}, d={d:.4f}")
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(d),
        "df": len(group1) - 1,
    }


def independent_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    equal_var: bool = True,
    alternative: str = 'two-sided',
) -> dict[str, float]:
    """Perform independent t-test between two groups.
    
    Args:
        group1: First group data.
        group2: Second group data.
        equal_var: Whether to assume equal variances (Welch's t-test if False).
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater').
    
    Returns:
        Dictionary with t-statistic, p-value, and effect size.
    """
    t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=equal_var, alternative=alternative)
    
    # Compute Cohen's d
    d = compute_cohens_d(group1, group2)
    
    logger.debug(f"Independent t-test: t={t_stat:.4f}, p={p_val:.6f}, d={d:.4f}")
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": float(d),
        "df": len(group1) + len(group2) - 2,
    }


def one_way_anova(
    *groups: np.ndarray,
) -> dict[str, Any]:
    """Perform one-way ANOVA across multiple groups.
    
    Args:
        *groups: Variable number of group arrays.
    
    Returns:
        Dictionary with F-statistic, p-value, and effect size (eta-squared).
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for ANOVA")
    
    f_stat, p_val = stats.f_oneway(*groups)
    
    # Compute eta-squared (effect size)
    # η² = SS_between / SS_total
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    
    # Between-group sum of squares
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    
    # Total sum of squares
    ss_total = np.sum((all_data - grand_mean) ** 2)
    
    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0
    
    logger.debug(f"One-way ANOVA: F={f_stat:.4f}, p={p_val:.6f}, η²={eta_squared:.4f}")
    
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "eta_squared": float(eta_squared),
        "df_between": len(groups) - 1,
        "df_within": len(all_data) - len(groups),
    }


def repeated_measures_anova(
    data: np.ndarray,
) -> dict[str, Any]:
    """Perform repeated measures ANOVA (Friedman test for non-parametric).
    
    Args:
        data: Data array, shape (n_subjects, n_conditions).
    
    Returns:
        Dictionary with chi-square statistic, p-value, and effect size.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array (subjects x conditions), got shape {data.shape}")
    
    # Friedman test (non-parametric alternative to repeated measures ANOVA)
    stat, p_val = stats.friedmanchisquare(*data.T)
    
    # Compute Kendall's W (effect size for Friedman test)
    n_subjects, n_conditions = data.shape
    
    # Rank data
    ranks = np.apply_along_axis(stats.rankdata, axis=1, arr=data)
    rank_sums = ranks.sum(axis=0)
    
    # Kendall's W
    mean_rank_sum = np.mean(rank_sums)
    ss_ranks = np.sum((rank_sums - mean_rank_sum) ** 2)
    w = (12 * ss_ranks) / (n_subjects ** 2 * (n_conditions ** 3 - n_conditions))
    
    logger.debug(f"Friedman test: χ²={stat:.4f}, p={p_val:.6f}, W={w:.4f}")
    
    return {
        "chi_square": float(stat),
        "p_value": float(p_val),
        "kendalls_w": float(w),
        "df": n_conditions - 1,
        "n_subjects": n_subjects,
        "n_conditions": n_conditions,
    }


def bonferroni_correction(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List or array of p-values.
        alpha: Family-wise error rate.
    
    Returns:
        Tuple of (corrected_p_values, corrected_alpha, significant_mask).
    """
    p_values = np.array(p_values)
    n_comparisons = len(p_values)
    
    # Corrected alpha level
    corrected_alpha = alpha / n_comparisons
    
    # Corrected p-values (multiply by number of comparisons, cap at 1.0)
    corrected_p = np.minimum(p_values * n_comparisons, 1.0)
    
    # Significant comparisons
    significant = corrected_p < alpha
    
    logger.info(f"Bonferroni correction: {n_comparisons} comparisons, α={corrected_alpha:.6f}")
    logger.info(f"Significant comparisons: {significant.sum()} / {n_comparisons}")
    
    return corrected_p, corrected_alpha, significant


def fdr_correction(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
    method: str = 'bh',
) -> tuple[np.ndarray, np.ndarray]:
    """Apply False Discovery Rate (FDR) correction for multiple comparisons.
    
    Args:
        p_values: List or array of p-values.
        alpha: Desired false discovery rate.
        method: FDR method ('bh' for Benjamini-Hochberg, 'by' for Benjamini-Yekutieli).
    
    Returns:
        Tuple of (corrected_p_values, significant_mask).
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    # Compute corrected p-values
    if method == 'bh':
        # Benjamini-Hochberg
        corrected_p = np.minimum.accumulate((sorted_p * n / np.arange(1, n + 1))[::-1])[::-1]
    elif method == 'by':
        # Benjamini-Yekutieli (more conservative)
        c_factor = np.sum(1.0 / np.arange(1, n + 1))
        corrected_p = np.minimum.accumulate((sorted_p * n * c_factor / np.arange(1, n + 1))[::-1])[::-1]
    else:
        raise ValueError(f"Unknown FDR method: {method}")
    
    # Restore original order
    corrected_p_original = np.empty(n)
    corrected_p_original[sorted_indices] = corrected_p
    
    # Significant comparisons
    significant = corrected_p_original < alpha
    
    logger.info(f"FDR correction ({method}): {n} comparisons, α={alpha}")
    logger.info(f"Significant comparisons: {significant.sum()} / {n}")
    
    return corrected_p_original, significant


def compute_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = 'pearson',
) -> dict[str, float]:
    """Compute correlation between two variables.
    
    Args:
        x: First variable.
        y: Second variable.
        method: Correlation method ('pearson', 'spearman', 'kendall').
    
    Returns:
        Dictionary with correlation coefficient and p-value.
    """
    if method == 'pearson':
        r, p_val = stats.pearsonr(x, y)
    elif method == 'spearman':
        r, p_val = stats.spearmanr(x, y)
    elif method == 'kendall':
        r, p_val = stats.kendalltau(x, y)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    logger.debug(f"{method.capitalize()} correlation: r={r:.4f}, p={p_val:.6f}")
    
    return {
        "correlation": float(r),
        "p_value": float(p_val),
        "method": method,
    }


def summarize_group_statistics(
    data: np.ndarray,
    axis: int = 0,
) -> dict[str, float]:
    """Compute summary statistics for a group.
    
    Args:
        data: Data array.
        axis: Axis along which to compute statistics.
    
    Returns:
        Dictionary with mean, std, sem, median, and confidence intervals.
    """
    mean = np.mean(data, axis=axis)
    std = np.std(data, axis=axis, ddof=1)
    n = data.shape[axis]
    sem = std / np.sqrt(n)
    median = np.median(data, axis=axis)
    
    # 95% confidence interval
    ci_95 = stats.t.ppf(0.975, n - 1) * sem
    
    return {
        "mean": float(mean) if np.ndim(mean) == 0 else mean,
        "std": float(std) if np.ndim(std) == 0 else std,
        "sem": float(sem) if np.ndim(sem) == 0 else sem,
        "median": float(median) if np.ndim(median) == 0 else median,
        "ci_95": float(ci_95) if np.ndim(ci_95) == 0 else ci_95,
        "min": float(np.min(data, axis=axis)),
        "max": float(np.max(data, axis=axis)),
        "n": n,
    }


def create_statistical_report(
    results_dict: dict[str, Any],
) -> str:
    """Create a formatted statistical report from results dictionary.
    
    Args:
        results_dict: Dictionary with statistical test results.
    
    Returns:
        Formatted report string.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL ANALYSIS REPORT")
    lines.append("=" * 80)
    
    for test_name, result in results_dict.items():
        lines.append(f"\n{test_name}:")
        lines.append("-" * 40)
        
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.6f}")
                else:
                    lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {result}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)
