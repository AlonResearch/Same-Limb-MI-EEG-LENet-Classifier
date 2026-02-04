"""Group-level EEG analysis orchestrator.

This module provides the main entry point for performing group-level
time-frequency analysis, topographical mapping, and statistical comparisons
across all subjects.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from mi3_eeg.analysis.metrics_aggregator import aggregate_all_metrics, load_subject_results
from mi3_eeg.analysis.statistical_tests import (
    bonferroni_correction,
    compute_correlation,
    create_statistical_report,
    fdr_correction,
    independent_t_test,
    one_way_anova,
    paired_t_test,
    summarize_group_statistics,
)
from mi3_eeg.analysis.time_frequency import (
    average_across_subjects,
    compute_band_power,
    compute_erd_ers_from_rest,
    process_subject_tfr,
    get_class_specific_power,
)
from mi3_eeg.analysis.topography import (
    create_mne_info,
    create_standard_montage,
    get_channel_subset,
    plot_topomap_comparison,
    save_topomap,
)
from mi3_eeg.config import CLASS_COLORS, GroupAnalysisConfig, Paths
from mi3_eeg.logger import logger


def _to_serializable(value):
    """Convert values to JSON-serializable types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _get_subject_mat_files(paths: Paths, subjects_subset: list[str] | None = None) -> list[Path]:
    """Get list of subject .mat files for analysis."""
    mat_files = sorted(paths.dataset_derivatives.glob("*_eeg200hz.mat"))

    def _is_git_lfs_pointer(path: Path) -> bool:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
            return first_line.startswith("version https://git-lfs.github.com/spec")
        except OSError:
            return False

    # Filter out git-lfs pointer files (not actual .mat content)
    mat_files = [mf for mf in mat_files if not _is_git_lfs_pointer(mf)]

    if subjects_subset is None:
        return mat_files

    subset_set = set(subjects_subset)
    filtered = [mf for mf in mat_files if mf.stem.split("_")[0] in subset_set]

    return filtered


def _compute_group_time_frequency_and_topography(
    mat_files: list[Path],
    config: GroupAnalysisConfig,
    output_dir: Path,
) -> None:
    """Compute group-level time-frequency maps and topographical distributions.

    This function streams subject processing to avoid storing large tensors in memory.
    """
    if not mat_files:
        logger.warning("No .mat files found for time-frequency analysis")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Frequency setup
    f_min, f_max, f_step = config.tfr_freqs
    freqs = np.arange(f_min, f_max + f_step, f_step)

    # Prepare montage and channel names
    montage = create_standard_montage()
    _, channel_names = get_channel_subset(montage, n_channels=62)

    # Electrode indices for TF maps
    electrodes = list(config.electrodes_of_interest)
    electrode_indices = [channel_names.index(e) for e in electrodes if e in channel_names]

    if not electrode_indices:
        logger.warning("No matching electrodes found for TF maps. Skipping TF maps.")
        return

    class_map = {
        "Rest": 0,
        "Elbow": 1,
        "Hand": 2,
    }

    # Accumulators for TF maps (ERD/ERS)
    tf_accumulator = {name: None for name in class_map.keys()}
    tf_count = 0

    # Accumulators for topography (band power)
    bands = {
        "alpha": (10.0, 12.0),
        "beta": (23.0, 25.0),
    }
    topo_accumulator = {
        band_name: {cls: None for cls in class_map.keys()} for band_name in bands.keys()
    }
    topo_count = 0

    times = None

    logger.info("Computing group-level time-frequency and topography...")

    for mat_file in tqdm(mat_files, desc="TFR per subject"):
        result = process_subject_tfr(
            mat_file,
            sfreq=config.sampling_rate,
            freqs=freqs,
            use_cache=config.use_cache,
        )

        power = result["power"]  # (epochs, channels, freqs, times)
        labels = result["labels"].flatten()

        if times is None:
            n_times = power.shape[-1]
            times = np.arange(n_times) / config.sampling_rate

        # Class-specific power
        rest_power = get_class_specific_power(result, class_map["Rest"])
        elbow_power = get_class_specific_power(result, class_map["Elbow"])
        hand_power = get_class_specific_power(result, class_map["Hand"])

        # ERD/ERS relative to rest baseline
        erd_rest = compute_erd_ers_from_rest(rest_power, rest_power, method=config.baseline_method)
        erd_elbow = compute_erd_ers_from_rest(elbow_power, rest_power, method=config.baseline_method)
        erd_hand = compute_erd_ers_from_rest(hand_power, rest_power, method=config.baseline_method)

        # Average across epochs -> (channels, freqs, times)
        erd_means = {
            "Rest": erd_rest.mean(axis=0),
            "Elbow": erd_elbow.mean(axis=0),
            "Hand": erd_hand.mean(axis=0),
        }

        for cls_name, cls_mean in erd_means.items():
            cls_electrode = cls_mean[electrode_indices, :, :]
            if tf_accumulator[cls_name] is None:
                tf_accumulator[cls_name] = cls_electrode.copy()
            else:
                tf_accumulator[cls_name] += cls_electrode

        tf_count += 1

        # Topography: compute band power for each class
        for band_name, band_range in bands.items():
            for cls_name, cls_label in class_map.items():
                cls_power = get_class_specific_power(result, cls_label)
                band_power = compute_band_power(cls_power, freqs, band_range)
                band_mean = band_power.mean(axis=(0, 2))  # mean over epochs and time -> (channels,)

                if topo_accumulator[band_name][cls_name] is None:
                    topo_accumulator[band_name][cls_name] = band_mean.copy()
                else:
                    topo_accumulator[band_name][cls_name] += band_mean

        topo_count += 1

    if tf_count == 0 or topo_count == 0:
        logger.warning("No subjects processed for TF/topography. Skipping plots.")
        return

    # Average accumulators
    tf_mean = {k: v / tf_count for k, v in tf_accumulator.items() if v is not None}
    topo_mean = {
        band: {cls: v / topo_count for cls, v in cls_dict.items() if v is not None}
        for band, cls_dict in topo_accumulator.items()
    }

    # === Time-Frequency Maps (ERD/ERS) ===
    tf_output = output_dir / "figures"
    tf_output.mkdir(parents=True, exist_ok=True)

    all_vals = np.concatenate([v.flatten() for v in tf_mean.values()])
    vlim = np.nanpercentile(np.abs(all_vals), 95)
    vmin, vmax = -vlim, vlim

    n_rows = len(class_map)
    n_cols = len(electrode_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    for r, cls_name in enumerate(["Hand", "Elbow", "Rest"]):
        cls_data = tf_mean[cls_name]
        for c, (elec_name, elec_idx) in enumerate(zip(electrodes, electrode_indices)):
            ax = axes[r, c]
            im = ax.imshow(
                cls_data[c],
                aspect="auto",
                origin="lower",
                extent=[times[0], times[-1], freqs[0], freqs[-1]],
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{cls_name} - {elec_name}", fontsize=11, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Time (s)")
            if c == 0:
                ax.set_ylabel("Frequency (Hz)")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75)
    cbar.set_label("ERD/ERS (%)")
    fig.suptitle("Group-Averaged Time-Frequency ERD/ERS Maps", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(tf_output / "group_time_frequency_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # === Topographical Maps (Alpha/Beta) ===
    info = create_mne_info(channel_names, sfreq=config.sampling_rate, montage=montage)
    topo_output = output_dir / "figures"
    topo_output.mkdir(parents=True, exist_ok=True)

    for band_name, class_dict in topo_mean.items():
        fig, _ = plot_topomap_comparison(
            class_dict,
            info=info,
            cmap="RdBu_r",
            title=f"Group Topography - {band_name.title()} Band",
            cbar_label="Power (a.u.)",
        )
        save_topomap(fig, topo_output / f"group_topomap_{band_name}.png")
        plt.close(fig)


def load_classification_metrics(
    metrics_dir: Path,
    model_name: str = "lenet",
) -> pd.DataFrame:
    """Load classification performance metrics for all subjects.
    
    Args:
        metrics_dir: Path to metrics directory.
        model_name: Name of the model.
    
    Returns:
        DataFrame with classification metrics per subject.
    """
    logger.info(f"Loading classification metrics for model '{model_name}'")
    
    df = aggregate_all_metrics(metrics_dir, model_name)
    
    if df.empty:
        logger.warning(f"No metrics found for model '{model_name}'")
        return df
    
    logger.info(f"Loaded metrics for {len(df)} subjects")
    
    return df


def create_performance_summary_plots(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str = "lenet",
) -> None:
    """Create summary plots of classification performance across subjects.
    
    Args:
        df: DataFrame with classification metrics.
        output_dir: Directory to save plots.
        model_name: Name of the model.
    """
    logger.info("Creating performance summary plots")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Overall accuracy distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    accuracies = df['Overall_Accuracy'].values * 100
    
    ax.hist(accuracies, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(accuracies.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {accuracies.mean():.1f}%')
    ax.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(accuracies):.1f}%')
    
    ax.set_xlabel('Overall Accuracy (%)', fontsize=12)
    ax.set_ylabel('Number of Subjects', fontsize=12)
    ax.set_title(f'Distribution of Classification Accuracy Across Subjects\n({model_name.upper()})', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_accuracy_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Subject ranking by accuracy
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_sorted = df.sort_values('Overall_Accuracy', ascending=True)
    subjects = df_sorted['Subject'].values
    accs = df_sorted['Overall_Accuracy'].values * 100
    
    colors = ['#e74c3c' if acc < 50 else '#f39c12' if acc < 60 else '#2ecc71' for acc in accs]
    
    ax.barh(range(len(subjects)), accs, color=colors, edgecolor='black')
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9)
    ax.set_xlabel('Overall Accuracy (%)', fontsize=12)
    ax.set_title(f'Classification Accuracy by Subject (Ranked)\n({model_name.upper()})', 
                 fontsize=14, fontweight='bold')
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5, label='Chance level (33.3%)')
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{model_name}_subject_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Per-class accuracy comparison
    class_cols = [col for col in df.columns if col.endswith('_Accuracy') and col != 'Overall_Accuracy']
    
    if class_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        class_data = []
        class_names = []
        for col in class_cols:
            class_name = col.replace('_Accuracy', '')
            class_names.append(class_name)
            class_data.append(df[col].values * 100)
        
        positions = range(len(class_names))
        bp = ax.boxplot(class_data, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Color boxes
        colors = [CLASS_COLORS.get(name, 'gray') for name in class_names]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(class_names, fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'Per-Class Classification Accuracy Distribution\n({model_name.upper()})', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(33.3, color='gray', linestyle='--', alpha=0.5, label='Chance level')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_class_accuracy_boxplot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Performance summary plots saved to {output_dir}")


def perform_classification_statistics(
    df: pd.DataFrame,
    output_dir: Path,
    model_name: str = "lenet",
) -> dict:
    """Perform statistical analysis of classification performance.
    
    Args:
        df: DataFrame with classification metrics.
        output_dir: Directory to save results.
        model_name: Name of the model.
    
    Returns:
        Dictionary with statistical test results.
    """
    logger.info("Performing statistical analysis of classification performance")
    
    results = {}
    
    # Overall accuracy statistics
    overall_acc = df['Overall_Accuracy'].values
    results['overall_accuracy'] = summarize_group_statistics(overall_acc)
    
    # Per-class accuracy statistics
    class_cols = [col for col in df.columns if col.endswith('_Accuracy') and col != 'Overall_Accuracy']
    
    class_accuracies = {}
    class_arrays = []
    class_names_list = []
    
    for col in class_cols:
        class_name = col.replace('_Accuracy', '')
        class_acc = df[col].values
        class_accuracies[class_name] = summarize_group_statistics(class_acc)
        class_arrays.append(class_acc)
        class_names_list.append(class_name)
    
    results['class_accuracies'] = class_accuracies
    
    # ANOVA across classes
    if len(class_arrays) >= 2:
        anova_result = one_way_anova(*class_arrays)
        results['class_anova'] = anova_result
        
        logger.info(f"ANOVA across classes: F={anova_result['f_statistic']:.4f}, "
                   f"p={anova_result['p_value']:.6f}")
    
    # Pairwise comparisons
    if len(class_arrays) >= 2:
        pairwise_results = {}
        
        for i in range(len(class_names_list)):
            for j in range(i + 1, len(class_names_list)):
                name1, name2 = class_names_list[i], class_names_list[j]
                
                # Paired t-test (same subjects across classes)
                t_result = paired_t_test(class_arrays[i], class_arrays[j])
                
                pairwise_results[f"{name1}_vs_{name2}"] = t_result
                
                logger.info(f"Paired t-test {name1} vs {name2}: t={t_result['t_statistic']:.4f}, "
                           f"p={t_result['p_value']:.6f}, d={t_result['cohens_d']:.4f}")
        
        results['pairwise_comparisons'] = pairwise_results
        
        # Apply multiple comparisons correction
        p_values = [res['p_value'] for res in pairwise_results.values()]
        corrected_p, significant = fdr_correction(p_values, alpha=0.05)
        
        for idx, (comp_name, comp_result) in enumerate(pairwise_results.items()):
            comp_result['p_value_corrected'] = corrected_p[idx]
            comp_result['significant_fdr'] = significant[idx]
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / f'{model_name}_classification_statistics.json', 'w') as f:
        json.dump(_to_serializable(results), f, indent=2)
    
    # Create text report
    report = create_statistical_report(results)
    with open(output_dir / f'{model_name}_classification_statistics.txt', 'w') as f:
        f.write(report)
    
    logger.info(f"Statistical results saved to {output_dir}")
    
    return results


def run_group_analysis(
    config: GroupAnalysisConfig | None = None,
    paths: Paths | None = None,
    model_name: str = "lenet",
    subjects_subset: list[str] | None = None,
) -> None:
    """Run complete group-level analysis pipeline.
    
    This function orchestrates:
    1. Loading classification metrics
    2. Creating performance summary plots
    3. Performing statistical tests
    4. Computing group-level time-frequency analysis (if enabled)
    5. Creating topographical maps (if enabled)
    
    Args:
        config: Group analysis configuration. If None, uses defaults.
        paths: Project paths. If None, uses default paths.
        model_name: Name of the model to analyze.
        subjects_subset: Optional list of subject IDs to analyze. If None, analyzes all.
    """
    if config is None:
        config = GroupAnalysisConfig()
    
    if paths is None:
        paths = Paths.from_here()
    
    logger.info("="*80)
    logger.info("STARTING GROUP-LEVEL ANALYSIS")
    logger.info("="*80)
    
    output_dir = paths.reports_group_analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === STEP 1: Load classification metrics ===
    logger.info("\n" + "="*80)
    logger.info("STEP 1: Loading Classification Metrics")
    logger.info("="*80)
    
    df_metrics = load_classification_metrics(paths.reports_metrics, model_name)
    
    if df_metrics.empty:
        logger.error("No metrics found. Aborting group analysis.")
        return
    
    # Filter to subset if provided
    if subjects_subset is not None:
        df_metrics = df_metrics[df_metrics['Subject'].isin(subjects_subset)]
        logger.info(f"Filtered to {len(df_metrics)} subjects from subset")
    
    logger.info(f"Analyzing {len(df_metrics)} subjects")
    
    # === STEP 2: Create performance visualizations ===
    logger.info("\n" + "="*80)
    logger.info("STEP 2: Creating Performance Summary Plots")
    logger.info("="*80)
    
    create_performance_summary_plots(df_metrics, output_dir / "figures", model_name)
    
    # === STEP 3: Perform statistical analysis ===
    logger.info("\n" + "="*80)
    logger.info("STEP 3: Performing Statistical Analysis")
    logger.info("="*80)
    
    stats_results = perform_classification_statistics(
        df_metrics, 
        output_dir / "statistics", 
        model_name
    )

    # === STEP 4: Time-Frequency & Topographical Analysis ===
    logger.info("\n" + "="*80)
    logger.info("STEP 4: Time-Frequency & Topographical Analysis")
    logger.info("="*80)

    mat_files = _get_subject_mat_files(paths, subjects_subset)
    _compute_group_time_frequency_and_topography(mat_files, config, output_dir)
    
    # === STEP 4: Summary ===
    logger.info("\n" + "="*80)
    logger.info("GROUP ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"  - Figures: {output_dir / 'figures'}")
    logger.info(f"  - Statistics: {output_dir / 'statistics'}")
    logger.info(f"  - Time-Frequency/Topography: {output_dir / 'figures'}")
    
    # Print summary statistics
    overall_stats = stats_results['overall_accuracy']
    logger.info(f"\nOverall Classification Performance:")
    logger.info(f"  Mean Accuracy: {overall_stats['mean']*100:.2f}% ± {overall_stats['std']*100:.2f}%")
    logger.info(f"  Median Accuracy: {overall_stats['median']*100:.2f}%")
    logger.info(f"  Range: {overall_stats['min']*100:.2f}% - {overall_stats['max']*100:.2f}%")
    logger.info(f"  95% CI: ± {overall_stats['ci_95']*100:.2f}%")
    
    if 'class_anova' in stats_results:
        anova = stats_results['class_anova']
        logger.info(f"\nClass Comparison (ANOVA):")
        logger.info(f"  F({anova['df_between']}, {anova['df_within']}) = {anova['f_statistic']:.4f}, "
                   f"p = {anova['p_value']:.6f}, η² = {anova['eta_squared']:.4f}")
        
        if anova['p_value'] < 0.05:
            logger.info("  *** Significant difference between classes (p < 0.05) ***")
        else:
            logger.info("  No significant difference between classes (p ≥ 0.05)")


def main():
    """Main entry point for group analysis script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run group-level EEG analysis')
    parser.add_argument('--model', type=str, default='lenet', help='Model name')
    parser.add_argument('--subjects', type=str, nargs='+', help='Subset of subjects to analyze')
    
    args = parser.parse_args()
    
    run_group_analysis(
        model_name=args.model,
        subjects_subset=args.subjects,
    )


if __name__ == "__main__":
    main()
