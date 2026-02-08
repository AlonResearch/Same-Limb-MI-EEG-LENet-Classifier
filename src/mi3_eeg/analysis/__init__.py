"""Analysis package for post-training analysis and visualization.

This package contains modules for:
- Visualization of training and evaluation results
- Metrics aggregation across subjects
- Time-frequency analysis (ERD/ERS patterns)
- Topographical brain mapping
- Group-level statistical analysis

Note: group_analysis.py should be executed via the __main__.py entry point
(python -m mi3_eeg.analysis) to properly handle module imports.
"""

from __future__ import annotations

from mi3_eeg.analysis.group_analysis import run_group_analysis
from mi3_eeg.analysis.metrics_aggregator import (
    aggregate_all_metrics,
    compute_metrics_from_confusion_matrix,
    create_summary_table,
    generate_metrics_report,
    load_subject_results,
    save_metrics_table,
)
from mi3_eeg.analysis.statistical_tests import (
    bonferroni_correction,
    compute_cohens_d,
    compute_correlation,
    create_statistical_report,
    fdr_correction,
    independent_t_test,
    one_way_anova,
    paired_t_test,
    repeated_measures_anova,
    summarize_group_statistics,
)
from mi3_eeg.analysis.time_frequency import (
    average_across_subjects,
    compute_band_power,
    compute_erd_ers,
    compute_erd_ers_from_rest,
    compute_morlet_tfr,
    get_class_specific_power,
    process_all_subjects_tfr,
    process_subject_tfr,
)
from mi3_eeg.analysis.topography import (
    create_mne_info,
    create_standard_montage,
    get_channel_subset,
    plot_topomap,
    plot_topomap_comparison,
    plot_topomap_series,
    save_topomap,
)
from mi3_eeg.analysis.visualization import (
    create_all_visualizations,
    plot_class_accuracies,
    plot_confusion_matrix,
    plot_overall_comparison,
    plot_training_curves,
)

__all__ = [
    # Visualization
    "plot_training_curves",
    "plot_confusion_matrix",
    "plot_class_accuracies",
    "plot_overall_comparison",
    "create_all_visualizations",
    # Metrics aggregation
    "load_subject_results",
    "compute_metrics_from_confusion_matrix",
    "aggregate_all_metrics",
    "create_summary_table",
    "save_metrics_table",
    "generate_metrics_report",
    # Time-frequency analysis
    "compute_morlet_tfr",
    "compute_band_power",
    "compute_erd_ers",
    "compute_erd_ers_from_rest",
    "average_across_subjects",
    "process_subject_tfr",
    "process_all_subjects_tfr",
    "get_class_specific_power",
    # Topography
    "create_standard_montage",
    "get_channel_subset",
    "create_mne_info",
    "plot_topomap",
    "plot_topomap_series",
    "plot_topomap_comparison",
    "save_topomap",
    # Statistical tests
    "compute_cohens_d",
    "paired_t_test",
    "independent_t_test",
    "one_way_anova",
    "repeated_measures_anova",
    "bonferroni_correction",
    "fdr_correction",
    "compute_correlation",
    "summarize_group_statistics",
    "create_statistical_report",
    # Group analysis orchestrator
    "run_group_analysis",
]
