"""Regenerate TFR visualizations from aggregated cached data.

This script is part of the mi3_eeg.analysis submodule and allows regenerating
TFR and topographical visualizations without recomputing expensive transformations.

Usage:
    python -m mi3_eeg.analysis.regenerate_visualizations
    
Or from root:
    python src/mi3_eeg/analysis/regenerate_visualizations.py
"""

import logging
from pathlib import Path
import pickle

from mi3_eeg.analysis.tfr_visualization import (
    plot_group_erd_ers_maps,
    plot_group_topomaps,
)
from mi3_eeg.config import GroupAnalysisConfig, Paths
from mi3_eeg.logger import setup_logger

logger = setup_logger(__name__)


def save_aggregated_data(
    tf_mean: dict,
    topo_mean: dict,
    times,
    freqs,
    electrodes: list[str],
    electrode_indices: list[int],
    channel_names: list[str],
    montage: str,
    cache_dir: Path | str | None = None,
) -> Path:
    """Save aggregated TFR data for later visualization regeneration.
    
    Called internally by group_analysis.py during full analysis run.
    
    Args:
        tf_mean: Averaged ERD/ERS maps for each class.
        topo_mean: Averaged band power for topography.
        times: Time vector.
        freqs: Frequency vector.
        electrodes: Electrode names for visualization.
        electrode_indices: Indices of electrodes to plot.
        channel_names: All channel names.
        montage: MNE montage name.
        cache_dir: Where to save. Defaults to ~/.cache/mi3_eeg/tfr/
    
    Returns:
        Path to saved file.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mi3_eeg" / "tfr"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        "tf_mean": tf_mean,
        "topo_mean": topo_mean,
        "times": times,
        "freqs": freqs,
        "electrodes": electrodes,
        "electrode_indices": electrode_indices,
        "channel_names": channel_names,
        "montage": montage,
    }
    
    output_file = cache_dir / "aggregated_tfr_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved aggregated TFR data to {output_file}")
    return output_file


def regenerate_from_cache(
    cache_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> tuple[Path, list[Path]]:
    """Regenerate TFR visualizations from cached aggregated data.
    
    Requires that group_analysis.py was run previously with caching enabled.
    
    Args:
        cache_dir: Path to cache directory.
                   Defaults to ~/.cache/mi3_eeg/tfr/
        output_dir: Path to save visualizations.
                    Defaults to reports/group_analysis/tfr_analysis/
    
    Returns:
        Tuple of (tfr_fig_path, [topo_fig_paths])
    
    Raises:
        FileNotFoundError: If cached data not found.
    """
    paths = Paths.from_here()
    config = GroupAnalysisConfig()
    
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "mi3_eeg" / "tfr"
    else:
        cache_dir = Path(cache_dir)
    
    if output_dir is None:
        output_dir = paths.reports_group_analysis / "tfr_analysis"
    else:
        output_dir = Path(output_dir)
    
    logger.info(f"Loading aggregated TFR data from: {cache_dir}")
    
    aggregated_file = cache_dir / "aggregated_tfr_data.pkl"
    if not aggregated_file.exists():
        logger.error(f"Aggregated data not found at {aggregated_file}")
        logger.info("Available files in cache:")
        if cache_dir.exists():
            for f in cache_dir.glob("*.pkl"):
                logger.info(f"  - {f.name}")
        raise FileNotFoundError(
            f"Aggregated TFR data not found. "
            f"Run group_analysis.py first to generate cache data."
        )
    
    logger.info("Loading aggregated TFR data...")
    with open(aggregated_file, 'rb') as f:
        data = pickle.load(f)
    
    # Extract data
    tf_mean = data.get("tf_mean")
    topo_mean = data.get("topo_mean")
    times = data.get("times")
    freqs = data.get("freqs")
    electrodes = data.get("electrodes")
    electrode_indices = data.get("electrode_indices")
    channel_names = data.get("channel_names")
    montage = data.get("montage")
    
    # Validate
    required_keys = [
        "tf_mean", "topo_mean", "times", "freqs",
        "electrodes", "electrode_indices", "channel_names", "montage"
    ]
    missing_keys = [k for k in required_keys if data.get(k) is None]
    if missing_keys:
        raise ValueError(f"Missing required data keys: {missing_keys}")
    
    logger.info(f"Loaded data for {len(tf_mean)} classes and {len(topo_mean)} frequency bands")
    
    # Generate visualizations
    logger.info("Generating TFR ERD/ERS maps...")
    tfr_fig_path = plot_group_erd_ers_maps(
        tf_mean=tf_mean,
        times=times,
        freqs=freqs,
        electrodes=electrodes,
        electrode_indices=electrode_indices,
        output_path=output_dir,
    )
    
    logger.info("Generating topographical maps...")
    topo_fig_paths = plot_group_topomaps(
        topo_mean=topo_mean,
        channel_names=channel_names,
        montage_name=montage,
        output_path=output_dir,
        sampling_rate=config.sampling_rate,
    )
    
    logger.info("✓ Regenerated visualizations successfully")
    logger.info(f"  TFR maps: {tfr_fig_path}")
    logger.info(f"  Topomaps: {len(topo_fig_paths)} figures")
    
    return tfr_fig_path, topo_fig_paths


def main():
    """Main entry point for visualization regeneration."""
    import sys
    module_name = 'mi3_eeg.analysis.regenerate_visualizations'
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    setup_logger()
    logger.info("=" * 80)
    logger.info("TFR Visualization Regeneration")
    logger.info("=" * 80)
    
    try:
        tfr_path, topo_paths = regenerate_from_cache()
        logger.info("\n✓ Regeneration complete!")
        logger.info(f"  TFR maps saved to: {tfr_path}")
        logger.info(f"  Topographical maps saved to:")
        for p in topo_paths:
            logger.info(f"    - {p}")
    except FileNotFoundError as e:
        logger.error(f"\n✗ Error: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
