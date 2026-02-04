"""Time-Frequency (TFR) visualization module.

Decoupled visualization of ERD/ERS maps and topographical plots.
Can be used standalone with cached TFR data or as part of group analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import mne

from mi3_eeg.analysis.topography import (
    plot_topomap_comparison,
    save_topomap,
    create_mne_info,
)

logger = logging.getLogger(__name__)


def plot_group_erd_ers_maps(
    tf_mean: Dict[str, np.ndarray],
    times: np.ndarray,
    freqs: np.ndarray,
    electrodes: list[str],
    electrode_indices: list[int],
    output_path: Path,
    dpi: int = 300,
) -> Path:
    """Plot group-averaged ERD/ERS time-frequency maps.
    
    Args:
        tf_mean: Dictionary with keys ["Hand", "Elbow", "Rest"], values are
                 (n_electrodes, n_freqs, n_times) ERD/ERS arrays.
        times: Time vector for the TFR, shape (n_times,).
        freqs: Frequency vector for the TFR, shape (n_freqs,).
        electrodes: List of electrode names, length n_electrodes.
        electrode_indices: List of indices for electrodes to plot.
        output_path: Path to save the figure.
        dpi: Resolution for saved figure.
    
    Returns:
        Path to the saved figure.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Exclude Rest from vmin/vmax calculation since it will be all zeros (Rest vs Rest baseline)
    all_vals = np.concatenate([v.flatten() for k, v in tf_mean.items() if k != "Rest"])
    vlim = np.nanpercentile(np.abs(all_vals), 95)
    vmin, vmax = -vlim, vlim
    
    logger.debug(f"TFR colorbar range: [{vmin:.2f}, {vmax:.2f}]")
    
    n_rows = 3  # Hand, Elbow, Rest
    n_cols = len(electrode_indices)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharex=True, sharey=True)
    
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)
    
    im = None
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
    
    # Add colorbar to the right of all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("ERD/ERS (%)", fontsize=11)
    fig.suptitle("Group-Averaged Time-Frequency ERD/ERS Maps\n(Note: Rest shows zero as it's baseline condition)", 
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    
    fig_path = output_path / "group_time_frequency_maps.png"
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved TFR ERD/ERS maps to {fig_path}")
    return fig_path


def plot_group_topomaps(
    topo_mean: Dict[str, Dict[str, np.ndarray]],
    channel_names: list[str],
    montage_name: str,
    output_path: Path,
    sampling_rate: float = 160.0,
    dpi: int = 300,
) -> list[Path]:
    """Plot group-averaged topographical maps for frequency bands.
    
    Args:
        topo_mean: Nested dictionary: {band_name: {class_name: power_array}}.
                   E.g., {"alpha": {"Rest": array, "Elbow": array, "Hand": array}, ...}
                   Power arrays should be shape (n_channels,).
        channel_names: List of all channel names.
        montage_name: Name of MNE montage (e.g., "standard_1020").
        output_path: Path to save figures.
        sampling_rate: Sampling rate in Hz (for MNE Info object).
        dpi: Resolution for saved figures.
    
    Returns:
        List of paths to saved figures.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create MNE info object for electrode positioning
    info = create_mne_info(channel_names, sfreq=sampling_rate, montage=montage_name)
    
    saved_paths = []
    for band_name, class_dict in topo_mean.items():
        logger.debug(f"Plotting topography for {band_name} band")
        
        fig, _ = plot_topomap_comparison(
            class_dict,
            info=info,
            cmap="RdBu_r",
            title=f"Group Topography - {band_name.title()} Band",
            cbar_label="Power (a.u.)",
        )
        
        fig_path = output_path / f"group_topomap_{band_name}.png"
        save_topomap(fig, fig_path)
        saved_paths.append(fig_path)
        
        logger.info(f"Saved topography map to {fig_path}")
    
    return saved_paths


if __name__ == "__main__":
    """Example usage of visualization functions."""
    logger.info("TFR Visualization Module")
    logger.info("Use: from mi3_eeg.analysis.tfr_visualization import plot_group_erd_ers_maps, plot_group_topomaps")
