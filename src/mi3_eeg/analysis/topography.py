"""Topographical brain mapping for EEG analysis.

This module provides functions for creating topographical maps (topomaps)
of EEG power distributions across the scalp.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mi3_eeg.logger import logger


def create_standard_montage(n_channels: int = 62) -> mne.channels.DigMontage:
    """Create a standard 10-20 EEG montage for 62 channels.
    
    Args:
        n_channels: Number of EEG channels (default: 62).
    
    Returns:
        MNE DigMontage object with electrode positions.
    """
    # Use MNE's standard 1020 montage (covers up to 94 channels)
    montage = mne.channels.make_standard_montage('standard_1020')
    
    logger.info(f"Created standard 10-20 montage with {len(montage.ch_names)} channels")
    
    return montage


def get_channel_subset(
    montage: mne.channels.DigMontage,
    n_channels: int = 62,
) -> tuple[mne.channels.DigMontage, list[str]]:
    """Get a subset of channels from the montage.
    
    Args:
        montage: Full montage object.
        n_channels: Desired number of channels.
    
    Returns:
        Tuple of (subset_montage, channel_names).
    """
    # Common 62-channel subset (excludes reference, ground, and some peripheral electrodes)
    # This is a typical research setup
    common_62_channels = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'FC5', 'FC1', 'FC2', 'FC6',
        'T7', 'C3', 'Cz', 'C4', 'T8',
        'CP5', 'CP1', 'CP2', 'CP6',
        'P7', 'P3', 'Pz', 'P4', 'P8',
        'PO7', 'PO3', 'POz', 'PO4', 'PO8',
        'O1', 'Oz', 'O2',
        'AF7', 'AF3', 'AF4', 'AF8',
        'F5', 'F1', 'F2', 'F6',
        'FT7', 'FC3', 'FC4', 'FT8',
        'C5', 'C1', 'C2', 'C6',
        'TP7', 'CP3', 'CPz', 'CP4', 'TP8',
        'P5', 'P1', 'P2', 'P6',
        'PO5', 'PO1', 'PO2', 'PO6',
        'FPz', 'Iz'
    ]
    
    # Filter to available channels
    available_channels = [ch for ch in common_62_channels if ch in montage.ch_names]
    
    if len(available_channels) < n_channels:
        logger.warning(
            f"Only {len(available_channels)} channels available from requested {n_channels}. "
            f"Using all available channels."
        )
        selected_channels = available_channels
    else:
        selected_channels = available_channels[:n_channels]
    
    logger.info(f"Selected {len(selected_channels)} channels for topography")
    
    return montage, selected_channels


def create_mne_info(
    ch_names: list[str],
    sfreq: float = 200.0,
    montage: mne.channels.DigMontage | None = None,
) -> mne.Info:
    """Create MNE Info object for topographical plotting.
    
    Args:
        ch_names: List of channel names.
        sfreq: Sampling frequency in Hz.
        montage: Optional montage. If None, creates standard montage.
    
    Returns:
        MNE Info object.
    """
    if montage is None:
        montage = create_standard_montage()
    
    # Create info structure
    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Set montage
    info.set_montage(montage, on_missing='warn')
    
    logger.debug(f"Created MNE Info with {len(ch_names)} channels at {sfreq} Hz")
    
    return info


def plot_topomap(
    data: np.ndarray,
    info: mne.Info,
    times: np.ndarray | None = None,
    time_idx: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = 'RdBu_r',
    contours: int = 6,
    title: str | None = None,
    cbar_label: str = 'Power (dB)',
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot topographical map of EEG power distribution.
    
    Args:
        data: Data to plot, shape (n_channels,) or (n_channels, n_times).
        info: MNE Info object with channel positions.
        times: Time vector (only needed if data is 2D).
        time_idx: Time index to plot (only needed if data is 2D).
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        cmap: Colormap name.
        contours: Number of contour lines.
        title: Plot title.
        cbar_label: Colorbar label.
        ax: Optional matplotlib axes.
    
    Returns:
        Tuple of (figure, axes).
    """
    # Handle 2D data (channels x time)
    if data.ndim == 2:
        if time_idx is None:
            # Average across time
            data_to_plot = data.mean(axis=1)
            if title and times is not None:
                title = f"{title} (averaged)"
        else:
            data_to_plot = data[:, time_idx]
            if title and times is not None:
                title = f"{title} (t={times[time_idx]:.3f}s)"
    else:
        data_to_plot = data
    
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    
    # Plot topomap using MNE
    im, cn = mne.viz.plot_topomap(
        data_to_plot,
        info,
        axes=ax,
        show=False,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        contours=contours,
        sensors=True,
        names=None,  # Don't show channel names to avoid clutter
    )
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(cbar_label, fontsize=12)
    
    # Set title
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    return fig, ax


def plot_topomap_series(
    data: np.ndarray,
    info: mne.Info,
    times: np.ndarray,
    time_points: list[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = 'RdBu_r',
    title: str | None = None,
    cbar_label: str = 'Power (dB)',
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a series of topomaps at different time points.
    
    Args:
        data: Data to plot, shape (n_channels, n_times).
        info: MNE Info object with channel positions.
        times: Time vector.
        time_points: List of time points (in seconds) to plot. If None, plots 4 evenly-spaced points.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        cmap: Colormap name.
        title: Overall title for the figure.
        cbar_label: Colorbar label.
        figsize: Figure size (width, height).
    
    Returns:
        Tuple of (figure, axes_array).
    """
    if time_points is None:
        # Select 4 evenly-spaced time points
        n_points = 4
        time_indices = np.linspace(0, len(times) - 1, n_points, dtype=int)
        time_points = times[time_indices]
    else:
        # Find nearest time indices
        time_indices = [np.argmin(np.abs(times - tp)) for tp in time_points]
    
    n_plots = len(time_points)
    
    if figsize is None:
        figsize = (4 * n_plots, 5)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for idx, (time_idx, time_val) in enumerate(zip(time_indices, time_points)):
        plot_topomap(
            data,
            info,
            times=times,
            time_idx=time_idx,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=f"t = {time_val:.2f}s",
            cbar_label=cbar_label if idx == n_plots - 1 else None,  # Only last plot gets colorbar
            ax=axes[idx],
        )
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig, np.array(axes)


def plot_topomap_comparison(
    data_dict: dict[str, np.ndarray],
    info: mne.Info,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = 'RdBu_r',
    title: str | None = None,
    cbar_label: str = 'Power (dB)',
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot comparison of topomaps for different conditions.
    
    Args:
        data_dict: Dictionary mapping condition names to data arrays (n_channels,) or (n_channels, n_times).
        info: MNE Info object with channel positions.
        vmin: Minimum value for color scale.
        vmax: Maximum value for color scale.
        cmap: Colormap name.
        title: Overall title for the figure.
        cbar_label: Colorbar label.
        figsize: Figure size (width, height).
    
    Returns:
        Tuple of (figure, axes_array).
    """
    n_conditions = len(data_dict)
    
    if figsize is None:
        figsize = (5 * n_conditions, 5)
    
    fig, axes = plt.subplots(1, n_conditions, figsize=figsize)
    
    if n_conditions == 1:
        axes = [axes]
    
    # Compute global vmin/vmax if not provided
    if vmin is None or vmax is None:
        all_data = np.concatenate([d.flatten() for d in data_dict.values()])
        if vmin is None:
            vmin = np.percentile(all_data, 2)
        if vmax is None:
            vmax = np.percentile(all_data, 98)
    
    for idx, (condition, data) in enumerate(data_dict.items()):
        plot_topomap(
            data,
            info,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            title=condition,
            cbar_label=cbar_label if idx == n_conditions - 1 else None,
            ax=axes[idx],
        )
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    return fig, np.array(axes)


def save_topomap(
    fig: plt.Figure,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Save topographical map to file.
    
    Args:
        fig: Matplotlib figure.
        save_path: Path to save the figure.
        dpi: Resolution in dots per inch.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    logger.info(f"Topomap saved to: {save_path}")
