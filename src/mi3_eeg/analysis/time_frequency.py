"""Time-frequency analysis for EEG signals using Morlet wavelets.

This module provides functions for computing time-frequency representations,
calculating ERD/ERS patterns, and averaging across subjects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mne
import numpy as np
from joblib import Memory, Parallel, delayed
from scipy import signal
from tqdm import tqdm

from mi3_eeg.logger import logger

# Setup caching for expensive computations
CACHE_DIR = Path.home() / ".cache" / "mi3_eeg" / "tfr"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)


def compute_morlet_tfr(
    data: np.ndarray,
    sfreq: float,
    freqs: np.ndarray | None = None,
    n_cycles: float | np.ndarray = 7.0,
    use_fft: bool = True,
    zero_mean: bool = True,
    decim: int | None = None,
    compute_itc: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Compute time-frequency representation using Morlet wavelets.
    
    Args:
        data: EEG data, shape (n_epochs, n_channels, n_times).
        sfreq: Sampling frequency in Hz.
        freqs: Frequencies of interest. If None, uses 4-40 Hz in 1 Hz steps.
        n_cycles: Number of cycles for the Morlet wavelet.
        use_fft: Whether to use FFT-based convolution (faster).
        zero_mean: Whether to zero-mean the signal before analysis.
    
    Returns:
        Tuple of (power, itc, freqs) where:
            - power: shape (n_epochs, n_channels, n_freqs, n_times)
            - itc: Inter-trial coherence, shape (n_channels, n_freqs, n_times)
            - freqs: Frequency vector
    """
    if freqs is None:
        freqs = np.arange(4, 41, 1.0)  # 4-40 Hz in 1 Hz steps
    
    logger.info(f"Computing TFR for {data.shape[0]} epochs, {data.shape[1]} channels")
    logger.info(f"Frequency range: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz ({len(freqs)} freqs)")
    
    # Use MNE's tfr_array_morlet for standardized computation
    kwargs = {
        'data': data,
        'sfreq': sfreq,
        'freqs': freqs,
        'n_cycles': n_cycles,
        'use_fft': use_fft,
        'zero_mean': zero_mean,
        'output': 'power',
        'n_jobs': -1,  # Use all available cores
        'verbose': 'WARNING'
    }
    if decim is not None:
        kwargs['decim'] = decim
    
    power = mne.time_frequency.tfr_array_morlet(**kwargs)

    itc = None
    if compute_itc:
        # Compute inter-trial coherence (average phase consistency across trials)
        complex_tfr = mne.time_frequency.tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=use_fft,
            zero_mean=zero_mean,
            decim=decim,
            output='complex',
            n_jobs=-1,
            verbose='WARNING'
        )
        itc = np.abs(np.mean(complex_tfr / np.abs(complex_tfr), axis=0))

    if itc is not None:
        logger.info(f"TFR computed: power shape {power.shape}, ITC shape {itc.shape}")
    else:
        logger.info(f"TFR computed: power shape {power.shape}")

    return power, itc, freqs


def compute_band_power(
    power: np.ndarray,
    freqs: np.ndarray,
    band: tuple[float, float],
) -> np.ndarray:
    """Extract power in a specific frequency band.
    
    Args:
        power: Power array, shape (n_epochs, n_channels, n_freqs, n_times).
        freqs: Frequency vector.
        band: Tuple of (low_freq, high_freq) in Hz.
    
    Returns:
        Band power, shape (n_epochs, n_channels, n_times).
    """
    freq_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = power[:, :, freq_mask, :].mean(axis=2)
    
    logger.debug(f"Extracted {band[0]}-{band[1]} Hz band power: shape {band_power.shape}")
    
    return band_power


def compute_erd_ers(
    power: np.ndarray,
    baseline_power: np.ndarray | None = None,
    baseline_indices: tuple[int, int] | None = None,
    method: str = "percent",
) -> np.ndarray:
    """Compute ERD/ERS (Event-Related Desynchronization/Synchronization).
    
    ERD/ERS measures the relative power change from baseline:
    - Negative values = ERD (desynchronization, power decrease)
    - Positive values = ERS (synchronization, power increase)
    
    Args:
        power: Power array, shape (n_epochs, n_channels, n_freqs, n_times) or
               (n_epochs, n_channels, n_times) for band-limited data.
        baseline_power: Pre-computed baseline power. If None, computed from baseline_indices.
        baseline_indices: Tuple of (start, end) time indices for baseline period.
        method: Method for computing relative change:
            - "percent": ((P - B) / B) * 100
            - "ratio": P / B
            - "db": 10 * log10(P / B)
    
    Returns:
        ERD/ERS values, same shape as power.
    """
    if baseline_power is None:
        if baseline_indices is None:
            raise ValueError("Either baseline_power or baseline_indices must be provided")
        
        # Extract baseline period and average across time
        baseline_power = power[..., baseline_indices[0]:baseline_indices[1]].mean(axis=-1, keepdims=True)
    
    # Ensure baseline has correct dimensions for broadcasting
    if baseline_power.ndim < power.ndim:
        # Add time dimension if missing
        baseline_power = np.expand_dims(baseline_power, axis=-1)
    
    # Compute relative change
    if method == "percent":
        erd_ers = ((power - baseline_power) / baseline_power) * 100
    elif method == "ratio":
        erd_ers = power / baseline_power
    elif method == "db":
        erd_ers = 10 * np.log10(power / baseline_power)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'percent', 'ratio', or 'db'")
    
    logger.debug(f"Computed ERD/ERS using method '{method}': shape {erd_ers.shape}")
    
    return erd_ers


def compute_erd_ers_from_rest(
    task_power: np.ndarray,
    rest_power: np.ndarray,
    method: str = "percent",
) -> np.ndarray:
    """Compute ERD/ERS using Rest trials as baseline.
    
    Args:
        task_power: Power during task (Hand/Elbow), shape (n_epochs, n_channels, ...).
        rest_power: Power during rest, shape (n_epochs, n_channels, ...).
        method: Method for computing relative change.
    
    Returns:
        ERD/ERS values relative to rest baseline.
    """
    # Average rest power across epochs to get baseline
    baseline_power = rest_power.mean(axis=0, keepdims=True)
    
    # Compute ERD/ERS for task relative to rest
    erd_ers = compute_erd_ers(task_power, baseline_power=baseline_power, method=method)
    
    logger.info(f"Computed ERD/ERS from Rest baseline: task={task_power.shape}, rest={rest_power.shape}")
    
    return erd_ers


def average_across_subjects(
    subject_powers: list[np.ndarray],
    subject_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Average time-frequency power across subjects.
    
    Args:
        subject_powers: List of power arrays, each shape (n_epochs, n_channels, n_freqs, n_times).
        subject_ids: Optional list of subject identifiers for logging.
    
    Returns:
        Tuple of (mean_power, std_power).
    """
    if not subject_powers:
        raise ValueError("subject_powers list is empty")
    
    # Stack along new subject dimension
    all_powers = np.stack(subject_powers, axis=0)  # (n_subjects, n_epochs, n_channels, n_freqs, n_times)
    
    # Average across subjects and epochs
    mean_power = all_powers.mean(axis=(0, 1))  # (n_channels, n_freqs, n_times)
    std_power = all_powers.std(axis=(0, 1))  # (n_channels, n_freqs, n_times)
    
    n_subjects = len(subject_powers)
    logger.info(f"Averaged power across {n_subjects} subjects: mean shape {mean_power.shape}")
    
    return mean_power, std_power


@memory.cache
def load_and_compute_tfr_cached(
    mat_file: Path,
    sfreq: float,
    freqs: np.ndarray | None = None,
) -> dict[str, Any]:
    """Load EEG data and compute TFR (cached for performance).
    
    Args:
        mat_file: Path to .mat file with EEG data.
        sfreq: Sampling frequency in Hz.
        freqs: Frequencies of interest.
    
    Returns:
        Dictionary with 'power', 'itc', 'freqs', 'labels', 'data_shape'.
    """
    from mi3_eeg.dataset import load_mat_from_derivatives
    
    logger.info(f"Loading and computing TFR for {mat_file.name} (cached)")
    
    # Load data
    bundle = load_mat_from_derivatives(mat_file)
    
    # Compute TFR
    power, itc, freqs_out = compute_morlet_tfr(
        bundle.data,
        sfreq=sfreq,
        freqs=freqs,
    )
    
    return {
        "power": power,
        "itc": itc,
        "freqs": freqs_out,
        "labels": bundle.labels,
        "data_shape": bundle.data.shape,
        "class_distribution": bundle.class_distribution,
    }


def process_subject_tfr(
    mat_file: Path,
    sfreq: float = 200.0,
    freqs: np.ndarray | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Process a single subject's EEG data for time-frequency analysis.
    
    Args:
        mat_file: Path to subject's .mat file.
        sfreq: Sampling frequency in Hz.
        freqs: Frequencies of interest.
        use_cache: Whether to use cached results.
    
    Returns:
        Dictionary with TFR results and metadata.
    """
    if use_cache:
        return load_and_compute_tfr_cached(mat_file, sfreq, freqs)
    else:
        from mi3_eeg.dataset import load_mat_from_derivatives
        
        logger.info(f"Loading and computing TFR for {mat_file.name} (no cache)")
        
        bundle = load_mat_from_derivatives(mat_file)
        power, itc, freqs_out = compute_morlet_tfr(
            bundle.data,
            sfreq=sfreq,
            freqs=freqs,
        )
        
        return {
            "power": power,
            "itc": itc,
            "freqs": freqs_out,
            "labels": bundle.labels,
            "data_shape": bundle.data.shape,
            "class_distribution": bundle.class_distribution,
        }


def process_all_subjects_tfr(
    mat_files: list[Path],
    sfreq: float = 200.0,
    freqs: np.ndarray | None = None,
    use_cache: bool = True,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Process multiple subjects in parallel for time-frequency analysis.
    
    Args:
        mat_files: List of paths to .mat files.
        sfreq: Sampling frequency in Hz.
        freqs: Frequencies of interest.
        use_cache: Whether to use cached results.
        n_jobs: Number of parallel jobs (-1 for all cores).
    
    Returns:
        List of TFR result dictionaries, one per subject.
    """
    logger.info(f"Processing {len(mat_files)} subjects for TFR analysis")
    
    if n_jobs == 1:
        # Sequential processing with progress bar
        results = []
        for mat_file in tqdm(mat_files, desc="Processing subjects"):
            result = process_subject_tfr(mat_file, sfreq, freqs, use_cache)
            results.append(result)
    else:
        # Parallel processing
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_subject_tfr)(mat_file, sfreq, freqs, use_cache)
            for mat_file in tqdm(mat_files, desc="Processing subjects")
        )
    
    logger.info(f"Completed TFR processing for {len(results)} subjects")
    
    return results


def get_class_specific_power(
    tfr_result: dict[str, Any],
    class_label: int,
) -> np.ndarray:
    """Extract power for a specific class from TFR results.
    
    Args:
        tfr_result: TFR result dictionary from process_subject_tfr.
        class_label: Class label (0=Rest, 1=Elbow, 2=Hand).
    
    Returns:
        Power array for specified class, shape (n_epochs, n_channels, n_freqs, n_times).
    """
    labels = tfr_result["labels"].flatten()
    power = tfr_result["power"]
    
    class_mask = labels == class_label
    class_power = power[class_mask]
    
    logger.debug(f"Extracted {class_mask.sum()} epochs for class {class_label}")
    
    return class_power
