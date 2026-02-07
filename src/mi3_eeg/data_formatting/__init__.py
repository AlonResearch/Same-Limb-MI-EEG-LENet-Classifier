"""Data formatting module for converting raw MI3 EEG files to standardized format.

Provides conversion utilities and CLI tools for reformatting raw MI3 EEG data
into standardized model-ready format.
"""

from mi3_eeg.data_formatting.dataformatter import (
    convert_raw_format,
    detect_format,
    format_and_save,
)

__all__ = [
    # Core conversion functions
    "convert_raw_format",
    "detect_format",
    "format_and_save",
    # CLI modules (run as: python -m mi3_eeg.data_formatting.convert_subject/convert_batch)
    "convert_subject",
    "convert_batch",
]
