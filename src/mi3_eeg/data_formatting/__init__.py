"""Data formatting module for converting raw MI3 EEG files to standardized format."""

from mi3_eeg.data_formatting.dataformatter import (
    convert_raw_format,
    detect_format,
    format_and_save,
)

__all__ = [
    "convert_raw_format",
    "detect_format",
    "format_and_save",
]
