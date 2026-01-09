"""Centralized logging configuration for the MI3 EEG project.

This module provides a configured logger instance that can be imported
and used throughout the project for consistent logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "mi3_eeg",
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Set up and configure the project logger.
    
    Args:
        name: Logger name.
        level: Logging level (e.g., logging.DEBUG, logging.INFO).
        log_file: Optional path to log file. If None, logs only to console.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger
    
    # Format for log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - "
        "%(funcName)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger instance for the project
logger = setup_logger()
