"""Tests for logging module."""

from __future__ import annotations

import logging
from pathlib import Path

from mi3_eeg.logger import logger, setup_logger


def test_default_logger_exists() -> None:
    """Test that default logger is properly initialized."""
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "mi3_eeg"


def test_setup_logger_basic() -> None:
    """Test basic logger setup."""
    test_logger = setup_logger(name="test_logger", level=logging.DEBUG)
    
    assert test_logger.name == "test_logger"
    assert test_logger.level == logging.DEBUG
    assert len(test_logger.handlers) > 0


def test_setup_logger_with_file(tmp_path: Path) -> None:
    """Test logger setup with file handler."""
    log_file = tmp_path / "test.log"
    test_logger = setup_logger(
        name="test_file_logger",
        level=logging.INFO,
        log_file=log_file,
    )
    
    # Write a test message
    test_logger.info("Test message")
    
    # Verify log file was created and contains message
    assert log_file.exists()
    log_content = log_file.read_text(encoding="utf-8")
    assert "Test message" in log_content
    assert "test_file_logger" in log_content


def test_logger_no_duplicate_handlers() -> None:
    """Test that calling setup_logger multiple times doesn't add duplicate handlers."""
    initial_logger = setup_logger(name="duplicate_test")
    initial_handler_count = len(initial_logger.handlers)
    
    # Call setup again with same name
    same_logger = setup_logger(name="duplicate_test")
    
    # Should be the same logger instance with same handlers
    assert same_logger is initial_logger
    assert len(same_logger.handlers) == initial_handler_count


def test_logger_levels() -> None:
    """Test different logging levels."""
    test_logger = setup_logger(name="level_test", level=logging.WARNING)
    
    # Logger should be at WARNING level
    assert test_logger.level == logging.WARNING
    
    # Should log WARNING and above
    assert test_logger.isEnabledFor(logging.WARNING)
    assert test_logger.isEnabledFor(logging.ERROR)
    
    # Should not log INFO and below
    assert not test_logger.isEnabledFor(logging.INFO)
    assert not test_logger.isEnabledFor(logging.DEBUG)
