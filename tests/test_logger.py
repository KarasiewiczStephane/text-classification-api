"""Tests for the structured logging module."""

import logging

from src.utils.logger import setup_logger


class TestSetupLogger:
    """Tests for setup_logger()."""

    def test_creates_logger_with_name(self) -> None:
        """Logger is created with the specified name."""
        logger = setup_logger("test.module")
        assert logger.name == "test.module"

    def test_sets_log_level(self) -> None:
        """Logger respects the configured level."""
        logger = setup_logger("test.level", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_adds_console_handler(self) -> None:
        """Logger has at least one StreamHandler."""
        logger = setup_logger("test.console")
        handler_types = [type(h) for h in logger.handlers]
        assert logging.StreamHandler in handler_types

    def test_adds_file_handler(self, tmp_path: str) -> None:
        """Logger creates a file handler when log_file is given."""
        log_file = str(tmp_path / "logs" / "test.log")
        logger = setup_logger("test.file", log_file=log_file)
        handler_types = [type(h) for h in logger.handlers]
        assert logging.FileHandler in handler_types

    def test_creates_log_directory(self, tmp_path: str) -> None:
        """Logger creates parent directories for the log file."""
        log_file = str(tmp_path / "nested" / "dir" / "test.log")
        setup_logger("test.mkdir", log_file=log_file)
        from pathlib import Path

        assert Path(log_file).parent.exists()

    def test_no_duplicate_handlers(self) -> None:
        """Calling setup_logger twice does not add duplicate handlers."""
        name = "test.no_dup"
        logger1 = setup_logger(name)
        handler_count = len(logger1.handlers)
        logger2 = setup_logger(name)
        assert len(logger2.handlers) == handler_count

    def test_log_output_format(self, tmp_path: str) -> None:
        """Log messages follow the expected format."""
        log_file = str(tmp_path / "format.log")
        logger = setup_logger("test.format", log_file=log_file)
        logger.info("test message")

        with open(log_file) as f:
            content = f.read()
        assert "test.format" in content
        assert "INFO" in content
        assert "test message" in content
