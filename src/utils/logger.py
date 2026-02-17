"""Structured logging setup for the application.

Provides a factory function to create loggers with consistent formatting,
optional file output, and configurable log levels.
"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str,
    log_file: str | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create and configure a logger with console and optional file output.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.
        log_file: Optional file path for persisting log output.
        level: Logging level (default: ``logging.INFO``).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
