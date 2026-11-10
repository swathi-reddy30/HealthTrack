"""Logging setup: console output plus a rotating log file.

Privacy rule for the whole project: log counts, timings, row numbers and
patient IDs only. Never log complete patient records or health values.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_FILE_NAME = "healthtrack.log"
MAX_LOG_FILE_BYTES = 1_000_000  # rotate at ~1 MB
LOG_BACKUP_COUNT = 3  # keep 3 old files so logs can never fill the disk

_is_configured = False


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    global _is_configured
    # The app (and the test suite) may start several times in one process.
    # Configuring only once avoids every log line being printed twice.
    if _is_configured:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        log_dir / LOG_FILE_NAME,
        maxBytes=MAX_LOG_FILE_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    _is_configured = True
