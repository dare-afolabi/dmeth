#!/usr/bin/env python
# coding: utf-8


"""
Centralised logging utilities for the DMeth DNA-methylation analysis suite.

This module configures a unified logger with the following features:

Features
--------
- Timestamped log files automatically saved to ``<output_dir>/log/``
- Simultaneous console (stdout) and file output
- Consistent formatting across all package modules
- A custom :class:`ProgressAwareLogger` that seamlessly integrates \
``tqdm`` progress bars:
    - ``logger.progress("Processing samples", total=n)`` starts a progress bar
    - ``logger.progress_update(k)`` advances it
    - Any regular log call (info/warning/error/etc.) automatically closes the active \
    bar so that log messages are never corrupted by overlapping tqdm output

All other ``dmeth`` modules import the logger via ``get_logger()``.
"""


from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import Optional

from tqdm import tqdm


class ProgressAwareLogger(logging.Logger):
    """
    Custom logger class that supports a temporary progress bar.
    The progress bar stays active until the next normal log call.
    """

    def __init__(self, name) -> None:
        super().__init__(name)
        self._pbar = None  # active progress bar

    def _close_pbar(self) -> None:
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def progress(self, msg: str, total: Optional[int] = None) -> None:
        """
        Start or replace the active tqdm progress bar.

        When ``total`` is provided, a determinate progress bar is shown; otherwise an
        indeterminate (spinning) bar is displayed. Any subsequent regular log call
        (info, warning, error, debug) automatically closes the bar to avoid overlapping
        output with tqdm.

        Parameters
        ----------
        msg : str
            Description displayed to the left of the progress bar.
        total : int, optional
            Expected total number of iterations. If ``None``, an \
            indeterminate bar is used.
        """
        self._close_pbar()
        total_val = total if total is not None else 0
        self._pbar = tqdm(
            total=total_val, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
        )
        self._pbar.set_description(msg)

    def progress_update(self, n: int = 1) -> None:
        """
        Advance the active progress bar by the specified number of steps.

        If no progress bar is currently active, the call is ignored silently.

        Parameters
        ----------
        n : int, default 1
            Number of steps to advance the bar (typically 1).
        """
        if self._pbar is not None:
            try:
                self._pbar.update(n)
            except Exception as e:
                logger.warning(f"Failed to advance progress bar: {e}")

    # Override logging methods so they auto-close the pbar
    def info(self, msg: str, *a: object, **k: object) -> None:
        self._close_pbar()
        super().info(msg, *a, **k)

    def warning(self, msg: str, *a: object, **k: object) -> None:
        self._close_pbar()
        super().warning(msg, *a, **k)

    def error(self, msg: str, *a: object, **k: object) -> None:
        self._close_pbar()
        super().error(msg, *a, **k)

    def debug(self, msg: str, *a: object, **k: object) -> None:
        self._close_pbar()
        super().debug(msg, *a, **k)


logging.setLoggerClass(ProgressAwareLogger)


def _configure_logger(
    name: str = "dmeth", output_dir: str = "output"
) -> logging.Logger:
    """
    Configure and return the central DMeth logger instance.

    Parameters
    ----------
    name : str, default "dmeth"
        Logger name. Typically left as the default.
    output_dir : str, default "output"
        Base directory for log storage. Log files are written to
        ``<output_dir>/log/``.

    Returns
    -------
    logging.Logger
        A :class:`ProgressAwareLogger` instance set to ``INFO`` level,
        equipped with console and file handlers.
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = os.path.join(output_dir, "log")
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Instantiate logger once
logger = _configure_logger()


def get_logger(name: str = "dmeth") -> logging.Logger:
    """
    Return the central DMeth logger instance.

    Parameters
    ----------
    name : str, default "dmeth"
        Logger name. Typically left as the default.

    Returns
    -------
    logging.Logger
        The configured ProgressAwareLogger instance.
    """
    return logging.getLogger(name)
