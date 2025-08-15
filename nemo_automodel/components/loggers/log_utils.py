# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from functools import partial
from logging import Filter, LogRecord
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


class RankFilter(logging.Filter):
    """
    A logging filter that controls log output based on the process rank.

    This filter allows log messages only for rank 0 by default.
    """

    def filter(self, record):
        """Decide whether to log the provided record.

        Args:
            record (logging.LogRecord): The log record to be evaluated.

        Returns:
            bool: True if the log record should be logged, False otherwise.
        """
        # TODO(@akoumparouli): make this PP aware.
        if "RANK" in os.environ:
            rank = int(os.environ.get("RANK"))
            # permantly disable logging for rank != 0
            if rank > 0:
                logging.disable(logging.CRITICAL)
                return False
        return True


# ANSI color codes for log levels
_COLOR_RESET = "\x1b[0m"
_LEVEL_TO_COLOR = {
    logging.DEBUG: "\x1b[36m",  # cyan
    logging.INFO: "\x1b[32m",  # green
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",  # red
    logging.CRITICAL: "\x1b[31;1m",  # bright/bold red
}


class ColorFormatter(logging.Formatter):
    """Logging formatter that colorizes the level name and includes date/time.

    The date is included via asctime with a default format of YYYY-MM-DD HH:MM:SS.
    Colors can be disabled by setting the NO_COLOR env var, or forced with FORCE_COLOR.
    """

    def __init__(self, fmt: str | None = None, datefmt: str | None = None, use_color: bool = True):
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_color = bool(use_color) and self._stream_supports_color()

    def _stream_supports_color(self) -> bool:
        if os.getenv("FORCE_COLOR"):
            return True
        if os.getenv("NO_COLOR"):
            return False
        try:
            return (hasattr(sys.stderr, "isatty") and sys.stderr.isatty()) or (
                hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
            )
        except Exception:
            return False

    def format(self, record: LogRecord) -> str:
        original_levelname = record.levelname
        if self.use_color:
            color = _LEVEL_TO_COLOR.get(record.levelno)
            if color:
                record.levelname = f"{color}{record.levelname}{_COLOR_RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


def warning_filter(record: LogRecord) -> bool:
    """
    Logging filter to exclude WARNING level messages.

    Args:
        record: The logging record to check.

    Returns:
        False if the record level is WARNING, True otherwise.
    """
    return record.levelno != logging.WARNING


def module_filter(record: LogRecord, modules_to_filter: list[str]) -> bool:
    """
    Logging filter to exclude messages from specific modules.

    Args:
        record: The logging record to check.
        modules_to_filter: A list of module name prefixes to filter out.

    Returns:
        False if the record's logger name starts with any of the specified
        module prefixes, True otherwise.
    """
    for module in modules_to_filter:
        if record.name.startswith(module):
            return False
    return True


def add_filter_to_all_loggers(filter: Union[Filter, Callable[[LogRecord], bool]]) -> None:
    """
    Add a filter to the root logger and all existing loggers.

    Args:
        filter: A logging filter instance or callable to add.
    """
    # Get the root logger
    root = logging.getLogger()
    root.addFilter(filter)

    # Add handler to all existing loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        logger.addFilter(filter)


def _ensure_root_handler_with_formatter(formatter: logging.Formatter) -> None:
    """Ensure the root logger has at least one StreamHandler with the given formatter.

    If handlers already exist on the root logger, set their formatter to the provided
    formatter. Otherwise, create a StreamHandler, attach the formatter and RankFilter,
    and add it to the root logger.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(RankFilter())
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            try:
                handler.setFormatter(formatter)
            except Exception:
                # Best-effort; skip handlers that don't accept formatters
                pass


def setup_logging(
    logging_level: int = logging.INFO,
    filter_warning: bool = True,
    modules_to_filter: Optional[list[str]] = None,
    set_level_for_all_loggers: bool = False,
) -> None:
    """
    Set up logging level and filters for the application.

    Configures the logging level based on arguments, environment variables,
    or defaults. Optionally adds filters to suppress warnings or messages
    from specific modules.

    Logging Level Precedence:
    1. Env var `LOGGING_LEVEL`
    2. `logging_level` argument
    3. Default: `logging.INFO`

    Also configures a colorized formatter that includes the date/time.
    """
    env_logging_level = os.getenv("LOGGING_LEVEL", None)
    if env_logging_level is not None:
        logging_level = int(env_logging_level)

    # Set levels on root and optionally all other known loggers first
    logging.getLogger().setLevel(logging_level)

    for _logger_name in logging.root.manager.loggerDict:
        if _logger_name.startswith("nemo") or set_level_for_all_loggers:
            _logger = logging.getLogger(_logger_name)
            _logger.setLevel(logging_level)

    # Install formatter (includes date) and ensure a handler exists
    formatter = ColorFormatter()
    _ensure_root_handler_with_formatter(formatter)

    # Filters
    if filter_warning:
        add_filter_to_all_loggers(warning_filter)
    add_filter_to_all_loggers(RankFilter())
    root = logging.getLogger()
    for h in root.handlers:
        h.addFilter(RankFilter())
    if modules_to_filter:
        add_filter_to_all_loggers(partial(module_filter, modules_to_filter=modules_to_filter))

    logger.info(f"Setting logging level to {logging_level}")
