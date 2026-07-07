import logging
import os
import sys

DINGO_LOG_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


def check_directory_exists_and_if_not_mkdir(directory, logger):
    """Checks if the given directory exists and creates it if it does not exist

    Parameters
    ----------
    directory: str
        Name of the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.debug(f"Making directory {directory}")
    else:
        logger.debug(f"Directory {directory} exists")


def setup_logger(outdir=None, label=None, log_level="INFO", use_bilby: bool = True):
    """Setup logging output: call at the start of the script to use.

    Sets up the dingo logger and, optionally, also calls bilby's setup_logger so
    that both loggers share the same file handler when a log file is requested.

    Parameters
    ----------
    outdir: str, optional
        If supplied, write the logging output to this directory.
    label: str, optional
        Name of the log file without suffix. If omitted and outdir is supplied,
        the entrypoint name is used.
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
    use_bilby:
        If true and bilby is available, share bilby's file handler.
    """
    if "-v" in sys.argv or "--verbose" in sys.argv:
        log_level = "DEBUG"

    if isinstance(log_level, str):
        try:
            level = getattr(logging, log_level.upper())
        except AttributeError:
            raise ValueError(f"log_level {log_level} not understood")
    else:
        level = int(log_level)

    formatter = logging.Formatter(DINGO_LOG_FORMAT)
    save_log = outdir is not None
    entrypoint = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    if not entrypoint or entrypoint.startswith("-"):
        entrypoint = "dingo"
    log_label = label or entrypoint

    # Set up bilby's logger first so we can share its file handler.
    bilby_logger = None
    if use_bilby and save_log:
        try:
            from bilby.core.utils.log import setup_logger as bilby_setup_logger

            bilby_setup_logger(outdir=outdir, label=log_label, log_level=log_level)
            bilby_logger = logging.getLogger("bilby")
        except ImportError:
            pass

    logger = logging.getLogger("dingo")
    logger.propagate = False
    logger.setLevel(level)

    # Add a stream handler if not already present.
    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    # Share bilby's file handler so both loggers write to the same log file.
    if bilby_logger is not None:
        for handler in bilby_logger.handlers:
            handler.setFormatter(formatter)
            handler.setLevel(level)
            if isinstance(handler, logging.FileHandler):
                if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                    logger.addHandler(handler)

    # Fall back to a standalone file handler if bilby is unavailable.
    if save_log and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        check_directory_exists_and_if_not_mkdir(outdir, logger)
        log_file = os.path.join(outdir, f"{log_label}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)
