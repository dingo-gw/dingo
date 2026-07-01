import logging
import os
import sys


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


def setup_logger(outdir=None, label=None, log_level="INFO"):
    """Setup logging output: call at the start of the script to use.

    Sets up the dingo logger and, if bilby is available, also calls bilby's
    setup_logger so that both loggers share the same file handler and their
    entries appear together in the same log file.

    Parameters
    ----------
    outdir, label: str
        If supplied, write the logging output to outdir/label.log
    log_level: str, optional
        ['debug', 'info', 'warning']
        Either a string from the list above, or an integer as specified
        in https://docs.python.org/2/library/logging.html#logging-levels
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

    # Set up bilby's logger first so we can share its file handler.
    bilby_logger = None
    try:
        from bilby.core.utils.log import setup_logger as bilby_setup_logger

        bilby_setup_logger(outdir=outdir or ".", label=label, log_level=log_level)
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
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(name)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
            )
        )
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)

    # Share bilby's file handler so both loggers write to the same log file.
    if bilby_logger is not None:
        for handler in bilby_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
                    logger.addHandler(handler)

    # Fall back to a standalone file handler if bilby is unavailable.
    if label and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        if outdir:
            check_directory_exists_and_if_not_mkdir(outdir, logger)
        else:
            outdir = "."
        log_file = f"{outdir}/{label}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s: %(message)s", datefmt="%H:%M"
            )
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    for handler in logger.handlers:
        handler.setLevel(level)


setup_logger()
logger = logging.getLogger("dingo")
