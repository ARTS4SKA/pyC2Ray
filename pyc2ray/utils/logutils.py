import logging
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Iterator, Union

from mpi4py import MPI

PathType = Union[str, os.PathLike]


@contextmanager
def disable_newline() -> Iterator[None]:
    """Context manager to temporarily disable terminating character in ALL StreamHandlers."""
    end = logging.StreamHandler.terminator
    logging.StreamHandler.terminator = ""

    try:
        yield
    finally:
        logging.StreamHandler.terminator = end


class MaxLevelFilter:
    """Filter to allow only messages up to a specific level."""

    def __init__(self, level: int) -> None:
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.level


def configure_logger(
    logfile: PathType | None = None,
    verbose: bool = False,
    allow_reconfigure: bool = True,
) -> None:
    """Configure this module's logger.

    Parameters
    ----------
    logfile : optional log file.
    verbose : print DEBUG messages when True, otherwise only INFO message
    allow_reconfigure : if False, raises a warning and skips reconfiguration if the logger already has handlers.
    """
    # Grab this module's logger and set level
    module_logger = logging.getLogger(__name__.partition(".")[0])

    # Logger was already configured
    if module_logger.handlers:
        if not allow_reconfigure:
            warnings.warn(
                f"Logger {module_logger.name} has {len(module_logger.handlers)} existing handlers. "
                "To allow reconfiguration, set allow_reconfigure=True.",
            )
            return
        for hand in module_logger.handlers:
            hand.close()
        module_logger.handlers.clear()

    lev0 = logging.INFO if not verbose else logging.DEBUG
    module_logger.setLevel(lev0)

    # Set up console handlers for info messages to stdout
    cout = logging.StreamHandler(sys.stdout)
    cout.addFilter(MaxLevelFilter(logging.INFO))
    module_logger.addHandler(cout)

    # Set up console handlers for warning and error messages to stderr
    cerr = logging.StreamHandler(sys.stderr)
    cerr.setLevel(logging.WARNING)
    cerr.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    module_logger.addHandler(cerr)

    # Optionally set up a more comprehensive file handler
    if logfile is not None:
        fout = logging.FileHandler(logfile, mode="a")
        fout.setLevel(lev0)
        fout.setFormatter(
            logging.Formatter("%(asctime)s %(name)-12s %(levelname)-4s: %(message)s")
        )
        module_logger.addHandler(fout)

    # Logging is only enabled on one MPI process
    if MPI.COMM_WORLD.Get_rank() != 0:
        module_logger.disabled = True
