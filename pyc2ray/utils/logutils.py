import logging
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from mpi4py import MPI

PathType = str | os.PathLike


@contextmanager
def disable_newline() -> Iterator[None]:
    """Context manager to temporarily disable terminating character in ALL StreamHandlers."""
    end = logging.StreamHandler.terminator
    logging.StreamHandler.terminator = ""

    try:
        yield
    finally:
        logging.StreamHandler.terminator = end


@contextmanager
def allow_rank_logging(rank: int) -> Iterator[None]:
    """Context manager to temporarily allow logging on the specified MPI rank."""
    default = MPIRankFilter.RANK
    MPIRankFilter.RANK = rank

    try:
        yield
    finally:
        MPIRankFilter.RANK = default


class MaxLevelFilter:
    """Filter to allow only messages up to a specific level."""

    def __init__(self, level: int) -> None:
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.level


class MPIRankFilter:
    """Filter to allow only messages from the given MPI rank."""

    # Global rank variable that can be set to filter for a different rank.
    RANK: int = 0

    def __init__(self) -> None:
        self._this_rank = MPI.COMM_WORLD.Get_rank()

    def filter(self, record: logging.LogRecord) -> bool:
        return self._this_rank == MPIRankFilter.RANK


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

    # Grab this module's logger and set level
    lev0 = logging.INFO if not verbose else logging.DEBUG
    module_logger.setLevel(lev0)

    # Set up console handlers for info messages to stdout
    cout = logging.StreamHandler(sys.stdout)
    cout.setLevel(lev0)
    cout.addFilter(MaxLevelFilter(logging.INFO))
    cout.addFilter(MPIRankFilter())
    module_logger.addHandler(cout)

    # Set up console handlers for warning and error messages to stderr
    cerr = logging.StreamHandler(sys.stderr)
    cerr.setLevel(logging.WARNING)
    cerr.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    cerr.addFilter(MPIRankFilter())
    module_logger.addHandler(cerr)

    # Optionally set up a more comprehensive file handler
    if logfile is not None:
        # TODO: check if we want to have the folder creation here
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        fout = logging.FileHandler(logfile, mode="a")
        fout.setLevel(lev0)
        fout.setFormatter(
            logging.Formatter("%(asctime)s %(name)-12s %(levelname)-4s: %(message)s")
        )
        fout.addFilter(MPIRankFilter())
        module_logger.addHandler(fout)
