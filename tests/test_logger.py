import logging

import pytest
from mpi4py import MPI

from pyc2ray.utils.logutils import allow_rank_logging, configure_logger, disable_newline


def logging_function(logger: logging.Logger):
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("pyc2ray.test")


def test_configure_logger(logger, caplog, capsys):
    configure_logger()
    logging_function(logger)
    assert len(caplog.records) == 3
    out, err = capsys.readouterr()
    assert out == "This is an info message\n"
    assert (
        err == "WARNING: This is a warning message\nERROR: This is an error message\n"
    )


def test_configure_logger_verbose(logger, caplog, capsys):
    configure_logger(None, True)
    logging_function(logger)
    assert len(caplog.records) == 4
    out, err = capsys.readouterr()
    assert out == "This is a debug message\nThis is an info message\n"
    assert (
        err == "WARNING: This is a warning message\nERROR: This is an error message\n"
    )


def test_configure_logger_file(logger, caplog, capsys, tmp_path):
    logfile = tmp_path / "pyc2ray.log"
    configure_logger(logfile)
    logging_function(logger)

    assert len(caplog.records) == 3
    out, err = capsys.readouterr()
    assert out == "This is an info message\n"
    assert (
        err == "WARNING: This is a warning message\nERROR: This is an error message\n"
    )

    with open(logfile) as f:
        text = f.read()

    assert "DEBUG: This is a debug message" not in text
    assert "INFO: This is an info message" in text
    assert "WARNING: This is a warning message" in text
    assert "ERROR: This is an error message" in text


def test_configure_logger_file_debug(logger, caplog, capsys, tmp_path):
    logfile = tmp_path / "pyc2ray.log"
    configure_logger(logfile, True)
    logging_function(logger)

    assert len(caplog.records) == 4
    out, err = capsys.readouterr()
    assert out == "This is a debug message\nThis is an info message\n"
    assert (
        err == "WARNING: This is a warning message\nERROR: This is an error message\n"
    )

    with open(logfile) as f:
        text = f.read()

    assert "DEBUG: This is a debug message" in text
    assert "INFO: This is an info message" in text
    assert "WARNING: This is a warning message" in text
    assert "ERROR: This is an error message" in text


def test_disable_newline(logger, caplog, capsys):
    configure_logger()

    with disable_newline():
        logger.info("First part ")
    logger.info("Second part")
    logger.info("Third part")

    assert len(caplog.records) == 3
    out, _ = capsys.readouterr()
    assert out == "First part Second part\nThird part\n"


def test_allow_rank_logging(logger, caplog, capsys):
    configure_logger()

    rank = MPI.COMM_WORLD.Get_rank()
    with allow_rank_logging(rank):
        logger.info(f"From rank {rank}")
    logger.info("From root")

    assert len(caplog.records) == 2
    out, _ = capsys.readouterr()
    if rank == 0:
        assert out == "From rank 0\nFrom root\n"
    else:
        assert out == f"From rank {rank}\n"
