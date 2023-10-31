__all__ = ["loguru_setup", "get_module_logger"]


# setup a simple loguru logger, to be imported by other modules


def loguru_setup(logfile=None, level="info"):
    """Setup a loguru logger with a default configuration.

    Parameters:
    -----------
    logfile : str (optional):
        Path to log file. Defaults to None.
    level : str (optional):
        Log level. Defaults to 'INFO'."""

    import sys
    from loguru import logger as _logger

    _logger.remove()
    if logfile is not None:
        _logger.add(logfile, level=level.upper(), enqueue=True)
    _logger.add(sys.stderr, level=level.upper(), enqueue=True)
    return _logger


import logging


def get_module_logger(mod_name, level=logging.INFO):
    logger = logging.getLogger(mod_name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
