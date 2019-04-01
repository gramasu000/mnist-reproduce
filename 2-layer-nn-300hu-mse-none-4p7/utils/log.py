"""Module for Logging Setup"""

import logging


LOG = None
"""module-level global logger, uninitialized"""

def init_logger(log_filepath, log_level):
    """Initialize module-level global logger

    Args:
        log_filepath (str): Filepath of log file
        log_level (int): Log levels (Ex. logging.DEBUG, logging.INFO, logging.WARNING, etc.) 
    """
    global LOG
    logging.basicConfig(format="%(asctime)s, %(name)-12s, %(levelname)-8s: %(message)s",
                        level=log_level,
                        handlers=[logging.FileWriter(log_filepath), logging.StreamWriter()])
    LOG = logging.getLogger()
