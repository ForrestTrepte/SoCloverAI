import logging
import os
from typing import Optional

logger = logging.getLogger("SoCloverAI")
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

file_handler = None


def log_to_file(filename: Optional[str]) -> None:
    global file_handler
    if file_handler:
        logger.removeHandler(file_handler)
    if filename:
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(message)s")
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
