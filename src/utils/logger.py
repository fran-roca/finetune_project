import logging
import sys
from typing import Optional
from pathlib import Path

def setup_logger(name: str, level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configures and returns a logger with the specified name and level.

    Args:
        name (str): The logger name.
        level (int): Logging level (default is logging.INFO).
        log_file (Optional[str]): Path to log file if file logging is desired.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Optional file handler
        if log_file:
            # Ensure the directory exists
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger
