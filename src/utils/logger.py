import logging
import sys
from datetime import datetime

def setup_logger(name: str = "rapid_foxin", level: int = logging.INFO) -> logging.Logger:
    """Setup logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        return logger
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(f'logs/rapid_foxin_{datetime.now().strftime("%Y%m%d")}.log')
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = "rapid_foxin") -> logging.Logger:
    """Get existing logger or create new one"""
    return logging.getLogger(name)