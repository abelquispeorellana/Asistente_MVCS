import logging
from pathlib import Path


def setup_logger(name: str, log_file: str = "logs/app.log", level: str = "INFO") -> logging.Logger:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger
