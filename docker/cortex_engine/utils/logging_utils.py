import logging

def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logging()
    return logger

