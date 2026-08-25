"""Loguru-based logging setup, called once from main.py on startup."""
import sys

from loguru import logger

from src.config import get_settings


def configure_logging() -> None:
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
        ),
    )
    logger.add(
        "logs/paraai.log",
        rotation="10 MB",
        retention="14 days",
        level=settings.log_level,
    )
