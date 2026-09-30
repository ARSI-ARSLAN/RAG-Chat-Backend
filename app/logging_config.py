"""Logging setup (single place so every module just calls logging.getLogger)."""
import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        force=True,
    )
    # Keep third-party libraries quiet unless we are debugging.
    for noisy in ("httpx", "httpcore", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
