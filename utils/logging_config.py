import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    log = logging.getLogger()
    if log.handlers:
        return
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/app.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    log.addHandler(fh)
