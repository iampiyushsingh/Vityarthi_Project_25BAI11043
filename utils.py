import os
import sys
import pickle
import logging
from datetime import datetime

# Ensure we import local config, not a global `config` package
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import LOGS_DIR
 
# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a logger that writes to both
    the console and a log file inside logs/.
    Usage: logger = get_logger(__name__)
    """
    os.makedirs(LOGS_DIR, exist_ok=True)
 
    log_filename = os.path.join(
        LOGS_DIR,
        f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    )
 
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
 
    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
 
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
 
        # File handler
        fh = logging.FileHandler(log_filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
 
    return logger
 
 
# ──────────────────────────────────────────────
# MODEL SAVE / LOAD
# ──────────────────────────────────────────────
def save_model(model, path: str) -> None:
    """Saves a sklearn model or any object to a .pkl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[utils] Model saved to: {path}")
 
 
def load_model(path: str):
    """Loads a .pkl model from disk. Raises FileNotFoundError if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[utils] Model not found at: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[utils] Model loaded from: {path}")
    return model
 
 
# ──────────────────────────────────────────────
# DATA HELPERS
# ──────────────────────────────────────────────
def ensure_dir(path: str) -> None:
    """Creates a directory if it doesn't already exist."""
    os.makedirs(path, exist_ok=True)
 
 
def file_exists(path: str) -> bool:
    """Returns True if a file exists at the given path."""
    return os.path.isfile(path)