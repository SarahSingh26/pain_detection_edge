# logger.py
import logging
import os
from datetime import datetime
from config import LOG_PATH

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()          # also prints to terminal
    ]
)
logger = logging.getLogger("PainDetector")

def log_inference(frame_id, label, confidence, inference_ms, fps):
    logger.info(
        f"Frame {frame_id:05d} | {label} | conf={confidence:.3f} "
        f"| infer={inference_ms:.2f}ms | fps={fps:.1f}"
    )

def log_info(msg):
    logger.info(msg)

def log_error(msg):
    logger.error(msg)