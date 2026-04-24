# config.py
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_RAW     = os.path.join(BASE_DIR, "dataset", "raw")
DATASET_PROC    = os.path.join(BASE_DIR, "dataset", "raw")
MODEL_DIR       = os.path.join(BASE_DIR, "models", "saved")
LOG_PATH        = os.path.join(BASE_DIR, "logs", "inference_log.txt")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

# ─── Model ───────────────────────────────────────────────────────────────────
BEST_MODEL_PATH  = os.path.join(MODEL_DIR, "best_model.pth")
MODEL_INPUT_SIZE = (48, 48)
NUM_CLASSES      = 2
PAIN_THRESHOLD   = 0.5

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE     = 32
EPOCHS         = 20
LEARNING_RATE  = 0.001
TRAIN_SPLIT    = 0.8

# ─── Input source ────────────────────────────────────────────────────────────
INPUT_SOURCE   = 0

# ─── Display ─────────────────────────────────────────────────────────────────
SHOW_FPS        = True
SHOW_CONFIDENCE = True