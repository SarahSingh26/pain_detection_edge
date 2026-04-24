# utils.py
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from config import MODEL_INPUT_SIZE

def load_image(path):
    """Load an image from disk and convert to RGB."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resize_image(img, size=MODEL_INPUT_SIZE):
    """Resize image to target size (H, W)."""
    return cv2.resize(img, (size[1], size[0]))

def normalize_image(img):
    """Normalize pixel values to [0, 1]."""
    return img.astype(np.float32) / 255.0

def draw_label(frame, label, confidence, inference_ms, fps):
    """Draw pain label, confidence, FPS, and inference time on frame."""
    color = (0, 0, 255) if label == "PAIN" else (0, 255, 0)
    cv2.putText(frame, f"{label}: {confidence:.2f}",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Infer: {inference_ms:.1f}ms",
                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    return frame

def ensure_dirs():
    """Create all required directories if they don't exist."""
    from config import DATASET_PROC, MODEL_DIR, LOG_PATH, OUTPUT_DIR
    for d in [DATASET_PROC, MODEL_DIR, OUTPUT_DIR, os.path.dirname(LOG_PATH)]:
        os.makedirs(d, exist_ok=True)

def plot_training_graphs(train_acc, val_acc, train_loss, val_loss, model_name):
    """Save accuracy and loss graphs for the README."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_acc)+1)

    ax1.plot(epochs, train_acc, label='Train Acc')
    ax1.plot(epochs, val_acc,   label='Val Acc')
    ax1.set_title(f'{model_name} – Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend()

    ax2.plot(epochs, train_loss, label='Train Loss')
    ax2.plot(epochs, val_loss,   label='Val Loss')
    ax2.set_title(f'{model_name} – Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend()

    plt.tight_layout()
    path = f"models/saved/{model_name}_training_graph.png"
    plt.savefig(path)
    print(f"Graph saved: {path}")
    plt.close()