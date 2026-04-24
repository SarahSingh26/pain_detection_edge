# preprocessing.py
import cv2
import numpy as np
import os
from config import DATASET_RAW, DATASET_PROC, MODEL_INPUT_SIZE
from utils import normalize_image, ensure_dirs

# OpenCV Haar cascade for face detection (built into OpenCV, no download needed)
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_and_crop_face(frame_bgr):
    """
    Detect face in a BGR frame.
    Returns cropped grayscale face region resized to MODEL_INPUT_SIZE,
    or None if no face found.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]                      # take the first (largest) face
    face_roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]))
    return face_resized

def preprocess_frame(frame_bgr):
    """
    Full preprocessing pipeline for a single frame (used in inference).
    Returns a float32 numpy array of shape (1, 1, H, W) ready for PyTorch.
    Returns None if no face found.
    """
    face = detect_and_crop_face(frame_bgr)
    if face is None:
        return None
    face_norm = normalize_image(face)
    # shape: (H, W) → (1, 1, H, W)  [batch=1, channels=1 grayscale]
    return face_norm[np.newaxis, np.newaxis, :, :]

def extract_hog_features(face_gray):
    """Extract HOG features from a grayscale face image."""
    win_size   = (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])
    block_size = (16, 16)
    block_stride = (8, 8)
    cell_size  = (8, 8)
    nbins      = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
    return hog.compute(face_gray).flatten()

def preprocess_dataset():
    """
    Walk through DATASET_RAW, detect faces, save processed images.
    UNBC dataset structure expected:
        dataset/raw/<subject_id>/<sequence>/<frame>.png  (label in filename or CSV)
    """
    ensure_dirs()
    processed_count = 0
    for root, _, files in os.walk(DATASET_RAW):
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            src = os.path.join(root, fname)
            img = cv2.imread(src)
            if img is None:
                continue
            face = detect_and_crop_face(img)
            if face is None:
                continue
            # Mirror the subfolder structure inside processed/
            rel = os.path.relpath(root, DATASET_RAW)
            dst_dir = os.path.join(DATASET_PROC, rel)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            cv2.imwrite(dst, face)
            processed_count += 1

    print(f"Preprocessing done. {processed_count} faces saved to {DATASET_PROC}")

if __name__ == "__main__":
    preprocess_dataset()