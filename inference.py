# inference.py
import cv2, torch, time, numpy as np
from collections import deque
from torchvision import transforms
from config import (BEST_MODEL_PATH, MODEL_INPUT_SIZE, PAIN_THRESHOLD,
                    NUM_CLASSES, SHOW_FPS, SHOW_CONFIDENCE)
from preprocessing import FACE_CASCADE
from utils import draw_label
from logger import log_inference, log_info, log_error
from training import CustomCNN, get_mobilenet, get_resnet50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def load_best_model():
    for model_fn, name in [
        (get_resnet50,               "ResNet50"),
        (get_mobilenet,              "MobileNetV2"),
        (lambda: CustomCNN(NUM_CLASSES), "CustomCNN"),
    ]:
        try:
            model = model_fn().to(DEVICE)
            model.load_state_dict(
                torch.load(BEST_MODEL_PATH, map_location=DEVICE)
            )
            model.eval()
            log_info(f"Loaded best model: {name}")
            return model
        except RuntimeError:
            continue
    raise RuntimeError("Could not load model!")

def preprocess_face(face_bgr):
    """Match training preprocessing exactly."""
    # Convert to grayscale like FER2013
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(
        face, (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])
    )
    face = face.astype(np.float32) / 255.0
    # Grayscale to 3 channel
    face   = np.stack([face]*3, axis=0)
    tensor = torch.tensor(face).unsqueeze(0).to(DEVICE)
    tensor = NORMALIZE(tensor)
    return tensor

def run_inference(source=0):
    model = load_best_model()
    cap   = cv2.VideoCapture(source)
    if not cap.isOpened():
        log_error(f"Cannot open source: {source}")
        return

    frame_id    = 0
    fps_time    = time.time()
    fps         = 0.0
    pain_buffer = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id     += 1
        display_frame = frame.copy()
        gray          = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Detect faces ──
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1,
            minNeighbors=5, minSize=(30, 30)
        )

        label, confidence, infer_ms = "NO FACE", 0.0, 0.0

        if len(faces) > 0:
            # Use largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_bgr    = frame[y:y+h, x:x+w]

            # Preprocess matching training exactly
            tensor = preprocess_face(face_bgr)

            t0 = time.time()
            with torch.no_grad():
                out       = model(tensor)
                probs     = torch.softmax(out, dim=1)[0]
                pain_conf = probs[1].item()
            infer_ms = (time.time() - t0) * 1000

            # Smooth over last 10 frames
            pain_buffer.append(pain_conf)
            avg_conf = sum(pain_buffer) / len(pain_buffer)

            label      = "PAIN" if avg_conf >= PAIN_THRESHOLD else "NO PAIN"
            confidence = avg_conf

            # Draw rectangle
            color = (0, 0, 255) if label == "PAIN" else (0, 255, 0)
            cv2.rectangle(
                display_frame,
                (x, y), (x+w, y+h),
                color, 2
            )

        # ── FPS ──
        now      = time.time()
        fps      = 1.0 / (now - fps_time + 1e-9)
        fps_time = now

        # ── Draw overlays ──
        display_frame = draw_label(
            display_frame, label, confidence, infer_ms, fps
        )

        # ── Log every 30 frames ──
        if frame_id % 30 == 0:
            log_inference(
                frame_id, label, confidence, infer_ms, fps
            )

        cv2.imshow("Pain Detection - Edge", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_info("Inference session ended.")

if __name__ == "__main__":
    from config import INPUT_SOURCE
    run_inference(INPUT_SOURCE)