# Pain Detection from Facial Expressions
### Affective Computing in Healthcare — Edge Computing on Jetson Nano



## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Pipeline](#system-pipeline)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Results and Model Comparison](#results-and-model-comparison)
- [Real-World Performance](#real-world-performance)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Jetson Nano Deployment](#jetson-nano-deployment)
- [Edge Computing Advantages](#edge-computing-advantages)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Introduction

This project implements an automated, real-time pain detection system using deep learning analysis of facial expressions, deployed on a Jetson Nano edge device. Pain is a critical healthcare signal — yet many patients (unconscious, neonatal, or cognitively impaired) cannot verbally communicate it. This system provides an objective, automated alternative using facial expressions as the detection medium, running entirely on-device without any cloud dependency.

---

## Problem Statement

Current pain assessment in clinical settings relies entirely on patient self-reporting, which fails for:

- Unconscious or post-surgical patients
- Neonates and infants in ICU
- Patients with dementia or cognitive impairment
- Sedated or mechanically ventilated patients

**Our solution:** A camera-based deep learning system that detects pain from facial expressions in real-time, deployed on Jetson Nano edge hardware for privacy-preserving, offline clinical monitoring.

---

## Dataset

### FER2013 — Facial Expression Recognition 2013

| Property | Details |
|---|---|
| Source | [Kaggle — msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| Total Images | 35,887 grayscale face images |
| Image Size | 48 × 48 pixels |
| Original Classes | 7 emotion classes |
| Our Classes | 2 — Pain / No Pain |
| Training Samples Used | 4,000 pain + 4,000 no pain = 8,000 balanced |
| Train / Val Split | 80% / 20% |

### Emotion-to-Pain Mapping

Based on FACS (Facial Action Coding System) research — pain expressions share the same facial muscle activations (AU4, AU6, AU9) as these emotions:

| Emotion | Mapped Class | Action Units | Scientific Reason |
|---|---|---|---|
| Angry | **PAIN** | AU4 | Brow Lowerer |
| Disgust | **PAIN** | AU9 | Nose Wrinkler |
| Fear | **PAIN** | AU4 + AU20 | Overlaps with pain |
| Sad | **PAIN** | AU15 + AU17 | Similar to pain |
| Happy | **NO PAIN** | AU6 + AU12 | Opposite muscles |
| Neutral | **NO PAIN** | None | No pain activation |
| Surprise | **NO PAIN** | AU1 + AU2 | Different muscles |

> **Why FER2013?** The UNBC-McMaster dataset (originally planned) was found to have label mismatches in its Kaggle version — PSPI scores did not correctly correspond to image frames, making supervised training unreliable. FER2013 provides correctly labelled, diverse facial expression data suitable for transfer learning-based pain detection.

---

## System Pipeline

```
Webcam / Video Input
        |
        v
Face Detection (OpenCV Haar Cascade)
  scaleFactor=1.1 | minNeighbors=5 | minSize=(30,30)
        |
        |--- No face detected ---> Skip frame
        |
        v
Preprocessing
  1. Crop largest face ROI
  2. Resize to 48 x 48 pixels
  3. Normalize pixel values: divide by 255.0 → [0.0, 1.0]
  4. Convert to 3-channel by stacking grayscale x3
  5. Convert to PyTorch tensor [1, 3, 48, 48]
  6. Apply ImageNet normalization
        |
        v
Deep Learning Model Inference
  (MobileNetV2 — best model)
        |
        v
Softmax → Pain Probability (0.0 to 1.0)
        |
        v
10-Frame Smoothing Buffer (average last 10 predictions)
        |
        v
Threshold = 0.5
  >= 0.5 → PAIN    (red bounding box)
   < 0.5 → NO PAIN (green bounding box)
        |
        v
Display: Label + Confidence Score + FPS + Inference Time
Log: Every 30 frames → inference_log.txt
```

---

## Model Architecture

Three deep learning models were trained and compared:

### Model 1 — Custom CNN (Built from scratch)

```
Input (3 x 48 x 48)
    ↓
Conv2d(3→32) + BatchNorm + ReLU + MaxPool(2) + Dropout2d(0.25)
    ↓
Conv2d(32→64) + BatchNorm + ReLU + MaxPool(2) + Dropout2d(0.25)
    ↓
Conv2d(64→128) + BatchNorm + ReLU + MaxPool(2) + Dropout2d(0.25)
    ↓
Flatten → Linear(128×6×6 → 256) → ReLU → Dropout(0.5)
    ↓
Linear(256 → 2) → Output
```

- Parameters: ~500K
- Built entirely from scratch — no pre-trained weights
- BatchNorm for stable training, Dropout to prevent overfitting

---

### Model 2 — MobileNetV2 (Transfer Learning) ⭐ Best Model

```
Input (3 x 48 x 48)
    ↓
Frozen layers 1–10 (ImageNet features preserved)
    ↓
Inverted Residual Blocks (fine-tuned)
  Depthwise separable convolutions
  Linear bottlenecks
    ↓
Global Average Pooling → 1280-d feature vector
    ↓
Dropout(0.2) → Linear(1280 → 2) → Output
```

- Parameters: 3.4M
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- First 10 parameter groups frozen during fine-tuning
- Designed for efficient edge device inference

---

### Model 3 — ResNet50 (Transfer Learning)

```
Input (3 x 48 x 48)
    ↓
Frozen parameter groups 1–6 (ImageNet features preserved)
    ↓
Residual Blocks (fine-tuned)
  Skip connections prevent vanishing gradient
  4 layer groups
    ↓
Average Pooling → 2048-d feature vector
    ↓
Linear(2048 → 2) → Output
```

- Parameters: 25M
- Pre-trained on ImageNet with deep residual architecture
- Final FC layer replaced with 2-class classifier

---

## Training Details

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Device | CPU |
| Optimizer | Adam (weight_decay = 1e-4) |
| Learning Rate | 0.001 |
| LR Scheduler | StepLR (step=5, gamma=0.5) |
| Epochs | 20 |
| Batch Size | 32 |
| Loss Function | CrossEntropyLoss (weighted) |
| Pain Class Weight | 2.0 |
| No-Pain Class Weight | 1.0 |
| Train / Val Split | 80% / 20% |
| Input Size | 48 × 48 grayscale |

### Why Weighted Loss?

The pain class is weighted 2× higher than no-pain. In clinical settings, **failing to detect pain (false negative) is more dangerous** than a false alarm (false positive). The weighted loss penalizes the model more when it misses a pain event, making it more sensitive to pain detection.

### Learning Rate Schedule

```
Epochs 1–5:   LR = 0.001000
Epochs 6–10:  LR = 0.000500
Epochs 11–15: LR = 0.000250
Epochs 16–20: LR = 0.000125
```

---

## Results and Model Comparison

| Model | Validation Accuracy | Parameters | Inference Time | Complexity |
|---|---|---|---|---|
| Custom CNN | ~75.00% | ~500K | ~15ms | Low |
| **MobileNetV2** | **76.44% ⭐** | **3.4M** | **~21ms** | **Low** |
| ResNet50 | 75.75% | 25M | ~35ms | High |

### Why MobileNetV2 is the Best Choice

- **Highest accuracy** — 76.44% validation accuracy
- **Lightweight** — only 3.4M parameters, ideal for Jetson Nano
- **Fast inference** — ~21ms per frame, enabling ~15 FPS real-time detection
- **Efficient architecture** — inverted residual blocks designed for edge deployment

### Training Graphs

Training accuracy and loss graphs for all three models are available in `models/saved/`:

```
models/saved/
├── CustomCNN_training_graph.png
├── MobileNetV2_training_graph.png
└── ResNet50_training_graph.png
```

---

## Real-World Performance

The deployed MobileNetV2 model demonstrates the following behaviour under real webcam conditions:

| Expression | Expected Result | Actual Result |
|---|---|---|
| Neutral / calm face | NO PAIN | ✅ NO PAIN |
| Happy / smiling | NO PAIN | ✅ NO PAIN |
| Angry / furrowed brows | PAIN | ✅ PAIN |
| Fear / eye squinting | PAIN | ✅ PAIN |
| Mouth open only | NO PAIN | ✅ NO PAIN (Fixed) |
| Extreme pain expression | PAIN | ✅ PAIN |

> **Threshold sensitivity:** The classification threshold can be adjusted in `config.py`. Lowering from 0.5 to 0.3 increases sensitivity to mild pain expressions.

---

## Project Structure

```
pain_detection_edge/
│
├── main.py                  ← Entry point — run this file
├── config.py                ← All settings: paths, hyperparameters, thresholds
├── preprocessing.py         ← Haar Cascade face detection and cropping
├── training.py              ← 3 model definitions + training loop + comparison
├── inference.py             ← Real-time webcam detection with smoothing
├── utils.py                 ← Helper functions: normalize, draw labels, plot graphs
├── logger.py                ← Dual logging to screen and file
├── requirements.txt         ← Python dependencies
├── README.md                ← This file
│
├── models/
│   └── saved/
│       ├── best_model.pth              ← Trained MobileNetV2 (best model)
│       ├── MobileNetV2.pth             ← MobileNetV2 weights
│       ├── CustomCNN.pth               ← CustomCNN weights
│       ├── MobileNetV2_training_graph.png
│       └── CustomCNN_training_graph.png
│
├── logs/
│   └── inference_log.txt    ← Auto-generated inference log
│
└── outputs/                 ← Demo videos and output files
```

### File Descriptions

| File | Purpose |
|---|---|
| `main.py` | CLI entry point — accepts `--mode preprocess/train/infer` |
| `config.py` | Central configuration — all paths, hyperparameters, thresholds in one place |
| `preprocessing.py` | Haar Cascade face detection, ROI cropping, dataset preprocessing |
| `training.py` | All 3 model architectures, training loop, model comparison, best model selection |
| `inference.py` | Real-time webcam inference with temporal smoothing and on-screen display |
| `utils.py` | Shared utilities — normalize images, draw labels, save training graphs, create folders |
| `logger.py` | Writes timestamped logs to both terminal and `inference_log.txt` |

---

## Setup and Installation

### Requirements

- Python 3.6+
- Webcam (USB or built-in)
- 4GB+ RAM

### Installation on PC (Windows / Linux / Mac)

```bash
# 1. Clone the repository
git clone https://github.com/SarahSingh26/pain_detection_edge.git
cd pain_detection_edge

# 2. Create virtual environment
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Linux/Mac:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### Run Inference (Real-time webcam detection)

```bash
python main.py --mode infer
```

**What happens:**
- Opens your webcam
- Detects face using Haar Cascade
- Classifies pain/no-pain in real-time
- Shows label, confidence score, FPS, and inference time on screen
- Logs results every 30 frames to `logs/inference_log.txt`
- Press **Q** to quit

### Run Training (Train all 3 models and compare)

```bash
python main.py --mode train
```

**What happens:**
- Trains CustomCNN, MobileNetV2, and ResNet50
- Saves each model to `models/saved/`
- Saves training graphs for each model
- Compares all 3 and saves the best as `best_model.pth`

### Run Preprocessing (Face detection on dataset)

```bash
python main.py --mode preprocess
```

### Configuration

All settings can be changed in `config.py`:

```python
PAIN_THRESHOLD   = 0.5    # Lower = more sensitive to pain (try 0.3)
MODEL_INPUT_SIZE = (48, 48)
BATCH_SIZE       = 32
EPOCHS           = 20
LEARNING_RATE    = 0.001
INPUT_SOURCE     = 0      # 0 = webcam, or path to video file
```

---

## Jetson Nano Deployment

### Prerequisites on Jetson

- JetPack OS installed on SD card
- USB webcam connected
- Connected to same WiFi network as development PC

### Step 1 — Connect via SSH (from PC)

```bash
# Find Jetson IP address (run on Jetson terminal)
hostname -I

# Connect from PC using PuTTY or terminal
ssh nvidia@<JETSON_IP>
```

### Step 2 — Clone Repository on Jetson

```bash
cd ~
git clone https://github.com/SarahSingh26/pain_detection_edge.git
cd pain_detection_edge
```

### Step 3 — Install Dependencies on Jetson

```bash
# Install OpenCV (Jetson-optimized version)
sudo apt install python3-opencv -y

# Install PyTorch for Jetson (special ARM64 wheel — NOT regular pip install)
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
     -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# Install remaining requirements
pip3 install numpy matplotlib pillow scikit-learn tqdm
```

### Step 4 — Fix Haar Cascade Path (Jetson-specific)

On Jetson, OpenCV is an older version and `cv2.data` may not exist. Find the cascade file:

```bash
find / -name "haarcascade_frontalface_default.xml" 2>/dev/null
```

Then update `preprocessing.py`:

```python
# Replace this:
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# With the actual path found above, for example:
FACE_CASCADE = cv2.CascadeClassifier(
    "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
)
```

### Step 5 — Set Display and Run

```bash
# Set display for SSH windowed output
export DISPLAY=:0
xhost +

# Run inference
python3 main.py --mode infer
```

### For CSI Camera (Jetson native camera)

Change `INPUT_SOURCE` in `config.py`:

```python
INPUT_SOURCE = "nvarguscamerasrc ! video/x-raw(memory:NVMM), \
    width=640, height=480 ! nvvidconv ! \
    video/x-raw, format=BGRx ! videoconvert ! appsink"
```

### Updating Code on Jetson

```bash
# Pull latest changes from GitHub
cd ~/pain_detection_edge
git pull origin main
```

---

## Edge Computing Advantages

| Feature | Cloud Approach | Edge (Jetson Nano) |
|---|---|---|
| Internet Required | Yes — always | No — fully offline |
| Latency | High (seconds) | Low (~21ms) |
| Patient Privacy | Risk of data breach | Data stays on device |
| Works Offline | No | Yes |
| Real-Time Processing | Limited | Yes (~15 FPS) |
| Hospital Compliance | Complex | HIPAA/GDPR friendly |

**Why edge computing matters for healthcare:**
Patient facial video data is among the most sensitive categories of personal health information. By performing all inference on the Jetson Nano device located within the clinical environment, no patient data is ever transmitted to external servers, ensuring complete privacy and regulatory compliance.

---

## Limitations

1. **Dataset approximation** — FER2013 emotions are used as proxies for clinical pain. Genuine clinical datasets with PSPI annotations would provide superior results.

2. **Binary classification** — The system classifies pain as present or absent. Multi-level pain intensity scoring (mild/moderate/severe) would provide greater clinical utility.

3. **Frontal face constraint** — Haar Cascade is optimized for frontal faces. Non-frontal orientations may reduce detection reliability.

4. **CPU training** — Training was performed on CPU, limiting dataset size and epoch count. GPU training would enable larger-scale experiments.

5. **No TensorRT optimization** — The deployed model has not been optimized with TensorRT, which could provide 2–4× inference speedup on Jetson Nano.

---

## Future Work

- Train on the official UNBC-McMaster dataset with clinically validated PSPI pain scores
- Implement multi-level pain intensity scoring aligned with clinical pain scales
- Apply TensorRT optimization for faster Jetson Nano inference
- Incorporate deep learning-based face detection (RetinaFace / MTCNN) for non-frontal faces
- Develop real-time alert/notification system for sustained pain detection
- Conduct clinical validation studies comparing system output against nurse-assessed pain scores

---

## References

1. Prkachin, K. M., & Solomon, P. E. (2008). The structure, reliability and validity of pain expression. *Pain*, 139(2), 267–274.

2. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System*. Consulting Psychologists Press.

3. Lucey, P., et al. (2011). Painful data: The UNBC-McMaster shoulder pain expression archive database. *IEEE FG*.

4. Goodfellow, I., et al. (2013). Challenges in representation learning: A report on three machine learning contests. *ICML 2013*.

5. Sandler, M., Howard, A., et al. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. *CVPR*.

6. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*.

7. Franklin, D., & Treichler, J. (2019). NVIDIA Jetson Nano delivers 472 GFLOPS for edge AI. *NVIDIA Developer Blog*.

---

## Authors

**Sarah Singh** — Roll No: 1024240071
**Harshil Bansal** — Roll No: 1024240040

Department of Computer Science and Engineering
Thapar Institute of Engineering & Technology, Patiala — 147004

**Supervisor:** Mr. Manav Malhotra

---

## GitHub Repository

```
https://github.com/SarahSingh26/pain_detection_edge
```

---

*This project was developed as part of an Edge Computing course assignment focusing on deploying AI models on resource-constrained edge devices for real-world healthcare applications.*
