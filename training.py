# training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
import os, cv2, numpy as np, random
from config import (DATASET_PROC, MODEL_DIR, BATCH_SIZE, EPOCHS,
                    LEARNING_RATE, TRAIN_SPLIT, NUM_CLASSES, MODEL_INPUT_SIZE)
from utils import plot_training_graphs, ensure_dirs
from logger import log_info

ensure_dirs()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_info(f"Training on: {DEVICE}")

# ─── Dataset ─────────────────────────────────────────────────────────────────
class PainDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform

        for label_name, label_idx in [("pain", 1), ("no_pain", 0)]:
            folder = os.path.join(root, label_name)
            if not os.path.exists(folder):
                log_info(f"Folder not found: {folder}")
                continue
            for f in os.listdir(folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append(
                        (os.path.join(folder, f), label_idx)
                    )

        # ── Balanced dataset — 4000 each ──
        pain_samples    = [(p, l) for p, l in self.samples if l == 1]
        no_pain_samples = [(p, l) for p, l in self.samples if l == 0]
        limit = 4000
        random.shuffle(pain_samples)
        random.shuffle(no_pain_samples)
        pain_samples    = pain_samples[:limit]
        no_pain_samples = no_pain_samples[:limit]
        self.samples    = pain_samples + no_pain_samples
        random.shuffle(self.samples)
        log_info(f"Balanced: {len(pain_samples)} pain + "
                 f"{len(no_pain_samples)} no_pain = "
                 f"{len(self.samples)} total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        # ── FER2013 is grayscale 48x48 ──
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(
                (MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1]),
                dtype=np.float32
            )
        else:
            img = cv2.resize(
                img, (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0])
            )
            img = img.astype(np.float32) / 255.0

        # Grayscale to 3 channel
        img    = np.stack([img]*3, axis=0)
        tensor = torch.tensor(img)
        if self.transform:
            tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)

# ─── Model 1: Custom CNN ──────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ─── Model 2: MobileNetV2 ─────────────────────────────────────────────────────
def get_mobilenet():
    m = models.mobilenet_v2(pretrained=True)
    for i, param in enumerate(m.features.parameters()):
        if i < 10:
            param.requires_grad = False
    m.classifier[1] = nn.Linear(m.last_channel, NUM_CLASSES)
    return m

# ─── Model 3: ResNet50 ────────────────────────────────────────────────────────
def get_resnet50():
    m = models.resnet50(pretrained=True)
    for i, (name, param) in enumerate(m.named_parameters()):
        if i < 6:
            param.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

# ─── Training Loop ────────────────────────────────────────────────────────────
def train_model(model, model_name, train_loader, val_loader):
    model     = model.to(DEVICE)
    weights   = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    best_val_acc = 0.0
    train_accs, val_accs     = [], []
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == labels).sum().item()
            total   += labels.size(0)
        t_acc  = correct / total
        t_loss = running_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        vtotal, vcorrect, vloss = 0, 0, 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out   = model(imgs)
                vloss += criterion(out, labels).item()
                _, pred = torch.max(out, 1)
                vcorrect += (pred == labels).sum().item()
                vtotal   += labels.size(0)
        v_acc  = vcorrect / vtotal
        v_loss = vloss / len(val_loader)

        train_accs.append(t_acc);    val_accs.append(v_acc)
        train_losses.append(t_loss); val_losses.append(v_loss)
        scheduler.step()

        log_info(f"[{model_name}] Epoch {epoch+1}/{EPOCHS} | "
                 f"Train Acc={t_acc:.4f} | Val Acc={v_acc:.4f} | "
                 f"Train Loss={t_loss:.4f} | Val Loss={v_loss:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(
                model.state_dict(),
                os.path.join(MODEL_DIR, f"{model_name}.pth")
            )

    plot_training_graphs(
        train_accs, val_accs,
        train_losses, val_losses,
        model_name
    )
    log_info(f"[{model_name}] Best Val Accuracy: {best_val_acc:.4f}")
    return best_val_acc

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    dataset  = PainDataset(DATASET_PROC, transform=transform)
    n_train  = int(len(dataset) * TRAIN_SPLIT)
    n_val    = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )
    log_info(f"Train={n_train} | Val={n_val}")

    results = {}
    for name, model in [("CustomCNN",   CustomCNN()),
                         ("MobileNetV2", get_mobilenet()),
                         ("ResNet50",    get_resnet50())]:
        log_info(f"\n{'='*40}\nTraining {name}\n{'='*40}")
        acc = train_model(model, name, train_loader, val_loader)
        results[name] = acc

    # ── Select best model ──
    best_name = max(results, key=results.get)
    import shutil
    shutil.copy(
        os.path.join(MODEL_DIR, f"{best_name}.pth"),
        os.path.join(MODEL_DIR, "best_model.pth")
    )
    log_info("\n===== MODEL COMPARISON =====")
    for name, acc in results.items():
        marker = " <- BEST" if name == best_name else ""
        log_info(f"  {name}: {acc:.4f}{marker}")
    log_info(f"Best model '{best_name}' saved as best_model.pth")