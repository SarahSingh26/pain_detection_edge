# main.py
"""
Pain Detection from Face – Edge Computing (Jetson Nano)
Usage:
    python main.py --mode train       # train all 3 models & compare
    python main.py --mode infer       # run inference on webcam
    python main.py --mode preprocess  # preprocess raw dataset
"""
import argparse
from utils import ensure_dirs
from logger import log_info

def main():
    parser = argparse.ArgumentParser(description="Pain Detection – Edge")
    parser.add_argument("--mode", choices=["preprocess", "train", "infer"],
                        required=True, help="Which mode to run")
    args = parser.parse_args()
    ensure_dirs()

    if args.mode == "preprocess":
        log_info("Starting dataset preprocessing...")
        from preprocessing import preprocess_dataset
        preprocess_dataset()

    elif args.mode == "train":
        log_info("Starting model training...")
        import training  # runs the if __name__ == '__main__' block is not called
        # re-call explicitly:
        from training import (PainDataset, CustomCNN, get_mobilenet,
                               get_resnet50, train_model, DataLoader,
                               random_split, DEVICE)
        import torch
        from torchvision import transforms
        from config import DATASET_PROC, TRAIN_SPLIT, BATCH_SIZE, MODEL_DIR
        import shutil, os

        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset   = PainDataset(DATASET_PROC, transform=transform)
        n_train   = int(len(dataset) * TRAIN_SPLIT)
        n_val     = len(dataset) - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        tl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
        vl = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

        results = {}
        for name, model in [("CustomCNN",   CustomCNN()),
                             ("MobileNetV2", get_mobilenet()),
                             ("ResNet50",    get_resnet50())]:
            results[name] = train_model(model, name, tl, vl)

        best = max(results, key=results.get)
        shutil.copy(os.path.join(MODEL_DIR, f"{best}.pth"),
                    os.path.join(MODEL_DIR, "best_model.pth"))
        log_info(f"Best model: {best} ({results[best]:.4f}) -> saved as best_model.pth")

    elif args.mode == "infer":
        log_info("Starting inference...")
        from inference import run_inference
        from config import INPUT_SOURCE
        run_inference(INPUT_SOURCE)

if __name__ == "__main__":
    main()