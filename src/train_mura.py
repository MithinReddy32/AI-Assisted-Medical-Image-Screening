"""Reproducible MURA baseline; existing specialist checkpoints are preserved."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from .evaluate_mura import mura_metrics
from .evaluation import collect_predictions, sha256, subset
from .mura_dataset import (
    MURA_DIR, MuraDataset, load_mura_metadata, train_transform, validate_mura_patient_split,
)
from .paths import OUTPUT_DIR
from .train import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--no-pretrained", action="store_true", help="Offline smoke training without ImageNet downloads.")
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "mura_retrained.pth")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.lr <= 0 or min(args.max_train, args.max_val, args.workers) < 0:
        parser.error("Invalid training arguments.")
    set_seed(42)
    torch.set_num_threads(args.threads)
    train, valid = load_mura_metadata("train"), load_mura_metadata("valid")
    validate_mura_patient_split(train, valid)
    train, valid = subset(train, args.max_train), subset(valid, args.max_val)
    loader = DataLoader(MuraDataset(train, train_transform), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers)
    model = models.resnet18(weights=None if args.no_pretrained else models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, history = -1.0, []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, targets = images.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        _, scores = collect_predictions(model, MuraDataset(valid), args.batch_size, args.workers, args.device)
        metrics, _, _ = mura_metrics(valid, scores)
        image, study = metrics["image"], metrics["study_mean_score"]
        history.append({"epoch": epoch, "train_loss": total_loss / len(train),
                        "image_auc": image["auroc"], "image_f1": image["f1"],
                        "study_auc": study["auroc"], "study_f1": study["f1"]})
        auc = study["auroc"]
        if epoch == 1 or (auc is not None and auc > best_auc):
            best_auc = auc if auc is not None else -1.0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                        "val_image_auc": image["auroc"], "val_study_auc": auc,
                        "architecture": "resnet18", "task": "mura_abnormality",
                        "seed": 42, "study_aggregation": "mean_sigmoid_score", "image_size": 224,
                        "subset_training": bool(args.max_train or args.max_val),
                        "training_args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        "train_labels_sha256": sha256(MURA_DIR / "train_labeled_studies.csv"),
                        "valid_labels_sha256": sha256(MURA_DIR / "valid_labeled_studies.csv")}, args.checkpoint)
        print(history[-1], flush=True)
        pd.DataFrame(history).to_csv(args.checkpoint.with_suffix(".history.csv"), index=False)
    print("Checkpoint:", args.checkpoint)


if __name__ == "__main__":
    main()
