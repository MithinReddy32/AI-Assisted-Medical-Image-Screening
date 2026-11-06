
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .chest_dataset import (
    FINDINGS,
    OUTPUT_DIR,
    ChestDataset,
    prepare_metadata,
    train_transform,
    eval_transform,
)


def build_model(pretrained=True):
    weights = (
        models.ResNet18_Weights.DEFAULT
        if pretrained else None
    )

    model = models.resnet18(weights=weights)

    # NIH ChestX-ray14 has 14 possible findings.
    model.fc = nn.Linear(
        model.fc.in_features,
        len(FINDINGS)
    )

    return model


def calculate_macro_auc(targets, probabilities):
    auc_scores = []

    for i in range(len(FINDINGS)):
        # AUROC needs both positive and negative examples.
        if len(np.unique(targets[:, i])) < 2:
            continue

        auc = roc_auc_score(
            targets[:, i],
            probabilities[:, i]
        )

        auc_scores.append(auc)

    if not auc_scores:
        return float("nan")

    return float(np.mean(auc_scores))


def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probabilities = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            probabilities = torch.sigmoid(logits)

            all_targets.append(
                targets.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    targets = np.concatenate(all_targets)
    probabilities = np.concatenate(all_probabilities)

    avg_loss = total_loss / total_samples

    macro_auc = calculate_macro_auc(
        targets,
        probabilities
    )

    return avg_loss, macro_auc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs", type=int, default=5
    )

    parser.add_argument(
        "--batch-size", type=int, default=4
    )

    parser.add_argument(
        "--lr", type=float, default=0.0001
    )

    parser.add_argument(
        "--workers", type=int, default=0
    )

    parser.add_argument(
        "--max-train", type=int, default=0
    )

    parser.add_argument(
        "--max-val", type=int, default=0
    )

    parser.add_argument(
        "--smoke-test", action="store_true"
    )
    parser.add_argument("--checkpoint", type=Path, default=OUTPUT_DIR / "chest_retrained.pth")
    parser.add_argument("--no-pretrained", action="store_true")

    args = parser.parse_args()
    if min(args.epochs, args.batch_size) < 1 or args.lr <= 0 or min(args.max_train, args.max_val, args.workers) < 0:
        parser.error("Invalid training arguments.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Training device:", device)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load the existing patient-level NIH splits.
    df = prepare_metadata()

    train_df = df[
        df["split"] == "train"
    ].copy()

    val_df = df[
        df["split"] == "val"
    ].copy()

    # Optional small subsets for testing.
    if args.max_train > 0:
        train_df = train_df.sample(
            n=min(args.max_train, len(train_df)),
            random_state=42
        )

    if args.max_val > 0:
        val_df = val_df.sample(
            n=min(args.max_val, len(val_df)),
            random_state=42
        )

    print("Training images:", len(train_df))
    print("Validation images:", len(val_df))

    train_loader = DataLoader(
        ChestDataset(
            train_df,
            train_transform
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    val_loader = DataLoader(
        ChestDataset(
            val_df,
            eval_transform
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )

    print("Initializing ResNet18...")

    model = build_model(
        pretrained=not args.no_pretrained
    ).to(device)

    # Multi-label classification.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_auc = -1.0
    best_val_loss = float("inf")

    history = []

    checkpoint_path = (
        OUTPUT_DIR / "chest_smoke_test.pth"
        if args.smoke_test
        else args.checkpoint
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):

        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}"
        )

        for images, targets in progress:

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                targets
            )

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            total_train_loss += (
                loss.item() * batch_size
            )

            total_train_samples += batch_size

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            total_train_loss / total_train_samples
        )

        val_loss, val_auc = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"\nEpoch {epoch + 1}/{args.epochs}"
        )

        print(
            f"Training Loss: {train_loss:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f}"
        )

        print(
            f"Validation Macro-AUROC: {val_auc:.4f}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_auc": val_auc
        })

        # Save the best model by AUROC.
        # If AUROC is undefined on a tiny smoke-test
        # subset, use validation loss instead.
        if np.isfinite(val_auc):
            improved = val_auc > best_auc
        else:
            improved = best_auc < 0 and val_loss < best_val_loss

        if improved:
            if np.isfinite(val_auc):
                best_auc = val_auc

            best_val_loss = val_loss

            torch.save({
                "model_state_dict": model.state_dict(),
                "findings": FINDINGS,
                "epoch": epoch + 1,
                "val_macro_auc": val_auc,
                "val_loss": val_loss,
                "seed": 42,
                "image_size": 224,
                "subset_training": bool(args.max_train or args.max_val),
            }, checkpoint_path)

            print(
                "Saved best model:",
                checkpoint_path
            )

    history_path = (
        OUTPUT_DIR / "chest_smoke_history.csv"
        if args.smoke_test
        else checkpoint_path.with_suffix(".history.csv")
    )

    pd.DataFrame(history).to_csv(
        history_path,
        index=False
    )

    print("\nTraining finished.")
    print("Training history:", history_path)
    print("Best checkpoint:", checkpoint_path)


if __name__ == "__main__":
    main()
