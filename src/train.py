
import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from .paths import DATA_DIR as RAW_DATA_DIR, OUTPUT_DIR
from .class_names import CLASS_NAMES


DATA_DIR = RAW_DATA_DIR / "UNIFESP"
CSV_PATH = DATA_DIR / "train_df.csv"

NUM_CLASSES = 22
EPOCHS = 5
BATCH_SIZE = 4
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224
SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_image_path(image_path):
    path = str(image_path).strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return DATA_DIR / path


def parse_target(value):
    labels = []
    for item in str(value).split():
        class_id = int(item)
        if not 0 <= class_id < NUM_CLASSES:
            raise ValueError(f"Invalid class ID: {class_id}")
        labels.append(class_id)
    if not labels:
        raise ValueError("Empty target")
    return labels


class XrayDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = get_image_path(row["image_path"])

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        target = torch.zeros(NUM_CLASSES, dtype=torch.float32)

        for class_id in parse_target(row["Target"]):
            target[class_id] = 1.0

        return image, target


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

            probabilities = torch.sigmoid(logits)
            predicted = (probabilities >= 0.5).int()

            predictions.append(predicted.cpu().numpy())
            targets.append(labels.int().cpu().numpy())

    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)

    micro_f1 = f1_score(
        y_true, y_pred,
        average="micro",
        zero_division=0
    )

    return total_loss / len(loader.dataset), micro_f1


def load_metadata(check_images=True):
    df = pd.read_csv(
        CSV_PATH,
        dtype={"Target": str, "PatientID": str}
    )

    required_columns = ["image_path", "Target", "PatientID"]
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df = df.dropna(subset=required_columns).copy()
    df["Target"] = df["Target"].str.strip()
    df["PatientID"] = df["PatientID"].str.strip()

    df = df[
        (df["Target"] != "") &
        (df["PatientID"] != "")
    ].copy()

    for target in df["Target"]:
        parse_target(target)

    missing_files = [
        str(get_image_path(path))
        for path in df["image_path"]
        if check_images and not get_image_path(path).is_file()
    ]

    if missing_files:
        print("Example missing image:", missing_files[0])
        raise FileNotFoundError(
            f"{len(missing_files)} image files were not found."
        )

    if df["image_path"].duplicated().any():
        raise ValueError("Duplicate body-part image paths.")
    return df


def split_metadata(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=SEED
    )

    train_idx, val_idx = next(
        splitter.split(df, groups=df["PatientID"])
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    return train_df, val_df


def main():
    parser = argparse.ArgumentParser(description="Train the UNIFESP body-part baseline.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--checkpoint", type=str, default=str(OUTPUT_DIR / "bodypart_retrained.pth"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch size, and learning rate must be positive")
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    train_df, val_df = split_metadata(load_metadata())
    pd.concat([train_df.assign(split="train"), val_df.assign(split="val")]).to_csv(
        OUTPUT_DIR / "bodypart_splits.csv", index=False)

    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = XrayDataset(train_df, transform)
    val_dataset = XrayDataset(val_df, transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Training device: {device}")
    print("Loading pretrained ResNet18...")

    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        NUM_CLASSES
    )

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_f1 = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Batch {batch_idx + 1}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}",
                    flush=True
                )

        train_loss = running_loss / len(train_dataset)

        val_loss, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Micro-F1: {val_f1:.4f}",
            flush=True
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val_loss, "validation_micro_f1": val_f1})
        pd.DataFrame(history).to_csv(Path(args.checkpoint).with_suffix(".history.csv"), index=False)

        if val_f1 > best_f1:
            best_f1 = val_f1

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "epoch": epoch + 1,
                "validation_micro_f1": val_f1,
                "image_size": IMAGE_SIZE,
                "threshold": 0.5
            }
            checkpoint["class_names"] = CLASS_NAMES
            checkpoint["seed"] = SEED

            torch.save(
                checkpoint,
                args.checkpoint
            )

            print("Best model saved!", flush=True)

    print("\nTraining completed.")
    print(f"Best validation Micro-F1: {best_f1:.4f}")
    print("Model:", args.checkpoint)


if __name__ == "__main__":
    main()
