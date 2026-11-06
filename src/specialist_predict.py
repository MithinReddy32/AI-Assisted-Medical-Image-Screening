
import csv
import math

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

from .paths import ROOT, OUTPUT_DIR
from .chest_dataset import FINDINGS


OUTPUTS = OUTPUT_DIR

CHEST_PATH = OUTPUTS / "chest_resnet18.pth"
MURA_PATH = OUTPUTS / "mura_resnet18.pth"
THRESHOLDS_PATH = OUTPUTS / "chest_thresholds.csv"

CHEST_CLASSES = FINDINGS

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def load_checkpoint(path):
    # The legacy MURA file stores two NumPy float64 metric scalars. Allow
    # exactly those types while retaining PyTorch's restricted weight loader.
    with torch.serialization.safe_globals([
        np._core.multiarray.scalar, np.dtype, type(np.dtype("float64")),
    ]):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_specialist(path, num_outputs):
    checkpoint = load_checkpoint(path)
    if num_outputs == len(CHEST_CLASSES) and checkpoint.get("findings") != CHEST_CLASSES:
        raise ValueError("Chest checkpoint finding order does not match this application.")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_outputs)

    # Both checkpoints were saved as dictionaries.
    state = checkpoint["model_state_dict"]
    model.load_state_dict(state)
    model.eval()

    return model


def load_chest_model(path=CHEST_PATH):
    return load_specialist(path, len(CHEST_CLASSES))


def load_mura_model(path=MURA_PATH):
    return load_specialist(path, 1)


def load_chest_thresholds(path=THRESHOLDS_PATH):
    thresholds = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        name_col = next(
            (c for c in columns
             if c.lower().strip() in
             {"finding", "class", "label", "class_name"}),
            None
        )
        threshold_col = next(
            (c for c in columns
             if "threshold" in c.lower()),
            None
        )

        if name_col is None or threshold_col is None:
            raise ValueError(
                f"Unexpected chest thresholds columns: {columns}"
            )

        for row in reader:
            name = row[name_col].strip()
            if name not in CHEST_CLASSES or name in thresholds:
                raise ValueError(f"Unknown or duplicate chest finding: {name}")
            value = float(row[threshold_col])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid threshold for {name}: {value}")
            thresholds[name] = value

    missing = set(CHEST_CLASSES) - thresholds.keys()
    if missing:
        raise ValueError(f"Chest threshold file is missing findings: {sorted(missing)}")

    return thresholds


def predict_chest(model, image, thresholds):
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        probabilities = torch.sigmoid(model(tensor))[0].tolist()

    results = [
        {
            "finding": name,
            "score": float(score),
            "threshold": float(thresholds[name]),
            "flagged": score >= thresholds[name],
        }
        for name, score in zip(CHEST_CLASSES, probabilities)
    ]

    return sorted(results, key=lambda x: x["score"], reverse=True)


def predict_mura(model, image):
    tensor = transform(image.convert("RGB")).unsqueeze(0)

    with torch.inference_mode():
        score = torch.sigmoid(model(tensor)).item()

    return float(score)
