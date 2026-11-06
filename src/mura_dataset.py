"""Read MURA's official image and study manifests without inferring labels."""
from pathlib import PurePosixPath

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .paths import DATA_DIR
from .chest_dataset import eval_transform

MURA_DIR = DATA_DIR / "MURA-v1.1"
train_transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_mura_metadata(split):
    if split not in {"train", "valid"}:
        raise ValueError("MURA provides train and valid manifests here; no held-out test labels are available.")
    images = pd.read_csv(MURA_DIR / f"{split}_image_paths.csv", header=None, names=["image_path"])
    studies = pd.read_csv(MURA_DIR / f"{split}_labeled_studies.csv", header=None, names=["study_id", "label"])
    if images.isna().any().any() or studies.isna().any().any():
        raise ValueError("Missing entries in MURA manifests.")
    images["image_path"] = images["image_path"].str.replace("\\", "/", regex=False)
    studies["study_id"] = studies["study_id"].str.replace("\\", "/", regex=False).str.rstrip("/")
    if images["image_path"].duplicated().any() or studies["study_id"].duplicated().any():
        raise ValueError("Duplicate MURA image or study entries.")
    if not studies["label"].isin([0, 1]).all():
        raise ValueError("MURA study labels must be 0 or 1.")
    for path in images["image_path"]:
        parts = PurePosixPath(path).parts
        if len(parts) != 6 or parts[:2] != ("MURA-v1.1", split) or ".." in parts:
            raise ValueError(f"Unexpected MURA image path: {path}")
    images["study_id"] = images["image_path"].map(lambda p: str(PurePosixPath(p).parent))
    if set(images["study_id"]) != set(studies["study_id"]):
        raise ValueError("MURA image and study manifests disagree.")
    images = images.merge(studies, on="study_id", how="left", validate="many_to_one")
    images["patient_id"] = images["image_path"].map(lambda p: PurePosixPath(p).parts[-3])
    images["region"] = images["image_path"].map(lambda p: PurePosixPath(p).parts[-4])
    return images


def validate_mura_patient_split(train, valid):
    overlap = set(train["patient_id"]) & set(valid["patient_id"])
    if overlap:
        raise ValueError(f"Patient overlap between MURA train and valid: {len(overlap)}")


class MuraDataset(Dataset):
    def __init__(self, frame, transform=eval_transform):
        self.df = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        with Image.open(DATA_DIR / row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor([row["label"]], dtype=torch.float32)
