
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit

from .paths import DATA_DIR, OUTPUT_DIR, ROOT

NIH_DIR = DATA_DIR / "NIH" / "Chest"
CSV_PATH = NIH_DIR / "Data_Entry_2017.csv"
SPLIT_PATH = OUTPUT_DIR / "chest_splits.csv"

FINDINGS = [
    "Atelectasis", "Cardiomegaly", "Effusion",
    "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening",
    "Hernia"
]

IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def prepare_metadata():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if SPLIT_PATH.exists():
        df = pd.read_csv(SPLIT_PATH)
        validate_patient_splits(df)
        return df

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"NIH metadata missing: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required = {
        "Image Index", "Finding Labels", "Patient ID"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    image_map = {}

    for path in NIH_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in {
            ".png", ".jpg", ".jpeg"
        }:
            if path.name in image_map:
                raise ValueError(
                    f"Duplicate image filename: {path.name}"
                )

                image_map[path.name] = path.relative_to(ROOT).as_posix()

    df["image_path"] = df["Image Index"].map(image_map)

    print("Total CSV rows:", len(df))
    print("Images found:", df["image_path"].notna().sum())

    df = df.dropna(subset=["image_path"]).copy()

    if df.empty:
        raise RuntimeError(
            "No NIH images found. Check your folder layout."
        )

    allowed = set(FINDINGS) | {"No Finding"}

    for labels in df["Finding Labels"]:
        parsed = set(str(labels).split("|"))
        unknown = parsed - allowed

        if unknown:
            raise ValueError(f"Unknown labels: {unknown}")

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42
    )

    train_idx, remaining_idx = next(
        first_split.split(df, groups=df["Patient ID"])
    )

    train_df = df.iloc[train_idx].copy()
    remaining_df = df.iloc[remaining_idx].copy()

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )

    val_idx, test_idx = next(
        second_split.split(
            remaining_df,
            groups=remaining_df["Patient ID"]
        )
    )

    val_df = remaining_df.iloc[val_idx].copy()
    test_df = remaining_df.iloc[test_idx].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    combined = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    validate_patient_splits(combined)

    combined.to_csv(SPLIT_PATH, index=False)

    print(combined["split"].value_counts())

    return combined


def validate_patient_splits(df):
    required = {"Image Index", "Finding Labels", "Patient ID", "image_path", "split"}
    if not required.issubset(df.columns) or df[list(required)].isna().any().any():
        raise ValueError("Chest split manifest has missing columns or values.")
    if set(df["split"]) != {"train", "val", "test"}:
        raise ValueError("Chest manifest must contain train, val, and test splits.")
    if df["Image Index"].duplicated().any():
        raise ValueError("Duplicate images in chest split manifest.")
    if (df.groupby("Patient ID")["split"].nunique() > 1).any():
        raise ValueError("Patient overlap between chest dataset splits.")
    allowed = set(FINDINGS) | {"No Finding"}
    for labels in df["Finding Labels"].unique():
        if set(str(labels).split("|")) - allowed:
            raise ValueError(f"Unknown finding in chest split manifest: {labels}")


def resolve_image_path(value):
    path = Path(value)
    # Relocate legacy absolute paths after copying the project to another PC.
    normalized = str(value).replace("\\", "/")
    marker = "data/raw/NIH/Chest/"
    if marker in normalized:
        return NIH_DIR / normalized.split(marker, 1)[1]
    if path.is_absolute():
        return path
    return ROOT / path


class ChestDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform or eval_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(resolve_image_path(row["image_path"])) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        labels = set(
            str(row["Finding Labels"]).split("|")
        )

        target = torch.tensor(
            [
                float(name in labels)
                for name in FINDINGS
            ],
            dtype=torch.float32
        )

        return image, target
