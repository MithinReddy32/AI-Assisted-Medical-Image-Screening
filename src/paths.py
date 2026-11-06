
"""Project paths and automatic model downloads."""

from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"

HF_REPO = "Miyamura12/AI-Medical-Xray-Models"

MODEL_FILES = [
    "bodypart_resnet18.pth",
    "chest_resnet18.pth",
    "mura_resnet18.pth",
    "chest_thresholds.csv",
]


def ensure_models():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for filename in MODEL_FILES:
        destination = OUTPUT_DIR / filename

        if destination.is_file() and destination.stat().st_size > 0:
            continue

        url = (
            f"https://huggingface.co/{HF_REPO}/"
            f"resolve/main/{filename}"
        )

        temporary = destination.with_name(filename + ".download")

        try:
            urlretrieve(url, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
