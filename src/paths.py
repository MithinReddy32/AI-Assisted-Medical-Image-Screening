"""Project paths are independent of the process working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "outputs"
EVALUATION_DIR = OUTPUT_DIR / "evaluation"
