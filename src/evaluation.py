"""Common metrics, inference, and provenance for offline evaluation."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from sklearn.metrics import (
    average_precision_score, confusion_matrix,
    precision_recall_curve, roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(checkpoint, dataset, split, sample_limit=None):
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {"path": str(Path(checkpoint).resolve()), "sha256": sha256(checkpoint)},
        "dataset_manifest": {"path": str(Path(dataset).resolve()), "sha256": sha256(dataset)},
        "split": split,
        "sample_limit": sample_limit,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "preprocessing": {"resize": [224, 224], "channels": "RGB",
                          "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def binary_metrics(targets, scores, threshold=0.5):
    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=float)
    if targets.ndim != 1 or scores.shape != targets.shape:
        raise ValueError("Binary targets and scores must be equally sized vectors.")
    if not np.isin(targets, [0, 1]).all() or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Expected binary targets and finite probability scores in [0, 1].")
    targets = targets.astype(int)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid classification threshold.")
    tn, fp, fn, tp = confusion_matrix(targets, scores >= threshold, labels=[0, 1]).ravel()
    both = len(np.unique(targets)) == 2
    divide = lambda a, b: float(a / b) if b else None
    return {
        "samples": len(targets), "positives": int(tp + fn), "negatives": int(tn + fp),
        "threshold": float(threshold),
        "auroc": float(roc_auc_score(targets, scores)) if both else None,
        "average_precision": float(average_precision_score(targets, scores)) if both else None,
        "precision": divide(tp, tp + fp), "recall": divide(tp, tp + fn),
        "specificity": divide(tn, tn + fp), "f1": divide(2 * tp, 2 * tp + fp + fn),
        "true_positive": int(tp), "false_positive": int(fp),
        "true_negative": int(tn), "false_negative": int(fn),
    }


def finding_metrics(targets, scores, names, thresholds):
    rows = [{"finding": name, **binary_metrics(targets[:, i], scores[:, i], thresholds[name])}
            for i, name in enumerate(names)]
    aucs = [r["auroc"] for r in rows if r["auroc"] is not None]
    return {"macro_auroc": float(np.mean(aucs)) if aucs else None,
            "classes_with_defined_auroc": len(aucs), "per_finding": rows}


def fit_f1_thresholds(targets, scores, names):
    """Call only with validation data; test evaluation never calls this."""
    rows = []
    for i, name in enumerate(names):
        y = targets[:, i]
        if len(np.unique(y)) != 2:
            raise ValueError(f"Cannot calibrate {name}: validation needs both labels.")
        precision, recall, thresholds = precision_recall_curve(y, scores[:, i])
        f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                       out=np.zeros_like(precision[:-1]), where=(precision[:-1] + recall[:-1]) > 0)
        best = int(np.argmax(f1))
        rows.append({"Finding": name, "Threshold": float(thresholds[best]),
                     "Validation F1": float(f1[best])})
    return pd.DataFrame(rows)


def collect_predictions(model, dataset, batch_size=16, workers=0, device="cpu"):
    if not len(dataset):
        raise ValueError("Evaluation dataset is empty.")
    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    scores, targets = [], []
    with torch.inference_mode():
        for images, labels in tqdm(loader, desc="Evaluating", leave=False, mininterval=5):
            scores.append(torch.sigmoid(model(images.to(device))).cpu().numpy())
            targets.append(labels.numpy())
    return np.concatenate(targets), np.concatenate(scores)


def add_runtime_arguments(parser):
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="A deterministic subset for smoke checks; 0 evaluates the full split.")


def configure_runtime(args):
    if args.batch_size < 1 or args.workers < 0 or args.threads < 1 or args.max_samples < 0:
        raise ValueError("Invalid batch size, workers, threads, or sample limit.")
    torch.set_num_threads(args.threads)


def subset(frame, limit):
    return frame.sample(n=min(limit, len(frame)), random_state=42).reset_index(drop=True) if limit else frame.reset_index(drop=True)
