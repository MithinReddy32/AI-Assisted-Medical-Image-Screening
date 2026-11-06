"""Evaluate the original patient-grouped UNIFESP validation split."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score, hamming_loss, precision_score, recall_score

from .class_names import CLASS_NAMES
from .evaluation import add_runtime_arguments, collect_predictions, configure_runtime, provenance, save_json, subset
from .paths import EVALUATION_DIR
from .pipeline_metrics import routing_metrics
from .predict import MODEL_PATH, load_model, transform
from .train import CSV_PATH, NUM_CLASSES, XrayDataset, load_metadata, split_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_DIR / "bodypart")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    configure_runtime(args)
    train, valid = split_metadata(load_metadata())
    if set(train["PatientID"]) & set(valid["PatientID"]):
        raise ValueError("Patient leakage in body-part split.")
    frame = subset(valid, args.max_samples)
    targets, scores = collect_predictions(load_model(args.checkpoint), XrayDataset(frame, transform),
                                          args.batch_size, args.workers, args.device)
    predicted = scores >= 0.5
    report = provenance(args.checkpoint, CSV_PATH, "val", args.max_samples or None)
    report.update({"images": len(frame), "patients": int(frame["PatientID"].nunique()),
                   "patient_overlap_train_val": 0, "classification_threshold": 0.5,
                   "split_seed": 42, "split_method": "GroupShuffleSplit(test_size=0.2, grouped by PatientID)",
                   "micro_f1": float(f1_score(targets, predicted, average="micro", zero_division=0)),
                   "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
                   "micro_precision": float(precision_score(targets, predicted, average="micro", zero_division=0)),
                   "micro_recall": float(recall_score(targets, predicted, average="micro", zero_division=0)),
                   "hamming_loss": float(hamming_loss(targets, predicted)),
                   "evaluation_note": "Validation was used for checkpoint selection; this is not an independent test estimate."})
    per_class = classification_report(targets, predicted, labels=list(range(NUM_CLASSES)),
                                       target_names=list(CLASS_NAMES.values()), output_dict=True, zero_division=0)
    routing, routes = routing_metrics(scores, targets)
    report["routing"] = routing
    report["small_validation_classes"] = [name for name in CLASS_NAMES.values() if per_class[name]["support"] < 20]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_class).T.to_csv(args.output_dir / "class_report.csv")
    predictions = pd.DataFrame(scores, columns=[f"prob_class_{i}" for i in range(NUM_CLASSES)])
    predictions.insert(0, "image_path", frame["image_path"].to_numpy())
    predictions.to_csv(args.output_dir / "predictions.csv", index=False)
    routes.insert(0, "image_path", frame["image_path"].to_numpy())
    routes.to_csv(args.output_dir / "routing_decisions.csv", index=False)
    pd.concat([train.assign(split="train"), valid.assign(split="val")]).to_csv(args.output_dir / "split_manifest.csv", index=False)
    save_json(args.output_dir / "report.json", report)
    print(f"Micro-F1: {report['micro_f1']:.4f}; Macro-F1: {report['macro_f1']:.4f}")
    print(f"Routing: {routing['routed_images']}/{len(frame)} images; {routing['wrong_specialist_images']} anatomy mismatches")
    print("Report:", args.output_dir / "report.json")


if __name__ == "__main__":
    main()
