"""Evaluate single-image MURA predictions and mean-score study aggregation."""
import argparse
from pathlib import Path

from .evaluation import (
    add_runtime_arguments, binary_metrics, collect_predictions, configure_runtime,
    provenance, save_json, sha256, subset,
)
from .mura_dataset import MURA_DIR, MuraDataset, load_mura_metadata, validate_mura_patient_split
from .paths import EVALUATION_DIR
from .specialist_predict import MURA_PATH, load_mura_model


def mura_metrics(frame, scores, threshold=0.5):
    scored = frame.assign(score=scores.reshape(-1))
    studies = scored.groupby("study_id", sort=False).agg(
        label=("label", "first"), score=("score", "mean"), images=("score", "size"))
    return {
        "image": binary_metrics(scored["label"], scored["score"], threshold),
        "study_mean_score": binary_metrics(studies["label"], studies["score"], threshold),
        "study_aggregation": "Arithmetic mean of image sigmoid scores within each study.",
        "per_region_image": {region: binary_metrics(group["label"], group["score"], threshold)
                             for region, group in scored.groupby("region")},
    }, scored, studies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=MURA_PATH)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_DIR / "mura")
    parser.add_argument("--threshold", type=float, default=0.5)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    configure_runtime(args)
    train, valid = load_mura_metadata("train"), load_mura_metadata("valid")
    validate_mura_patient_split(train, valid)
    frame = subset(valid, args.max_samples)
    _, scores = collect_predictions(load_mura_model(args.checkpoint), MuraDataset(frame),
                                    args.batch_size, args.workers, args.device)
    report, scored, studies = mura_metrics(frame, scores, args.threshold)
    report.update(provenance(args.checkpoint, MURA_DIR / "valid_labeled_studies.csv",
                             "valid", args.max_samples or None))
    report["image_manifest_sha256"] = sha256(MURA_DIR / "valid_image_paths.csv")
    report["patient_overlap_train_valid"] = 0
    report["evaluation_note"] = "Validation data was used for model selection; this is not an independent test estimate."
    if args.max_samples:
        report["evaluation_note"] += " Subset study scores may omit other views and are smoke checks only."
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output_dir / "image_predictions.csv", index=False)
    studies.to_csv(args.output_dir / "study_predictions.csv")
    save_json(args.output_dir / "report.json", report)
    print(f"Image AUROC: {report['image']['auroc']}; study AUROC: {report['study_mean_score']['auroc']}")
    print("Report:", args.output_dir / "report.json")


if __name__ == "__main__":
    main()
