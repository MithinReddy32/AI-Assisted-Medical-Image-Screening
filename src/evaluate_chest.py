"""Calibrate on validation data, then evaluate frozen thresholds on test data."""
import argparse
import json
from pathlib import Path

from .chest_dataset import FINDINGS, SPLIT_PATH, ChestDataset, prepare_metadata
from .evaluation import (
    add_runtime_arguments, collect_predictions, configure_runtime, finding_metrics,
    fit_f1_thresholds, provenance, save_json, sha256, subset,
)
from .paths import EVALUATION_DIR
from .specialist_predict import CHEST_PATH, THRESHOLDS_PATH, load_chest_model, load_chest_thresholds


def threshold_provenance(path, checkpoint):
    sidecar = Path(str(path) + ".metadata.json")
    if not sidecar.exists():
        return {"status": "legacy_unverified", "note": "Threshold calibration provenance was not saved with this file."}
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if meta["checkpoint"]["sha256"] != sha256(checkpoint) or meta["thresholds_sha256"] != sha256(path):
        raise ValueError("Threshold provenance does not match the checkpoint or threshold file.")
    if meta["split"] != "val" or meta["dataset_manifest"]["sha256"] != sha256(SPLIT_PATH):
        raise ValueError("Thresholds must be calibrated on this manifest's validation split.")
    if meta.get("sample_limit"):
        raise ValueError("Subset-calibrated thresholds cannot be used for a full evaluation.")
    return {"status": "verified_validation_calibration", "metadata": meta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "evaluate"])
    parser.add_argument("--checkpoint", type=Path, default=CHEST_PATH)
    parser.add_argument("--thresholds", type=Path, default=None,
                        help="Calibration output or evaluation input. Calibration defaults to a separate file.")
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_DIR / "chest")
    add_runtime_arguments(parser)
    args = parser.parse_args()
    configure_runtime(args)
    if args.mode == "calibrate" and args.max_samples:
        parser.error("Threshold calibration requires the complete validation split; omit --max-samples.")
    split = "val" if args.mode == "calibrate" else "test"
    frame = subset(prepare_metadata().query("split == @split"), args.max_samples)
    targets, scores = collect_predictions(load_chest_model(args.checkpoint), ChestDataset(frame),
                                          args.batch_size, args.workers, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    meta = provenance(args.checkpoint, SPLIT_PATH, split, args.max_samples or None)
    if args.mode == "calibrate":
        path = args.thresholds or args.output_dir / "calibrated_thresholds.csv"
        table = fit_f1_thresholds(targets, scores, FINDINGS)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        meta.update({"thresholds_sha256": sha256(path), "criterion": "maximum validation F1; lowest threshold breaks ties"})
        save_json(str(path) + ".metadata.json", meta)
        print("Validation-calibrated thresholds:", path)
    else:
        path = args.thresholds or THRESHOLDS_PATH
        thresholds = load_chest_thresholds(path)
        calibration = threshold_provenance(path, args.checkpoint)
        report = {**meta, **finding_metrics(targets, scores, FINDINGS, thresholds),
                  "images": len(frame), "patients": int(frame["Patient ID"].nunique()),
                  "thresholds": {"path": str(path.resolve()), "sha256": sha256(path), "calibration": calibration},
                  "patient_overlap_between_splits": 0}
        save_json(args.output_dir / "report.json", report)
        print("Test macro-AUROC:", report["macro_auroc"])
        print("Report:", args.output_dir / "report.json")
    predictions = frame[["Image Index", "Patient ID"]].reset_index(drop=True).copy()
    for i, finding in enumerate(FINDINGS):
        predictions[f"target_{finding}"] = targets[:, i].astype(int)
        predictions[f"score_{finding}"] = scores[:, i]
    predictions.to_csv(args.output_dir / f"{split}_predictions.csv", index=False)


if __name__ == "__main__":
    main()
