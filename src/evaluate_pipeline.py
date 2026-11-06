"""Replay routing or evaluate routing and specialist flag yield on labeled cohorts."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .chest_dataset import FINDINGS, SPLIT_PATH, ChestDataset, prepare_metadata
from .class_names import CLASS_NAMES
from .evaluate_chest import threshold_provenance
from .evaluation import (
    add_runtime_arguments, collect_predictions, configure_runtime, finding_metrics,
    provenance, save_json, sha256, subset,
)
from .mura_dataset import MURA_DIR, MuraDataset, load_mura_metadata, validate_mura_patient_split
from .paths import EVALUATION_DIR, OUTPUT_DIR
from .pipeline_metrics import flag_yield, routing_metrics
from .predict import MODEL_PATH, load_model
from .routing import MIN_GAP, MIN_SCORE, decide_routing
from .specialist_predict import (
    CHEST_PATH, MURA_PATH, THRESHOLDS_PATH, load_chest_model, load_chest_thresholds, load_mura_model,
)
from .train import CSV_PATH, load_metadata, parse_target, split_metadata


def replay(predictions_path):
    predictions = pd.read_csv(predictions_path)
    _, valid = split_metadata(load_metadata(check_images=False))
    normalize = lambda s: s.str.replace("\\", "/", regex=False).str.removeprefix("./")
    valid = valid.assign(key=normalize(valid["image_path"])).set_index("key")
    keys = normalize(predictions["image_path"])
    if keys.duplicated().any() or not keys.isin(valid.index).all():
        raise ValueError("Saved predictions must match unique images in the original validation split.")
    valid = valid.loc[keys]
    scores = predictions[[f"prob_class_{i}" for i in CLASS_NAMES]].to_numpy()
    targets = np.zeros_like(scores)
    for row, target in enumerate(valid["Target"]):
        targets[row, parse_target(target)] = 1
    report, routes = routing_metrics(scores, targets)
    report.update({"prediction_file": str(predictions_path.resolve()),
                   "prediction_sha256": sha256(predictions_path), "labels_sha256": sha256(CSV_PATH),
                   "split": "val", "prediction_checkpoint_verified": False,
                   "note": "Replay of supplied scores; checkpoint identity cannot be established from this legacy CSV. Disease outcomes are not available in UNIFESP."})
    routes.insert(0, "image_path", predictions["image_path"])
    return report, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=["routing", "chest", "mura"])
    parser.add_argument("--predictions", type=Path, default=OUTPUT_DIR / "bodypart_validation_predictions.csv")
    parser.add_argument("--bodypart-checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--specialist-checkpoint", type=Path, default=None)
    parser.add_argument("--thresholds", type=Path, default=THRESHOLDS_PATH)
    parser.add_argument("--mura-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=None)
    add_runtime_arguments(parser)
    args = parser.parse_args()
    configure_runtime(args)
    output = args.output_dir or EVALUATION_DIR / f"pipeline_{args.cohort}"
    output.mkdir(parents=True, exist_ok=True)
    if args.cohort == "routing":
        if args.max_samples:
            parser.error("Routing replay evaluates all supplied scores; subset the input CSV explicitly.")
        report, routes = replay(args.predictions)
    else:
        if args.cohort == "chest":
            frame = subset(prepare_metadata().query("split == 'test'"), args.max_samples)
            dataset, manifest, split = ChestDataset(frame), SPLIT_PATH, "test"
            names, thresholds = FINDINGS, load_chest_thresholds(args.thresholds)
            checkpoint = args.specialist_checkpoint or CHEST_PATH
            specialist = load_chest_model(checkpoint)
            calibration = threshold_provenance(args.thresholds, checkpoint)
            identifiers = frame["Image Index"].to_numpy()
        else:
            train, valid = load_mura_metadata("train"), load_mura_metadata("valid")
            validate_mura_patient_split(train, valid)
            frame = subset(valid, args.max_samples)
            dataset = MuraDataset(frame)
            manifest, split = MURA_DIR / "valid_labeled_studies.csv", "valid"
            names, thresholds = ["Abnormality"], {"Abnormality": args.mura_threshold}
            checkpoint = args.specialist_checkpoint or MURA_PATH
            specialist = load_mura_model(checkpoint)
            identifiers = frame["image_path"].to_numpy()
            calibration = {"status": "fixed_experimental_threshold", "threshold": args.mura_threshold}
        targets, body_scores = collect_predictions(load_model(args.bodypart_checkpoint), dataset,
                                                   args.batch_size, args.workers, args.device)
        _, scores = collect_predictions(specialist, dataset, args.batch_size, args.workers, args.device)
        decisions = [decide_routing([{"class_id": i, "score": float(p)} for i, p in enumerate(row)])
                     for row in body_scores]
        eligible = np.array([d["specialist"] == args.cohort for d in decisions])
        report = provenance(checkpoint, manifest, split, args.max_samples or None)
        report.update({"bodypart_checkpoint": {"path": str(args.bodypart_checkpoint.resolve()),
                                               "sha256": sha256(args.bodypart_checkpoint)},
                       "images": len(frame), "routing_min_score": MIN_SCORE, "routing_min_gap": MIN_GAP,
                       "correct_specialist_images": int(eligible.sum()),
                       "correct_specialist_coverage": float(eligible.mean()),
                       "wrong_specialist_images": sum(d["status"] == "ready" and d["specialist"] != args.cohort for d in decisions),
                       "deferred_images": sum(d["status"] != "ready" for d in decisions),
                       "oracle_specialist_all_images": finding_metrics(targets, scores, names, thresholds),
                       "correctly_routed_subset": finding_metrics(targets[eligible], scores[eligible], names, thresholds),
                       "end_to_end_flag_yield": flag_yield(targets, scores, eligible, names, thresholds),
                       "threshold_provenance": calibration,
                       "interpretation": "No correct-specialist decision is made on deferred or misrouted images. Flag recall includes these positive cases as unflagged; abstentions are never reported as normal. Oracle inference runs the correct specialist on every image for comparison. Wrong-specialist disease outputs cannot be evaluated with this cohort's labels."})
        if args.cohort == "mura":
            report["evaluation_note"] = "Image-level metrics use inherited study labels; validation was used for model selection. This is not an independent test estimate."
            report["image_manifest_sha256"] = sha256(MURA_DIR / "valid_image_paths.csv")
        else:
            report["thresholds_sha256"] = sha256(args.thresholds)
        routes = pd.DataFrame({"image_id": identifiers,
                               "status": [d["status"] for d in decisions],
                               "specialist": [d["specialist"] for d in decisions],
                               "correct_specialist": eligible})
        for i, name in enumerate(names):
            routes[f"target_{name}"] = targets[:, i].astype(int)
            routes[f"oracle_score_{name}"] = scores[:, i]
            routes[f"pipeline_flag_{name}"] = eligible & (scores[:, i] >= thresholds[name])
    routes.to_csv(output / "decisions.csv", index=False)
    save_json(output / "report.json", report)
    print("Report:", output / "report.json")


if __name__ == "__main__":
    main()
