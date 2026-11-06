import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from XrayScreening_V2.src.chest_dataset import NIH_DIR, resolve_image_path, validate_patient_splits
from XrayScreening_V2.src.evaluate_mura import mura_metrics
from XrayScreening_V2.src.evaluation import binary_metrics, fit_f1_thresholds, sha256
from XrayScreening_V2.src.mura_dataset import validate_mura_patient_split
from XrayScreening_V2.src.pipeline_metrics import flag_yield, routing_metrics
from XrayScreening_V2.src.reporting import verified_report
from XrayScreening_V2.src.routing import decide_routing
from XrayScreening_V2.src.specialist_predict import load_chest_thresholds


class EvaluationTests(unittest.TestCase):
    def test_legacy_image_paths_follow_the_current_project(self):
        old = "C:\\another_computer\\project\\data\\raw\\NIH\\Chest\\images_001\\images\\sample.png"
        self.assertEqual(resolve_image_path(old), NIH_DIR / "images_001/images/sample.png")

    def test_ambiguous_and_unsupported_images_are_not_routed(self):
        self.assertEqual(decide_routing([{"class_id": 3, "score": .9}, {"class_id": 9, "score": .85}])["status"], "ambiguous")
        self.assertEqual(decide_routing([{"class_id": 11, "score": .95}])["status"], "unsupported")

    def test_nonfinite_scores_cannot_bypass_routing(self):
        with self.assertRaises(ValueError):
            decide_routing([{"class_id": 3, "score": float("nan")}])

    def test_wrong_specialist_is_counted_even_when_prediction_is_confident(self):
        targets = np.zeros((2, 22)); targets[:, 3] = 1
        scores = np.zeros_like(targets); scores[0, 3] = .99; scores[1, 9] = .99
        report, _ = routing_metrics(scores, targets)
        self.assertEqual(report["routed_images"], 2)
        self.assertEqual(report["wrong_specialist_images"], 1)
        self.assertEqual(report["supported_anatomy_routing_recall"], .5)

    def test_deferred_positive_is_in_end_to_end_recall_denominator(self):
        rows = flag_yield(np.array([[1], [1], [0]]), np.array([[.9], [.9], [.8]]),
                          np.array([True, False, True]), ["Disease"], {"Disease": .5})
        self.assertEqual(rows[0]["end_to_end_flag_recall"], .5)
        self.assertEqual(rows[0]["positive_images_not_routed_to_correct_specialist"], 1)
        self.assertEqual(rows[0]["false_positive_flags"], 1)

    def test_metrics_keep_undefined_auc_and_empty_denominators_explicit(self):
        metrics = binary_metrics([0, 0], [.1, .2])
        self.assertIsNone(metrics["auroc"])
        self.assertIsNone(metrics["recall"])
        self.assertEqual(metrics["specificity"], 1)

    def test_fractional_labels_are_not_silently_rounded(self):
        with self.assertRaises(ValueError):
            binary_metrics([0, .5], [.1, .9])

    def test_study_aggregation_does_not_treat_views_as_independent_studies(self):
        frame = pd.DataFrame({"study_id": ["a", "a", "b"], "label": [0, 0, 1], "region": ["wrist"] * 3})
        report, _, _ = mura_metrics(frame, np.array([[.1], [.9], [.6]]))
        self.assertEqual(report["image"]["samples"], 3)
        self.assertEqual(report["study_mean_score"]["samples"], 2)
        self.assertEqual(report["study_mean_score"]["auroc"], 1)

    def test_calibration_requires_both_labels(self):
        with self.assertRaises(ValueError):
            fit_f1_thresholds(np.zeros((2, 1)), np.array([[.2], [.3]]), ["Disease"])

    def test_chest_split_rejects_same_patient_in_train_and_test(self):
        frame = pd.DataFrame({"Image Index": ["a", "b", "c"], "Finding Labels": ["No Finding"] * 3,
                              "Patient ID": [1, 2, 1], "image_path": ["a", "b", "c"],
                              "split": ["train", "val", "test"]})
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            validate_patient_splits(frame)

    def test_mura_split_rejects_patient_overlap(self):
        with self.assertRaises(ValueError):
            validate_mura_patient_split(pd.DataFrame({"patient_id": ["p"]}), pd.DataFrame({"patient_id": ["p"]}))

    def test_missing_thresholds_are_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.csv"
            path.write_text("Finding,Threshold\nAtelectasis,0.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing findings"):
                load_chest_thresholds(path)

    def test_nonfinite_and_duplicate_thresholds_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "thresholds.csv"
            for lines in ["Atelectasis,nan", "Atelectasis,0.5\nAtelectasis,0.6"]:
                path.write_text("Finding,Threshold\n" + lines + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_chest_thresholds(path)

    def test_sidebar_rejects_stale_or_subset_results(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, path = Path(directory) / "model.pth", Path(directory) / "report.json"
            checkpoint.write_bytes(b"model1")
            report = {"checkpoint": {"sha256": sha256(checkpoint)}, "sample_limit": None}
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertIsNotNone(verified_report(path, checkpoint))
            checkpoint.write_bytes(b"model2")
            self.assertIsNone(verified_report(path, checkpoint))
            report["checkpoint"]["sha256"] = sha256(checkpoint); report["sample_limit"] = 8
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertIsNone(verified_report(path, checkpoint))


if __name__ == "__main__":
    unittest.main()
