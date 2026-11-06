"""Routing coverage and disease-flag yield; abstentions remain explicit."""
from collections import Counter

import numpy as np
import pandas as pd

from .class_names import CLASS_NAMES
from .routing import MIN_GAP, MIN_SCORE, decide_routing, specialist_for_class


def routing_metrics(scores, targets, min_score=MIN_SCORE, min_gap=MIN_GAP):
    scores, targets = np.asarray(scores), np.asarray(targets)
    if scores.ndim != 2 or scores.shape != targets.shape or scores.shape[1] != len(CLASS_NAMES) or not len(scores):
        raise ValueError("Routing evaluation needs nonempty, matching 22-class score and target matrices.")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Routing targets must be binary.")
    rows = []
    for probabilities, truth in zip(scores, targets):
        results = [{"class_id": i, "score": float(p)} for i, p in enumerate(probabilities)]
        route = decide_routing(results, min_score, min_gap)
        top = int(np.argmax(probabilities))
        true_ids = np.flatnonzero(truth).tolist()
        supported = {specialist_for_class(i) for i in true_ids} - {None}
        rows.append({"predicted_body_part": CLASS_NAMES[top], "top_score": float(probabilities[top]),
                     "true_class_ids": " ".join(map(str, true_ids)), "status": route["status"],
                     "specialist": route["specialist"], "top1_correct": bool(truth[top]),
                     "supported_anatomy": bool(supported),
                     "correct_specialist": bool(route["specialist"] in supported)})
    frame = pd.DataFrame(rows)
    routed = frame["status"].eq("ready")
    supported = frame["supported_anatomy"]
    correct = frame["correct_specialist"]
    n, nr = len(frame), int(routed.sum())
    report = {
        "images": n, "min_score": min_score, "min_gap": min_gap,
        "status_counts": dict(Counter(frame["status"])),
        "specialist_counts": dict(Counter(frame.loc[routed, "specialist"])),
        "top1_accuracy": float(frame["top1_correct"].mean()),
        "routed_images": nr, "routing_coverage_all_images": nr / n,
        "supported_anatomy_images": int(supported.sum()),
        "correctly_routed_supported_images": int(correct.sum()),
        "supported_anatomy_routing_recall": float(correct.sum() / supported.sum()) if supported.any() else None,
        "wrong_specialist_images": int((routed & ~correct).sum()),
        "routed_anatomy_precision": float(correct.sum() / nr) if nr else None,
        "per_true_class": {},
    }
    for i, name in CLASS_NAMES.items():
        mask = targets[:, i].astype(bool)
        report["per_true_class"][name] = {
            "images": int(mask.sum()), "routed": int(routed[mask].sum()),
            "correct_specialist": int(correct[mask].sum()),
            "top1_correct": int(frame.loc[mask, "top1_correct"].sum()),
        }
    report["interpretation"] = "Routing accuracy measures anatomy only. UNIFESP provides no specialist disease labels. Per-class counts overlap for multi-label images."
    return report, frame


def flag_yield(targets, scores, eligible, names, thresholds):
    """All positive cases form the denominator, including deferred cases."""
    rows = []
    for i, name in enumerate(names):
        positives = targets[:, i].astype(bool)
        flagged = eligible & (scores[:, i] >= thresholds[name])
        tp = int((flagged & positives).sum())
        count = int(positives.sum())
        rows.append({"finding": name, "positive_images": count,
                     "positive_images_flagged": tp,
                     "positive_images_not_routed_to_correct_specialist": int((positives & ~eligible).sum()),
                     "positive_images_routed_but_below_threshold": int((positives & eligible & ~flagged).sum()),
                     "false_positive_flags": int((flagged & ~positives).sum()),
                     "end_to_end_flag_recall": tp / count if count else None})
    return rows
