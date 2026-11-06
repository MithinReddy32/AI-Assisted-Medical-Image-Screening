"""Shared automatic routing policy for the UI and offline evaluation."""
import math

from .class_names import CLASS_NAMES

MIN_SCORE = 0.80
MIN_GAP = 0.20
CHEST_CLASSES = {"Chest"}
UPPER_LIMB_CLASSES = {"Shoulder", "Elbow", "Forearm", "Wrist", "Hand", "Finger"}


def specialist_for_class(class_id):
    name = CLASS_NAMES.get(class_id)
    if name in CHEST_CLASSES:
        return "chest"
    if name in UPPER_LIMB_CLASSES:
        return "mura"
    return None


def decide_routing(all_results, min_score=MIN_SCORE, min_gap=MIN_GAP):
    if not 0 <= min_score <= 1 or not 0 <= min_gap <= 1:
        raise ValueError("Routing cutoffs must be between 0 and 1.")
    if not all_results:
        return {"status": "unavailable", "specialist": None,
                "reason": "No predictions were returned."}
    if any(r["class_id"] not in CLASS_NAMES or not math.isfinite(r["score"])
           or not 0 <= r["score"] <= 1 for r in all_results):
        raise ValueError("Invalid body-part class or score.")
    ranked = sorted(all_results, key=lambda r: r["score"], reverse=True)
    top = ranked[0]
    gap = top["score"] - (ranked[1]["score"] if len(ranked) > 1 else 0.0)
    if top["score"] < min_score:
        return {"status": "uncertain", "specialist": None,
                "reason": f"Top score {top['score']:.2%} is below the routing minimum of {min_score:.0%}."}
    if gap < min_gap:
        return {"status": "ambiguous", "specialist": None,
                "reason": f"Top two scores differ by only {gap * 100:.2f} percentage points."}
    specialist = specialist_for_class(top["class_id"])
    if specialist:
        return {"status": "ready", "specialist": specialist,
                "reason": f"{'Chest' if specialist == 'chest' else 'Upper-limb'} routing criteria satisfied."}
    return {"status": "unsupported", "specialist": None,
            "reason": f"No specialist model is available for {CLASS_NAMES[top['class_id']]}."}
