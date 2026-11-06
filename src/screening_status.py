"""Plain-language non-chest results without implying a medical all-clear."""
import math

from .routing import UPPER_LIMB_CLASSES


def non_chest_status(body_part, routing, score=None, threshold=0.5):
    if body_part == "Chest":
        return None

    unavailable = {
        "title": "Unable to assess — consult a doctor",
        "level": "warning",
        "message": "Have a doctor interpret the X-ray.",
    }
    if body_part not in UPPER_LIMB_CLASSES:
        return {**unavailable,
                "message": f"Abnormality screening is not available for {body_part}. Have a doctor interpret the X-ray."}
    if routing["status"] != "ready" or routing["specialist"] != "mura":
        return {**unavailable,
                "message": "The body-part prediction was not reliable enough to run abnormality screening. Have a doctor interpret the X-ray."}
    if (score is None or not math.isfinite(score) or not 0 <= score <= 1
            or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        return {**unavailable,
                "message": "An abnormality result could not be obtained. Have a doctor interpret the X-ray."}
    if score >= threshold:
        return {
            "title": "Possible abnormality — consult a doctor",
            "level": "warning",
            "message": "The model flagged a possible abnormality. A doctor needs to review the X-ray to determine what it means.",
        }
    return {
        "title": "No abnormality flagged",
        "level": "info",
        "message": "This experimental result cannot confirm that the image is normal. Have a doctor interpret the X-ray.",
    }
