"""Create a human-readable summary of complete, checkpoint-matched reports."""
from .paths import EVALUATION_DIR
from .predict import MODEL_PATH
from .reporting import verified_report
from .specialist_predict import CHEST_PATH, MURA_PATH


def main():
    body = verified_report(EVALUATION_DIR / "bodypart/report.json", MODEL_PATH)
    chest = verified_report(EVALUATION_DIR / "chest/report.json", CHEST_PATH)
    mura = verified_report(EVALUATION_DIR / "mura/report.json", MURA_PATH)
    lines = ["# Evaluation results", "", "Only complete reports matching the current checkpoint SHA-256 are included.", ""]
    if body:
        routing = body["routing"]
        lines += ["## Body-part validation", "",
                  f"- Images: {body['images']}; patients: {body['patients']}.",
                  f"- Micro-F1: {body['micro_f1']:.4f}; Macro-F1: {body['macro_f1']:.4f}.",
                  f"- Specialist routing: {routing['routed_images']}/{body['images']} images.",
                  f"- Incorrect specialist anatomy: {routing['wrong_specialist_images']}.",
                  f"- Supported-anatomy routing recall: {routing['supported_anatomy_routing_recall']:.2%}.",
                  "- This validation split was used for checkpoint selection.",
                  "- Small class samples limit conclusions; see [per-class results](bodypart/class_report.csv).", ""]
    else:
        lines += ["Body-part evaluation: no complete matching report.", ""]
    if chest:
        lines += ["## Chest test", "", f"- Images: {chest['images']}.",
                  f"- Macro-AUROC: {chest['macro_auroc']:.4f}.",
                  f"- Threshold provenance: {chest['thresholds']['calibration']['status']}.",
                  "- Per-finding sensitivity, specificity, precision, F1, and positive counts are in [the report](chest/report.json).", ""]
    else:
        lines += ["Chest evaluation: no complete matching report. Subset smoke results do not establish test performance.", ""]
    if mura:
        lines += ["## MURA validation", "", f"- Images: {mura['image']['samples']}; studies: {mura['study_mean_score']['samples']}.",
                  f"- Image AUROC: {mura['image']['auroc']:.4f}; image F1: {mura['image']['f1']:.4f}.",
                  f"- Study AUROC (mean image scores): {mura['study_mean_score']['auroc']:.4f}.",
                  "- Image-level labels are inherited from studies; the app accepts one image.",
                  "- Validation was used for checkpoint selection, so these are not independent test estimates.", ""]
    else:
        lines += ["MURA evaluation: no complete matching report.", ""]
    lines += ["## Scope", "", "These datasets evaluate separate stages. UNIFESP contains no specialist disease outcomes. Pipeline reports on NIH/MURA measure routing coverage and disease-flag recall on those respective cohorts. Deferred and misrouted images remain explicit; they are not normal results. External clinical validation is still outstanding.", ""]
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALUATION_DIR / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
