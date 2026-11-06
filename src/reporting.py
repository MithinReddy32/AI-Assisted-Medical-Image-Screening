"""Read complete reports only when their checkpoint fingerprint still matches."""
import json
from pathlib import Path

from .evaluation import sha256


def verified_report(path, checkpoint):
    try:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if report.get("sample_limit") is not None:
            return None
        if report["checkpoint"]["sha256"] != sha256(checkpoint):
            return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None
