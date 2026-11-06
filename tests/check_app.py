"""Headless startup check; execute from any working directory."""
from pathlib import Path
from io import BytesIO
import sys
import json
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.screening_status import non_chest_status as current_screening_status


def legacy_screening_status(*args, **kwargs):
    # Reproduce an already imported helper from before the graph was added.
    result = current_screening_status(*args, **kwargs)
    return None if result is None else {key: result[key] for key in ("title", "level", "message")}


def check_non_chest_results():
    image = BytesIO()
    Image.new("RGB", (32, 32)).save(image, format="PNG")
    image_bytes = image.getvalue()
    cases = [
        ("flagged", 21, .99, .8, False, "Possible abnormality — consult a doctor"),
        ("below_threshold", 9, .99, .2, False, "No abnormality flagged"),
        ("unsupported", 11, .99, .2, False, "Unable to assess — consult a doctor"),
        ("deferred", 21, .6, .2, False, "Unable to assess — consult a doctor"),
        ("model_error", 21, .99, .2, True, "Unable to assess — consult a doctor"),
        ("invalid_score", 21, .99, float("nan"), False, "Unable to assess — consult a doctor"),
        ("chest", 3, .99, .2, False, None),
    ]
    for name, class_id, confidence, score, fails, expected in cases:
        predictions = sorted([
            {"class_id": i, "score": confidence if i == class_id else .001}
            for i in range(22)
        ], key=lambda r: r["score"], reverse=True)
        chest_results = [{"finding": "Cardiomegaly", "score": .8, "threshold": .5, "flagged": True}]
        with (patch("streamlit.file_uploader", side_effect=lambda *a, **kw: BytesIO(image_bytes)),
              patch("src.predict.load_model", return_value=object()),
              patch("src.predict.predict_image", return_value=([predictions[0]], predictions)),
              patch("src.specialist_predict.load_mura_model", return_value=object()),
              patch("src.specialist_predict.predict_mura", return_value=score,
                    side_effect=RuntimeError("Test inference failure") if fails else None) as predict,
              patch("src.specialist_predict.load_chest_model", return_value=object()),
              patch("src.specialist_predict.load_chest_thresholds", return_value={}),
              patch("src.specialist_predict.predict_chest", return_value=chest_results),
              patch("src.screening_status.non_chest_status", side_effect=legacy_screening_status)):
            app = AppTest.from_file(str(PROJECT_DIR / "app.py"), default_timeout=90).run()
            assert not app.exception, (name, [error.value for error in app.exception])
            if not fails:
                assert not app.error, (name, [error.value for error in app.error])
            if expected:
                banners = [element.value for element in list(app.warning) + list(app.info)]
                assert expected in banners, (name, banners)
                assert any(element.value == f"Screening result: {expected}" for element in app.markdown), name
            else:
                assert "Screening Result" not in [element.value for element in app.subheader]
                assert "Top 3 Chest Finding Scores" in [element.value for element in app.subheader]
            if name in {"unsupported", "deferred", "chest"}:
                predict.assert_not_called()
            if name == "below_threshold":
                assert "Model score details" in [element.label for element in app.expander]
            chart_specs = [json.loads(chart.proto.spec) for chart in app.get("vega_lite_chart")]
            abnormality_charts = [spec for spec in chart_specs
                                  if spec.get("usermeta", {}).get("chart") == "non_chest_abnormality"]
            assert len(abnormality_charts) == (1 if name in {"flagged", "below_threshold"} else 0), name
            if name == "below_threshold":
                # Moving the cutoff must update the graph and the screening decision together.
                app.sidebar.slider[1].set_value(.1).run()
                assert not app.exception
                assert any(element.value == "Possible abnormality — consult a doctor" for element in app.warning)
                updated_specs = [json.loads(chart.proto.spec) for chart in app.get("vega_lite_chart")]
                updated = next(spec for spec in updated_specs
                               if spec.get("usermeta", {}).get("chart") == "non_chest_abnormality")
                assert "threshold 10%" in updated["description"]
        print(f"Screening display passed: {name}")


if __name__ == "__main__":
    app_path = PROJECT_DIR / "app.py"
    app = AppTest.from_file(str(app_path), default_timeout=90).run()
    assert not app.exception, [error.value for error in app.exception]
    assert not app.error, [error.value for error in app.error]
    assert len(app.title) == 1
    assert len(app.sidebar.radio) == 1
    app.sidebar.radio[0].set_value("Manual Testing").run()
    assert not app.exception
    assert len(app.sidebar.selectbox) == 1
    print("App startup and manual routing controls passed.")
    check_non_chest_results()
