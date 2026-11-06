
import math

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError
from src.paths import ensure_models

ensure_models()
from src.predict import load_model, predict_image
from src.class_names import CLASS_NAMES
from src.paths import EVALUATION_DIR
from src.predict import MODEL_PATH
from src.specialist_predict import CHEST_PATH, MURA_PATH
from src.reporting import verified_report
from src.routing import decide_routing, MIN_SCORE, MIN_GAP
from src.screening_status import non_chest_status
from src.abnormality_chart import build_abnormality_chart
from src.specialist_predict import (
    load_chest_model,
    load_mura_model,
    load_chest_thresholds,
    predict_chest,
    predict_mura,
)


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Multi-Body-Part X-ray Screening",
    page_icon="🩻",
    layout="wide",
)

MANUAL_OPTIONS = ["Chest", "Shoulder", "Elbow", "Forearm", "Wrist", "Hand", "Finger"]


@st.cache_data
def get_evaluation_reports(fingerprints):
    # Cache keys include file mtimes so a newly evaluated model refreshes the UI.
    return {
        name: verified_report(EVALUATION_DIR / name / "report.json", checkpoint)
        for name, checkpoint in [("bodypart", MODEL_PATH), ("chest", CHEST_PATH), ("mura", MURA_PATH)]
    }


@st.cache_resource
def get_bodypart_model():
    return load_model()


@st.cache_resource
def get_chest_model():
    return load_chest_model()


@st.cache_resource
def get_mura_model():
    return load_mura_model()


@st.cache_data
def get_chest_thresholds():
    return load_chest_thresholds()


# =====================================================
# HEADER
# =====================================================

st.title("🩻 Multi-Body-Part X-ray Screening")

st.markdown(
    "### Deep Learning-Based X-ray Classification and Screening"
)

st.info(
    "Stage 1: Body-part classification. "
    "Stage 2: Specialist model inference."
)

st.warning(
    "Research prototype only. Model outputs are experimental "
    "scores, not clinical diagnoses. Do not use this application "
    "for medical decisions."
)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Settings")

    threshold = st.slider(
        "Body-part display threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    mura_threshold = st.slider(
        "MURA abnormality threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    st.divider()

    st.subheader("Routing Mode")

    routing_mode = st.radio(
        "Choose routing method",
        ["Automatic", "Manual Testing"],
    )

    manual_body_part = None

    if routing_mode == "Manual Testing":

        manual_body_part = st.selectbox(
            "Select X-ray body part",
            MANUAL_OPTIONS,
        )

        st.warning(
            "Manual Testing bypasses automatic routing. "
            "Select the actual anatomical region shown "
            "in the uploaded image."
        )

    st.divider()

    st.subheader("Automatic Routing Criteria")

    st.write(f"Minimum top score: {MIN_SCORE:.0%}")
    st.write(f"Minimum score gap: {MIN_GAP * 100:.0f} percentage points")

    st.caption(
        "These are experimental routing thresholds, "
        "not clinically validated cutoffs."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("Body Part: UNIFESP ResNet18")
    st.write("Chest: NIH ChestX-ray14 ResNet18")
    st.write("Upper Limb: MURA ResNet18")

    report_files = [EVALUATION_DIR / name / "report.json" for name in ["bodypart", "chest", "mura"]]
    model_files = [MODEL_PATH, CHEST_PATH, MURA_PATH]
    fingerprints = tuple((str(p), p.stat().st_mtime_ns if p.exists() else None)
                         for p in report_files + model_files)
    reports = get_evaluation_reports(fingerprints)
    body_report, chest_report, mura_report = (reports[name] for name in ["bodypart", "chest", "mura"])
    if body_report:
        st.caption(f"Body-part validation Micro-F1: {body_report['micro_f1']:.2%} ({body_report['images']} images)")
    if chest_report and chest_report["macro_auroc"] is not None:
        st.caption(f"Chest test Macro-AUROC: {chest_report['macro_auroc']:.4f} ({chest_report['images']} images)")
    if mura_report and mura_report["image"]["auroc"] is not None:
        st.caption(f"MURA validation Image-AUROC: {mura_report['image']['auroc']:.4f} ({mura_report['image']['samples']} images)")
    if not all(reports.values()):
        st.caption("Complete evaluation results are unavailable for some models.")
    st.caption("These evaluate individual stages; they do not measure overall screening performance.")


# =====================================================
# LOAD BODY-PART MODEL
# =====================================================

try:
    body_model = get_bodypart_model()

except Exception as error:
    st.error(
        f"Unable to load body-part model: {error}"
    )
    st.stop()


# =====================================================
# UPLOAD IMAGE
# =====================================================

uploaded_file = st.file_uploader(
    "Upload an X-ray image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is None:
    st.info("Upload an X-ray image to begin.")
    st.stop()

try:
    image = Image.open(uploaded_file)
    image.load()
    image = image.convert("RGB")

except (UnidentifiedImageError, OSError, ValueError):
    st.error("The uploaded file is not a valid image.")
    st.stop()


# =====================================================
# MAIN LAYOUT
# =====================================================

left, right = st.columns([1, 1], gap="large")

with left:

    st.subheader("Uploaded X-ray")

    st.image(
        image,
        use_container_width=True,
    )


with right:

    # -------------------------------------------------
    # BODY-PART CLASSIFICATION
    # -------------------------------------------------

    st.subheader("Body-Part Classification")

    try:
        selected, all_results = predict_image(
            body_model,
            image,
            threshold=threshold,
        )

    except Exception as error:
        st.error(
            f"Body-part prediction failed: {error}"
        )
        st.stop()

    if not all_results:
        st.error("No body-part predictions returned.")
        st.stop()

    best_prediction = all_results[0]

    body_part = CLASS_NAMES.get(
        best_prediction["class_id"],
        f"Unknown Class {best_prediction['class_id']}",
    )

    confidence = best_prediction["score"]

    st.success(
        f"Top predicted body part: {body_part}"
    )

    st.metric(
        "Top Model Score",
        f"{confidence * 100:.2f}%",
    )

    if not selected:
        st.warning(
            "No body-part score exceeded the display threshold."
        )

    elif len(selected) > 1:
        st.info(
            "Multiple body-part scores exceeded the display "
            "threshold. Automatic routing uses the top-two "
            "score difference."
        )

    st.subheader("Body-Part Predictions")

    if selected:

        for result in selected:

            name = CLASS_NAMES.get(
                result["class_id"],
                f"Class {result['class_id']}",
            )

            st.write(
                f"{name}: {result['score'] * 100:.2f}%"
            )

    else:
        st.write("No predictions above the display threshold.")

    st.divider()

    # -------------------------------------------------
    # ROUTING DECISION
    # -------------------------------------------------

    st.subheader("Specialist Routing")

    if routing_mode == "Manual Testing":

        specialist = (
            "chest"
            if manual_body_part == "Chest"
            else "mura"
        )

        routing = {
            "status": "ready",
            "specialist": specialist,
            "reason": (
                f"Manual testing override: "
                f"{manual_body_part} selected by user."
            ),
        }

    else:
        routing = decide_routing(all_results)

    st.write(
        f"Routing mode: **{routing_mode}**"
    )

    st.write(
        f"Routing decision: **{routing['status'].title()}**"
    )

    st.caption(routing["reason"])

    specialist_status = "Not run"
    specialist_summary = None
    mura_score = None
    non_chest_result = None

    # -------------------------------------------------
    # NO ROUTING
    # -------------------------------------------------

    if routing["status"] == "unsupported":

        specialist_status = "Unsupported anatomy"

        st.info(
            f"No specialist model is available for {body_part}."
        )

    elif routing["status"] != "ready":

        specialist_status = (
            f"Routing deferred: {routing['status']}"
        )

        st.warning(
            "Specialist inference was deferred because "
            "the body-part prediction did not satisfy "
            "the automatic routing criteria."
        )

    # -------------------------------------------------
    # CHEST SPECIALIST
    # -------------------------------------------------

    elif routing["specialist"] == "chest":

        st.info("Routing to NIH ChestX-ray14 model")

        try:

            chest_model = get_chest_model()
            chest_thresholds = get_chest_thresholds()

            chest_results = predict_chest(
                chest_model,
                image,
                chest_thresholds,
            )

            flagged = [
                result
                for result in chest_results
                if result["flagged"]
            ]

            specialist_status = "Chest model executed"

            if flagged:

                st.write(
                    "**Findings above configured thresholds:**"
                )

                for result in flagged:

                    st.write(
                        f"{result['finding']}: "
                        f"{result['score'] * 100:.2f}%"
                    )

            else:

                st.info(
                    "No chest finding exceeded its configured "
                    "threshold. This does not rule out disease."
                )

            st.subheader("Top 3 Chest Finding Scores")

            for result in chest_results[:3]:

                st.write(
                    f"{result['finding']}: "
                    f"{result['score'] * 100:.2f}% "
                    f"(threshold: "
                    f"{result['threshold'] * 100:.2f}%)"
                )

            specialist_summary = pd.DataFrame([
                {
                    "Finding": result["finding"],
                    "Model Score (%)": round(
                        result["score"] * 100, 2
                    ),
                    "Threshold (%)": round(
                        result["threshold"] * 100, 2
                    ),
                    "Flagged": result["flagged"],
                }
                for result in chest_results
            ])

        except Exception as error:

            specialist_status = "Chest model error"

            st.error(
                f"Chest model could not run: {error}"
            )

    # -------------------------------------------------
    # MURA SPECIALIST
    # -------------------------------------------------

    elif routing["specialist"] == "mura":

        st.info("Routing to MURA upper-limb model")

        try:

            mura_model = get_mura_model()

            mura_score = predict_mura(
                mura_model,
                image,
            )

            specialist_status = "MURA model executed"

        except Exception as error:

            specialist_status = "MURA model error"

            st.error(
                f"MURA model could not run: {error}"
            )

    screening_body_part = (
        manual_body_part if routing_mode == "Manual Testing" else body_part
    )
    non_chest_result = non_chest_status(
        screening_body_part, routing, mura_score, mura_threshold
    )
    if non_chest_result is not None:
        st.subheader("Screening Result")
        if non_chest_result["level"] == "warning":
            st.warning(non_chest_result["title"])
        else:
            st.info(non_chest_result["title"])
        st.caption(non_chest_result["message"])

        # Derive chart availability from the completed inference. A long-running
        # Streamlit session may still hold the older three-field status helper.
        has_abnormality_score = (
            routing["status"] == "ready"
            and routing["specialist"] == "mura"
            and specialist_status == "MURA model executed"
            and mura_score is not None
            and math.isfinite(mura_score)
            and 0 <= mura_score <= 1
            and math.isfinite(mura_threshold)
            and 0 <= mura_threshold <= 1
        )
        if has_abnormality_score:
            st.subheader("Abnormality Score")
            st.altair_chart(
                build_abnormality_chart(mura_score, mura_threshold, screening_body_part),
                width="stretch",
            )
            st.caption("Dashed line: screening threshold. This is an experimental model score.")

            with st.expander("Model score details"):
                st.metric("Abnormality Model Score", f"{mura_score * 100:.2f}%")
                st.caption(f"Configured abnormality threshold: {mura_threshold:.0%}")
                st.caption(
                    "MURA was trained with study-level labels. "
                    "This single-image score is exploratory."
                )


# =====================================================
# TOP 5 BODY-PART PREDICTIONS
# =====================================================

st.divider()

st.subheader("Top 5 Body-Part Predictions")

top_five = pd.DataFrame([
    {
        "Body Part": CLASS_NAMES.get(
            result["class_id"],
            f"Class {result['class_id']}",
        ),
        "Model Score (%)": round(
            result["score"] * 100, 2
        ),
    }
    for result in all_results[:5]
])

st.dataframe(
    top_five,
    hide_index=True,
    use_container_width=True,
)

st.bar_chart(
    top_five.set_index("Body Part")["Model Score (%)"]
)


# =====================================================
# CHEST RESULTS TABLE
# =====================================================

if specialist_summary is not None:

    st.divider()

    st.subheader("All 14 Chest Finding Scores")

    st.dataframe(
        specialist_summary,
        hide_index=True,
        use_container_width=True,
    )


# =====================================================
# SCREENING SUMMARY
# =====================================================

st.divider()

st.subheader("Screening Summary")

st.write(
    f"Top predicted body part: {body_part}"
)

st.write(
    f"Top body-part model score: {confidence * 100:.2f}%"
)

st.write(
    f"Routing mode: {routing_mode}"
)

if routing_mode == "Manual Testing":
    st.write(
        f"Manually selected body part: {manual_body_part}"
    )

st.write(
    f"Routing decision: {routing['status'].title()}"
)

st.write(
    f"Routing reason: {routing['reason']}"
)

st.write(
    f"Specialist status: {specialist_status}"
)

if non_chest_result is not None:
    st.write(
        f"Screening result: {non_chest_result['title']}"
    )

st.caption(
    "Body-part and specialist scores are not calibrated "
    "medical probabilities. This research application "
    "cannot confirm or exclude disease."
)

st.divider()

st.caption(
    "Multi-Body-Part X-ray Screening | "
    "ResNet18 | UNIFESP | NIH | MURA | "
    "Research Prototype"
)
