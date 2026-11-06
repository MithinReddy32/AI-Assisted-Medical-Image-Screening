"""A score-versus-threshold chart for an assessed upper-limb image."""
import math

import altair as alt
import pandas as pd


def build_abnormality_chart(score, threshold, body_part):
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in (score, threshold)):
        raise ValueError("Chart scores and thresholds must be finite values in [0, 1].")

    values = pd.DataFrame({
        "measure": ["Abnormality score"], "start": [0], "maximum": [100],
        "score": [score * 100], "score_label": [f"{score:.1%}"],
    })
    cutoff = pd.DataFrame({"threshold": [threshold * 100],
                           "label": [f"Threshold: {threshold:.0%}"]})
    x = alt.X("score:Q", title="Model score (%)", scale=alt.Scale(domain=[0, 100]),
              axis=alt.Axis(values=[0, 25, 50, 75, 100], grid=False))
    y = alt.Y("measure:N", axis=None)
    base = alt.Chart(values)
    track = base.mark_bar(color="#e5e7eb", size=34, cornerRadius=5).encode(
        x=alt.X("maximum:Q", title="Model score (%)", scale=alt.Scale(domain=[0, 100])),
        x2="start:Q", y=y)
    bar = base.mark_bar(color="#b45309" if score >= threshold else "#2563eb",
                        size=34, cornerRadius=5).encode(
        x=x, x2="start:Q", y=y,
        tooltip=[alt.Tooltip("score:Q", title="Abnormality model score (%)", format=".1f")])
    value_label = base.mark_text(
        align="right" if score >= .85 else "left", dx=-7 if score >= .85 else 7,
        color="white" if score >= .85 else "#111827", fontSize=14, fontWeight="bold",
    ).encode(x=x, y=y, text="score_label:N")
    rule = alt.Chart(cutoff).mark_rule(
        color="#64748b", strokeDash=[5, 4], strokeWidth=2,
    ).encode(x=alt.X("threshold:Q", title="Model score (%)", scale=alt.Scale(domain=[0, 100])),
             tooltip=[alt.Tooltip("threshold:Q", title="Screening threshold (%)", format=".1f")])
    rule_label = alt.Chart(cutoff).mark_text(
        align="right" if threshold >= .75 else "left", dx=-5 if threshold >= .75 else 5,
        baseline="top", fontSize=12, color="#64748b",
    ).encode(x=alt.X("threshold:Q", title="Model score (%)", scale=alt.Scale(domain=[0, 100])),
             y=alt.value(4), text="label:N")

    return alt.layer(track, bar, rule, rule_label, value_label).properties(
        height=110, title=f"{body_part} X-ray",
        description=f"Abnormality model score {score:.1%}; screening threshold {threshold:.0%}.",
        usermeta={"chart": "non_chest_abnormality"},
    ).configure_view(stroke=None)
