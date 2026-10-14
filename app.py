"""Streamlit dashboard for saved benchmark evaluation results."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(page_title="AI Cyber Threat Detection", page_icon="🛡️", layout="wide")
st.title("AI-Powered Cybersecurity Threat Detection")
st.caption("Educational network intrusion detection prototype — saved benchmark predictions")


@st.cache_data
def load_results():
    with open(OUTPUTS / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    predictions = pd.read_csv(OUTPUTS / "predictions.csv")
    return metrics, predictions


if not (OUTPUTS / "metrics.json").exists() or not (OUTPUTS / "predictions.csv").exists():
    st.error("Evaluation outputs are missing. Run `python train_model.py` and then `python evaluate_model.py`.")
    st.stop()

metrics, df = load_results()
threshold = st.sidebar.slider("Alert threshold", 0.0, 1.0, 0.50, 0.05)
view_mode = st.sidebar.selectbox("Show", ["All records", "Predicted attacks", "Predicted benign"])

df["alert_triggered"] = df["attack_probability"] >= threshold
if view_mode == "Predicted attacks":
    view = df[df["predicted_label"] == 1].copy()
elif view_mode == "Predicted benign":
    view = df[df["predicted_label"] == 0].copy()
else:
    view = df.copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Records in view", f"{len(view):,}")
c2.metric("Predicted attacks in view", f"{(view['predicted_label'] == 1).sum():,}")
c3.metric("Alerts at selected threshold", f"{view['alert_triggered'].sum():,}")
c4.metric("F1-score", f"{metrics['f1_score']:.3f}")

st.subheader("Classifier evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("Precision", f"{metrics['precision']:.3f}")
m3.metric("Recall", f"{metrics['recall']:.3f}")
m4.metric("F1-score", f"{metrics['f1_score']:.3f}")

left, right = st.columns(2)
with left:
    st.subheader("Attack score distribution")
    fig = px.histogram(view, x="attack_probability", color="predicted_label", nbins=30)
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Confusion matrix")
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual benign", "Actual attack"],
                         columns=["Predicted benign", "Predicted attack"])
    st.plotly_chart(px.imshow(cm_df, text_auto=True, aspect="auto"), use_container_width=True)

st.subheader("Investigation records")
view = view.sort_values("attack_probability", ascending=False)
st.dataframe(view, use_container_width=True, hide_index=True)
st.download_button(
    "Download visible records as CSV",
    data=view.to_csv(index=False).encode("utf-8"),
    file_name="investigation_records.csv",
    mime="text/csv",
)
st.info(
    "These are predictions on a labeled public benchmark dataset, not live network telemetry. "
    "Scores are model outputs, not calibrated probabilities of real-world compromise. "
    "An alert requires analyst review."
)
