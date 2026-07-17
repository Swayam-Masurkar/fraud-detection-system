"""
Streamlit demo UI for the fraud detection pipeline.

Run:
    streamlit run app_streamlit.py
"""
import io
import os
import tempfile
import textwrap
from datetime import datetime

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Insurance Fraud Detector", page_icon="🔍", layout="centered")

MODEL_PATH = "models/fraud_pipeline.pkl"

# Risk thresholds and the recommended action for each band
LOW_THRESHOLD = 0.3
HIGH_THRESHOLD = 0.6


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_PATH)


def risk_band(proba: float):
    if proba < LOW_THRESHOLD:
        return "LOW", "Approve claim automatically"
    elif proba < HIGH_THRESHOLD:
        return "MEDIUM", "Flag for manual review"
    else:
        return "HIGH", "Escalate to fraud investigation team"


def generate_pdf_report(claim: dict, proba: float, risk_level: str, action: str, image_arr=None) -> bytes:
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    gs = fig.add_gridspec(6, 2, height_ratios=[0.5, 0.5, 1.6, 1.6, 0.3, 1.8], hspace=0.6)

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(0, 0.6, "Insurance Fraud Risk Report", fontsize=20, fontweight="bold")
    ax_title.text(0, 0.05, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9, color="gray")

    ax_pred = fig.add_subplot(gs[1, :])
    ax_pred.axis("off")
    is_fraud = proba >= 0.5
    verdict = "FRAUDULENT CLAIM" if is_fraud else "GENUINE CLAIM"
    color = "#c0392b" if is_fraud else "#27ae60"
    confidence = proba if is_fraud else 1 - proba
    ax_pred.text(0, 0.65, verdict, fontsize=16, fontweight="bold", color=color)
    ax_pred.text(
        0, 0.15,
        f"Confidence: {confidence:.1%}    Risk level: {risk_level}    Recommended action: {action}",
        fontsize=10,
    )

    ax_claim = fig.add_subplot(gs[2:4, 0])
    ax_claim.axis("off")
    ax_claim.text(0, 1.0, "Claim details", fontsize=12, fontweight="bold", va="top")
    field_labels = {
        "claim_amount": "Claim amount",
        "num_prev_claims": "Previous claims",
        "days_since_policy": "Days since policy started",
        "location": "Location",
        "disaster_nearby": "Disaster nearby",
        "num_payments": "Number of payments",
        "payment_irregularity": "Payment irregularity",
    }
    lines = []
    for key, label in field_labels.items():
        val = claim.get(key)
        if key in ("disaster_nearby", "payment_irregularity"):
            val = "Yes" if val else "No"
        if key == "claim_amount":
            val = f"${val:,.2f}"
        lines.append(f"{label}: {val}")
    ax_claim.text(0, 0.88, "\n\n".join(lines), fontsize=9.5, va="top", linespacing=1.8)

    ax_img = fig.add_subplot(gs[2:4, 1])
    ax_img.axis("off")
    ax_img.set_title("Claim photo", fontsize=10)
    if image_arr is not None:
        ax_img.imshow(image_arr)
    else:
        ax_img.text(0.5, 0.5, "No image provided", ha="center", va="center", fontsize=9, color="gray")

    ax_narr = fig.add_subplot(gs[5, :])
    ax_narr.axis("off")
    ax_narr.text(0, 1.0, "Claim narrative", fontsize=12, fontweight="bold", va="top")
    wrapped = "\n".join(textwrap.wrap(claim.get("narrative", ""), width=95))
    ax_narr.text(0, 0.8, wrapped, fontsize=9.5, va="top")

    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


st.title("🔍 AI-Based Insurance Fraud Detection")
st.caption("Multimodal demo — structured claim data + narrative text + claim photo")

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    with col1:
        claim_amount = st.number_input("Claim amount ($)", min_value=0.0, value=8000.0, step=100.0)
        num_prev_claims = st.number_input("Previous claims", min_value=0, value=0, step=1)
        days_since_policy = st.number_input("Days since policy started", min_value=0, value=400, step=1)
        location = st.selectbox("Location", ["urban", "suburban", "rural", "coastal", "mountain"])
    with col2:
        disaster_nearby = st.radio("Disaster nearby?", ["No", "Yes"], horizontal=True)
        num_payments = st.number_input("Number of payments", min_value=0, value=1, step=1)
        payment_irregularity = st.radio("Payment irregularity flagged?", ["No", "Yes"], horizontal=True)

    narrative = st.text_area(
        "Claim narrative",
        value="Storm damaged the roof and water entered the house. Photos and a contractor estimate are attached.",
    )

    st.markdown("**Upload Damage Image**")
    uploaded_image = st.file_uploader("Choose Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_image is not None:
        st.success(f"✓ {uploaded_image.name} uploaded")

    submitted = st.form_submit_button("Score this claim")

if submitted:
    pipe = load_pipeline()

    # Save the uploaded image to a temp file so the pipeline's image
    # feature extractor (which reads from disk) can use it.
    image_arr = None
    if uploaded_image is not None:
        pil_img = Image.open(uploaded_image).convert("RGB")
        image_arr = np.array(pil_img)
        suffix = os.path.splitext(uploaded_image.name)[1] or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        cv2.imwrite(tmp.name, cv2.cvtColor(image_arr, cv2.COLOR_RGB2BGR))
        image_path_for_model = tmp.name
    else:
        # no image uploaded — the extractor degrades gracefully (zeros) if
        # the path is missing, so an empty/nonexistent path is safe
        image_path_for_model = ""

    claim = dict(
        claim_amount=claim_amount,
        num_prev_claims=num_prev_claims,
        days_since_policy=days_since_policy,
        location=location,
        disaster_nearby=1 if disaster_nearby == "Yes" else 0,
        num_payments=num_payments,
        payment_irregularity=1 if payment_irregularity == "Yes" else 0,
        narrative=narrative,
        image_path=image_path_for_model,
    )
    row = pd.DataFrame([claim])
    proba = float(pipe.predict_proba(row)[:, 1][0])
    is_fraud = proba >= 0.5
    confidence = proba if is_fraud else 1 - proba
    risk_level, action = risk_band(proba)

    if uploaded_image is not None:
        st.image(image_arr, caption="Uploaded claim photo", use_container_width=True)

    st.markdown("### Prediction")
    verdict_col, conf_col = st.columns(2)
    with verdict_col:
        if is_fraud:
            st.error("🚩 Fraudulent Claim")
        else:
            st.success("✅ Genuine Claim")
    with conf_col:
        st.metric("Confidence", f"{confidence:.1%}")

    risk_col, action_col = st.columns(2)
    with risk_col:
        st.metric("Risk Level", risk_level)
    with action_col:
        st.markdown(f"**Recommended Action**\n\n{action}")

    pdf_bytes = generate_pdf_report(claim, proba, risk_level, action, image_arr=image_arr)
    st.download_button(
        label="⬇️ Download Fraud Report (PDF)",
        data=pdf_bytes,
        file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
    )
