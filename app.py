import sys
from PIL import Image
import streamlit as st

from src.pipeline.prediction_pipeline import ChestXrayClassifier
from src.pipeline.training_pipeline import TrainPipeline
from src.exception import MyException
from src.logger import logging


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chest X-Ray Classifier",
    page_icon="🫁",
    layout="centered",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🫁 Chest X-Ray Pneumonia Classifier")
st.markdown(
    "Upload a chest X-ray image and the model will predict whether it shows "
    "**NORMAL** lungs or **PNEUMONIA**."
)
st.divider()

# ── Sidebar — Training ────────────────────────────────────────────────────────
st.sidebar.title("Pipeline Controls")
st.sidebar.markdown("Trigger model retraining from S3 dataset.")

if st.sidebar.button("🚀 Train Model", use_container_width=True):
    with st.sidebar:
        with st.spinner("Training in progress..."):
            try:
                pipeline = TrainPipeline()
                pipeline.run_pipeline()
                st.success("Training complete!")
                logging.info("Training pipeline triggered from Streamlit.")
            except Exception as e:
                st.error(f"Training failed: {e}")
                raise MyException(e, sys)

# ── Main — Prediction ─────────────────────────────────────────────────────────
st.subheader("Upload X-Ray Image")

uploaded_file = st.file_uploader(
    label="Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"],
    help="Upload a chest X-ray in JPG or PNG format.",
)

if uploaded_file is not None:

    # ── Show uploaded image ───────────────────────────────────────────────
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded X-Ray",
            use_container_width=True,
        )

    # ── Run prediction ────────────────────────────────────────────────────
    with col2:
        st.markdown("### Prediction")

        with st.spinner("Analysing X-ray..."):
            try:
                classifier = ChestXrayClassifier()
                result     = classifier.predict(image=image)
                logging.info(
                    f"Prediction made for uploaded image: {result}"
                )

                # ── Display result ────────────────────────────────────────
                if result == "PNEUMONIA":
                    st.error(
                        "🔴 **PNEUMONIA DETECTED**\n\n"
                        "The model has detected signs of pneumonia "
                        "in this X-ray. Please consult a doctor.",
                        icon="⚠️",
                    )
                else:
                    st.success(
                        "🟢 **NORMAL**\n\n"
                        "The model found no signs of pneumonia "
                        "in this X-ray.",
                        icon="✅",
                    )

                # ── Confidence note ───────────────────────────────────────
                st.info(
                    "**Note:** This is an AI-assisted prediction tool. "
                    "Always consult a qualified medical professional "
                    "for diagnosis.",
                    icon="ℹ️",
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                raise MyException(e, sys)

else:
    # Placeholder when no image uploaded
    st.info(
        "Please upload a chest X-ray image above to get a prediction.",
        icon="👆",
    )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Chest X-Ray Classifier — ResNet50 | MLOPs Project")