import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AML Blood Cell Detector",
    page_icon="🩸",
    layout="centered"
)

IMG_SIZE = (224, 224)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("aml_wbc_classifier.keras")
    return model

model = load_model()

# ==========================
# TITLE
# ==========================
st.title("🩸 AML vs White Blood Cell Classifier")
st.write(
    "Upload a microscopic blood smear image. "
    "The AI model predicts whether it is **AML (Acute Myeloid Leukemia)** "
    "or a **Normal White Blood Cell**."
)

st.markdown("---")

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "Upload Blood Cell Image",
    type=["jpg", "jpeg", "png", "tiff"]
)

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_image(img):

    # resize
    img = img.resize(IMG_SIZE)

    # convert to numpy
    img_array = np.array(img, dtype=np.float32)

    # SAME preprocessing used in training
    img_array = img_array / 255.0

    # batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    prediction = model.predict(img_array, verbose=0)

    return float(prediction[0][0])

# ==========================
# RUN PREDICTION
# ==========================
if uploaded_file is not None:

    # open image
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:

        with st.spinner("Analyzing image..."):

            confidence = predict_image(image)

            # IMPORTANT:
            # AML = 0
            # NORMAL = 1

            if confidence < 0.5:

                prediction = "⚠ AML CELL DETECTED"

                # if model says AML
                confidence_score = (1 - confidence) * 100

                st.error(prediction)

            else:

                prediction = "✅ NORMAL WHITE BLOOD CELL"

                confidence_score = confidence * 100

                st.success(prediction)

            st.subheader("Confidence Score")

            st.progress(confidence_score / 100)

            st.write(f"**{confidence_score:.2f}% confidence**")

            # debug (remove later)
            st.write("Raw Model Output:", round(confidence, 4))

else:
    st.info("Please upload an image to start prediction.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption(
    "Educational / Research Use Only • Not a Medical Diagnostic Tool"
)
