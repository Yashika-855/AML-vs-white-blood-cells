import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# ================= CONFIG =================
st.set_page_config(
    page_title="AML vs WBC Classifier",
    page_icon="🩸",
    layout="centered"
)

IMG_SIZE = (224, 224)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("aml_wbc_classifier.keras")
    return model

model = load_model()

# ================= UI =================
st.title("🩸 AML vs White Blood Cell Classifier")
st.markdown(
    "Upload a microscopic blood smear image and the model will classify it as **AML Cancer Cell** or **Normal White Blood Cell**."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png", "tiff"]
)

# ================= PREDICTION =================
if uploaded_file is not None:

    try:
        # show uploaded image
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.image(img, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.write("Processing...")

            progress = st.progress(0)

            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)

            # preprocessing (same as training)
            img_resized = img.resize(IMG_SIZE)

            img_array = np.array(img_resized, dtype=np.float32)

            # IMPORTANT: same preprocessing as training
            img_array = img_array / 255.0

            img_array = np.expand_dims(img_array, axis=0)

            # prediction
            prediction = model.predict(img_array, verbose=0)

            confidence = float(prediction[0][0])

            st.write("Raw model output:", confidence)

            # class mapping
            # AML = 0
            # NORMAL = 1

            if confidence < 0.5:
                label = "⚠ AML CELL DETECTED"
                prob = (1 - confidence) * 100

                st.error(label)

            else:
                label = "✅ NORMAL WHITE BLOOD CELL"
                prob = confidence * 100

                st.success(label)

            st.subheader("Confidence")

            st.progress(prob / 100)

            st.write(f"{prob:.2f}% confidence")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Upload an image to begin.")

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "For educational / research use only. Not a medical diagnostic tool."
)
