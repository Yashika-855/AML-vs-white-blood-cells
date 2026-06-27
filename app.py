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
# THEME / CUSTOM STYLING
# ==========================
# Palette is drawn from the subject itself: a subtle pale pink
# background (like a stained slide held up to bright light) with accent
# colors pulled from a Wright-Giemsa blood stain -- teal for the field,
# a saturated red reserved only for the AML-positive alert so it
# carries real diagnostic weight against the soft backdrop.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #FBF0EF;
        --surface: #FFFFFF;
        --surface-raised: #FDF6F5;
        --border: #ECD9D7;
        --text: #2E2326;
        --text-dim: #8C7679;
        --teal: #3D8C7D;
        --teal-dim: #BFE0D9;
        --alert: #C5392E;
        --alert-dim: #F2C9C5;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, var(--bg) 0%, #F6E4E2 100%);
        color: var(--text);
    }

    /* Hide default chrome that fights the custom look */
    #MainMenu, footer, header { visibility: hidden; }

    /* ---- Hero header ---- */
    .lab-header {
        border-bottom: 1px solid var(--border);
        padding-bottom: 1.4rem;
        margin-bottom: 1.8rem;
    }
    .lab-eyebrow {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--teal);
        margin-bottom: 0.4rem;
    }
    .lab-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: var(--text);
        line-height: 1.15;
        margin-bottom: 0.6rem;
        letter-spacing: -0.01em;
    }
    .lab-subtitle {
        font-size: 0.98rem;
        color: var(--text-dim);
        line-height: 1.55;
        max-width: 38rem;
    }
    .lab-subtitle b { color: var(--text); font-weight: 600; }

    /* ---- Upload zone ---- */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface);
        border: 1.5px dashed var(--border);
        border-radius: 10px;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--teal-dim);
    }

    /* ---- Result card ---- */
    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        border: 1px solid var(--border);
        background: var(--surface);
        position: relative;
        overflow: hidden;
    }
    .result-card.is-normal { border-color: var(--teal-dim); }
    .result-card.is-aml { border-color: var(--alert-dim); }

    .result-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--text-dim);
        margin-bottom: 0.5rem;
    }
    .result-verdict {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.9rem;
    }
    .result-verdict.is-normal { color: var(--teal); }
    .result-verdict.is-aml { color: var(--alert); }

    .confidence-readout {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
    }
    .confidence-caption {
        font-size: 0.78rem;
        color: var(--text-dim);
        margin-top: 0.2rem;
    }

    /* Progress bar recolor */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--teal-dim), var(--teal));
    }

    /* ---- Footer ---- */
    .lab-footer {
        margin-top: 2.4rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--text-dim);
        letter-spacing: 0.02em;
        text-align: center;
    }

    /* Scoped overrides for default widgets to match the dark surface */
    .stAlert {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
    }
    [data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("aml_wbc_classifier.keras")
    return model

model = load_model()

# ==========================
# HEADER
# ==========================
st.markdown(
    """
    <div class="lab-header">
        <div class="lab-eyebrow">HEMATOLOGY · IMAGE CLASSIFIER</div>
        <div class="lab-title">🩸 AML vs White Blood Cell Classifier</div>
        <div class="lab-subtitle">
            Upload a microscopic blood smear image. The model predicts whether
            the sample shows <b>AML (Acute Myeloid Leukemia)</b> or a
            <b>normal white blood cell</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================
# FILE UPLOAD
# ==========================
uploaded_file = st.file_uploader(
    "Upload blood cell image",
    type=["jpg", "jpeg", "png", "tiff"]
)

# ==========================
# PREDICTION FUNCTION
# ==========================
def predict_image(img):
    # resize
    img = img.resize(IMG_SIZE)
    # convert to numpy, keep RAW 0-255 pixel values
    # NOTE: the model has its own built-in Rescaling + Normalization
    # layers (EfficientNetB0 preprocessing), so do NOT divide by 255
    # here -- doing so double-normalizes the input and corrupts predictions.
    img_array = np.array(img, dtype=np.float32)
    # batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # prediction
    prediction = model.predict(img_array, verbose=0)
    return float(prediction[0][0])

# ==========================
# RUN PREDICTION
# ==========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Analyzing image..."):
            confidence = predict_image(image)

            # IMPORTANT:
            # AML = 0
            # NORMAL = 1
            if confidence < 0.5:
                verdict_class = "is-aml"
                verdict_text = "⚠ AML cell detected"
                confidence_score = (1 - confidence) * 100
            else:
                verdict_class = "is-normal"
                verdict_text = "✅ Normal white blood cell"
                confidence_score = confidence * 100

            st.markdown(
                f"""
                <div class="result-card {verdict_class}">
                    <div class="result-label">Model Verdict</div>
                    <div class="result-verdict {verdict_class}">{verdict_text}</div>
                    <div class="confidence-readout">{confidence_score:.1f}%</div>
                    <div class="confidence-caption">confidence score</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            st.progress(confidence_score / 100)

            with st.expander("Show raw model output"):
                st.code(f"sigmoid_output = {confidence:.4f}", language="text")
                st.caption("Output < 0.5 → AML · Output ≥ 0.5 → Normal")

else:
    st.info("Please upload an image to start prediction.")

# ==========================
# FOOTER
# ==========================
st.markdown(
    """
    <div class="lab-footer">
        EDUCATIONAL / RESEARCH USE ONLY · NOT A MEDICAL DIAGNOSTIC TOOL
    </div>
    """,
    unsafe_allow_html=True
)
