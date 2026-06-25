import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import time

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="AML Detect | Blood Cell Classifier",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600;8..60,700&family=Source+Sans+3:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
    }

    .main {
        background: #FBF1EF;
        color: #3A2E2C;
    }
    .stApp {
        background: #FBF1EF;
    }

    h1, h2, h3 {
        font-family: 'Source Serif 4', Georgia, serif;
        color: #3A2E2C;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    h1 {
        text-align: center;
        font-size: 2.4rem !important;
        border-bottom: 1px solid #E3C9C5;
        padding-bottom: 0.6rem;
    }
    h5, .stMarkdown p em {
        font-family: 'Source Serif 4', Georgia, serif;
        font-style: italic;
        color: #8C6B66;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #F6E4E1;
        border-right: 1px solid #E3C9C5;
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.5rem !important;
        border-bottom: none;
        text-align: left;
    }

    .upload-section {
        border: 1.5px dashed #C99A94;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
        background: #FEFBFA;
        transition: all 0.25s ease;
    }
    .upload-section:hover {
        border-color: #B5413A;
        background: #FCEFED;
    }

    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'Source Serif 4', Georgia, serif;
        box-shadow: 0 4px 16px rgba(150, 60, 50, 0.08);
        border: 1px solid rgba(0,0,0,0.04);
    }
    .aml-card {
        background: #FBE2DF;
        color: #7A2E27;
        border-left: 4px solid #B5413A;
    }
    .normal-card {
        background: #E9F2E7;
        color: #355E3B;
        border-left: 4px solid #5C8A5C;
    }
    .confidence-bar {
        height: 10px;
        background: #EFE0DD;
        border-radius: 9999px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 9999px;
        background: #B5413A;
        transition: width 1.5s ease;
    }

    /* Streamlit's own progress bar accent */
    div[data-testid="stProgress"] > div > div > div {
        background-color: #B5413A;
    }

    div[data-testid="stFileUploader"] section {
        background: transparent;
    }
    </style>
""", unsafe_allow_html=True)

# ====================== MODEL CONFIG ======================
# This model uses a MobileNetV2 backbone trained on 224x224 RGB input.
# MobileNetV2 requires its own preprocess_input (scales pixels to [-1, 1]) --
# using a plain /255.0 normalization, or the wrong image size, will silently
# produce unreliable predictions without ever raising an error.
IMG_SIZE = (224, 224)

# ====================== LOAD MODEL ======================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("aml_model.keras")

model = load_model()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/blood-test.png", width=80)
    st.title("🩸 AML Detect")
    st.markdown("**Acute Myeloid Leukemia Detection**")

    st.divider()

    st.subheader("About the Model")
    st.info(
        "This AI model classifies blood smear images as **AML Blast Cells** or "
        "**Normal White Blood Cells** using a MobileNetV2-based CNN."
    )

    st.subheader("How to Use")
    st.markdown("""
    1. Upload a clear blood smear image
    2. Wait for prediction
    3. Review confidence score
    """)

    st.divider()
    st.caption("Built with ❤️ using Streamlit + TensorFlow")

# ====================== MAIN HEADER ======================
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("# AML Blood Cell Classifier")
    st.markdown("##### *Early detection through AI-powered image analysis*")

st.divider()

# ====================== UPLOAD SECTION ======================
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload Blood Smear Image",
    type=["jpg", "png", "jpeg"],
    help="Supported formats: JPG, PNG, JPEG | Image will be resized to 224x224 for the model"
)
st.markdown('</div>', unsafe_allow_html=True)

# ====================== PREDICTION ======================
if uploaded_file is not None:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize(IMG_SIZE)

        col_img1, col_img2 = st.columns([1, 2])

        with col_img1:
            st.image(img, caption="📸 Uploaded Blood Smear", use_container_width=True)

        with col_img2:
            st.markdown("### Processing Image...")
            progress_bar = st.progress(0)

            for i in range(100):
                time.sleep(0.005)
                progress_bar.progress(i + 1)

            # Prepare image for the model exactly the way MobileNetV2 expects:
            # float32 array, then preprocess_input scales it to [-1, 1].
            img_array = np.array(img_resized, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            confidence = float(prediction[0][0])

            # Result — sigmoid output: >0.5 assumed to mean "AML".
            # NOTE: if real-world results look inverted, the training labels
            # were likely flipped (0=AML, 1=Normal) -- swap this comparison.
            is_aml = confidence > 0.5
            confidence_percent = confidence * 100 if is_aml else (1 - confidence) * 100

            st.markdown("### Prediction Result")

            if is_aml:
                st.markdown("""
                    <div class="result-card aml-card">
                        <h2 style="margin:0">⚠️ AML BLAST CELL DETECTED</h2>
                        <p style="margin:8px 0 0 0; opacity:0.9">Immediate medical attention recommended</p>
                    </div>
                """, unsafe_allow_html=True)
                st.error("**High Risk Finding**")
            else:
                st.markdown("""
                    <div class="result-card normal-card">
                        <h2 style="margin:0">✅ NORMAL WHITE BLOOD CELL</h2>
                        <p style="margin:8px 0 0 0; opacity:0.9">No AML blast cells detected</p>
                    </div>
                """, unsafe_allow_html=True)
                st.success("**Negative for AML**")

            # Confidence Visualization
            st.markdown("**Confidence Level**")
            st.progress(confidence_percent / 100)
            st.caption(f"{confidence_percent:.1f}% confidence")

            # Raw probability
            st.info(f"Model raw output: **{confidence:.4f}**")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    st.info("👆 Upload a blood smear image to begin analysis", icon="ℹ️")

# ====================== FOOTER ======================
st.divider()
st.markdown("""
    <div style="text-align: center; color: #A6837D; font-size: 0.9rem; font-family: 'Source Serif 4', Georgia, serif; font-style: italic;">
        Disclaimer: This tool is for educational and research purposes only.
        Not a substitute for professional medical diagnosis.
    </div>
""", unsafe_allow_html=True)
