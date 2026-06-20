import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
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
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
        color: #e2e8f0;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e2937 100%);
    }
    h1 {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(90deg, #f87171, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 700;
    }
    .upload-section {
        border: 2px dashed #64748b;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-section:hover {
        border-color: #f87171;
        background: rgba(248, 113, 113, 0.1);
    }
    .result-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .aml-card {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: white;
    }
    .normal-card {
        background: linear-gradient(135deg, #166534, #14532d);
        color: white;
    }
    .confidence-bar {
        height: 12px;
        background: #334155;
        border-radius: 9999px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 1.5s ease;
    }
    </style>
""", unsafe_allow_html=True)

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
        "**Normal White Blood Cells** using a deep learning CNN."
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
    help="Supported formats: JPG, PNG, JPEG | Recommended: 128x128 or higher resolution"
)
st.markdown('</div>', unsafe_allow_html=True)

# ====================== PREDICTION ======================
if uploaded_file is not None:
    try:
        # Load and display image
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((128, 128))
        
        col_img1, col_img2 = st.columns([1, 2])
        
        with col_img1:
            st.image(img, caption="📸 Uploaded Blood Smear", use_column_width=True)
        
        with col_img2:
            st.markdown("### Processing Image...")
            progress_bar = st.progress(0)
            
            # Simulate processing
            for i in range(100):
                time.sleep(0.015)
                progress_bar.progress(i + 1)
            
            # Prepare image for model
            img_array = image.img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Make prediction
            prediction = model.predict(img_array, verbose=0)
            confidence = prediction[0][0]
            
            # Result
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
    <div style="text-align: center; color: #64748b; font-size: 0.9rem;">
        Disclaimer: This tool is for educational and research purposes only. 
        Not a substitute for professional medical diagnosis.
    </div>
""", unsafe_allow_html=True)
