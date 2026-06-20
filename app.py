import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# load model
model = tf.keras.models.load_model("aml_model.keras")

st.title("AML Blood Cell Classification")

uploaded_file = st.file_uploader("Upload Blood Smear Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(128,128))
    st.image(img, caption="Uploaded Image")

    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("AML Blast Cell Detected")
    else:
        st.success("Normal White Blood Cell")
