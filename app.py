import streamlit as st
import gdown
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input

st.title("🚀 Hybrid VGG19 + ResNet50 Model Deployment")

FILE_ID = "1qb4k0OdZhvTHs4AjEP69lnwT8pnTB74d"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
OUTPUT = "best_model.h5"

if not os.path.exists(OUTPUT):
    with st.spinner("Downloading model from Google Drive... ⏳"):
        gdown.download(URL, OUTPUT, quiet=False, fuzzy=True)

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(OUTPUT)
st.success("✅ Model loaded successfully!")

if st.button("Run Dummy Prediction"):
    dummy_input = np.random.rand(1, 224, 224, 3)
    dummy_input = preprocess_input(dummy_input)
    pred = model.predict(dummy_input)
    st.write("Prediction shape:", pred.shape)
