import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
from PIL import Image

# Load model and label encoder once (cache for performance)
@st.cache_resource
def load_model_and_encoder():
    model = load_model('best_skin_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

model, le = load_model_and_encoder()

def preprocess_image(img: Image.Image, img_size=224):
    img = img.resize((img_size, img_size))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("Skin Disease Classification Web App")

option = st.radio("Select input type:", ("Single Image", "Multiple Images (Folder)"))

if option == "Single Image":
    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess_image(img)
        preds = model.predict(img_array)
        pred_class = le.inverse_transform([np.argmax(preds)])[0]
        st.success(f"Predicted disease: {pred_class}")

elif option == "Multiple Images (Folder)":
    uploaded_files = st.file_uploader("Upload multiple images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            st.image(img, caption=uploaded_file.name, use_column_width=True)

            img_array = preprocess_image(img)
            preds = model.predict(img_array)
            pred_class = le.inverse_transform([np.argmax(preds)])[0]
            st.write(f"**{uploaded_file.name}**: Predicted disease: {pred_class}")
