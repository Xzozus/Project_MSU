{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "a7324aef-0b10-40a2-b6db-4c48c79231fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from tensorflow.keras.models import load_model\n",
    "from tensorflow.keras.preprocessing import image\n",
    "import numpy as np\n",
    "import pickle\n",
    "from PIL import Image\n",
    "\n",
    "#Load model and label encoder once (cache for performance)\n",
    "@st.cache_resource\n",
    "def load_model_and_encoder():\n",
    "    model = load_model('best_skin_model.h5')\n",
    "    with open('label_encoder.pkl', 'rb') as f:\n",
    "        le = pickle.load(f)\n",
    "    return model, le\n",
    "\n",
    "model, le = load_model_and_encoder()\n",
    "\n",
    "def preprocess_image(img: Image.Image, img_size=224):\n",
    "    img = img.resize((img_size, img_size))\n",
    "    img_array = image.img_to_array(img) / 255.0\n",
    "    img_array = np.expand_dims(img_array, axis=0)\n",
    "    return img_array\n",
    "\n",
    "st.title(\"Skin Disease Classification Web App\")\n",
    "\n",
    "option = st.radio(\"Select input type:\", (\"Single Image\", \"Multiple Images (Folder)\"))\n",
    "\n",
    "if option == \"Single Image\":\n",
    "    uploaded_file = st.file_uploader(\"Upload an image\", type=['jpg', 'jpeg', 'png'])\n",
    "    if uploaded_file is not None:\n",
    "        img = Image.open(uploaded_file)\n",
    "        st.image(img, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "        img_array = preprocess_image(img)\n",
    "        preds = model.predict(img_array)\n",
    "        pred_class = le.inverse_transform([np.argmax(preds)])[0]\n",
    "        st.success(f\"Predicted disease: {pred_class}\")\n",
    "\n",
    "elif option == \"Multiple Images (Folder)\":\n",
    "    uploaded_files = st.file_uploader(\"Upload multiple images\", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)\n",
    "    if uploaded_files:\n",
    "        for uploaded_file in uploaded_files:\n",
    "            img = Image.open(uploaded_file)\n",
    "            st.image(img, caption=uploaded_file.name, use_column_width=True)\n",
    "\n",
    "            img_array = preprocess_image(img)\n",
    "            preds = model.predict(img_array)\n",
    "            pred_class = le.inverse_transform([np.argmax(preds)])[0]\n",
    "            st.write(f\"{uploaded_file.name}: Predicted disease: {pred_class}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
