import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from huggingface_hub import hf_hub_download

st.title("Skin Cancer Classifier")

# نزل الموديل من Hugging Face
model_path = hf_hub_download(
    repo_id="Nesma333/skin-cancer-resnet50", 
    filename="skin_cancer_Vgg16_finetuned1.h5"                 
)
model = tf.keras.models.load_model(model_path, compile=False)

st.write("### Upload an image of a skin lesion")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224,3))  # غيّري على حسب حجم الموديل
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    if st.button("Predict"):
        preds = model.predict(img_array)
        class_idx = np.argmax(preds)
        st.success(f"Prediction: Class {class_idx}")
