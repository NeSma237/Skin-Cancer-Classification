import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#تحميل الموديل
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("skin_model.h5", compile=False)
    return model

model = load_model()

#ترويسة
st.title("Skin Cancer Classification")

#رفع صورة
uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

#لو في صورة مرفوعة
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # تجهيز الصورة
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # توقع
    prediction = model.predict(img_array)
    class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Prediction: {predicted_class}")
