import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download

from tensorflow.keras.models import load_model

model = load_model(model_path,compile=False)
model.summary()


st.title("Skin Cancer Classifier")

# 1. حمل النموذج من Hugging Face
repo_id = "Nesma333/skin-cancer-resnet50"
filename = "skin_cancer_resnet50_finetuned.h5"

model_path = hf_hub_download(repo_id=repo_id, filename=filename)

# 2. حمل النموذج (compile=False لتفادي المشاكل)
model = tf.keras.models.load_model(model_path, compile=False)

# 3. ارفع صورة للتجربة
uploaded_file = st.file_uploader("Upload a skin image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    import numpy as np
    from PIL import Image

    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # 4. حضر الصورة للنموذج
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 5. اعمل التنبؤ
    prediction = model.predict(img_array)
    st.write("Prediction:", prediction)
