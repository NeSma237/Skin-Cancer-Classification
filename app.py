import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =========================
# تحميل الموديل
# =========================
@st.cache_resource  # علشان الموديل ميعدش يحمل كل مرة
def load_model():
    model = tf.keras.models.load_model("skin_cancer_resnet50_finetuned.h5", compile=False)
    return model

model = load_model()

# =========================
# إعداد الصفحة
# =========================
st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")
st.title("Skin Cancer Classification using ResNet50")
st.write("ارفع صورة للجلد وسيقوم النموذج بالتنبؤ بالتصنيف المناسب")

# =========================
# رفع الصورة
# =========================
uploaded_file = st.file_uploader("ارفع صورة هنا", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # قراءة الصورة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="الصورة المرفوعة", use_column_width=True)

    # =========================
    # Preprocessing
    # =========================
    img_size = (224, 224)  # نفس حجم التدريب
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    # =========================
    # Prediction
    # =========================
    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    # =========================
    # عرض النتيجة
    # =========================
    st.subheader("النتيجة:")
    st.write(f"التصنيف المتوقع: **{predicted_class}**")
    st.write(f"نسبة الثقة: **{confidence*100:.2f}%**")
