import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

# Klasör ve dosya yollarını tanımlayın
model_path = 'v1_alpaca_model.h5'
weights_path = 'v1_alpaca_weights.h5'

# Modeli yükleyin
if not os.path.exists(model_path) or not os.path.exists(weights_path):
    st.error(f"Model files not found.")
    st.stop()

try:
    model = tf.keras.models.load_model(model_path)
    model.load_weights(weights_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Sınıf kodlarını ve isimlerini tanımlayın
code = {'actinic keratosis': 3, 'basal cell carcinoma': 5, 'dermatofibroma': 7, 'melanoma': 1, 'nevus': 8, 'pigmented benign keratosis': 0, 'seborrheic keratosis': 6, 'squamous cell carcinoma': 7, 'vascular lesion': 2}

def getcode(n):
    for x, y in code.items():
        if n == y:
            return x

# Görüntüyü hazırlayın
def prepare_image(img, target_size=(200, 200)):
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Görüntüyü tahmin edin
def predict_skin_disease(model, img_array):
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return getcode(predicted_class)

# Streamlit uygulaması
st.title("Skin Cancer Detection")
st.markdown("<h2 style='text-align: center; color: #4CAF50;'>Please upload a skin image:</h2>", unsafe_allow_html=True)

# Kişisel veri işleme onayı
consent_given = st.checkbox("I agree to the processing of my personal data")

if consent_given:
    # Dosya yükleme alanı
    uploaded_file = st.file_uploader("Upload image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        try:
            # Görüntüyü yükleyin ve tahmin yapın
            img = Image.open(uploaded_file)
            img_array = prepare_image(img)
            result = predict_skin_disease(model, img_array)

            # Görüntüyü gösterin ve sonucu belirtin
            st.image(img, caption='Uploaded image', use_column_width=True)
            st.markdown(f"<h3 style='text-align: center; color: #FF5733;'>Forecast Result: {result}</h3>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing the image: {e}")
else:
    st.warning("Please agree to the processing of your personal data")

# Sayfa stili
st.markdown("""
    <style>
        .reportview-container {
            background: #f5f5dc; /* Kum rengi arka plan */
            color: #333;
            overflow: hidden; /* Scrollbars'ı kaldırır */
        }
        .sidebar .sidebar-content {
            background: #f5f5dc; /* Kum rengi arka plan */
        }
        .stTitle {
            color: #4CAF50;
        }
        .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stCheckbox>div>div>label {
            color: #4CAF50;
        }
        .stFileUploader>label {
            color: #4CAF50;
        }
        .stAlert {
            background-color: #ffcccc;
            color: #990000;
            border-radius: 5px;
            padding: 10px;
        }
        /* Sayfanın yüksekliğini tam ekran yapar */
        .block-container {
            padding: 2rem;
            max-width: 100%;
        }
        .stApp {
            min-height: 100vh;
        }
    </style>
    """, unsafe_allow_html=True)
