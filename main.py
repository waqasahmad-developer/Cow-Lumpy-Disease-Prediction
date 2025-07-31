import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load model
model = load_model('lumpy_cow_classifier_model.keras')

IMG_HEIGHT, IMG_WIDTH = 256, 256
class_labels = ['Healthy Cow', 'Lumpy Cow']

def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.set_page_config(page_title="Cow Health Predictor", layout="centered")

st.title("Healthy VS Lumpy Cow 🐄 Predictor")
st.markdown("Upload an image and click **Predict** to classify the cow's health status.")

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Predict"):
        try:
            img_array = preprocess_image(image)
            preds = model.predict(img_array)
            confidence = float(preds[0][0])
            label = class_labels[1] if confidence >= 0.5 else class_labels[0]

            st.success(f"**Prediction:** {label}")
            

        except Exception as e:
            st.error(f"Error processing the image: {e}")



st.markdown("---")


st.markdown("<h3 style='text-align: center;'>Connect with me</h3>", unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    
    st.markdown(
        """
        <div style="text-align: center;">
            <a href="https://github.com/waqasahmad-developer" target="_blank" 
               style="
                background-color: #333; 
                color: white; 
                padding: 10px 25px; 
                text-decoration: none; 
                border-radius: 5px; 
                font-weight: bold; 
                margin-right: 15px;
                display: inline-block;
                ">
                🐙 GitHub
            </a>
            <a href="https://www.linkedin.com/in/waqasahmad-wa/" target="_blank" 
               style="
                background-color: #0A66C2; 
                color: white; 
                padding: 10px 25px; 
                text-decoration: none; 
                border-radius: 5px; 
                font-weight: bold;
                display: inline-block;
                ">
                💼 LinkedIn
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )