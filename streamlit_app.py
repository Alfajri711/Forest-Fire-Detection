import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "forest_fire_model.h5"
model = load_model(MODEL_PATH)

# Function to make predictions
def predict_image(model, img, target_size=(64, 64)):
    # Preprocess the image
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize pixel values to [0,1]
    
    # Predict using the model
    prediction = model.predict(img_array)
    class_label = "Non Fire" if prediction[0][0] > 0.5 else "Fire"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    return class_label, confidence

# Streamlit app
st.title("🔥 Forest Fire Detection 🔥")
st.markdown(
    """
    This application uses a trained Convolutional Neural Network (CNN) to detect 
    whether an uploaded image contains **fire** or not. 
    Upload an image and let the model analyze it.
    """
)

# File uploader
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Make a prediction
    st.write("Analyzing the image...")
    label, confidence = predict_image(model, img)
    
    # Display results
    if label == "Fire":
        st.error(f"🔥 **Prediction: {label}!**")
        st.warning(f"Confidence: **{confidence:.2f}**")
        st.markdown("⚠️ **Action:** Fire detected. Please take necessary precautions!")
    else:
        st.success(f"🌳 **Prediction: {label}!**")
        st.info(f"Confidence: **{confidence:.2f}**")
        st.markdown("✅ **Action:** No fire detected. The area seems safe.")
    
    # Display a progress bar for confidence
    st.progress(int(confidence * 100))
