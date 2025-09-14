import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Fraud Detection", layout="wide")
MODEL_PATH = 'saved_model/fraud_detector.h5'
IMG_SIZE = 224
# These names must match the folder names in your training data, in alphabetical order.
# 'fraud' comes before 'non_fraud' alphabetically.
CLASS_NAMES = ['Fraud', 'Non-Fraud']

# --- Model Loading ---
# Use st.cache_resource to load the model only once.
@st.cache_resource
def load_fraud_model():
    """Loads the trained Keras model from disk."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        # Display an error if the model file is not found or corrupted
        st.error(f"Error loading model: {e}\n"
                 f"Please ensure the model file exists at '{MODEL_PATH}'. "
                 "You may need to run the `train_model.py` script first.")
        return None

model = load_fraud_model()

# --- Prediction Function ---
def predict(image_data, model):
    """
    Takes a PIL image, preprocesses it, and returns the model's prediction score.
    """
    # Resize image
    image = image_data.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert image to numpy array
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    
    # Expand dimensions to create a batch of 1
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Make a prediction. The output is a single value from the sigmoid neuron.
    prediction_score = model.predict(image_batch)[0][0]
    
    return prediction_score

# --- Streamlit App UI ---

st.title("🚗 Auto Insurance Fraud Detection")
st.write("Upload an image of vehicle damage to check for potential fraudulent claims.")

# Main layout with two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a vehicle damage image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None and model is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

with col2:
    st.header("Analysis Result")
    if uploaded_file is not None and model is not None:
        with st.spinner('Analyzing claim...'):
            # Get the prediction score
            score = predict(image, model)

        st.subheader("Prediction:")
        
        # The model outputs a probability. A value close to 0 means 'fraud' (class 0)
        # and a value close to 1 means 'non-fraud' (class 1).
        threshold = 0.5
        if score < threshold:
            prediction_class = CLASS_NAMES[0] # Fraud
            confidence = 1 - score
            st.error(f"**Status: {prediction_class} Claim Detected**")
        else:
            prediction_class = CLASS_NAMES[1] # Non-Fraud
            confidence = score
            st.success(f"**Status: {prediction_class} Claim**")
        
        st.write(f"**Confidence:** `{confidence:.2%}`")
        
        st.subheader("Recommendation:")
        if prediction_class == 'Fraud':
            st.warning("This claim shows characteristics similar to known fraudulent submissions. It is highly recommended to flag this for manual review by a claims specialist.")
        else:
            st.info("This claim appears legitimate. Standard processing is recommended.")
    elif model is None:
        st.error("The prediction model is not loaded. Please check the setup.")
    else:
        st.info("Please upload an image to see the analysis.")

st.sidebar.header("About")
st.sidebar.info(
    
    "It uses a fine-tuned `EfficientNetV2-B0` model to classify car damage images as fraudulent or legitimate. "
    "The model was trained to handle severe class imbalance, prioritizing the detection of fraudulent claims (high recall)."
)
