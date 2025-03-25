import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import keras
from PIL import Image

def load_model():
    model = tf.keras.models.load_model("saved_model.keras")
    #model = keras.layers.TFSMLayer("saved_model", call_endpoint='serving_default')
    return model

def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize to [0,1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def main():
    st.title("Cat vs Dog Classifier")
    st.write("Upload an image to classify it as a Cat or a Dog.")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        image = preprocess_image(image)
        
        prediction = model.predict(image)[0][0]  # Get the first prediction value
        
        if prediction > 0.5:
            st.write("### Prediction: 🐶 Dog")
        else:
            st.write("### Prediction: 🐱 Cat")

if __name__ == "__main__":
    main()
