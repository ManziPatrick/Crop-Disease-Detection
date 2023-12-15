
import streamlit as st
import cv2
import numpy as np
from tensorflow import keras

def load_model():
    
    model = keras.models.load_model("model.h5")
    return model

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

def predict(image, model):
    try:
        # Placeholder function for the prediction logic (without TensorFlow)
        # You can replace this with any logic based on your requirements
        # For simplicity, let's say it always predicts 'Not Affected'
        return 'Not Affected'

    except Exception as e:
        return str(e)

def main():
    st.title('Crop Disease Detection')

    # File uploader for the model
    model_file = st.file_uploader("Upload a model file (.h5)", type=["h5"])

    if model_file is not None:
        # Load the model
        model = load_model(model_file)

        # File uploader for the image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            try:
                # Read the image
                image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

                # Display the uploaded image
                st.image(image, caption='Uploaded Image', use_column_width=True)

                # Make prediction (placeholder)
                prediction = predict(image, model)

                # Display the prediction result
                st.write(f'Prediction: {prediction}')

            except Exception as e:
                # Display any exception that occurs during image processing
                st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()
