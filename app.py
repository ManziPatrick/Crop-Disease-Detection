import streamlit as st
import tensorflow as tf
import cv2
import numpy as np


# Load the Keras model
model = model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0 
    return image

def predict(image):
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Make prediction
        input_image_reshaped = np.reshape(preprocessed_image, [1, 128, 128, 3])
        prediction = model.predict(input_image_reshaped)
        result = np.argmax(prediction)

        if result == 1:
            prediction_label = 'Affected'
        else:
            prediction_label = 'Not Affected'

        return prediction_label

    except Exception as e:
        return str(e)

def main():
    st.title('Crop Disease Detection')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Read the image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Make prediction
            prediction = predict(image)

            # Display the prediction result
            st.write(f'Prediction: {prediction}')

        except Exception as e:
            # Display any exception that occurs during image processing
            st.error(f"Error: {str(e)}")
