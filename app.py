import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import tensorflow as tf

# Function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    return image

# Function to load and make predictions with the model
def predict(image):
    try:
        # Load the Keras model
        model = tf.keras.models.load_model('model.h5')
        
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

# Function to display content for Link 1
def display_link1_content():
    st.write("hello.")

# Function to display content for Link 2
def display_link2_content():
    st.write("world.")

# Main function to create Streamlit app
def main():
    st.title('Crop Disease Detection Demo :blue[cassava] :sunglasses:')

    # Sidebar selectbox
    add_selectbox = st.sidebar.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone")
    )

    # Add clickable links to the sidebar
    st.sidebar.write('### Links')
    link1_clicked = st.sidebar.button('home')
    link2_clicked = st.sidebar.button('contact')

    # Display content based on which link is clicked
    if link1_clicked:
        display_link1_content()
    elif link2_clicked:
        display_link2_content()

    # File uploader for image
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

# Call the main function to run the Streamlit app
if __name__ == '__main__':
    main()
