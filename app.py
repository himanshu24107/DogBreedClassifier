import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


model = tf.keras.models.load_model('keras_model.h5')

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img = np.array(img)  # Convert the image to a numpy array
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add an extra dimension for the batch
    return img

def main():
    st.title("Dog Breed Classifier")
    st.text("Available Dog breed: American Hunting Dog, Cardigen, Poodle, Maxican Hairless, Dingo, Dhole")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=False)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)

        # Load the class labels or names
        # Replace with your own list of dog breed names
        class_names = ["American Hunting Dog","Cardigen", "Poodle", "Maxican Hairless", "Dingo", "Dhole"]
        
        st.write("Prediction:", class_names[predicted_class])


main()
