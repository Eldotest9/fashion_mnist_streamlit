import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('fashion_mnist_model.h5')
    return model

model = load_model()

# Class names for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = ImageOps.grayscale(image)  # Convert image to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img)  # Convert image to numpy array
    img = img / 255.0  # Normalize the image
    img = img.reshape(1, 28, 28, 1)  # Reshape to fit the model input
    return img

# Streamlit app
st.title('Fashion MNIST Classifier')
st.write('Upload an image to classify it as one of the Fashion MNIST categories.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Image uploaded successfully!")

    if st.button('Run Model'):
        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Perform prediction
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.write(f'Prediction: {predicted_class}')
        st.write(f'Confidence: {confidence:.2f}')
