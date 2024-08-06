import streamlit as st
from PIL import Image

# Streamlit app
st.title('Fashion MNIST Classifier')
st.write('Upload an image to classify it as one of the Fashion MNIST categories.')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Image uploaded successfully!")
