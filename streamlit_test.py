import streamlit as st

st.title('Streamlit Test App')
st.write("This is a simple test of Streamlit.")
slider_value = st.slider('Select a value', min_value=0, max_value=100, value=50)
st.write(f'Slider value: {slider_value}')
