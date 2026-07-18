import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("digit_model.keras")

st.title("✍️ Handwritten Digit Recognition")

uploaded_file = st.file_uploader(
    "Upload an image of a handwritten digit",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    st.image(image, caption="Uploaded Image", width=200)

    img_array = np.array(image)

    img_array = 255 - img_array

    img_array = img_array / 255.0

    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {digit}")

    st.write(f"Confidence: {confidence:.2f}%")