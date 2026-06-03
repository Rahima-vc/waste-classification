import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
model = tf.keras.models.load_model("waste_classifier_cnn.h5")
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
tips = {
    "plastic": "Place in the recycling bin after cleaning.",
    "paper": "Keep dry before recycling.",
    "glass": "Handle carefully and recycle separately.",
    "metal": "Recycle in the metal collection bin.",
    "cardboard": "Flatten before recycling."}
st.title("♻️ Smart Waste Classification AI")
st.write("Upload an image to identify the waste type.")
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"])
if uploaded_file is not None:

    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            uploaded_file,
            caption="Uploaded Image",
            width=300
        )

    with col2:
        st.subheader("Prediction Result")
        st.success(f"♻️ Waste Type: {predicted_class.capitalize()}")
        st.metric("Confidence", f"{confidence:.2f}%")
        st.progress(confidence / 100)
        st.info(f"💡 {tips[predicted_class]}")
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100
    st.subheader("Prediction Result")
    st.success(f"♻️ Waste Type: {predicted_class.capitalize()}")
    st.metric("Confidence", f"{confidence:.2f}%")
    st.progress(float(confidence / 100))
    st.info(f"💡 {tips[predicted_class]}")