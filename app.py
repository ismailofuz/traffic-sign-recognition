import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("MODEL_TRAFFIC_SIGN_RECOGNITION.h5")

# Class list
classes = [
"Speed limit (20km/h)",
"Speed limit (30km/h)",
"Speed limit (50km/h)",
"Speed limit (60km/h)",
"Speed limit (70km/h)",
"Speed limit (80km/h)",
"End of speed limit (80km/h)",
"Speed limit (100km/h)",
"Speed limit (120km/h)",
"No passing",
"No passing vehicles > 3.5 tons",
"Right-of-way at intersection",
"Priority road",
"Yield",
"Stop",
"No vehicles",
"Vehicles > 3.5 tons prohibited",
"No entry",
"General caution",
"Dangerous curve left",
"Dangerous curve right",
"Double curve",
"Bumpy road",
"Slippery road",
"Road narrows on the right",
"Road work",
"Traffic signals",
"Pedestrians",
"Children crossing",
"Bicycles crossing",
"Beware of ice/snow",
"Wild animals crossing",
"End speed + passing limits",
"Turn right ahead",
"Turn left ahead",
"Ahead only",
"Go straight or right",
"Go straight or left",
"Keep right",
"Keep left",
"Roundabout mandatory",
"End of no passing",
"End of no passing vehicles > 3.5 tons"
]

st.title("🚦 Traffic Sign Recognition Web App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    
    st.image(uploaded_file, width=300, caption="Uploaded image")

    img = image.load_img(uploaded_file, target_size=(30,30))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    pred = np.argmax(model.predict(img), axis=1)[0]

    if pred < len(classes):
        st.success(f"Prediction: {classes[pred]}")
    else:
        st.error("This image does not appear to be a traffic sign.")
