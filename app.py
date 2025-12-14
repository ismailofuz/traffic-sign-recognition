import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ----------------------------
# Load trained model
# ----------------------------
model = tf.keras.models.load_model("MODEL_TRAFFIC_SIGN_RECOGNITION.h5")

# ----------------------------
# Traffic sign class labels
# ----------------------------
CLASSES = [
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

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")
st.title("🚦 Traffic Sign Recognition")
st.write("Upload an image. The model will classify the traffic sign.")

# ----------------------------
# File uploader (SAFE handling)
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload an image (JPG, PNG, JPEG)",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False
)

if uploaded_file is None:
    st.info("Please upload a traffic sign image to get a prediction.")
    st.stop()

# ----------------------------
# Display uploaded image
# ----------------------------
st.image(uploaded_file, width=300, caption="Uploaded image")

# ----------------------------
# Image preprocessing
# ----------------------------
img = image.load_img(uploaded_file, target_size=(30, 30))
img = image.img_to_array(img)
img = img / 255.0              # normalization (IMPORTANT)
img = np.expand_dims(img, axis=0)

# ----------------------------
# Model prediction
# ----------------------------
predictions = model.predict(img)
confidence = float(np.max(predictions))
predicted_class = int(np.argmax(predictions))

# ----------------------------
# Confidence-based rejection
# ----------------------------
CONFIDENCE_THRESHOLD = 0.6

if confidence < CONFIDENCE_THRESHOLD:
    st.error("❌ This image does not appear to be a traffic sign.")
    st.caption(f"Model confidence: {confidence:.2f}")
else:
    st.success(f"✅ Prediction: {CLASSES[predicted_class]}")
    st.caption(f"Confidence: {confidence:.2f}")
