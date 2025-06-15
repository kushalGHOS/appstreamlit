import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from gtts import gTTS
import tempfile
import os
from PIL import Image


model = tf.keras.models.load_model("isl_gesture_recognition_cnn.h5")


classes = [chr(i) for i in range(65, 91)] 


st.title(" Real-time ISL Gesture Recognition")
st.markdown("This app uses **OpenCV**, **CNN**, and **GTTS** for gesture prediction and voice output.")


FRAME_WINDOW = st.image([])
pred_placeholder = st.empty()
audio_placeholder = st.empty()


start_camera = st.checkbox("Start Camera")


@st.cache_resource
def get_camera():
    return cv2.VideoCapture(0)


if "last_pred" not in st.session_state:
    st.session_state.last_pred = "A"

if start_camera:
    cap = get_camera()
    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        roi = frame[100:324, 200:424]
        roi_resized = cv2.resize(roi, (64, 64))
        roi_normalized = roi_resized / 255.0
        roi_input = np.expand_dims(roi_normalized, axis=0)

        pred = model.predict(roi_input, verbose=0)
        pred_index = np.argmax(pred)
        pred_label = classes[pred_index] if pred_index < len(classes) else "?"

        st.session_state.last_pred = pred_label

        
        cv2.rectangle(frame, (200, 100), (424, 324), (255, 0, 0), 2)
        cv2.putText(frame, f'Prediction: {pred_label}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)
        pred_placeholder.markdown(f"###  Predicted Letter: **{pred_label}**")


if st.button(" Speak Prediction"):
    tts = gTTS(f"The letter is {st.session_state.last_pred}")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_placeholder.audio(fp.name, format="audio/mp3")
