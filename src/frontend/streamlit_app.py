# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import requests

st.set_page_config(page_title="ASL Real-Time Detection", layout="wide")

st.title("ASL Real-Time Detection")
st.write("Webcam-based American Sign Language recognition using FastAPI backend.")

API_URL = "http://127.0.0.1:8000/predict"
CONF_THRESHOLD = 60  # Confidence threshold

# Sidebar options
st.sidebar.header("Settings")
frame_interval = st.sidebar.slider("Frames per prediction", 1, 60, 50)

# Start webcam
run = st.button("Open Camera")

if run:
    stframe = st.empty()  # placeholder for video
    cap = cv2.VideoCapture(0)

    frame_count = 0
    label = "Waiting..."
    confidence = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        frame_count += 1

        # Send every N frames to API
        if frame_count % frame_interval == 0:
            _, img_encoded = cv2.imencode(".jpg", frame)
            try:
                response = requests.post(API_URL, files={"file": img_encoded.tobytes()})
                result = response.json()
                label = result.get("label", "Error")
                confidence = result.get("confidence", 0)
                if confidence < CONF_THRESHOLD:
                    label = "Uncertain"
            except requests.exceptions.RequestException:
                label = "Server Error"
                confidence = ""

        # Overlay label/confidence on frame
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Prediction: {label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
        cv2.putText(display_frame, f"Confidence: {confidence:.2f}%", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_frame, channels="RGB", use_column_width=True)

    cap.release()
