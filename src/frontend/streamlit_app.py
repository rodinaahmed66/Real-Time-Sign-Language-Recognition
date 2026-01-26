
import streamlit as st
import cv2
import numpy as np
import requests

st.set_page_config(page_title="ASL Real-Time Detection", layout="wide")
st.title("ASL Real-Time Detection")
st.write("Webcam-based ASL recognition using FastAPI backend.")

# Sidebar options
st.sidebar.header("Settings")
frame_interval = st.sidebar.slider("Frames per prediction", 1, 60, 50)
confidence_threshold = st.sidebar.slider("Confidence Threshold (%)", 0, 100, 60)

# API URL
API_URL = st.text_input("Enter FastAPI URL:", "http://127.0.0.1:8000/predict")

# Start webcam
run = st.button("Open Camera")

if run:
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    pred_text = "Detecting..."
    conf_text = ""
    frame_count = 0

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
                conf = result.get("confidence", 0)

                # Confidence check
                try:
                    conf_value = float(conf)
                    if conf_value < confidence_threshold:
                        pred_text = "Uncertain"
                        conf_text = ""
                    else:
                        pred_text = f"Prediction: {label}"
                        conf_text = f"Confidence: {conf_value:.2f}%"
                except (ValueError, TypeError):
                    pred_text = label
                    conf_text = str(conf)
            except requests.exceptions.RequestException:
                pred_text = "Server Error"
                conf_text = ""

        # Overlay text
        display_frame = frame.copy()
        cv2.putText(display_frame, pred_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(display_frame, conf_text, (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        stframe.image(display_frame, channels="RGB", use_column_width=True)

    cap.release()
