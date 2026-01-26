
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model(r"enter path")

# MUST match training label order
classes = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','bye','c','d','e','good','good morning',
    'hello','little bit','no','pardon','please',
    'project','whats up','yes'
]

cap = cv2.VideoCapture(1)  # iVCam

pred_text = "Detecting..."
conf_text = ""

frame_count = 0
PRED_EVERY_N_FRAMES = 50

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run prediction every N frames
    if frame_count % PRED_EVERY_N_FRAMES == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (128, 128))
        resized = resized.astype("float32") / 255.0
        input_tensor = np.expand_dims(resized, axis=0)

        preds = model.predict(input_tensor, verbose=0)
        idx = np.argmax(preds)
        conf = preds[0][idx]

        if conf > 0.60:  # confidence threshold
            pred_text = f"Prediction: {classes[idx]}"
            conf_text = f"Confidence: {conf*100:.2f}%"
        else:
            pred_text = "Uncertain"
            conf_text = ""

    # Overlay text
    cv2.putText(frame, pred_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.putText(frame, conf_text, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

    cv2.imshow("ASL Real-Time Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
