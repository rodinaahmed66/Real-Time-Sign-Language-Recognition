from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = FastAPI(
    title="ASL Recognition API",
    description="Real-time American Sign Language Recognition",
    version="1.0"
)

model = load_model(r"enter path")

CLASSES = [
    '0','1','2','3','4','5','6','7','8','9',
    'a','b','bye','c','d','e','good','good morning',
    'hello','little bit','no','pardon','please',
    'project','whats up','yes'
]

def preprocess(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess(image_bytes)

    preds = model.predict(input_tensor)
    idx = int(np.argmax(preds))
    conf = float(preds[0][idx])

    return {
        "label": CLASSES[idx],
        "confidence": round(conf * 100, 2)
    }
