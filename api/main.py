"""
AI Face Emotion Game - emotion detection API.

Standalone FastAPI backend for the deployed version. It deliberately does NOT
import DeepFace/TensorFlow (those need ~1GB RAM and are meant for the desktop
game). Instead it uses:
  - OpenCV Haar cascade for face detection
  - the same eNet-B0 / AffectNet ONNX model as emotion_system/hsemotion_backend.py
    for classification (downloads ~16 MB once into ~/.hsemotion/)

This fits Render's free 512MB tier easily. The deployed frontend (Vercel)
sends a photo here and displays whatever emotion we return.

Endpoint: POST /predict  (multipart form field: file)
Responds with the 7-emotion probability distribution + top emotion + confidence.
"""

import io
import os
import tempfile
import urllib.request
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

MODEL_NAME = os.environ.get("HSEMOTION_MODEL", "enet_b0_8_best_vgaf")
MODEL_URL = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition/"
    "blob/main/models/affectnet_emotions/onnx/{name}.onnx?raw=true"
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# eNet-B0 predicts 8 classes (7 emotions + contempt) - we fold contempt into
# neutral, exactly like the desktop game does.
ENET_INDEX_TO_OURS = {
    0: "angry",
    1: "neutral",   # contempt
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprise",
}
OUR_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

app = FastAPI(title="AI Face Emotion Game API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Vercel page needs to call us from the browser
    allow_methods=["*"],
    allow_headers=["*"],
)

_session = None
_input_name: Optional[str] = None
_onnx_error: Optional[str] = None


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _load_onnx():
    global _session, _input_name, _onnx_error
    if _session is not None:
        return True
    import onnxruntime as ort

    path = os.path.join(tempfile.gettempdir(), "hsemotion", f"{MODEL_NAME}.onnx")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL.format(name=MODEL_NAME), path)
        except Exception as e:  # pragma: no cover - network dependent
            _onnx_error = f"download failed: {e}"
            return False
    try:
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        return True
    except Exception as e:  # pragma: no cover - env dependent
        _onnx_error = str(e)
        return False


@app.get("/health")
def health() -> Dict[str, str]:
    """Cheap liveness probe (Render uses it on free tier too)."""
    status = "ready" if _session is not None else (
        "ok" if _load_onnx() else "model-error"
    )
    return {"status": status, "model": f"hsemotion/{MODEL_NAME}", "onnx": _onnx_error or ""}


@app.post("/predict")
def predict(file: UploadFile = File(...)) -> Dict:
    """Analyze one face photo and return the emotion distribution."""
    contents = file.file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"not an image: {e}")

    frame_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    # Downscale very large phone photos so detection actually works.
    h, w = frame_bgr.shape[:2]
    if max(h, w) > 1200:
        scale = 1200 / max(h, w)
        frame_bgr = cv2.resize(
            frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    if not _load_onnx():
        raise HTTPException(
            status_code=503, detail=f"emotion model unavailable: {_onnx_error}"
        )

    # Haar cascade ships inside OpenCV - no extra downloads, no TensorFlow.
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    faces = cascade.detectMultiScale(
        gray_eq, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )

    if len(faces) == 0:
        return {
            "status": "no_face",
            "emotion": "uncertain",
            "confidence": 0.0,
            "emotions": {},
            "detail": "No clear face found. Try a brighter, closer, single-face photo.",
        }
    if len(faces) > 1:
        return {
            "status": "multiple_faces",
            "emotion": "uncertain",
            "confidence": 0.0,
            "emotions": {},
            "detail": "More than one face detected. Please upload one clear face.",
        }

    x, y, w_, h_ = [int(v) for v in faces[0]]
    pad = max(int(0.15 * w_), 8)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + w_ + pad)
    y1 = min(frame_bgr.shape[0], y + h_ + pad)
    roi = frame_bgr[y0:y1, x0:x1]
    if roi.size == 0:
        raise HTTPException(status_code=422, detail="face crop empty")

    rgb = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
    xin = rgb.astype(np.float32) / 255.0
    for c in range(3):
        xin[..., c] = (xin[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    blob = xin.transpose(2, 0, 1)[np.newaxis, ...]

    out = _session.run(None, {_input_name: blob})[0][0]
    prob = _softmax(out.astype(np.float32))

    acc = np.zeros(len(OUR_LABELS), dtype=float)
    for i, p in enumerate(prob):
        acc[OUR_LABELS.index(ENET_INDEX_TO_OURS[i])] += float(p)
    s = acc.sum()
    if s <= 0:
        raise HTTPException(status_code=422, detail="emotion model returned nothing")

    emotions: Dict[str, float] = {lab: float(v / s) for lab, v in zip(OUR_LABELS, acc)}
    top = max(emotions, key=emotions.get)
    confidence = emotions[top]

    return {
        "status": "success",
        "emotion": top,
        "confidence": round(confidence * 100, 1),
        "emotions": {k: round(v * 100, 1) for k, v in emotions.items()},
        "detail": "",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))