"""
HSEmotion / eNet-B0 emotion backend (pre-trained AffectNet model, OPT-IN).

Uses DeepFace for face detection and the EfficientNet-B0 model fine-tuned on
AffectNet (Savchenko et al.) for the actual emotion classification.

Measured on this project's demo photo set (`image/`): eNet-B0 reaches ~29%
accuracy vs ~41% for DeepFace's FER2013 CNN, so the game keeps DeepFace as
its primary model and this backend is provided as an alternative for
experiments (enable with EMOTION_BACKEND=hsemotion).

The ONNX file (~16 MB) is downloaded once into ~/.hsemotion/ on first use
(HSEMOTION_MODEL_PATH can point at an already-downloaded copy). If the ONNX
model is missing or cannot be loaded, the backend transparently falls back to
DeepFace emotion scores so the app keeps working offline.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Dict, Optional, Tuple

import numpy as np

from . import config
from .deepface_backend import DeepFaceEmotionBackend

_ENET_INDEX_TO_OURS = {
    0: "angry",
    1: "neutral",   # contempt
    2: "disgust",
    3: "fear",
    4: "happy",
    5: "neutral",
    6: "sad",
    7: "surprise",
}

_OUR_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

_MODEL_URL = (
    "https://github.com/HSE-asavchenko/face-emotion-recognition/"
    "blob/main/models/affectnet_emotions/onnx/{name}.onnx?raw=true"
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def default_hsemotion_path(model_name: str) -> str:
    env = os.environ.get("HSEMOTION_MODEL_PATH")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".hsemotion", f"{model_name}.onnx")


class HSEmotionEmotionBackend(DeepFaceEmotionBackend):
    """DeepFace (detection) + eNet-B0/AffectNet (emotion classification)."""

    def __init__(self):
        super().__init__()
        self._session = None
        self._input_name: Optional[str] = None
        self._onnx_error: Optional[str] = None
        self._model_name = config.HSEMOTION_MODEL
        self._img_size = 224
        self._clahe = config.HSEMOTION_CLAHE

    # ------------------------------------------------------------------
    # model loading
    # ------------------------------------------------------------------
    def _download(self, path: str) -> bool:
        url = _MODEL_URL.format(name=self._model_name)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url, path)
            return os.path.exists(path) and os.path.getsize(path) > 1000
        except Exception as e:  # pragma: no cover - env dependent
            self._onnx_error = f"download failed: {e}"
            return False

    def _load_onnx(self) -> bool:
        if self._session is not None:
            return True
        import onnxruntime as ort

        path = default_hsemotion_path(self._model_name)
        if not os.path.exists(path):
            if not self._download(path):
                return False
        try:
            self._session = ort.InferenceSession(
                path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            return True
        except Exception as e:  # pragma: no cover - env dependent
            self._onnx_error = str(e)
            self._session = None
            return False

    def _load(self) -> bool:
        loaded = super()._load()
        if loaded:
            # Best effort: the ONNX file is optional (fallback to DeepFace).
            self._load_onnx()
        return loaded

    @property
    def report(self) -> str:
        """Short human-readable status used in logs / UI."""
        if self._session is not None:
            return f"hsemotion/{self._model_name}"
        if self._onnx_error is not None:
            return f"deepface-fallback (hsemotion unavailable: {self._onnx_error})"
        return "deepface-fallback"

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def _enet_dist(self, frame_bgr: np.ndarray, box: tuple) -> Optional[Dict[str, float]]:
        """Run eNet-B0 on the detected face crop; return our 7-label
        probability distribution or None."""
        import cv2

        x, y, w, h = [int(v) for v in box]
        pad = max(int(0.15 * w), 8)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(frame_bgr.shape[1], x + w + pad)
        y1 = min(frame_bgr.shape[0], y + h + pad)
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        if self._clahe:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        rgb = cv2.resize(rgb, (self._img_size, self._img_size),
                         interpolation=cv2.INTER_AREA)
        x = rgb.astype(np.float32) / 255.0
        for c in range(3):
            x[..., c] = (x[..., c] - _IMAGENET_MEAN[c]) / _IMAGENET_STD[c]
        blob = x.transpose(2, 0, 1)[np.newaxis, ...]

        out = self._session.run(None, {self._input_name: blob})[0][0]
        prob = _softmax(out.astype(np.float32))

        acc = np.zeros(len(_OUR_LABELS), dtype=float)
        for i, p in enumerate(prob):
            acc[_OUR_LABELS.index(_ENET_INDEX_TO_OURS[i])] += float(p)
        s = acc.sum()
        if s <= 0:
            return None
        return {lab: float(v / s) for lab, v in zip(_OUR_LABELS, acc)}

    def analyze_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[Dict[str, float]], Optional[tuple], Optional[list], str]:
        """DeepFace finds the face first (reliable detection), then eNet-B0
        classifies the crop. Falls back to DeepFace emotion scores if the
        ONNX model is unavailable."""
        dist, box, landmarks, status = super().analyze_frame(frame_bgr)
        if status != "success" or dist is None or box is None:
            return dist, box, landmarks, status
        if not self._session and not self._load_onnx():
            return dist, box, landmarks, status

        enet = self._enet_dist(frame_bgr, box)
        if enet is None:
            return dist, box, landmarks, status
        return enet, box, landmarks, status


def get_hsemotion_backend():
    return HSEmotionEmotionBackend()