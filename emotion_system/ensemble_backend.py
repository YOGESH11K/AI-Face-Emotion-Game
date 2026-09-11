"""
Ensemble emotion backend (pre-trained models only).

Combines two independent, pre-trained emotion models on every frame so a
single model's bias ("everything is neutral/angry" on webcams) cannot
dominate the output:

    * DeepFace FER2013 CNN  (full-frame, SSD detection + internal alignment)
    * FER+ ISB ONNX model   (grayscale face crop, FER+ crowd-labelled training)

Probabilities are averaged per emotion (DeepFace weighted 0.6, FER+ 0.4 by
default) and re-normalized. The FER+ 'contempt' class is folded into
'neutral' to match the 7-emotion label space. Webcam crops are contrast
normalized with CLAHE because washed-out auto-exposure is the main reason
generic FER models read webcams as a flat 'neutral'/'angry' blob.

This backend deliberately trains nothing and downloads nothing at runtime
(beyond DeepFace's own weights); the FER+ ONNX file is expected at
~/.deepface/weights/emotion-ferplus-8.onnx and is optional - if it is
missing the backend transparently falls back to DeepFace alone.
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from . import config
from .deepface_backend import DeepFaceEmotionBackend

# FER+ model output order (8 classes):
#   neutral, happy, surprise, sad, angry, disgust, fear, contempt
_FERPLUS_TO_OURS = {
    0: "neutral",
    1: "happy",
    2: "surprise",
    3: "sad",
    4: "angry",
    5: "disgust",
    6: "fear",
    7: "neutral",  # contempt folds into neutral
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


def default_ferplus_path() -> str:
    env = os.environ.get("FERPLUS_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    return os.path.join(
        os.path.expanduser("~"), ".deepface", "weights", "emotion-ferplus-8.onnx"
    )


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class EnsembleEmotionBackend(DeepFaceEmotionBackend):
    """DeepFace + FER+ ensemble with CLAHE-enhanced webcam crops."""

    def __init__(self):
        super().__init__()
        self._ferplus_net = None
        self._ferplus_error: Optional[str] = None
        self.w_deepface = float(
            os.environ.get("ENSEMBLE_DEEPFACE_WEIGHT", "0.6")
        )
        self.w_ferplus = float(
            os.environ.get("ENSEMBLE_FERPLUS_WEIGHT", "0.4")
        )

    def _load_ferplus(self) -> bool:
        if self._ferplus_net is not None:
            return True
        import cv2

        path = default_ferplus_path()
        if not os.path.exists(path):
            self._ferplus_error = f"FER+ weights not found at {path}"
            return False
        try:
            self._ferplus_net = cv2.dnn.readNetFromONNX(path)
            return True
        except Exception as e:  # pragma: no cover - env dependent
            self._ferplus_error = str(e)
            return False

    def _load(self) -> bool:
        loaded = super()._load()
        if loaded:
            self._load_ferplus()
        return loaded

    def _ferplus_dist(self, frame_bgr: np.ndarray, box: tuple) -> Dict[str, float]:
        """Run FER+ on the SSD-detected face crop with CLAHE contrast fix."""
        import cv2

        x, y, w, h = [int(v) for v in box]
        pad = max(int(0.2 * w), 8)
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1 = min(frame_bgr.shape[1], x + w + pad)
        y1 = min(frame_bgr.shape[0], y + h + pad)
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blob = (gray.astype(np.float32) / 255.0 * 2.0 - 1.0)[None, None, :, :]
        self._ferplus_net.setInput(blob)
        out = self._ferplus_net.forward()[0]
        prob = _softmax(out)

        acc = np.zeros(len(_OUR_LABELS), dtype=float)
        for fi, p in enumerate(prob):
            lab = _FERPLUS_TO_OURS[fi]
            acc[_OUR_LABELS.index(lab)] += float(p)
        s = acc.sum()
        if s <= 0:
            return None
        return {lab: float(v / s) for lab, v in zip(_OUR_LABELS, acc)}

    def analyze_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[Dict[str, float]], Optional[tuple], Optional[list], str]:
        """DeepFace first (face detection + alignment + CNN), then FER+ on the
        detected face crop; return the probability-weighted average."""
        dist, box, landmarks, status = super().analyze_frame(frame_bgr)
        if status != "success" or dist is None or box is None:
            return dist, box, landmarks, status
        if not self._ferplus_net and not self._load_ferplus():
            return dist, box, landmarks, status

        fer = self._ferplus_dist(frame_bgr, box)
        if fer is None:
            return dist, box, landmarks, status

        out = {}
        for lab in _OUR_LABELS:
            out[lab] = (
                self.w_deepface * float(dist.get(lab, 0.0))
                + self.w_ferplus * float(fer.get(lab, 0.0))
            )
        norm = sum(out.values())
        if norm <= 0:
            return dist, box, landmarks, status
        out = {lab: float(v / norm) for lab, v in out.items()}
        return out, box, landmarks, status