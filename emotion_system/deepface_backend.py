"""
DeepFace emotion backend.

Wraps DeepFace.analyze for emotion-only inference and normalizes its output
into a clean per-emotion probability distribution. Also handles face
alignment (rotation normalization) and returns the detected face region.

DeepFace raw output is *not* a probability distribution (per-class scores do
not sum to 1); we always softmax-normalize it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config
from . import emotion_utils as eu


class DeepFaceEmotionBackend:
    """Lazy-loaded DeepFace wrapper. Keep model loaded across frames."""

    def __init__(self):
        self._deepface = None
        self._loaded = False
        self._load_error: Optional[str] = None
        self._detector_backend: str = config.DETECTOR_BACKEND
        if self._detector_backend not in ("opencv", "ssd", "mtcnn", "retinaface", "dlib", "mediapipe"):
            self._detector_backend = "opencv"

    @property
    def detector_backend(self) -> str:
        return self._detector_backend

    def _load(self) -> bool:
        if self._loaded:
            return True
        try:
            from deepface import DeepFace  # type: ignore

            self._deepface = DeepFace
            self._loaded = True
            return True
        except Exception as e:  # pragma: no cover - env dependent
            self._load_error = str(e)
            return False

    def analyze_frame(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[Optional[Dict[str, float]], Optional[tuple], Optional[list], str]:
        """
        Run emotion analysis on a BGR frame.

        Returns
        -------
        (dist, face_box, landmarks, status)
          dist       : normalized probability dict or None
          face_box   : (x, y, w, h) or None
          landmarks  : list of (x, y) or None
          status     : "success" | "no_face" | "model_unavailable" | "error"
        """
        if not self._load():
            return None, None, None, "model_unavailable"

        try:
            # DeepFace requires an RGB or valid numpy array; BGR->RGB for safety.
            rgb = cv2_bgr_to_rgb(frame_bgr)
            attempt_result = None
            last_msg = ""
            # DeepFace detection is occasionally nondeterministic and fails
            # transiently ('NoneType not iterable'). Retry a couple of times
            # so a single flaky frame does not drop the window.
            for attempt in range(3):
                try:
                    attempt_result = self._deepface.analyze(
                        img_path=rgb,
                        actions=["emotion"],
                        detector_backend=self._detector_backend,
                        enforce_detection=True,
                        silent=True,
                    )
                    break
                except Exception as e:
                    last_msg = str(e)
                    if "Face could not be detected" in last_msg or "FaceNotDetected" in type(e).__name__:
                        return None, None, None, "no_face"
                    # If the configured detector (e.g. SSD) could not load its
                    # weights, fall back to the built-in Haar cascade so the
                    # app keeps working offline.
                    if self._detector_backend != "opencv" and (
                        "download" in last_msg.lower() or "weights" in last_msg.lower()
                    ):
                        self._detector_backend = "opencv"
                        last_msg = ""
                        continue
                    continue
            else:
                self.last_error = last_msg
                return None, None, None, "error"
            result = attempt_result

            if isinstance(result, list):
                if not result:
                    return None, None, None, "no_face"
                if len(result) > 1:
                    # More than one face: do not guess which to read.
                    return None, None, None, "multiple_faces"
                result = result[0]

            raw = result.get("emotion", {})
            dist = eu.softmax(raw, temperature=config.SOFTMAX_TEMPERATURE)

            region = result.get("region") or {}
            box = None
            if "x" in region and "y" in region and "w" in region and "h" in region:
                box = (int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"]))

            # DeepFace gives left_eye/right_eye in region; treat as pseudo
            # landmarks for alignment (not a full 68-point mesh). Eyes can be
            # None on some frames - guard against it.
            landmarks = None
            if "left_eye" in region and "right_eye" in region:
                le = region["left_eye"]
                re = region["right_eye"]
                if le is not None and re is not None:
                    landmarks = [
                        (float(le[0]), float(le[1])),
                        (float(re[0]), float(re[1])),
                    ]

            return dist, box, landmarks, "success"
        except Exception as e:  # FaceNotDetected & friends
            msg = str(e)
            self.last_error = msg
            if "Face could not be detected" in msg or "FaceNotDetected" in type(e).__name__:
                return None, None, None, "no_face"
            return None, None, None, "error"


def cv2_bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def get_backend(name: str = None):
    name = (name or config.BACKEND).lower()
    if name == "deepface":
        return DeepFaceEmotionBackend()
    if name == "ensemble":
        from .ensemble_backend import EnsembleEmotionBackend

        return EnsembleEmotionBackend()
    if name == "hsemotion":
        from .hsemotion_backend import HSEmotionEmotionBackend

        return HSEmotionEmotionBackend()
    raise ValueError(f"Unknown emotion backend: {name}")