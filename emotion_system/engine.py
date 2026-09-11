"""
Emotion engine: orchestrates the full pipeline.

    face detection -> face quality -> alignment/preprocessing
    -> emotion inference -> confidence filtering -> temporal smoothing
    -> final prediction

The engine is deliberately backend-agnostic so the rest of the code never
has to know whether DeepFace or a landmark-feature model is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from . import config
from . import emotion_utils as eu
from .deepface_backend import get_backend
from .face_quality import compute_face_quality
from .temporal_smoothing import TemporalEmotionSmoother


@dataclass
class Prediction:
    emotion: str = config.UNCERTAIN_LABEL
    confidence: float = 0.0
    stable: bool = False
    face_detected: bool = False
    status: str = "no_frame"
    quality_score: float = 0.0
    quality_reasons: List[str] = field(default_factory=list)
    headpose: Dict[str, float] = field(default_factory=dict)
    raw_emotion: Optional[str] = None
    raw_confidence: float = 0.0
    all_emotions: Dict[str, float] = field(default_factory=dict)
    window: List[str] = field(default_factory=list)
    face_box: Optional[tuple] = None
    landmark_count: int = 0

    def to_dict(self) -> dict:
        return {
            "emotion": self.emotion,
            "confidence": round(float(self.confidence), 4),
            "stable": bool(self.stable),
            "face_detected": bool(self.face_detected),
            "status": self.status,
            "raw_emotion": self.raw_emotion,
            "raw_confidence": round(self.raw_confidence, 4),
            "quality_score": round(self.quality_score, 4),
            "quality_reasons": list(self.quality_reasons),
            "headpose": dict(self.headpose),
            "all_emotions": {k: round(v, 4) for k, v in self.all_emotions.items()},
            "window": list(self.window),
        }

    def __repr__(self):
        return (f"Prediction(emotion={self.emotion}, confidence={self.confidence:.2f}, "
                f"stable={self.stable}, face={self.face_detected}, "
                f"status={self.status})")


class EmotionEngine:
    def __init__(
        self,
        backend: str = None,
        smoother: Optional[TemporalEmotionSmoother] = None,
        track_faces: bool = True,
    ):
        self.backend = get_backend(backend or config.BACKEND)
        self.smoother = smoother or TemporalEmotionSmoother()
        self.track_faces = track_faces
        self._last_quality = None
        self._last_prediction: Optional[Prediction] = None

    def reset(self) -> None:
        self.smoother.reset()

    def last_prediction(self) -> Optional[Prediction]:
        return self._last_prediction

    def _record(self, prediction: Prediction) -> Prediction:
        """Remember the last outcome so the UI can always explain WHY.
        Every outcome (no_face, error, poor_quality, success, ...) is stored
        so that skipped display frames show the real reason instead of the
        misleading blank 'skipped' state."""
        self._last_prediction = prediction
        return prediction

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> Prediction:
        """
        Full single-frame pipeline. `frame_bgr` is the raw webcam frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return self._record(
                Prediction(status="no_frame", emotion=config.UNCERTAIN_LABEL)
            )

        gray = _to_gray(frame_bgr)

        dist, box, landmarks, status = self.backend.analyze_frame(frame_bgr)

        if status == "model_unavailable":
            return self._record(
                Prediction(status="model_unavailable", emotion=config.UNCERTAIN_LABEL)
            )
        if status == "error":
            return self._record(
                Prediction(status="error", emotion=config.UNCERTAIN_LABEL)
            )
        if status == "no_face" or dist is None:
            return self._record(
                Prediction(
                    status="no_face",
                    emotion=config.UNCERTAIN_LABEL,
                    face_detected=False,
                )
            )
        if status == "multiple_faces":
            return self._record(
                Prediction(
                    status="multiple_faces",
                    emotion=config.UNCERTAIN_LABEL,
                    face_detected=True,
                )
            )

        # Face is detected; evaluate quality.
        quality = compute_face_quality(gray, box, landmarks)
        self._last_quality = quality

        if not quality.ok:
            # Face present but unreadable: feed no evidence into the smoother
            # so the committed label persists; report "uncertain" now.
            res = self.smoother.update({})
            prediction = Prediction(
                status="poor_quality",
                emotion=config.UNCERTAIN_LABEL,
                face_detected=True,
                confidence=0.0,
                stable=self.smoother.stable,
                quality_score=quality.quality_score,
                quality_reasons=quality.reasons,
                headpose=quality.headpose,
                face_box=box,
                landmark_count=len(landmarks) if landmarks else 0,
                window=res["window"],
            )
            return self._record(prediction)

        # DeepFace already internally detects + aligns the face for its
        # emotion model, so we trust its single full-frame inference. We keep
        # alignment only as auxiliary info (head pose, display), not as a
        # second inference input - this avoids doubling CPU cost per frame
        # and avoids accuracy regressions from upsampled crops.
        top1 = eu.top_emotion(dist)
        raw_conf = eu.confidence(dist)
        mar = eu.margin(dist)
        ent = eu.entropy(dist)

        # Temporal smoothing acts on the (weak or strong) evidence. The
        # confidence gate lives inside the smoother: it will never commit a
        # low-confidence emotion, it only reports "uncertain".
        res = self.smoother.update(dist)

        # Reliability penalty: a strongly rotated face is harder to read.
        # We reduce the reported confidence (soft gate) rather than fully
        # vetoing it - evidence still counts, just less.
        penalty = confidence_from_headpose(quality.headpose)
        confidence_out = float(np.clip(res["confidence"] * penalty, 0.0, 1.0))
        emotion_out = res["emotion"]
        if confidence_out < config.EMOTION_CONFIDENCE_THRESHOLD:
            emotion_out = config.UNCERTAIN_LABEL

        prediction = Prediction(
            emotion=emotion_out,
            confidence=confidence_out,
            stable=res["stable"],
            face_detected=True,
            status="success",
            quality_score=quality.quality_score,
            quality_reasons=quality.reasons,
            headpose=quality.headpose,
            raw_emotion=top1,
            raw_confidence=raw_conf,
            all_emotions=dict(dist),
            window=res["window"],
            face_box=box,
            landmark_count=len(landmarks) if landmarks else 0,
        )
        prediction.raw_margin = mar
        prediction.raw_entropy = ent
        return self._record(prediction)


def _to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def confidence_from_headpose(headpose: dict) -> float:
    """
    Soft reliability multiplier in [0.5, 1.0] from head-pose angles. A
    perfectly frontal face keeps full confidence; a strongly rotated face is
    discounted but never completely vetoed on pose alone.
    """
    yaw = abs(float(headpose.get("yaw", 0.0)))
    roll = abs(float(headpose.get("roll", 0.0)))
    # Normalize yaw by 90 deg and roll by 45 deg, then combine.
    penalty = max(
        yaw / 90.0,
        roll / 45.0,
    )
    return float(np.clip(1.0 - penalty * 0.5, 0.5, 1.0))