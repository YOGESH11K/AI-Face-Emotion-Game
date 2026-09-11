"""
Real-time facial emotion recognition system.

Pipeline:
    face detection -> face quality -> alignment/preprocessing
    -> emotion inference -> confidence filtering -> temporal smoothing
    -> final prediction

This package contains the recognition core only. The webcam loop, debug
overlay and still-image tester consume this package.
"""

from . import config
from . import emotion_utils
from . import face_quality
from . import temporal_smoothing
from . import deepface_backend
from .engine import EmotionEngine, Prediction
from .temporal_smoothing import TemporalEmotionSmoother

__all__ = [
    "config",
    "emotion_utils",
    "face_quality",
    "temporal_smoothing",
    "deepface_backend",
    "EmotionEngine",
    "Prediction",
    "TemporalEmotionSmoother",
]