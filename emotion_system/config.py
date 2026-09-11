"""
Configuration for the real-time facial emotion recognition system.

All tunable values live here so that nothing is hard-coded inside
the processing pipeline. Use the environment-variable overrides only
when you need to change behaviour without editing code.
"""

import os

# ------------------------------------------------------------------
# Emotion labels (order must match the model's label space)
# ------------------------------------------------------------------
EMOTION_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
    "uncertain",
]

# ------------------------------------------------------------------
# Confidence gating
# ------------------------------------------------------------------
# Overall minimum confidence for a prediction to be accepted.
EMOTION_CONFIDENCE_THRESHOLD = float(os.environ.get(
    "EMOTION_CONFIDENCE_THRESHOLD", "0.35"
))

# Minimum margin (top1 confidence minus second-best confidence) required
# before the top emotion is trusted. Prevents near-ties from being reported
# as a confident label.
MARGIN_THRESHOLD = float(os.environ.get("MARGIN_THRESHOLD", "0.10"))

# If the face is detected but this confidence is not reached, we report
# "uncertain" (with low confidence) instead of guessing.
UNCERTAIN_LABEL = "uncertain"
UNCERTAIN_MAX_CONFIDENCE = float(os.environ.get("UNCERTAIN_MAX_CONFIDENCE", "0.30"))

# Softmax temperature for sharpening / flattening probabilities.
SOFTMAX_TEMPERATURE = float(os.environ.get("SOFTMAX_TEMPERATURE", "1.0"))

# ------------------------------------------------------------------
# Temporal smoothing / stability
# ------------------------------------------------------------------
# Number of raw predictions kept in the rolling window.
SMOOTHING_WINDOW = int(os.environ.get("SMOOTHING_WINDOW", "10"))

# Minimum fraction of the window that must agree for an emotion to be
# considered "stable". This implements the debounce, i.e. one lonely frame
# is not enough to switch the output.
STABILITY_MAJORITY = float(os.environ.get("STABILITY_MAJORITY", "0.6"))

# Hysteresis: once an emotion is committed, a new emotion must beat the
# current one by this ratio before we switch. Prevents rapid churn.
# Tuned low enough (1.08) that clear real emotion changes commit quickly.
HYSTERESIS_RATIO = float(os.environ.get("HYSTERESIS_RATIO", "1.08"))

# Weights for the moving average of per-emotion scores.
MOVING_AVG_ALPHA = float(os.environ.get("MOVING_AVG_ALPHA", "0.6"))

# ------------------------------------------------------------------
# Face detection / processing
# ------------------------------------------------------------------
# Minimum face width (in pixels) for reliable emotion reading.
MIN_FACE_SIZE = int(os.environ.get("MIN_FACE_SIZE", "64"))

# Run inference only every N frames.
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "1"))

# Minimum seconds between inferences (time-based throttle).
PREDICTION_INTERVAL = float(os.environ.get("PREDICTION_INTERVAL", "0.15"))

# ------------------------------------------------------------------
# Voice announcement ("tell me" when the emotion changes)
# ------------------------------------------------------------------
# Speak the committed emotion out loud (Windows SAPI voice).
SPEAK_EMOTIONS = os.environ.get("SPEAK_EMOTIONS", "1") == "1"

# Minimum seconds between spoken announcements.
SPEAK_INTERVAL = float(os.environ.get("SPEAK_INTERVAL", "2.0"))

# ------------------------------------------------------------------
# Quality gating
# ------------------------------------------------------------------
# Below this variance-of-Laplacian the face is considered too blurry.
# Scaled by face resolution at runtime (reference 224px face crop).
BLUR_THRESHOLD = float(os.environ.get("BLUR_THRESHOLD", "20.0"))

# Mean luminance range considered "good lighting" for the whole frame.
MIN_FRAME_LUMINANCE = float(os.environ.get("MIN_FRAME_LUMINANCE", "40.0"))
MAX_FRAME_LUMINANCE = float(os.environ.get("MAX_FRAME_LUMINANCE", "230.0"))

# Face mean luminance range.
MIN_FACE_LUMINANCE = float(os.environ.get("MIN_FACE_LUMINANCE", "30.0"))
MAX_FACE_LUMINANCE = float(os.environ.get("MAX_FACE_LUMINANCE", "240.0"))

# ------------------------------------------------------------------
# Head pose
# ------------------------------------------------------------------
# Absolute head-pose angles (degrees) beyond which the face is considered
# too side-facing to read.
MAX_YAW_DEG = float(os.environ.get("MAX_YAW_DEG", "45.0"))
MAX_PITCH_DEG = float(os.environ.get("MAX_PITCH_DEG", "35.0"))
MAX_ROLL_DEG = float(os.environ.get("MAX_ROLL_DEG", "30.0"))

# ------------------------------------------------------------------
# Camera / webcam
# ------------------------------------------------------------------
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "1"))

# Seconds without a single detected face on the active camera before the app
# auto-switches to the next camera index (guards against wrong-index drift).
CAMERA_RENEGOTIATION_INTERVAL = float(os.environ.get(
    "CAMERA_RENEGOTIATION_INTERVAL", "10.0"
))

# OpenCV capture backend to try first. Options: "auto", "dshow", "msmf",
# "ffmpeg". The app probes every candidate backend across several camera
# indices at startup, so this only controls the preferred order.
CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND", "dshow").lower()

# How many camera indices to probe at startup before giving up.
CAMERA_PROBE_RANGE = int(os.environ.get("CAMERA_PROBE_RANGE", "4"))

# After this many consecutive failed cap.read() calls the app reports the
# camera as dead instead of silently looping.
CONSECUTIVE_READ_FAILURE_LIMIT = int(os.environ.get(
    "CONSECUTIVE_READ_FAILURE_LIMIT", "20"
))

# Render debug overlay (face box, landmarks, head pose, FPS, quality).
DEBUG_MODE = bool(os.environ.get("DEBUG_MODE", "1") == "1")

# ------------------------------------------------------------------
# Model backend
# ------------------------------------------------------------------
# "ensemble" (default) averages two independent pre-trained emotion models
# (DeepFace FER2013 CNN + FER+ ONNX) so no single model's webcam bias can
# dominate. "hsemotion" uses the pre-trained eNet-B0 (AffectNet) model for
# the emotion step (DeepFace does the detection; falls back to DeepFace
# scores if unavailable). "deepface" uses DeepFace alone. Override with
# EMOTION_BACKEND.
BACKEND = os.environ.get("EMOTION_BACKEND", "ensemble")

# ------------------------------------------------------------------
# HSEmotion / eNet-B0 (AffectNet) pre-trained model
# ------------------------------------------------------------------
# Model downloaded once into ~/.hsemotion/ (first run needs internet).
HSEMOTION_MODEL = os.environ.get("HSEMOTION_MODEL", "enet_b0_8_best_vgaf")

# Custom path override for the ONNX weights (skips auto-download).
HSEMOTION_MODEL_PATH = os.environ.get("HSEMOTION_MODEL_PATH")

# Apply light CLAHE contrast enhancement to the face crop before inference
# (helps washed-out webcam auto-exposure).
HSEMOTION_CLAHE = os.environ.get("HSEMOTION_CLAHE", "1") == "1"

# Weight of each pre-trained model in the ensemble (must sum to ~1).
ENSEMBLE_DEEPFACE_WEIGHT = float(os.environ.get(
    "ENSEMBLE_DEEPFACE_WEIGHT", "0.6"
))
ENSEMBLE_FERPLUS_WEIGHT = float(os.environ.get(
    "ENSEMBLE_FERPLUS_WEIGHT", "0.4"
))

# Face detector used by DeepFace to find faces BEFORE emotion inference.
# "opencv" (Haar cascade) is DeepFace's default but is very strict on live
# webcam frames (minNeighbors=10) and frequently fails to find a face that
# the user can clearly see. "ssd" (ResNet-10 SSD, weights auto-downloaded to
# ~/.deepface/weights) is far more reliable on webcams. Falls back to
# "opencv" automatically if the SSD weights cannot be loaded.
DETECTOR_BACKEND = os.environ.get("DETECTOR_BACKEND", "ssd").lower()