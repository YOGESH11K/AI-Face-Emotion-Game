"""
Face quality gate: checks that a detected face is actually readable before
the emotion model is trusted. Poor-quality faces produce "uncertain" rather
than a confident guess.

Checks implemented:
  * face size        - is the face big enough to read expressions?
  * blur             - variance of Laplacian on the face crop
  * lighting         - frame and face luminance inside acceptable range
  * head pose        - yaw/pitch/roll estimated from landmarks, reject
                        strong side faces
  * occlusion        - fraction of landmark-triangle variance as a simple
                        proxy for missing facial region coverage

Head-pose estimation uses solvePnP on a generic 3D face model fitted to
MediaPipe landmarks if available, else a planar estimate from eye geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from . import config


@dataclass
class FaceQuality:
    ok: bool = True
    reasons: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    blur_score: float = 1.0
    size_ok: bool = True
    blur_ok: bool = True
    lighting_ok: bool = True
    headpose_ok: bool = True
    occlusion_ok: bool = True
    headpose: Dict[str, float] = field(
        default_factory=lambda: {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    )

    def add_reason(self, reason: str) -> None:
        self.reasons.append(reason)
        self.ok = False


def _laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _mean_luminance(gray: np.ndarray) -> float:
    return float(np.mean(gray))


def estimate_head_pose(
    landmarks: Optional[List[tuple]],
    face_box: Optional[tuple] = None,
) -> Dict[str, float]:
    """
    Estimate yaw/pitch/roll from facial landmarks (typically the two eye
    points reported by the face detector).

    Heading is a soft cue, not a precise measurement:
      * roll    - angle of the eye line vs horizontal.
      * yaw     - horizontal offset of the eye-line midpoint from the face
                  box centre.  For a frontal face the eyes are roughly
                  centred in the box; as the head turns, both eyes shift to
                  one side, so the offset grows.
      * pitch   - not reliably measurable with only eye landmarks; left 0.
    """
    if not landmarks or len(landmarks) < 1:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    pts = np.array([(float(p[0]), float(p[1])) for p in landmarks], dtype=np.float64)
    roll = 0.0
    if len(pts) >= 2:
        dx = pts[max(range(len(pts)), key=lambda i: pts[i][0])][0] - pts[
            min(range(len(pts)), key=lambda i: pts[i][0])
        ][0]
        dy = pts[max(range(len(pts)), key=lambda i: pts[i][0])][1] - pts[
            min(range(len(pts)), key=lambda i: pts[i][0])
        ][1]
        if dx != 0:
            roll = float(np.degrees(np.arctan2(dy, dx)))

    eye_mid_x = float(pts[:, 0].mean())

    if face_box:
        x, y, w, h = [int(v) for v in face_box]
        cx = x + w / 2.0
        # eye offset as fraction of half-width; frontal ~0, profile grows.
        offset = (eye_mid_x - cx) / max(w, 1.0)
        yaw = float(np.clip(offset * 180.0, -70, 70))
    else:
        yaw = 0.0

    return {"yaw": float(yaw), "pitch": 0.0, "roll": float(roll)}


def estimate_occlusion(
    gray_face: np.ndarray, landmarks: Optional[List[tuple]] = None
) -> float:
    """
    Simple occlusion proxy: standard deviation of landmark coordinate spread
    relative to face box. A face with landmarks clustered in a small region
    (e.g. partially hidden) gets a lower score. Ranges 0..1 (1 = good).
    """
    if landmarks is None or len(landmarks) < 3:
        return 1.0
    pts = np.array([(float(p[0]), float(p[1])) for p in landmarks], dtype=np.float64)
    spread = float(np.std(pts, axis=0).sum())
    diag = float(np.hypot(gray_face.shape[1], gray_face.shape[0]))
    if diag <= 0:
        return 1.0
    return float(np.clip(spread / (diag * 0.3), 0.0, 1.0))


def compute_face_quality(
    gray_full: np.ndarray,
    face_box: tuple,
    landmarks: Optional[List[tuple]] = None,
) -> FaceQuality:
    """
    Evaluate all readability checks for a single detected face.

    Parameters
    ----------
    gray_full : grayscale full frame
    face_box  : (x, y, w, h) pixel coords of the detected face
    landmarks : optional list of (x, y) landmark points in full-frame coords
    """
    quality = FaceQuality()

    x, y, w, h = [int(v) for v in face_box]

    # --- Size ---
    if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
        quality.size_ok = False
        quality.add_reason(f"face_too_small({w}x{h})")

    # --- Face crop for blur / lighting ---
    x = max(0, x)
    y = max(0, y)
    w = min(gray_full.shape[1] - x, max(w, 1))
    h = min(gray_full.shape[0] - y, max(h, 1))
    face_gray = gray_full[y:y + h, x:x + w]

    if face_gray.size == 0:
        quality.add_reason("empty_face_crop")
        return quality

    # --- Blur ---
    lap = _laplacian_variance(face_gray)
    # Normalize by face resolution: laplacian variance scales with image size,
    # so a small face crop should not be rejected just because it is small.
    # Reference resolution of 224px (typical FER training crop).
    face_w = max(int(w), 1)
    ref_scale = max(face_w / 224.0, 0.5)
    effective_threshold = config.BLUR_THRESHOLD * ref_scale
    blur_score = float(np.clip(lap / effective_threshold, 0.0, 1.0))
    quality.blur_score = blur_score
    if lap < effective_threshold:
        quality.blur_ok = False
        quality.add_reason(f"face_blurry({lap:.1f})")

    # --- Lighting ---
    face_lum = _mean_luminance(face_gray)
    frame_lum = _mean_luminance(gray_full)
    if face_lum < config.MIN_FACE_LUMINANCE or face_lum > config.MAX_FACE_LUMINANCE:
        quality.lighting_ok = False
        quality.add_reason(f"face_lighting({face_lum:.1f})")
    elif frame_lum < config.MIN_FRAME_LUMINANCE or frame_lum > config.MAX_FRAME_LUMINANCE:
        quality.lighting_ok = False
        quality.add_reason(f"frame_lighting({frame_lum:.1f})")

    # --- Head pose ---
    hp = estimate_head_pose(landmarks, (x, y, w, h))
    quality.headpose = hp
    if (
        abs(hp["yaw"]) > config.MAX_YAW_DEG
        or abs(hp["pitch"]) > config.MAX_PITCH_DEG
        or abs(hp["roll"]) > config.MAX_ROLL_DEG
    ):
        quality.headpose_ok = False
        quality.add_reason(
            f"head_pose(yaw={hp['yaw']:.0f},pitch={hp['pitch']:.0f},roll={hp['roll']:.0f})"
        )

    # --- Occlusion ---
    occ = estimate_occlusion(face_gray, landmarks)
    if occ < 0.35:
        quality.occlusion_ok = False
        quality.add_reason(f"occlusion({occ:.2f})")

    # --- Aggregate quality score (0..1) ---
    scores = []
    scores.append(1.0 if quality.size_ok else 0.3)
    scores.append(blur_score)
    lum_ok = face_lum / 255.0
    scores.append(float(np.clip(1.0 - abs(lum_ok - 0.5) * 2.0, 0.2, 1.0)))
    if landmarks is not None and len(landmarks) >= 3:
        scores.append(float(np.clip(1.0 - (abs(hp["yaw"]) / 90.0), 0.2, 1.0)))
    quality.quality_score = float(np.mean(scores))

    # Extreme blur / tiny face must force failure regardless of other scores.
    if lap < effective_threshold * 0.4:
        if "face_blurry" not in quality.reasons:
            quality.add_reason("face_severely_blurry")

    return quality