"""
Emotion math utilities: probability normalization, confidence measures,
margin, entropy and label mapping.

DeepFace's raw "emotion" dict is NOT a normalized probability distribution
(values are the model outputs and do not sum to 1.0). We always convert to a
proper softmax distribution before using it, so confidence numbers are
comparable across faces and frames.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from . import config

# Canonical short label -> model label space.
# "uncertain" is our decision label, not a model output.
MODEL_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


def softmax(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """Convert arbitrary model scores into a normalized probability dict."""
    if not scores:
        return {}
    keys = list(scores.keys())
    t = temperature if temperature and temperature > 0 else 1.0
    values = [float(scores[k]) / t for k in keys]
    m = max(values)
    exps = [math.exp(v - m) for v in values]  # numerically stable
    total = sum(exps)
    if total <= 0:
        return {k: 1.0 / len(keys) for k in keys}
    return {k: e / total for k, e in zip(keys, exps)}


def normalize_distribution(dist: Dict[str, float]) -> Dict[str, float]:
    """Ensure a dict of scores is a valid, normalized probability dict."""
    if not dist:
        return {}
    total = sum(float(v) for v in dist.values())
    if total <= 0:
        return {k: 1.0 / len(dist) for k in dist}
    return {k: float(v) / total for k, v in dist.items()}


def top_emotion(dist: Dict[str, float]) -> Optional[str]:
    """Return argmax label."""
    if not dist:
        return None
    return max(dist, key=dist.get)


def top_emotions(dist: Dict[str, float], n: int = 2) -> List[tuple]:
    """Return top-n (label, prob) sorted descending."""
    if not dist:
        return []
    return sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:n]


def margin(dist: Dict[str, float]) -> float:
    """Top1 probability minus second-best probability (0 if <2 classes)."""
    if not dist or len(dist) < 2:
        return 0.0
    sorted_v = sorted(dist.values(), reverse=True)
    return sorted_v[0] - sorted_v[1]


def entropy(dist: Dict[str, float]) -> float:
    """Shannon entropy (bits). High entropy -> ambiguous distribution."""
    if not dist:
        return 0.0
    h = 0.0
    for p in dist.values():
        p = max(min(p, 1.0), 0.0)
        if p > 0:
            h -= p * math.log2(p)
    return h


def confidence(dist: Dict[str, float], temperature: float = 1.0) -> float:
    """
    Calibrated confidence for the dominant emotion.

    Combines the top-1 probability with distribution "certainty" (how far the
    distribution is from uniform). A peaked distribution gets a confidence
    near its top-1 probability; a flat-ish distribution gets pulled down so we
    never report inflated confidence on ambiguous faces.
    """
    if not dist:
        return 0.0
    top1 = max(dist.values())
    ent = entropy(dist)
    max_ent = math.log2(len(dist)) if len(dist) > 1 else 1.0
    certainty = 1.0 - (ent / max_ent) if max_ent > 0 else 1.0
    raw = top1 * (0.5 + 0.5 * certainty)
    return float(max(0.0, min(1.0, raw)))


def to_labels_enum() -> List[str]:
    return list(MODEL_LABELS)


def describe(dist: Dict[str, float]) -> str:
    """Short human-readable summary (for debug overlay)."""
    if not dist:
        return "n/a"
    parts = []
    for label, prob in sorted(dist.items(), key=lambda kv: -kv[1])[:3]:
        parts.append(f"{label}:{prob:.2f}")
    return " ".join(parts)


def is_uncertain_label(label: str) -> bool:
    return label == config.UNCERTAIN_LABEL