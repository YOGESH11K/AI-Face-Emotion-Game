"""
Temporal smoothing and stability.

A single frame must never change the reported emotion.  We keep a rolling
window of raw per-emotion distributions and:
  1. Compute a moving-average distribution over the window.
  2. Apply hysteresis so an emotion only changes when a new candidate
     becomes clearly stronger than the currently committed emotion.
  3. Report "uncertain" when the window is not yet warm, or when the
     best candidate's confidence is below the threshold.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from . import config
from . import emotion_utils as eu


class TemporalEmotionSmoother:
    def __init__(
        self,
        window_size: int = None,
        majority: float = None,
        hysteresis_ratio: float = None,
    ) -> None:
        self.window_size = int(window_size or config.SMOOTHING_WINDOW)
        self.majority = float(majority if majority is not None else config.STABILITY_MAJORITY)
        self.hysteresis_ratio = float(
            hysteresis_ratio if hysteresis_ratio is not None else config.HYSTERESIS_RATIO
        )
        self.window: Deque[Dict[str, float]] = deque(maxlen=self.window_size)
        self.labels: Deque[str] = deque(maxlen=self.window_size)
        self.committed: Optional[str] = None
        self.committed_score: float = 0.0
        self.stable: bool = False

    def reset(self) -> None:
        self.window.clear()
        self.labels.clear()
        self.committed = None
        self.committed_score = 0.0
        self.stable = False

    def _mean_distribution(self) -> Dict[str, float]:
        """Moving average of all distributions in the window."""
        if not self.window:
            return {}
        acc: Dict[str, float] = {}
        for dist in self.window:
            for label, p in dist.items():
                acc[label] = acc.get(label, 0.0) + p
        n = len(self.window)
        return {k: v / n for k, v in acc.items()}

    def _mode_label(self) -> Optional[str]:
        """Label that appears most often in the window."""
        if not self.labels:
            return None
        counts: Dict[str, int] = {}
        for lbl in self.labels:
            counts[lbl] = counts.get(lbl, 0) + 1
        best = max(counts, key=counts.get)
        frac = counts[best] / len(self.labels)
        if frac >= self.majority:
            return best
        return None

    def _entropy_weighted_candidate(self) -> Tuple[Optional[str], float]:
        """Candidate label + score from the averaged distribution."""
        mean = self._mean_distribution()
        if not mean:
            return None, 0.0
        label = eu.top_emotion(mean)
        score = eu.confidence(mean)
        return label, score

    def update(self, raw_dist: Dict[str, float]) -> Dict[str, object]:
        """
        Feed one raw distribution (softmax probabilities per emotion label)
        and get the stable, de-bounced output dict.

        Returns:
            {
              "emotion": str,
              "confidence": float,
              "stable": bool,
              "face_detected": bool,
              "raw_emotion": str,
              "raw_confidence": float,
              "window": [...labels],
            }
        """
        if not raw_dist:
            return {
                "emotion": config.UNCERTAIN_LABEL,
                "confidence": 0.0,
                "stable": False,
                "face_detected": False,
                "raw_emotion": None,
                "raw_confidence": 0.0,
                "window": list(self.labels),
            }

        label = eu.top_emotion(raw_dist)
        score = eu.confidence(raw_dist)

        self.window.append(dict(raw_dist))
        self.labels.append(label)

        # Mode + averaged candidate.
        mode_label = self._mode_label()
        cand_label, cand_score = self._entropy_weighted_candidate()

        # Decide transition using hysteresis.
        if self.committed is None:
            new_label = cand_label
        else:
            if cand_label is None:
                new_label = self.committed
            elif cand_label == self.committed:
                new_label = self.committed
            else:
                # Only switch when the new candidate beats the committed
                # emotion by the hysteresis ratio AND matches the window mode.
                if mode_label == cand_label and cand_score >= (
                    self.committed_score * self.hysteresis_ratio
                ):
                    new_label = cand_label
                else:
                    new_label = self.committed

        if new_label is None:
            new_label = config.UNCERTAIN_LABEL
            conf = 0.0
        else:
            conf = cand_score if cand_label == new_label else self.committed_score

        # Confidence gate: never commit a low-confidence emotion.
        if new_label != config.UNCERTAIN_LABEL and conf < config.EMOTION_CONFIDENCE_THRESHOLD:
            new_label = config.UNCERTAIN_LABEL
            # Keep the actual confidence so "uncertain" carries the honest value.
            conf = max(conf, 0.0)

        self.committed = new_label
        self.committed_score = conf
        self.stable = mode_label is not None and mode_label == new_label

        return {
            "emotion": new_label,
            "confidence": round(float(conf), 4),
            "stable": bool(self.stable),
            "face_detected": True,
            "raw_emotion": label,
            "raw_confidence": round(float(score), 4),
            "window": list(self.labels),
        }