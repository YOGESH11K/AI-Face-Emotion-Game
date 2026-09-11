"""
Voice announcements for detected emotions using Windows SAPI (no extra
packages). Speech runs in a detached PowerShell process so it never blocks
the webcam loop, and repeats are debounced so the app does not chatter.
"""

from __future__ import annotations

import base64
import subprocess
import time

from . import config

# Hides the PowerShell console window flash when a voice is fired.
CREATE_NO_WINDOW = 0x08000000


def _powershell_encoded(script: str) -> str:
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand {encoded}"


def _speak_script(text: str) -> str:
    safe = text.replace("'", "''")
    return (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak('{safe}'); "
        "$s.Dispose()"
    )


class Speaker:
    """Debounced, non-blocking TTS for emotion changes."""

    def __init__(self, enabled: bool = None, min_interval: float = None) -> None:
        self.enabled = bool(enabled if enabled is not None else config.SPEAK_EMOTIONS)
        self.min_interval = float(
            min_interval if min_interval is not None else config.SPEAK_INTERVAL
        )
        self._last_text: str | None = None
        self._last_spoke_at: float = 0.0

    def speak(self, text: str) -> bool:
        """Announce text unless it is a repeat or too soon after the last one.
        Never raises; returns True if a voice process was started."""
        if not self.enabled:
            return False
        now = time.time()
        if text == self._last_text or (now - self._last_spoke_at) < self.min_interval:
            return False
        self._last_text = text
        self._last_spoke_at = now
        try:
            cmd = _powershell_encoded(_speak_script(text))
            subprocess.Popen(
                ["cmd", "/c", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
            return True
        except Exception:  # pragma: no cover - defensive
            return False


PHRASES = {
    "happy": "You look happy!",
    "sad": "You look sad.",
    "angry": "You look angry.",
    "surprise": "You look surprised!",
    "fear": "You look scared.",
    "disgust": "You look disgusted.",
    "neutral": "You look neutral.",
}