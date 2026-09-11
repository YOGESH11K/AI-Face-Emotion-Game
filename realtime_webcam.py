"""
Real-time webcam emotion recognition with optional debug overlay.

Usage:
    py -3.12 realtime_webcam.py                           # run with defaults
    py -3.12 realtime_webcam.py --camera 1                # different camera index
    py -3.12 realtime_webcam.py --list-cameras            # probe and list cameras
    py -3.12 realtime_webcam.py --max-seconds 30 --frames 100   # auto-stop (testing)

Keys:
    q / Esc   quit
    d         toggle debug overlay
    r         reset temporal smoother

All camera diagnostics and prediction status changes are logged to
`realtime_webcam.log` next to this file.
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import cv2

from emotion_system import config
from emotion_system.engine import EmotionEngine, Prediction
from emotion_system.speech import Speaker, PHRASES

EMOTION_COLORS = {
    "happy": (0, 200, 0),
    "sad": (255, 120, 0),
    "angry": (0, 0, 255),
    "surprise": (0, 200, 255),
    "fear": (180, 0, 180),
    "disgust": (0, 150, 120),
    "neutral": (200, 200, 200),
    "uncertain": (0, 165, 255),
}

# STEM setup ----------------------------------------------------------------
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "realtime_webcam.log")
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("emotion_realtime")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter(LOG_FORMAT)
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = setup_logger()

# Camera backends -----------------------------------------------------------
_BACKENDS = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "ffmpeg": cv2.CAP_FFMPEG,
}


def _backend_name(mode: str) -> str:
    mapping = {v: k for k, v in _BACKENDS.items()}
    return mapping.get(mode, str(mode))


def _pick_backend_order(preferred: str):
    """Return backend candidates, preferred first, deduplicated."""
    order = [preferred, "auto", "dshow", "msmf", "ffmpeg"]
    seen, out = set(), []
    for name in order:
        if name in _BACKENDS and name not in seen:
            seen.add(name)
            out.append((name, _BACKENDS[name]))
    return out


def _frame_has_content(frame) -> bool:
    """A camera that "opens" but only returns blank frames (all-black or flat
    colour) is useless - typically caused by Windows privacy blocking or a
    driver/light failure. Return False for those so the app can skip the
    device and try the next one."""
    try:
        import numpy as np
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        return not (mean < 8.0 and std < 4.0)
    except Exception:  # pragma: no cover - defensive
        return True


def _try_capture(index: int, backend: str, wait_seconds: float = 3.0):
    """Open index+backend, verify isOpened() and that a real frame can be
    read. Returns (cap, frame) on success or (None, None). Caller owns cap."""
    cap = None
    try:
        cap = cv2.VideoCapture(index, _BACKENDS[backend])
        if not cap.isOpened():
            log.debug(f"  index={index} backend={backend}: failed to open")
            cap.release()
            return None, None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        deadline = time.time() + wait_seconds
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                if not _frame_has_content(frame):
                    log.warning(
                        f"  index={index} backend={backend}: opened but delivers "
                        f"BLANK frames (mean={int(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())}) - "
                        f"skipping (check Windows camera privacy settings)"
                    )
                    cap.release()
                    return None, None
                log.info(
                    f"  index={index} backend={backend}: opened, "
                    f"frame={frame.shape[1]}x{frame.shape[0]}"
                )
                return cap, frame
            time.sleep(0.05)
        log.debug(f"  index={index} backend={backend}: opened but no readable frame")
        cap.release()
        return None, None
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(f"  index={index} backend={backend}: error: {type(exc).__name__}: {exc}")
        if cap is not None:
            cap.release()
        return None, None


def open_camera(preferred_index: int, restrict_to: int | None = None) -> tuple:
    """Robust camera init: probe indices and backends until a camera that can
    actually deliver a frame is found.

    restrict_to: only this exact index is tried (used for auto-renegotiation
    after a camera index fails to produce faces).

    Returns (cap, info_dict) on success, or (None, info_dict) after every
    candidate has been exhausted. Never raises."""
    info = {"tried": [], "preferred_index": preferred_index, "winner": None}
    order = _pick_backend_order(config.CAMERA_BACKEND)

    if restrict_to is not None:
        indices = [restrict_to]
    else:
        # Try the preferred index first (all backends), then the rest.
        indices = [preferred_index]
        for i in range(config.CAMERA_PROBE_RANGE):
            if i not in indices:
                indices.append(i)

    for idx in indices:
        for name, _ in order:
            log.debug(f"probing camera index={idx} backend={name}")
            cap, frame = _try_capture(idx, name)
            info["tried"].append({"index": idx, "backend": name, "ok": cap is not None})
            if cap is not None:
                info["winner"] = {"index": idx, "backend": name}
                info["width"] = int(frame.shape[1])
                info["height"] = int(frame.shape[0])
                return cap, info
            time.sleep(0.2)  # give a slow camera time to release

    return None, info


def list_cameras() -> None:
    print(f"\nOpenCV {cv2.__version__} - probing camera indices/backends")
    for idx in range(config.CAMERA_PROBE_RANGE):
        row = []
        for name, _ in _pick_backend_order("auto"):
            cap, frame = _try_capture(idx, name, wait_seconds=2.0)
            if cap is not None:
                row.append(f"{name}(width={frame.shape[1]},height={frame.shape[0]})")
            else:
                row.append(f"{name}(-)")
            if cap is not None:
                cap.release()
        print(f"  index {idx}: {'  '.join(row)}")
    print("")


def _status_hint(pred: Prediction) -> str:
    """Secondary hint for cases where the main header already explains the
    primary state (no_face handled there) but extra detail is useful."""
    if pred.status == "multiple_faces":
        return "MULTIPLE FACES - ONE PERSON AT A TIME"
    if pred.status == "model_unavailable":
        return "EMOTION MODEL FAILED - see realtime_webcam.log"
    if pred.status == "error":
        return "INFERENCE ERROR - see realtime_webcam.log"
    if pred.status == "no_frame":
        return "NO FRAME FROM CAMERA - see realtime_webcam.log"
    if pred.status == "poor_quality" and pred.quality_reasons:
        return "FACE UNCLEAR: " + ", ".join(pred.quality_reasons)
    return ""


def _main_status(pred: Prediction) -> str:
    """Primary header text: real, honest state. NEVER 'UNCERTAIN'."""
    if pred.status == "no_face":
        return "NO FACE DETECTED"
    if pred.status == "no_frame":
        return "NO FRAME FROM CAMERA"
    if pred.status == "startup":
        return "STARTING..."
    if pred.status == "multiple_faces":
        return "MULTIPLE FACES"
    if pred.status == "model_unavailable":
        return "EMOTION MODEL FAILED"
    if pred.status == "error":
        return "INFERENCE ERROR"
    return ""


def draw_overlay(frame, pred: Prediction, fps: float, debug: bool,
                 shown_emotion: str | None, shown_conf: float) -> None:
    """Draw face box + emotion + confidence + optionally debug details.

    The main line NEVER shows 'UNCERTAIN': while the smoother is still
    gathering evidence we show the last committed emotion; when no face is
    detected we show the real reason instead."""
    box = pred.face_box
    if box is not None:
        x, y, w, h = [int(v) for v in box]
        color = EMOTION_COLORS.get(shown_emotion or pred.emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    if pred.face_detected and pred.status == "success":
        # Face present + model evidence recorded.
        if shown_emotion:
            pct = int(round(min(1.0, shown_conf) * 100))
            main_line = f"Emotion: {shown_emotion.upper()}"
            conf_line = f"Confidence: {pct}%"
        else:
            main_line = "Emotion: reading face..."
            conf_line = "Confidence: --"
    else:
        main_line = _main_status(pred) or "STATUS UNKNOWN"
        conf_line = "Confidence: --"

    conf_color = EMOTION_COLORS.get(shown_emotion or pred.emotion, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (0, 0, 0), -1)
    cv2.putText(frame, main_line, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                conf_color if pred.face_detected else (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, conf_line, (10, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)

    # Secondary hint for extra detail (never for no_face - already shown above)
    hint = _status_hint(pred)
    if hint:
        cv2.rectangle(frame, (0, 58), (frame.shape[1], 84), (0, 0, 0), -1)
        cv2.putText(frame, hint, (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2)

    if debug:
        y0 = 90
        lines = [
            f"raw:        {pred.raw_emotion} ({pred.raw_confidence:.2f})",
            f"quality:    {pred.quality_score:.2f}",
            f"reasons:    {', '.join(pred.quality_reasons) if pred.quality_reasons else 'ok'}",
            f"headpose:   yaw={pred.headpose.get('yaw', 0):.0f} "
            f"pitch={pred.headpose.get('pitch', 0):.0f} "
            f"roll={pred.headpose.get('roll', 0):.0f}",
            f"window:     {pred.window[-8:]}",
            f"status:     {pred.status}",
        ]
        # Draw debug background
        panel_h = 40 + len(lines) * 22
        cv2.rectangle(frame, (0, 80), (430, panel_h + 75), (0, 0, 0), -1)
        cv2.putText(frame, "-- DEBUG --", (10, y0 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (10, y0 + 40 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time face emotion recognition")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--debug", type=int, default=int(config.DEBUG_MODE))
    parser.add_argument("--frame-skip", type=int, default=config.FRAME_SKIP)
    parser.add_argument("--max-camera-width", type=int, default=900,
                        help="Resize wide frames for speed")
    parser.add_argument("--list-cameras", action="store_true",
                        help="Probe cameras and exit")
    parser.add_argument("--max-seconds", type=float, default=0.0,
                        help="Stop automatically after N seconds (testing)")
    parser.add_argument("--frames", type=int, default=0,
                        help="Stop automatically after N frames (testing)")
    parser.add_argument("--flip", action="store_true",
                        help="Mirror the displayed feed horizontally (cosmetic)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable spoken emotion announcements")
    args = parser.parse_args()

    if args.list_cameras:
        list_cameras()
        return

    if args.frame_skip <= 0:
        log.error("--frame-skip must be >= 1")
        return

    log.info("=" * 60)
    log.info("Starting real-time emotion recognition")
    log.info(f"OpenCV {cv2.__version__} | backend preference={config.CAMERA_BACKEND}")
    log.info(f"preferred camera index={args.camera} | frame-skip={args.frame_skip} "
             f"| prediction-interval={config.PREDICTION_INTERVAL}s")

    engine = EmotionEngine()
    speaker = Speaker(enabled=not args.no_voice)
    log.info("Loading emotion model... (first run may take several seconds)")
    log.info(f"face detector backend: {engine.backend.detector_backend}")
    log.info(f"voice announcements: {'ON' if speaker.enabled else 'OFF'}")
    t0 = time.time()
    engine.backend._load()
    log.info(f"Model loaded in {time.time() - t0:.1f}s")

    cam, cam_info = open_camera(args.camera)
    if cam is None:
        log.error("CAMERA FAILED TO OPEN")
        log.error("Tried:")
        for t in cam_info["tried"]:
            log.error(f"  index={t['index']} backend={t['backend']} "
                      f"{'OK (frame read)' if t['ok'] else 'failed'}")
        log.error("Suggested actions:")
        log.error("  1. Close other apps using the camera (Zoom, Teams, etc.)")
        log.error("  2. Check Windows privacy: Settings > Privacy > Camera >"
                  " allow apps")
        log.error("  3. Try a different index: py -3.12 realtime_webcam.py --camera 1")
        log.error("  4. Run py -3.12 realtime_webcam.py --list-cameras to probe")
        print("ERROR: camera could not be opened - see realtime_webcam.log for details.")
        return

    w = cam_info.get("width", 640)
    h = cam_info.get("height", 480)
    log.info(f"CAMERA OPENED SUCCESSFULLY: index={cam_info['winner']['index']} "
             f"backend={cam_info['winner']['backend']} resolution={w}x{h}")
    log.info("Press q/Esc to quit, d toggles debug, r resets smoother.")

    debug = bool(args.debug)
    fps = 0.0
    frame_count = 0
    consecutive_read_failures = 0
    last_infer_time = 0.0
    last_info_log_time = 0.0
    last_frame_time = time.time()
    start_time = time.time()
    last_status = None

    # Auto-renegotiation: if the active camera never produces a face within
    # CAMERA_RENEGOTIATION_INTERVAL seconds, switch to the next index.
    # A camera that has ever produced a face is locked in.
    active_camera_index = cam_info["winner"]["index"]
    camera_rotation_pos = 0
    camera_rotation_live = [i for i in range(config.CAMERA_PROBE_RANGE)
                            if i != active_camera_index]
    camera_locked = False
    last_face_seen_at = time.time()
    announced_emotion: str | None = None

    prediction = Prediction(status="startup", emotion=config.UNCERTAIN_LABEL)
    shown_emotion: str | None = None
    shown_conf = 0.0

    while True:
        success, frame = cam.read()
        if not success or frame is None:
            consecutive_read_failures += 1
            if consecutive_read_failures == 1:
                log.warning(f"FRAME READ FAILED (attempt {consecutive_read_failures}) "
                            f"- retrying {config.CONSECUTIVE_READ_FAILURE_LIMIT} times")
            if consecutive_read_failures >= config.CONSECUTIVE_READ_FAILURE_LIMIT:
                log.error("FRAME READ FAILED 20x in a row - camera is not delivering "
                          "frames. Releasing and exiting; check camera privacy settings "
                          "or that no other app holds the camera.")
                break
            time.sleep(0.05)
            continue
        consecutive_read_failures = 0

        frame_count += 1
        if args.max_camera_width and frame.shape[1] > args.max_camera_width:
            scale = args.max_camera_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        # Cosmetic display flip (applied BEFORE detection so the drawn face
        # box stays aligned with what is shown).
        if args.flip:
            frame = cv2.flip(frame, 1)

        now = time.time()
        # Inference throttled by frame-skip AND wall-clock interval so that
        # the model cost is bounded even on fast cameras.
        should_infer = (
            frame_count % args.frame_skip == 0
            and (now - last_infer_time) >= config.PREDICTION_INTERVAL
        )
        if should_infer:
            prediction = engine.process_frame(frame)
            last_infer_time = time.time()
        else:
            last = engine.last_prediction()
            if last is not None:
                prediction = last
            # else keep the "startup" prediction until first inference lands

        if prediction.face_detected:
            last_face_seen_at = now
            camera_locked = True  # this camera clearly works
            camera_rotation_pos = 0
            # Displayed emotion: follow the smoothed result. If the smoother
            # is still unsettled (emotion == UNCERTAIN) keep the last committed
            # label instead of flashing 'UNCERTAIN' on screen.
            if prediction.emotion not in (config.UNCERTAIN_LABEL, None):
                shown_emotion = prediction.emotion
                shown_conf = prediction.confidence
        else:
            shown_emotion = None
            shown_conf = 0.0

        # "Tell me": speak only a stable, committed, non-uncertain emotion
        # change (never every frame, never repeats).
        if (prediction.face_detected and prediction.stable
                and prediction.emotion not in (config.UNCERTAIN_LABEL, None)
                and prediction.emotion != announced_emotion):
            announced_emotion = prediction.emotion
            phrase = PHRASES.get(prediction.emotion, f"Emotion: {prediction.emotion}")
            if speaker.speak(phrase):
                log.info(f"ANNOUNCED: {phrase}")

        # Re-negotiate camera if it has never shown a face for a while.
        if (not camera_locked
                and (now - last_face_seen_at) >= config.CAMERA_RENEGOTIATION_INTERVAL
                and camera_rotation_pos < len(camera_rotation_live)):
            camera_rotation_pos += 1
            log.warning(f"NO FACE on camera index {active_camera_index} for "
                        f"{config.CAMERA_RENEGOTIATION_INTERVAL:.0f}s - switching to "
                        f"index {camera_rotation_live[camera_rotation_pos - 1]}")
            cam.release()
            next_idx = camera_rotation_live[camera_rotation_pos - 1]
            cam, cam_info = open_camera(next_idx, restrict_to=next_idx)
            last_face_seen_at = now
            if cam is not None and cam_info and cam_info.get("winner"):
                active_camera_index = cam_info["winner"]["index"]
                log.info(f"Now using camera index {active_camera_index} "
                         f"backend={cam_info['winner']['backend']}")
                engine.reset()
            else:
                log.error("Camera switch to index %s failed. Falling back to "
                          "index %s and locking it - no more auto-switching "
                          "this session.",
                          next_idx, active_camera_index)
                cam, cam_info = open_camera(active_camera_index)
                camera_locked = True  # stop probing; just keep this camera
                if cam is None or not (cam_info or {}).get("winner"):
                    log.error("Fallback camera could not be reopened either. "
                              "Exiting.")
                    break

        if prediction.status != last_status:
            last_status = prediction.status
            log.info(f"[#frames={frame_count}] status={prediction.status} "
                     f"emotion={prediction.emotion} conf={prediction.confidence:.2f} "
                     f"face={prediction.face_detected} raw={prediction.raw_emotion}")

        dt = now - last_frame_time
        last_frame_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))

        draw_overlay(frame, prediction, fps, debug, shown_emotion, shown_conf)

        cv2.imshow("Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('d'):
            debug = not debug
        if key == ord('r'):
            engine.reset()

        if now - last_info_log_time >= 10.0:
            last_info_log_time = now
            log.info(f"[health] frames={frame_count} fps={fps:.0f} "
                     f"status={prediction.status} emotion={prediction.emotion} "
                     f"conf={prediction.confidence:.2f}")
        if args.max_seconds and (now - start_time) >= args.max_seconds:
            log.info(f"[auto-stop] reached --max-seconds {args.max_seconds}")
            break
        if args.frames and frame_count >= args.frames:
            log.info(f"[auto-stop] reached --frames {args.frames}")
            break

    if cam is not None:
        cam.release()
    cv2.destroyAllWindows()
    log.info(f"Stopped. Frames read={frame_count} consecutive_read_failures="
             f"{consecutive_read_failures}")
    print("Stopped.")


if __name__ == "__main__":
    main()