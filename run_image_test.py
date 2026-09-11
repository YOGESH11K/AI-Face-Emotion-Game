"""
Still-image emotion tester.

Run the engine (with temporal smoothing) over a single image or a folder of
images and print the prediction. Useful for verifying behaviour without a
webcam, e.g. over the sample photos in ./image.

Usage:
    py -3.12 run_image_test.py image/happy_44820917.jpg
    py -3.12 run_image_test.py --dir image
    py -3.12 run_image_test.py --dir image --repeat 5   # simulate frames
"""

from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

from emotion_system.engine import EmotionEngine
from emotion_system.emotion_utils import describe


def analyze_image(path: str, repeat: int = 1, engine: EmotionEngine = None) -> list:
    img = cv2.imread(path)
    if img is None:
        print(f"[skip] cannot read {path}")
        return []
    created = engine is None
    engine = engine or EmotionEngine()
    if created:
        engine.backend._load()
    results = []
    for i in range(max(1, repeat)):
        pred = engine.process_frame(img)
        results.append(pred)
    if created:
        engine.reset()
    return results


def run_dir(indir: str, repeat: int) -> None:
    engine = EmotionEngine()
    engine.backend._load()
    files = sorted(glob.glob(os.path.join(indir, "*.jpg")) +
                   glob.glob(os.path.join(indir, "*.png")))
    for path in files:
        results = analyze_image(path, repeat, engine)
        if not results:
            continue
        p = results[-1]
        base = os.path.basename(path)
        print(f"{base.ljust(32)} -> {p.emotion.ljust(9)} conf={p.confidence:.3f} "
              f"stable={p.stable} status={p.status}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Test emotion engine on still images")
    ap.add_argument("path", nargs="?", help="single image path")
    ap.add_argument("--dir", help="folder of images")
    ap.add_argument("--repeat", type=int, default=1,
                    help="frames to simulate for smoothing")
    args = ap.parse_args()

    engine = EmotionEngine()
    engine.backend._load()

    if args.dir:
        run_dir(args.dir, args.repeat)
    elif args.path:
        results = analyze_image(args.path, args.repeat, engine)
        if results:
            p = results[-1]
            print(f"prediction: {p.emotion}")
            print(f"confidence: {p.confidence:.4f}")
            print(f"stable:     {p.stable}")
            print(f"status:     {p.status}")
            print(f"quality:    {p.quality_score:.4f}  reasons={p.quality_reasons}")
            print(f"headpose:   {p.headpose}")
            print(f"all:        {describe(p.all_emotions)}")
            print(f"window:     {p.window}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()