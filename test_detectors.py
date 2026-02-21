#!/usr/bin/env python3
"""
DeepFakeGuard — Smoke Test
Creates a synthetic video and verifies every detector can initialise and run.

Usage:
    python test_detectors.py                   # test all detectors with synthetic video
    python test_detectors.py --video path.mp4  # test with a real video
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure the local src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))
from deepfake_guard.core import DeepfakeGuard


# ── helpers ───────────────────────────────────────────────────────────
def create_synthetic_video(path: str, num_frames: int = 30, fps: int = 10) -> str:
    """Write a short colour-changing synthetic video for testing."""
    size = (224, 224)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(num_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        c = int(255 * i / num_frames)
        frame[:, :] = [c, 128, 255 - c]
        cx = int(size[0] * (0.3 + 0.4 * (i / num_frames)))
        cv2.circle(frame, (cx, size[1] // 2), 30, (255, 255, 255), -1)
        out.write(frame)
    out.release()
    return path


def run_detector(detector_type: str, video_path: str, weights: str | None = None):
    """Initialise one detector and analyse a video.  Returns a results dict."""
    record: dict = {"detector": detector_type, "success": False}
    try:
        t0 = time.perf_counter()
        guard = DeepfakeGuard(detector_type=detector_type, weights_path=weights)
        record["init_time"] = round(time.perf_counter() - t0, 3)

        t1 = time.perf_counter()
        result = guard.detect_video(video_path)
        record["inference_time"] = round(time.perf_counter() - t1, 3)

        record["label"] = result.get("overall_label", "UNKNOWN")
        record["score"] = round(result.get("overall_score", 0.0), 4)
        record["errors"] = result.get("errors", [])
        record["success"] = True

        # grab per-modality detail
        for _name, mdata in result.get("modality_results", {}).items():
            details = mdata.get("details", {})
            record["frame_count"] = details.get("frame_count", "N/A")
            record["det_backend"] = details.get("detector_type", detector_type)
    except Exception as exc:
        record["error"] = str(exc)
        import traceback
        traceback.print_exc()
    return record


# ── main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Test all DeepFakeGuard detectors")
    parser.add_argument("--video", "-v", help="Path to a real video file (otherwise synthetic)")
    args = parser.parse_args()

    header = "DeepFakeGuard — Detector Smoke Test"
    print(f"\n{'=' * 60}\n{header}\n{'=' * 60}")

    # Decide on video
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            sys.exit(f"Video not found: {video_path}")
        print(f"Using provided video: {video_path}")
    else:
        video_path = os.path.join("tmp_test_video.mp4")
        create_synthetic_video(video_path)
        print(f"Created synthetic test video: {video_path}")

    # Define detectors to test
    weights_file = "weights/dinov3_best_v3.pth"
    tests = [
        ("resnet18", None),
        ("ivyfake", None),
        ("d3", None),
        ("dinov3", weights_file if os.path.exists(weights_file) else None),
    ]

    results = []
    for det, weights in tests:
        print(f"\n{'—' * 60}")
        print(f"  {det.upper()}" + (f"  (weights: {weights})" if weights else ""))
        print(f"{'—' * 60}")
        r = run_detector(det, video_path, weights)
        results.append(r)
        if r["success"]:
            print(f"  OK   label={r['label']}  score={r['score']:.4f}  "
                  f"init={r['init_time']}s  infer={r['inference_time']}s")
            if r["errors"]:
                print(f"  WARN errors: {r['errors']}")
        else:
            print(f"  FAIL {r.get('error', 'unknown')}")

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    for r in results:
        tag = "PASS" if r["success"] else "FAIL"
        line = f"  [{tag}] {r['detector']:10}"
        if r["success"]:
            line += f"  label={r['label']:5}  score={r['score']:.4f}  infer={r['inference_time']}s"
        else:
            line += f"  {r.get('error', '')[:60]}"
        print(line)
    print(f"{'=' * 60}\n")

    # Cleanup synthetic video
    if not args.video and os.path.exists(video_path):
        os.unlink(video_path)

    passed = sum(1 for r in results if r["success"])
    total = len(results)
    print(f"{passed}/{total} detectors passed.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
