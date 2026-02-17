#!/usr/bin/env python3
"""
Test DINOv3 with actual weights
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepfake_guard.core import DeepfakeGuard
import time

print("="*60)
print("🧪 Testing DINOv3 with Weights")
print("="*60)

video_path = "/tmp/synthetic_test.mp4"

# Create video if not exists
import cv2
import numpy as np

if not Path(video_path).exists():
    print("\n🎬 Creating test video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10, (224, 224))
    
    for i in range(30):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        color_val = int(255 * (i / 30))
        frame[:, :] = [color_val, 128, 255 - color_val]
        center_x = int(224 * (0.3 + 0.4 * (i / 30)))
        cv2.circle(frame, (center_x, 112), 30, (255, 255, 255), -1)
        out.write(frame)
    
    out.release()
    print("✓ Video created")

print(f"\n🔍 Testing DINOv3...")
print(f"   Weights: weights/dinov3_best_v3.pth")

try:
    # Initialize
    init_start = time.time()
    guard = DeepfakeGuard(
        detector_type='dinov3',
        weights_path='weights/dinov3_best_v3.pth'
    )
    init_time = time.time() - init_start
    print(f"✓ Initialized in {init_time:.3f}s")
    
    # Detect
    detect_start = time.time()
    result = guard.detect_video(video_path)
    detect_time = time.time() - detect_start
    
    label = result.get("overall_label", "UNKNOWN")
    score = result.get("overall_score", 0.0)
    errors = result.get("errors", [])
    
    print(f"✓ Detection complete in {detect_time:.3f}s")
    print(f"\n📊 Results:")
    print(f"   Label: {label}")
    print(f"   Score: {score:.4f}")
    
    if errors:
        print(f"   ⚠️  Errors: {errors}")
    
    # Show details
    modality_results = result.get("modality_results", {})
    for modality, data in modality_results.items():
        details = data.get("details", {})
        print(f"\n   Detector: {details.get('detector_type', 'unknown')}")
        print(f"   Frames: {details.get('frame_count', 'N/A')}")
    
    print(f"\n{'='*60}")
    print("✅ DINOv3 Test Complete!")
    print(f"   Init Time: {init_time:.3f}s")
    print(f"   Inference: {detect_time:.3f}s")
    print('='*60)
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()