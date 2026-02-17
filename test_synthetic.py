#!/usr/bin/env python3
"""
Create synthetic test videos and benchmark all 3 detectors
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepfake_guard.core import DeepfakeGuard

def create_synthetic_video(filename, num_frames=30, fps=10, size=(224, 224)):
    """Create a synthetic test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, size)
    
    for i in range(num_frames):
        # Create a frame with changing colors/patterns
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Add some visual content
        color_val = int(255 * (i / num_frames))
        frame[:, :] = [color_val, 128, 255 - color_val]
        
        # Add a moving circle
        center_x = int(size[0] * (0.3 + 0.4 * (i / num_frames)))
        center_y = size[1] // 2
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    return filename


def test_detector(detector_type, video_path):
    """Test a detector and return metrics."""
    print(f"\n{'='*60}")
    print(f"🔍 Testing: {detector_type.upper()}")
    print('='*60)
    
    results = {
        "detector": detector_type,
        "success": False
    }
    
    try:
        # Initialize
        init_start = time.time()
        guard = DeepfakeGuard(detector_type=detector_type)
        init_time = time.time() - init_start
        results["init_time"] = round(init_time, 3)
        print(f"✓ Initialized in {init_time:.3f}s")
        
        # Detect
        detect_start = time.time()
        result = guard.detect_video(video_path)
        detect_time = time.time() - detect_start
        results["inference_time"] = round(detect_time, 3)
        
        # Extract results
        label = result.get("overall_label", "UNKNOWN")
        score = result.get("overall_score", 0.0)
        errors = result.get("errors", [])
        
        results["label"] = label
        results["score"] = round(score, 4)
        results["errors"] = errors
        results["success"] = True
        
        print(f"✓ Detection complete in {detect_time:.3f}s")
        print(f"   Label: {label}")
        print(f"   Score: {score:.4f}")
        
        # Show details
        modality_results = result.get("modality_results", {})
        for modality, data in modality_results.items():
            details = data.get("details", {})
            det_type = details.get("detector_type", "unknown")
            frame_count = details.get("frame_count", "N/A")
            print(f"   Detector: {det_type}")
            print(f"   Frames: {frame_count}")
            
            features = details.get("features", [])
            if features:
                print(f"   Features: {', '.join(features)}")
        
        if errors:
            print(f"   ⚠️  Errors: {errors}")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)
    
    return results


def main():
    print("="*60)
    print("🧪 DeepFakeGuard Synthetic Video Test")
    print("="*60)
    
    # Create synthetic video
    video_path = "/tmp/synthetic_test.mp4"
    print(f"\n🎬 Creating synthetic test video...")
    create_synthetic_video(video_path, num_frames=30)
    print(f"✓ Created: {video_path}")
    
    # Test all detectors
    detectors = ["resnet18", "ivyfake"]  # DINOv3 needs weights
    all_results = []
    
    for detector in detectors:
        result = test_detector(detector, video_path)
        all_results.append(result)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    for r in all_results:
        print(f"\n{r['detector'].upper()}:")
        if r['success']:
            print(f"  Init Time: {r['init_time']:.3f}s")
            print(f"  Inference: {r['inference_time']:.3f}s")
            print(f"  Label: {r['label']}")
            print(f"  Score: {r['score']:.4f}")
            if r['errors']:
                print(f"  ⚠️  Errors: {r['errors']}")
        else:
            print(f"  ✗ Failed: {r.get('error', 'Unknown')}")
    
    # Comparison
    print("\n" + "="*60)
    print("⚡ SPEED COMPARISON")
    print("="*60)
    
    successful = [r for r in all_results if r['success']]
    if successful:
        fastest = min(successful, key=lambda x: x['inference_time'])
        print(f"\nFastest Inference: {fastest['detector'].upper()} ({fastest['inference_time']:.3f}s)")
        
        for r in successful:
            speedup = r['inference_time'] / fastest['inference_time']
            print(f"  {r['detector']:12} : {r['inference_time']:.3f}s ({speedup:.1f}x)")
    
    # Cleanup
    try:
        import os
        os.unlink(video_path)
    except:
        pass
    
    print("\n" + "="*60)
    print("✅ Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()