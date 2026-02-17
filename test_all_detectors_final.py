#!/usr/bin/env python3
"""
Comprehensive test of all 3 detectors
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import time

from deepfake_guard.core import DeepfakeGuard

def create_test_video(path, num_frames=30):
    """Create a simple test video."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 10, (224, 224))
    
    for i in range(num_frames):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        color_val = int(255 * (i / num_frames))
        frame[:, :] = [color_val, 128, 255 - color_val]
        center_x = int(224 * (0.3 + 0.4 * (i / num_frames)))
        cv2.circle(frame, (center_x, 112), 30, (255, 255, 255), -1)
        out.write(frame)
    
    out.release()
    return path


def test_detector(detector_type, video_path, weights_path=None):
    """Test a detector."""
    results = {
        "detector": detector_type,
        "weights": weights_path,
        "success": False
    }
    
    try:
        init_start = time.time()
        guard = DeepfakeGuard(
            detector_type=detector_type,
            weights_path=weights_path
        )
        results["init_time"] = time.time() - init_start
        
        detect_start = time.time()
        result = guard.detect_video(video_path)
        results["inference_time"] = time.time() - detect_start
        
        results["label"] = result.get("overall_label", "UNKNOWN")
        results["score"] = result.get("overall_score", 0.0)
        results["errors"] = result.get("errors", [])
        results["success"] = True
        
        # Get detector info
        for modality, data in result.get("modality_results", {}).items():
            details = data.get("details", {})
            results["detector_type"] = details.get("detector_type", "unknown")
            results["frame_count"] = details.get("frame_count", "N/A")
            results["features"] = details.get("features", [])
            
    except Exception as e:
        results["error"] = str(e)
    
    return results


def main():
    print("="*70)
    print("🧪 DEEPFAKEGUARD - COMPREHENSIVE DETECTOR TEST REPORT")
    print("="*70)
    
    video_path = "/tmp/test_video.mp4"
    create_test_video(video_path)
    print(f"\n📹 Test video created: 30 frames, 224x224, 10fps")
    
    # Test all 3 detectors
    tests = [
        ("resnet18", None),
        ("ivyfake", None),
        ("dinov3", "weights/dinov3_best_v3.pth")
    ]
    
    all_results = []
    
    for detector, weights in tests:
        print(f"\n{'='*70}")
        print(f"🔍 TESTING: {detector.upper()}")
        if weights:
            print(f"   Weights: {weights}")
        print('='*70)
        
        result = test_detector(detector, video_path, weights)
        all_results.append(result)
        
        if result['success']:
            print(f"✅ SUCCESS")
            print(f"   Init Time:     {result['init_time']:.3f}s")
            print(f"   Inference:     {result['inference_time']:.3f}s")
            print(f"   Label:         {result['label']}")
            print(f"   Score:         {result['score']:.4f}")
            print(f"   Detector Type: {result.get('detector_type', 'N/A')}")
            print(f"   Frames:        {result.get('frame_count', 'N/A')}")
            if result.get('features'):
                print(f"   Features:      {', '.join(result['features'])}")
            if result['errors']:
                print(f"   ⚠️  Errors:    {result['errors']}")
        else:
            print(f"❌ FAILED: {result.get('error', 'Unknown error')}")
    
    # Summary Table
    print("\n" + "="*70)
    print("📊 COMPREHENSIVE RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Detector':<12} {'Status':<10} {'Init(s)':<10} {'Infer(s)':<10} {'Label':<8} {'Score':<8}")
    print("-"*70)
    
    for r in all_results:
        status = "✅ PASS" if r['success'] else "❌ FAIL"
        if r['success']:
            print(f"{r['detector']:<12} {status:<10} {r['init_time']:<10.3f} {r['inference_time']:<10.3f} {r['label']:<8} {r['score']:<8.4f}")
        else:
            print(f"{r['detector']:<12} {status:<10} {'N/A':<10} {'N/A':<10} {'N/A':<8} {'N/A':<8}")
    
    # Performance Analysis
    successful = [r for r in all_results if r['success']]
    
    if successful:
        print("\n" + "="*70)
        print("⚡ PERFORMANCE ANALYSIS")
        print("="*70)
        
        # Fastest init
        fastest_init = min(successful, key=lambda x: x['init_time'])
        print(f"\n🚀 Fastest Initialization: {fastest_init['detector'].upper()} ({fastest_init['init_time']:.3f}s)")
        
        # Fastest inference
        fastest_infer = min(successful, key=lambda x: x['inference_time'])
        print(f"⚡ Fastest Inference:      {fastest_infer['detector'].upper()} ({fastest_infer['inference_time']:.3f}s)")
        
        print("\nDetailed Timing:")
        for r in sorted(successful, key=lambda x: x['inference_time']):
            print(f"  {r['detector']:<12} Init: {r['init_time']:>6.3f}s | Infer: {r['inference_time']:>6.3f}s | Total: {r['init_time'] + r['inference_time']:>6.3f}s")
    
    # Detector Characteristics
    print("\n" + "="*70)
    print("🔎 DETECTOR CHARACTERISTICS")
    print("="*70)
    
    for r in all_results:
        print(f"\n{r['detector'].upper()}:")
        if r['success']:
            print(f"  ✅ Successfully initialized and ran")
            print(f"  ✅ Inference completed in {r['inference_time']:.3f}s")
            if r['errors']:
                print(f"  ⚠️  Note: {r['errors'][0]}")
            else:
                print(f"  ✅ No errors")
        else:
            print(f"  ❌ Failed: {r.get('error', 'Unknown')}")
    
    # Final Verdict
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    passed = sum(1 for r in all_results if r['success'])
    total = len(all_results)
    
    print(f"\nTests Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL DETECTORS WORKING!")
        print("\nSummary:")
        print("  🧠 DINOv3:  Face-based, requires trained weights, most accurate")
        print("  🎯 ResNet18: Full-frame, pretrained, fastest, baseline accuracy")
        print("  🌿 IvyFake:  CLIP-based, explainable, temporal/spatial analysis")
    else:
        print(f"\n⚠️  {total - passed} detector(s) failed. Check logs above.")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()