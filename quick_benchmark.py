#!/usr/bin/env python3
"""
Quick Benchmark - Run with available synthetic data
Provides relative detector performance comparison.
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from deepfake_guard.core import DeepfakeGuard

def run_quick_benchmark():
    """Run benchmark on synthetic data."""
    print("="*60)
    print("DeepFakeGuard Quick Benchmark")
    print("="*60)
    print("\n⚠️  Using synthetic data - for relative comparison only")
    print("   Real deepfake datasets require authentication\n")
    
    data_dir = Path(__file__).parent / "benchmark_data" / "synthetic"
    real_videos = list((data_dir / "real").glob("*.mp4"))
    fake_videos = list((data_dir / "fake").glob("*.mp4"))
    
    print(f"Videos: {len(real_videos)} real + {len(fake_videos)} fake = {len(real_videos) + len(fake_videos)} total\n")
    
    weights = "weights/dinov3_best_v3.pth"
    if not os.path.exists(weights):
        weights = None
        print("⚠️  DINOv3 weights not found, using random initialization\n")
    
    detectors = ["dinov3", "resnet18", "ivyfake", "d3"]
    results = {}
    
    for detector in detectors:
        print(f"Testing {detector.upper()}...")
        det_weights = weights if detector == "dinov3" and weights else None
        
        try:
            guard = DeepfakeGuard(detector_type=detector, weights_path=det_weights)
            
            scores_real = []
            scores_fake = []
            times = []
            
            # Test real videos
            for v in real_videos:
                start = time.time()
                result = guard.detect_video(str(v))
                times.append(time.time() - start)
                scores_real.append(result.get("overall_score", 0.5))
            
            # Test fake videos  
            for v in fake_videos:
                start = time.time()
                result = guard.detect_video(str(v))
                times.append(time.time() - start)
                scores_fake.append(result.get("overall_score", 0.5))
            
            # Calculate separation
            avg_real = sum(scores_real) / len(scores_real) if scores_real else 0.5
            avg_fake = sum(scores_fake) / len(scores_fake) if scores_fake else 0.5
            separation = abs(avg_fake - avg_real)
            
            results[detector] = {
                "avg_real_score": round(avg_real, 4),
                "avg_fake_score": round(avg_fake, 4),
                "separation": round(separation, 4),
                "avg_time": round(sum(times) / len(times), 4) if times else 0,
                "status": "OK"
            }
            
            print(f"  ✓ Real: {avg_real:.3f}, Fake: {avg_fake:.3f}, Separation: {separation:.3f}")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)[:60]}")
            results[detector] = {"status": "ERROR", "error": str(e)[:100]}
    
    # Save results
    with open("benchmark_results_quick.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"{'Detector':<12} {'Real':<8} {'Fake':<8} {'Sep':<8} {'Time':<10} {'Status'}")
    print("-"*60)
    
    for det, res in results.items():
        if res["status"] == "OK":
            print(f"{det:<12} {res['avg_real_score']:<8.3f} {res['avg_fake_score']:<8.3f} "
                  f"{res['separation']:<8.3f} {res['avg_time']:<10.3f} ✓")
        else:
            print(f"{det:<12} {'-':<8} {'-':<8} {'-':<8} {'-':<10} ✗ {res.get('error', '')[:30]}")
    
    print("\n" + "="*60)
    print("NOTES:")
    print("="*60)
    print("• Separation = difference between fake and real scores")
    print("  Higher = better detector discrimination")
    print("• These are SYNTHETIC results - not comparable to real deepfakes")
    print("• For publication-quality results, use FaceForensics++ or Celeb-DF")
    print("\nResults saved to: benchmark_results_quick.json")
    
    return results

if __name__ == "__main__":
    run_quick_benchmark()
