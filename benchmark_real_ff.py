#!/usr/bin/env python3
"""
Run benchmark on real FaceForensics++ data.
"""

import os
import sys
import json
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from deepfake_guard.core import DeepfakeGuard

def run_real_benchmark():
    """Run benchmark on real FaceForensics++ data."""
    print("="*60)
    print("DeepFakeGuard - REAL Benchmark (FaceForensics++)")
    print("="*60)
    
    # Paths
    ff_dir = Path(__file__).parent / "benchmark_data" / "faceforensics"
    real_dir = ff_dir / "original_sequences" / "youtube" / "c23" / "videos"
    fake_dir = ff_dir / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    
    real_videos = list(real_dir.glob("*.mp4"))
    fake_videos = list(fake_dir.glob("*.mp4"))
    
    print(f"\nReal videos: {len(real_videos)}")
    print(f"Fake videos: {len(fake_videos)}")
    print(f"Total: {len(real_videos) + len(fake_videos)}\n")
    
    if not real_videos or not fake_videos:
        print("❌ No videos found. Run download script first.")
        return
    
    weights = "weights/dinov3_best_v3.pth"
    if not os.path.exists(weights):
        print("⚠️  DINOv3 weights not found, using random initialization\n")
        weights = None
    
    detectors = ["dinov3", "resnet18", "ivyfake", "d3"]
    all_results = {}
    
    for detector in detectors:
        print(f"\n{'='*60}")
        print(f"Testing: {detector.upper()}")
        print(f"{'='*60}")
        
        det_weights = weights if detector == "dinov3" and weights else None
        
        try:
            guard = DeepfakeGuard(detector_type=detector, weights_path=det_weights)
            
            predictions = []
            scores = []
            ground_truth = []
            times = []
            
            # Test real videos
            for v in real_videos:
                print(f"  [REAL] {v.name}...", end=" ", flush=True)
                start = time.time()
                result = guard.detect_video(str(v))
                elapsed = time.time() - start
                times.append(elapsed)
                
                score = result.get("overall_score", 0.5)
                label = result.get("overall_label", "UNKNOWN")
                
                scores.append(score)
                predictions.append(1 if label == "FAKE" else 0)
                ground_truth.append(0)  # Real = 0
                
                print(f"Score: {score:.3f} ({elapsed:.2f}s)")
            
            # Test fake videos
            for v in fake_videos:
                print(f"  [FAKE] {v.name}...", end=" ", flush=True)
                start = time.time()
                result = guard.detect_video(str(v))
                elapsed = time.time() - start
                times.append(elapsed)
                
                score = result.get("overall_score", 0.5)
                label = result.get("overall_label", "UNKNOWN")
                
                scores.append(score)
                predictions.append(1 if label == "FAKE" else 0)
                ground_truth.append(1)  # Fake = 1
                
                print(f"Score: {score:.3f} ({elapsed:.2f}s)")
            
            # Calculate metrics
            try:
                auroc = roc_auc_score(ground_truth, scores)
                acc = accuracy_score(ground_truth, predictions)
                prec = precision_score(ground_truth, predictions, zero_division=0)
                rec = recall_score(ground_truth, predictions, zero_division=0)
                f1 = f1_score(ground_truth, predictions, zero_division=0)
                
                all_results[detector] = {
                    "auroc": round(auroc, 4),
                    "accuracy": round(acc, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "avg_time": round(np.mean(times), 4),
                    "num_videos": len(ground_truth),
                    "status": "OK"
                }
                
                print(f"\n  ✓ AUROC: {auroc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}")
                
            except Exception as e:
                print(f"\n  ✗ Metric calculation error: {e}")
                all_results[detector] = {"status": "ERROR", "error": str(e)}
                
        except Exception as e:
            print(f"\n  ✗ Detector error: {e}")
            all_results[detector] = {"status": "ERROR", "error": str(e)}
    
    # Save results
    output_file = "benchmark_results_real.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY - REAL FACEFORENSICS++ DATA")
    print("="*60)
    print(f"{'Detector':<12} {'AUROC':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'Time':<8}")
    print("-"*60)
    
    for det, res in all_results.items():
        if res.get("status") == "OK":
            print(f"{det:<12} {res['auroc']:<8.4f} {res['accuracy']:<8.4f} "
                  f"{res['precision']:<8.4f} {res['recall']:<8.4f} "
                  f"{res['f1_score']:<8.4f} {res['avg_time']:<8.3f}")
        else:
            print(f"{det:<12} ERROR: {res.get('error', 'Unknown')[:40]}")
    
    print("\n" + "="*60)
    print("✓ These are REAL results on FaceForensics++ data!")
    print(f"✓ Results saved to: {output_file}")
    print("="*60)
    
    return all_results

if __name__ == "__main__":
    run_real_benchmark()
