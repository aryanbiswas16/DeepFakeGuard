#!/usr/bin/env python3
"""
Celeb-DF Cross-Dataset Benchmark

Tests generalization by evaluating on Celeb-DF v2 (unseen during training).
Your detector was trained on FaceForensics++, now test on Celeb-DF.
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

def run_celebdf_benchmark():
    """Run cross-dataset benchmark on Celeb-DF v2."""
    print("="*60)
    print("Celeb-DF v2 Cross-Dataset Benchmark")
    print("="*60)
    print("\nTraining Data: FaceForensics++")
    print("Test Data: Celeb-DF v2 (CROSS-DATASET)")
    print("This measures TRUE generalization!\n")
    
    # Paths
    celebdf_dir = Path(__file__).parent / "benchmark_data" / "celebdf"
    real_dir = celebdf_dir / "Celeb-real"
    fake_dir = celebdf_dir / "Celeb-synthesis"
    
    if not real_dir.exists() or not fake_dir.exists():
        print("❌ Celeb-DF data not found!")
        print(f"\nExpected at: {celebdf_dir}")
        print("\nTo download:")
        print("1. Fill Google Form: https://forms.gle/2jYBby6y1FBU3u6q9")
        print("2. Wait for approval email")
        print("3. Extract videos to benchmark_data/celebdf/")
        print("\nOr run: python3 setup_celebdf.py")
        return None
    
    real_videos = list(real_dir.glob("*.mp4"))
    fake_videos = list(fake_dir.glob("*.mp4"))
    
    print(f"Real videos: {len(real_videos)}")
    print(f"Fake videos: {len(fake_videos)}")
    print(f"Total: {len(real_videos) + len(fake_videos)}\n")
    
    if not real_videos or not fake_videos:
        print("❌ No videos found in Celeb-DF directories")
        return None
    
    # Load DINOv3 with your trained weights
    weights = "weights/dinov3_best_v3.pth"
    if not os.path.exists(weights):
        print(f"⚠️  Warning: Weights not found at {weights}")
        print("   Using random initialization (results will be poor)\n")
        weights = None
    else:
        print(f"✓ Loading trained weights: {weights}\n")
    
    print("Testing DINOv3 on Celeb-DF (cross-dataset)...")
    print("-"*60)
    
    try:
        guard = DeepfakeGuard(detector_type="dinov3", weights_path=weights)
        
        predictions = []
        scores = []
        ground_truth = []
        times = []
        
        # Test real videos
        for v in real_videos[:50]:  # Limit to 50 for speed
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
        for v in fake_videos[:50]:  # Limit to 50 for speed
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
        auroc = roc_auc_score(ground_truth, scores)
        acc = accuracy_score(ground_truth, predictions)
        prec = precision_score(ground_truth, predictions, zero_division=0)
        rec = recall_score(ground_truth, predictions, zero_division=0)
        f1 = f1_score(ground_truth, predictions, zero_division=0)
        
        results = {
            "dataset": "Celeb-DF v2",
            "num_real": len([x for x in ground_truth if x == 0]),
            "num_fake": len([x for x in ground_truth if x == 1]),
            "auroc": round(auroc, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "avg_time": round(np.mean(times), 4),
            "status": "OK"
        }
        
        # Save results
        with open("benchmark_results_celebdf.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "="*60)
        print("CROSS-DATASET RESULTS (Celeb-DF v2)")
        print("="*60)
        print(f"AUROC:      {auroc:.4f}")
        print(f"Accuracy:   {acc:.4f}")
        print(f"Precision:  {prec:.4f}")
        print(f"Recall:     {rec:.4f}")
        print(f"F1 Score:   {f1:.4f}")
        print(f"Avg Time:   {np.mean(times):.3f}s")
        print("="*60)
        
        # Compare to literature
        print("\nCOMPARISON TO LITERATURE:")
        print(f"  Xception (baseline):     ~0.65-0.70 AUROC")
        print(f"  EfficientNet-B4:         ~0.70-0.75 AUROC")
        print(f"  Your DINOv3 (measured):  {auroc:.4f} AUROC")
        print(f"  Your DINOv3 (paper):     0.88 AUROC")
        
        if auroc >= 0.85:
            print("\n🎉 EXCELLENT! Your detector generalizes well!")
        elif auroc >= 0.75:
            print("\n✓ GOOD! Better than CNN baselines.")
        else:
            print("\n⚠️  Lower than expected. Check if weights loaded correctly.")
        
        print(f"\n✓ Results saved to: benchmark_results_celebdf.json")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_celebdf_benchmark()
