#!/usr/bin/env python3
"""
DeepFakeGuard — Real Benchmark Evaluation
Run detectors on real videos and compute actual metrics (AUROC, Accuracy, etc.)

Requirements:
- Real test videos with known labels (real/fake)
- Recommended: FaceForensics++ or Celeb-DF test sets

Usage:
    python benchmark_real.py --data_dir /path/to/test/videos --labels /path/to/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# Ensure the local src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))
from deepfake_guard.core import DeepfakeGuard


class BenchmarkRunner:
    """Run benchmarks on real video datasets with ground truth labels."""
    
    def __init__(self, weights_path: str = "weights/dinov3_best_v3.pth"):
        self.weights_path = weights_path
        self.results = {}
        
    def run_detector_on_dataset(
        self, 
        detector_type: str, 
        video_paths: List[str], 
        ground_truth: List[str]
    ) -> Dict:
        """
        Run detector on a list of videos and compute metrics.
        
        Args:
            detector_type: 'dinov3', 'resnet18', 'ivyfake', 'd3'
            video_paths: List of video file paths
            ground_truth: List of labels ('REAL' or 'FAKE')
        
        Returns:
            Dictionary with predictions and metrics
        """
        print(f"\n{'='*60}")
        print(f"Benchmarking: {detector_type.upper()}")
        print(f"{'='*60}")
        
        # Initialize detector
        weights = self.weights_path if detector_type == "dinov3" else None
        guard = DeepfakeGuard(detector_type=detector_type, weights_path=weights)
        
        predictions = []
        scores = []
        inference_times = []
        errors = []
        
        for i, (video_path, true_label) in enumerate(zip(video_paths, ground_truth)):
            print(f"[{i+1}/{len(video_paths)}] {os.path.basename(video_path)}...", end=" ")
            
            if not os.path.exists(video_path):
                print("SKIP (not found)")
                continue
            
            try:
                start = time.time()
                result = guard.detect_video(video_path)
                infer_time = time.time() - start
                
                pred_label = result.get("overall_label", "UNKNOWN")
                pred_score = result.get("overall_score", 0.5)
                
                predictions.append(pred_label)
                scores.append(pred_score)
                inference_times.append(infer_time)
                
                print(f"{pred_label} ({pred_score:.3f}) in {infer_time:.2f}s")
                
            except Exception as e:
                print(f"ERROR: {str(e)[:50]}")
                errors.append((video_path, str(e)))
                predictions.append("ERROR")
                scores.append(0.5)
        
        # Compute metrics
        metrics = self._compute_metrics(ground_truth, predictions, scores)
        metrics["detector"] = detector_type
        metrics["num_videos"] = len(video_paths)
        metrics["errors"] = len(errors)
        metrics["avg_inference_time"] = np.mean(inference_times) if inference_times else 0
        metrics["total_time"] = sum(inference_times) if inference_times else 0
        
        return metrics
    
    def _compute_metrics(self, ground_truth: List[str], predictions: List[str], scores: List[float]) -> Dict:
        """Compute classification metrics."""
        # Filter out errors
        valid_idx = [i for i, p in enumerate(predictions) if p != "ERROR" and p != "UNKNOWN"]
        
        if not valid_idx:
            return {"error": "No valid predictions"}
        
        y_true = [1 if ground_truth[i] == "FAKE" else 0 for i in valid_idx]
        y_pred = [1 if predictions[i] == "FAKE" else 0 for i in valid_idx]
        y_scores = [scores[i] for i in valid_idx]
        
        metrics = {}
        
        try:
            metrics["auroc"] = round(roc_auc_score(y_true, y_scores), 4)
        except:
            metrics["auroc"] = None
        
        try:
            metrics["accuracy"] = round(accuracy_score(y_true, y_pred), 4)
            metrics["precision"] = round(precision_score(y_true, y_pred, zero_division=0), 4)
            metrics["recall"] = round(recall_score(y_true, y_pred, zero_division=0), 4)
            metrics["f1_score"] = round(f1_score(y_true, y_pred, zero_division=0), 4)
        except Exception as e:
            metrics["error"] = str(e)
        
        return metrics
    
    def run_full_benchmark(self, video_paths: List[str], ground_truth: List[str]) -> Dict:
        """Run all detectors and compare."""
        detectors = ["resnet18", "ivyfake", "d3", "dinov3"]
        
        all_results = {}
        
        for detector in detectors:
            try:
                result = self.run_detector_on_dataset(detector, video_paths, ground_truth)
                all_results[detector] = result
            except Exception as e:
                print(f"\nERROR running {detector}: {e}")
                all_results[detector] = {"error": str(e)}
        
        return all_results


def load_labels_from_csv(csv_path: str) -> Tuple[List[str], List[str]]:
    """Load video paths and labels from CSV."""
    video_paths = []
    labels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_paths.append(row['video_path'])
            labels.append(row['label'])  # 'REAL' or 'FAKE'
    
    return video_paths, labels


def create_sample_data(data_dir: str, num_real: int = 5, num_fake: int = 5):
    """
    Create a sample CSV template for benchmark data.
    Users need to populate this with actual video paths.
    """
    csv_path = os.path.join(data_dir, "labels.csv")
    
    rows = []
    for i in range(num_real):
        rows.append({"video_path": f"real_video_{i}.mp4", "label": "REAL"})
    for i in range(num_fake):
        rows.append({"video_path": f"fake_video_{i}.mp4", "label": "FAKE"})
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["video_path", "label"])
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Created sample labels file: {csv_path}")
    print("Please populate with actual video paths and ensure videos are in the data directory.")


def main():
    parser = argparse.ArgumentParser(description="Run real benchmarks on DeepFakeGuard detectors")
    parser.add_argument("--data_dir", "-d", help="Directory containing test videos")
    parser.add_argument("--labels", "-l", help="CSV file with video paths and labels (video_path,label)")
    parser.add_argument("--output", "-o", default="benchmark_results.json", help="Output JSON file")
    parser.add_argument("--create_sample", action="store_true", help="Create sample data template")
    parser.add_argument("--weights", "-w", default="weights/dinov3_best_v3.pth", help="Path to DINOv3 weights")
    args = parser.parse_args()
    
    if args.create_sample:
        if not args.data_dir:
            print("Usage: python benchmark_real.py --create_sample --data_dir ./test_data")
            return 1
        os.makedirs(args.data_dir, exist_ok=True)
        create_sample_data(args.data_dir)
        return 0
    
    if not args.data_dir or not args.labels:
        print("Usage: python benchmark_real.py --data_dir ./test_videos --labels ./labels.csv")
        print("\nOr create a template:")
        print("  python benchmark_real.py --create_sample --data_dir ./test_data")
        return 1
    
    # Load labels
    print(f"Loading labels from: {args.labels}")
    video_paths, ground_truth = load_labels_from_csv(args.labels)
    
    # Make paths absolute
    video_paths = [os.path.join(args.data_dir, p) for p in video_paths]
    
    print(f"Found {len(video_paths)} videos ({ground_truth.count('REAL')} real, {ground_truth.count('FAKE')} fake)")
    
    # Run benchmark
    runner = BenchmarkRunner(weights_path=args.weights)
    results = runner.run_full_benchmark(video_paths, ground_truth)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    for detector, metrics in results.items():
        print(f"\n{detector.upper()}:")
        if "error" in metrics:
            print(f"  ERROR: {metrics['error']}")
        else:
            print(f"  AUROC: {metrics.get('auroc', 'N/A')}")
            print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
            print(f"  Precision: {metrics.get('precision', 'N/A')}")
            print(f"  Recall: {metrics.get('recall', 'N/A')}")
            print(f"  F1 Score: {metrics.get('f1_score', 'N/A')}")
            print(f"  Avg Inference: {metrics.get('avg_inference_time', 0):.3f}s")
    
    print(f"\nFull results saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
