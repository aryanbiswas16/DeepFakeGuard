#!/usr/bin/env python3
"""
Comprehensive Benchmark & Analysis Script for All 3 Detectors
Generates detailed performance report with metrics

Run locally:
    python benchmark_detectors.py --videos path/to/test/videos/
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepfake_guard.core import DeepfakeGuard


class DetectorBenchmark:
    """Benchmark and compare all 3 detectors."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "detectors": {},
            "comparisons": {}
        }
    
    def test_detector(self, detector_type: str, video_path: str, ground_truth: str = None) -> dict:
        """Test a single detector and collect metrics."""
        print(f"\n{'='*60}")
        print(f"🔍 Testing: {detector_type.upper()}")
        print(f"   Video: {os.path.basename(video_path)}")
        print(f"   Ground Truth: {ground_truth or 'Unknown'}")
        print('='*60)
        
        metrics = {
            "detector": detector_type,
            "video": os.path.basename(video_path),
            "ground_truth": ground_truth,
            "success": False
        }
        
        try:
            # Initialize
            init_start = time.time()
            guard = DeepfakeGuard(detector_type=detector_type)
            init_time = time.time() - init_start
            metrics["init_time_sec"] = round(init_time, 3)
            print(f"✓ Initialized in {init_time:.2f}s")
            
            # Detection
            detect_start = time.time()
            result = guard.detect_video(video_path)
            detect_time = time.time() - detect_start
            metrics["inference_time_sec"] = round(detect_time, 3)
            print(f"✓ Detection complete in {detect_time:.2f}s")
            
            # Extract results
            label = result.get("overall_label", "UNKNOWN")
            score = result.get("overall_score", 0.0)
            errors = result.get("errors", [])
            
            metrics["predicted_label"] = label
            metrics["confidence_score"] = round(score, 4)
            metrics["errors"] = errors
            
            # Check accuracy if ground truth provided
            if ground_truth:
                correct = (label == ground_truth)
                metrics["correct"] = correct
                metrics["accuracy"] = 1.0 if correct else 0.0
                print(f"   Prediction: {label} (Score: {score:.3f}) {'✓' if correct else '✗'}")
            else:
                print(f"   Prediction: {label} (Score: {score:.3f})")
            
            # Get detector-specific details
            modality_results = result.get("modality_results", {})
            for modality, data in modality_results.items():
                details = data.get("details", {})
                metrics["detector_details"] = {
                    "detector_type": details.get("detector_type"),
                    "frame_count": details.get("frame_count"),
                    "features": details.get("features", []),
                    "backbone": details.get("backbone"),
                    "note": details.get("note")
                }
            
            metrics["success"] = True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            metrics["error"] = str(e)
        
        return metrics
    
    def compare_detectors(self, video_path: str, ground_truth: str = None) -> dict:
        """Run all detectors on one video and compare."""
        print(f"\n{'='*70}")
        print(f"📹 COMPARISON TEST")
        print(f"   Video: {video_path}")
        print(f"   Ground Truth: {ground_truth or 'Unknown'}")
        print('='*70)
        
        detectors = ["dinov3", "resnet18", "ivyfake"]
        results = []
        
        for detector in detectors:
            result = self.test_detector(detector, video_path, ground_truth)
            results.append(result)
        
        # Comparison summary
        print(f"\n{'='*70}")
        print("📊 COMPARISON SUMMARY")
        print('='*70)
        
        if ground_truth:
            print(f"\nGround Truth: {ground_truth}")
            print("\nDetector Performance:")
            for r in results:
                if r.get("success"):
                    status = "✓ CORRECT" if r.get("correct") else "✗ WRONG"
                    print(f"  {r['detector']:12} -> {r['predicted_label']:6} (score: {r['confidence_score']:.3f}) {status}")
        else:
            print("\nDetector Predictions:")
            for r in results:
                if r.get("success"):
                    print(f"  {r['detector']:12} -> {r['predicted_label']:6} (score: {r['confidence_score']:.3f})")
        
        # Agreement analysis
        predictions = [r.get("predicted_label") for r in results if r.get("success")]
        if len(set(predictions)) == 1:
            print(f"\n✓ All detectors agree: {predictions[0]}")
        else:
            print(f"\n⚠ Detectors disagree:")
            for r in results:
                if r.get("success"):
                    print(f"  {r['detector']:12} -> {r['predicted_label']}")
        
        return results
    
    def generate_report(self, output_file: str = "benchmark_report.json"):
        """Generate detailed JSON report."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Report saved to: {output_file}")
    
    def print_summary(self):
        """Print overall benchmark summary."""
        print(f"\n{'='*70}")
        print("📈 OVERALL BENCHMARK SUMMARY")
        print('='*70)
        
        # Aggregate by detector
        by_detector = defaultdict(lambda: {"tests": 0, "correct": 0, "total_time": 0})
        
        for comparison in self.results.get("comparisons", {}).values():
            for test in comparison:
                det = test["detector"]
                by_detector[det]["tests"] += 1
                by_detector[det]["correct"] += test.get("correct", 0)
                by_detector[det]["total_time"] += test.get("inference_time_sec", 0)
        
        print("\nPerformance by Detector:")
        for detector, stats in by_detector.items():
            accuracy = (stats["correct"] / stats["tests"] * 100) if stats["tests"] > 0 else 0
            avg_time = stats["total_time"] / stats["tests"] if stats["tests"] > 0 else 0
            
            print(f"\n  {detector.upper()}:")
            print(f"    Tests: {stats['tests']}")
            print(f"    Accuracy: {stats['correct']}/{stats['tests']} ({accuracy:.1f}%)")
            print(f"    Avg Time: {avg_time:.2f}s")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark all DeepFakeGuard detectors")
    parser.add_argument("--video", "-v", help="Single video to test")
    parser.add_argument("--videos", "-d", help="Directory of videos to test")
    parser.add_argument("--ground-truth", "-g", choices=["FAKE", "REAL"], help="Ground truth label")
    parser.add_argument("--output", "-o", default="benchmark_report.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    benchmark = DetectorBenchmark()
    
    if args.video:
        # Single video comparison
        results = benchmark.compare_detectors(args.video, args.ground_truth)
        
    elif args.videos:
        # Directory of videos
        video_dir = Path(args.videos)
        video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mov")) + list(video_dir.glob("*.avi"))
        
        print(f"Found {len(video_files)} videos in {video_dir}")
        
        for video_path in video_files:
            # Try to infer ground truth from filename
            ground_truth = None
            if "fake" in video_path.name.lower() or "deepfake" in video_path.name.lower():
                ground_truth = "FAKE"
            elif "real" in video_path.name.lower() or "original" in video_path.name.lower():
                ground_truth = "REAL"
            
            results = benchmark.compare_detectors(str(video_path), ground_truth)
    
    else:
        print("Usage:")
        print("  python benchmark_detectors.py -v video.mp4 -g FAKE")
        print("  python benchmark_detectors.py -d /path/to/videos/")
        return
    
    # Generate report
    benchmark.generate_report(args.output)


if __name__ == "__main__":
    main()