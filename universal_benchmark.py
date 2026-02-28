#!/usr/bin/env python3
"""
DeepFakeGuard - Universal Benchmark System
Works with any available data: real datasets, synthetic, or mixed.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent / "src"))
from deepfake_guard.core import DeepfakeGuard


class UniversalBenchmark:
    """Benchmark system that adapts to available data."""
    
    def __init__(self, data_dir: str = "./benchmark_data"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(__file__).parent / "benchmark_results"
        self.results_dir.mkdir(exist_ok=True)
        
    def scan_for_data(self) -> Dict[str, List[Path]]:
        """Scan for any available video data."""
        data = {"real": [], "fake": []}
        
        # Check benchmark_data structure
        for label in ["real", "fake"]:
            folder = self.data_dir / label
            if folder.exists():
                data[label] = list(folder.glob("*.mp4")) + list(folder.glob("*.avi")) + list(folder.glob("*.mov"))
        
        # Also check root of benchmark_data
        if not data["real"] and not data["fake"]:
            all_videos = list(self.data_dir.glob("*.mp4")) + list(self.data_dir.glob("*.avi"))
            # Try to infer from filenames
            for v in all_videos:
                name = v.name.lower()
                if "real" in name or "original" in name:
                    data["real"].append(v)
                elif "fake" in name or "deep" in name or "synth" in name:
                    data["fake"].append(v)
        
        return data
    
    def generate_realistic_synthetic_data(self, num_real: int = 10, num_fake: int = 10) -> Dict[str, List[Path]]:
        """
        Generate synthetic videos with patterns that detectors can actually distinguish.
        These won't match real deepfakes, but they allow relative detector comparison.
        """
        print("Generating realistic synthetic benchmark data...")
        
        synth_dir = self.data_dir / "synthetic"
        synth_dir.mkdir(exist_ok=True)
        (synth_dir / "real").mkdir(exist_ok=True)
        (synth_dir / "fake").mkdir(exist_ok=True)
        
        generated = {"real": [], "fake": []}
        
        # Generate "real" videos - smooth natural motion
        for i in range(num_real):
            path = synth_dir / "real" / f"synth_real_{i:03d}.mp4"
            self._create_real_video(str(path), seed=i)
            generated["real"].append(path)
            print(f"  Created {path.name}")
        
        # Generate "fake" videos - with artifacts
        for i in range(num_fake):
            path = synth_dir / "fake" / f"synth_fake_{i:03d}.mp4"
            self._create_fake_video(str(path), seed=i)
            generated["fake"].append(path)
            print(f"  Created {path.name}")
        
        return generated
    
    def _create_real_video(self, path: str, seed: int = 0, num_frames: int = 30):
        """Create synthetic 'real' video with smooth natural patterns."""
        np.random.seed(seed)
        size = (224, 224)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, 10, size)
        
        # Smooth motion - like natural video
        base_color = np.array([100, 120, 140], dtype=np.float32)
        
        for frame_idx in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            # Smooth color transition (natural lighting change)
            t = frame_idx / num_frames
            color_shift = np.sin(t * np.pi * 2) * 20
            color = np.clip(base_color + color_shift, 0, 255).astype(np.uint8)
            frame[:, :] = color
            
            # Smooth moving object (natural motion)
            cx = int(size[0] * (0.3 + 0.4 * t))
            cy = size[1] // 2 + int(10 * np.sin(t * np.pi * 4))
            cv2.circle(frame, (cx, cy), 25, (255, 255, 255), -1)
            
            # Add subtle noise (camera sensor noise)
            noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            out.write(frame)
        
        out.release()
    
    def _create_fake_video(self, path: str, seed: int = 0, num_frames: int = 30):
        """Create synthetic 'fake' video with artifacts (jitter, inconsistencies)."""
        np.random.seed(seed + 1000)
        size = (224, 224)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, 10, size)
        
        base_color = np.array([100, 120, 140], dtype=np.float32)
        
        for frame_idx in range(num_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            # Abrupt color changes (inconsistent lighting)
            t = frame_idx / num_frames
            if frame_idx % 5 == 0:  # Periodic jumps
                color_shift = np.random.uniform(-40, 40)
            else:
                color_shift = np.sin(t * np.pi * 2) * 20
            
            color = np.clip(base_color + color_shift, 0, 255).astype(np.uint8)
            frame[:, :] = color
            
            # Jittery motion (unnatural)
            cx = int(size[0] * (0.3 + 0.4 * t))
            cy = size[1] // 2 + int(10 * np.sin(t * np.pi * 4))
            
            # Add position jitter
            cx += np.random.randint(-5, 5)
            cy += np.random.randint(-5, 5)
            
            cv2.circle(frame, (cx, cy), 25, (255, 255, 255), -1)
            
            # Add compression-like artifacts
            block_size = 8
            for y in range(0, size[1], block_size):
                for x in range(0, size[0], block_size):
                    if np.random.random() < 0.1:  # 10% blocks affected
                        block = frame[y:y+block_size, x:x+block_size]
                        # Slight color shift in block
                        shift = np.random.randint(-10, 10, size=3)
                        frame[y:y+block_size, x:x+block_size] = np.clip(block.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            
            out.write(frame)
        
        out.release()
    
    def run_detector(self, detector_type: str, video_path: Path, weights_path: str = None) -> Dict:
        """Run a single detector on a video."""
        try:
            guard = DeepfakeGuard(detector_type=detector_type, weights_path=weights_path)
            
            start = time.time()
            result = guard.detect_video(str(video_path))
            infer_time = time.time() - start
            
            return {
                "label": result.get("overall_label", "UNKNOWN"),
                "score": result.get("overall_score", 0.5),
                "time": infer_time,
                "success": True
            }
        except Exception as e:
            return {
                "label": "ERROR",
                "score": 0.5,
                "time": 0,
                "success": False,
                "error": str(e)
            }
    
    def run_benchmark(self, data: Dict[str, List[Path]], use_synthetic: bool = False) -> Dict:
        """Run full benchmark on available data."""
        print(f"\n{'='*60}")
        print(f"Running Benchmark")
        print(f"{'='*60}")
        print(f"Real videos: {len(data['real'])}")
        print(f"Fake videos: {len(data['fake'])}")
        
        if use_synthetic:
            print("⚠️  Using synthetic data - results are for relative comparison only")
        
        weights = "weights/dinov3_best_v3.pth" if os.path.exists("weights/dinov3_best_v3.pth") else None
        
        detectors = ["resnet18", "ivyfake", "d3", "dinov3"]
        results = {d: {"predictions": [], "scores": [], "times": [], "ground_truth": []} for d in detectors}
        
        # Test each detector
        for detector in detectors:
            print(f"\n{'-'*60}")
            print(f"Testing: {detector.upper()}")
            print(f"{'-'*60}")
            
            det_weights = weights if detector == "dinov3" else None
            
            # Test real videos
            for video in data["real"]:
                print(f"  [REAL] {video.name}...", end=" ")
                r = self.run_detector(detector, video, det_weights)
                results[detector]["predictions"].append(r["label"])
                results[detector]["scores"].append(r["score"])
                results[detector]["times"].append(r["time"])
                results[detector]["ground_truth"].append("REAL")
                print(f"{r['label']} ({r['score']:.3f})" if r["success"] else f"ERROR: {r.get('error', '')[:30]}")
            
            # Test fake videos
            for video in data["fake"]:
                print(f"  [FAKE] {video.name}...", end=" ")
                r = self.run_detector(detector, video, det_weights)
                results[detector]["predictions"].append(r["label"])
                results[detector]["scores"].append(r["score"])
                results[detector]["times"].append(r["time"])
                results[detector]["ground_truth"].append("FAKE")
                print(f"{r['label']} ({r['score']:.3f})" if r["success"] else f"ERROR: {r.get('error', '')[:30]}")
        
        return results
    
    def compute_metrics(self, results: Dict) -> Dict:
        """Compute metrics from predictions."""
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {}
        
        for detector, data in results.items():
            if not data["predictions"]:
                metrics[detector] = {"error": "No predictions"}
                continue
            
            y_true = [1 if gt == "FAKE" else 0 for gt in data["ground_truth"]]
            y_pred = [1 if p == "FAKE" else 0 for p in data["predictions"]]
            y_scores = data["scores"]
            
            try:
                metrics[detector] = {
                    "auroc": round(roc_auc_score(y_true, y_scores), 4),
                    "accuracy": round(accuracy_score(y_true, y_pred), 4),
                    "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                    "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                    "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
                    "avg_inference_time": round(np.mean(data["times"]), 4),
                    "num_samples": len(y_true)
                }
            except Exception as e:
                metrics[detector] = {"error": str(e)}
        
        return metrics
    
    def execute(self, force_synthetic: bool = False):
        """Main execution - automatically handles data availability."""
        print("="*60)
        print("DeepFakeGuard Universal Benchmark")
        print("="*60)
        
        # Scan for real data
        data = self.scan_for_data()
        use_synthetic = force_synthetic or (not data["real"] and not data["fake"])
        
        if use_synthetic:
            print("\n⚠️  No real data found. Generating synthetic benchmark...")
            print("   This allows detector comparison but won't match real deepfake performance.")
            data = self.generate_realistic_synthetic_data(num_real=10, num_fake=10)
        
        # Run benchmark
        raw_results = self.run_benchmark(data, use_synthetic=use_synthetic)
        metrics = self.compute_metrics(raw_results)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"benchmark_{timestamp}.json"
        
        full_results = {
            "timestamp": timestamp,
            "data_type": "synthetic" if use_synthetic else "real",
            "num_real": len(data["real"]),
            "num_fake": len(data["fake"]),
            "metrics": metrics,
            "raw_results": {k: {sk: sv for sk, sv in v.items() if sk != "ground_truth"} 
                          for k, v in raw_results.items()}
        }
        
        with open(output_file, "w") as f:
            json.dump(full_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(f"Data type: {'SYNTHETIC' if use_synthetic else 'REAL'}")
        print(f"Real videos: {len(data['real'])}")
        print(f"Fake videos: {len(data['fake'])}")
        
        if use_synthetic:
            print("\n⚠️  WARNING: These are synthetic results!")
            print("   Use for relative detector comparison only.")
            print("   For publication-quality results, use real deepfake datasets:")
            print("   - FaceForensics++: https://github.com/ondyari/FaceForensics")
            print("   - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics")
        
        print(f"\n{'Detector':<15} {'AUROC':<8} {'Acc':<8} {'Prec':<8} {'Recall':<8} {'Time':<10}")
        print("-" * 70)
        
        for detector, m in metrics.items():
            if "error" in m:
                print(f"{detector:<15} ERROR: {m['error'][:40]}")
            else:
                print(f"{detector:<15} {m.get('auroc', 'N/A'):<8} {m.get('accuracy', 'N/A'):<8} "
                      f"{m.get('precision', 'N/A'):<8} {m.get('recall', 'N/A'):<8} "
                      f"{m.get('avg_inference_time', 0):.3f}s")
        
        print(f"\n✓ Results saved to: {output_file}")
        
        return metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Universal DeepFakeGuard Benchmark")
    parser.add_argument("--data-dir", "-d", default="./benchmark_data", help="Data directory")
    parser.add_argument("--synthetic", "-s", action="store_true", help="Force synthetic data")
    args = parser.parse_args()
    
    benchmark = UniversalBenchmark(data_dir=args.data_dir)
    benchmark.execute(force_synthetic=args.synthetic)


if __name__ == "__main__":
    main()
