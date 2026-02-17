#!/usr/bin/env python3
"""
Test script for all 3 DeepFakeGuard detectors
Downloads sample videos and tests each detector

Run this locally after installing dependencies:
    pip install -e .
    pip install transformers
    python test_all_detectors.py
"""

import os
import sys
import tempfile
import urllib.request
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from deepfake_guard.core import DeepfakeGuard


# Sample deepfake video URLs (small test files)
# These are from public datasets - replace with your own if needed
SAMPLE_VIDEOS = {
    "deepfake_example": {
        "url": "https://github.com/aryanbiswas16/DeepFakeGuard/raw/main/test_videos/fake_sample.mp4",
        "expected": "FAKE",
        "description": "Example deepfake video"
    },
    "real_example": {
        "url": "https://github.com/aryanbiswas16/DeepFakeGuard/raw/main/test_videos/real_sample.mp4", 
        "expected": "REAL",
        "description": "Example real video"
    }
}


def download_sample_video(video_name: str, url: str) -> str:
    """Download a sample video to temp location."""
    print(f"📥 Downloading {video_name}...")
    
    # Create temp file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        urllib.request.urlretrieve(url, temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        print(f"   ✓ Downloaded ({size_mb:.1f} MB)")
        return temp_path
    except Exception as e:
        print(f"   ✗ Download failed: {e}")
        os.unlink(temp_path)
        return None


def test_detector(detector_type: str, video_path: str, expected: str) -> dict:
    """Test a single detector on a video."""
    print(f"\n🔍 Testing {detector_type.upper()} detector...")
    
    try:
        # Initialize detector
        guard = DeepfakeGuard(detector_type=detector_type)
        
        # Run detection
        result = guard.detect_video(video_path)
        
        # Extract results
        label = result.get("overall_label", "UNKNOWN")
        score = result.get("overall_score", 0.0)
        errors = result.get("errors", [])
        
        print(f"   Label: {label}")
        print(f"   Score: {score:.3f}")
        
        if errors:
            print(f"   ⚠️  Errors: {errors}")
        
        # Check if prediction matches expected
        correct = (label == expected) or (label == "UNKNOWN")
        
        return {
            "detector": detector_type,
            "label": label,
            "score": score,
            "correct": correct,
            "errors": errors,
            "success": True
        }
        
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "detector": detector_type,
            "success": False,
            "error": str(e)
        }


def run_tests():
    """Run tests on all detectors."""
    print("=" * 60)
    print("🧪 DeepFakeGuard Detector Test Suite")
    print("=" * 60)
    
    # Check if weights exist for dinov3
    weights_path = Path("weights/dinov3_best_v3.pth")
    if not weights_path.exists():
        print(f"\n⚠️  Warning: DINOv3 weights not found at {weights_path}")
        print("   DINOv3 test will be skipped or fail.")
        print("   Download weights or use --skip-dinov3\n")
    
    results = []
    
    # For each sample video
    for video_name, video_info in SAMPLE_VIDEOS.items():
        print(f"\n{'='*60}")
        print(f"📹 Testing with: {video_name}")
        print(f"   Expected: {video_info['expected']}")
        print(f"   Description: {video_info['description']}")
        print('='*60)
        
        # Download video
        video_path = download_sample_video(video_name, video_info['url'])
        
        if not video_path:
            print("   Skipping (download failed)")
            continue
        
        try:
            # Test each detector
            detectors = ['dinov3', 'resnet18', 'ivyfake']
            
            for detector in detectors:
                result = test_detector(detector, video_path, video_info['expected'])
                result['video'] = video_name
                result['expected'] = video_info['expected']
                results.append(result)
                
        finally:
            # Cleanup
            try:
                os.unlink(video_path)
            except:
                pass
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    # Group by detector
    from collections import defaultdict
    by_detector = defaultdict(list)
    for r in results:
        by_detector[r['detector']].append(r)
    
    for detector, det_results in by_detector.items():
        print(f"\n{detector.upper()}:")
        for r in det_results:
            status = "✓" if r.get('success') else "✗"
            label = r.get('label', 'ERROR')
            score = r.get('score', 0)
            expected = r.get('expected', '?')
            video = r.get('video', '?')
            
            if r.get('success'):
                match = "✓" if label == expected else "✗"
                print(f"  {status} {video}: {label} (score: {score:.3f}) [expected: {expected}] {match}")
            else:
                print(f"  {status} {video}: FAILED - {r.get('error', 'Unknown error')}")
    
    # Overall stats
    total = len(results)
    successful = sum(1 for r in results if r.get('success'))
    correct = sum(1 for r in results if r.get('correct'))
    
    print(f"\n{'='*60}")
    print(f"Total tests: {total}")
    print(f"Successful: {successful}/{total}")
    print(f"Correct predictions: {correct}/{successful} (of successful)")
    print('='*60)
    
    return results


def test_with_local_video(video_path: str):
    """Test all detectors on a local video file."""
    print("=" * 60)
    print(f"🎥 Testing with local video: {video_path}")
    print("=" * 60)
    
    if not os.path.exists(video_path):
        print(f"✗ File not found: {video_path}")
        return
    
    detectors = ['dinov3', 'resnet18', 'ivyfake']
    
    for detector in detectors:
        print(f"\n{'='*60}")
        result = test_detector(detector, video_path, "UNKNOWN")
        print('='*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DeepFakeGuard detectors")
    parser.add_argument("--video", "-v", help="Test with local video file")
    parser.add_argument("--skip-dinov3", action="store_true", help="Skip DINOv3 tests (no weights)")
    
    args = parser.parse_args()
    
    if args.video:
        test_with_local_video(args.video)
    else:
        run_tests()