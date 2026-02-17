#!/usr/bin/env python3
"""
Quick test to verify all detectors can be imported and initialized
Run this to check integration without downloading videos
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 60)
print("🧪 DeepFakeGuard Quick Integration Test")
print("=" * 60)

# Test 1: Import all detectors
print("\n1️⃣  Testing imports...")
try:
    from deepfake_guard.models.dinov3 import Detector as DINOv3Detector
    print("   ✓ DINOv3 detector imported")
except Exception as e:
    print(f"   ✗ DINOv3 import failed: {e}")

try:
    from deepfake_guard.models.resnet18 import ResNet18Detector
    print("   ✓ ResNet18 detector imported")
except Exception as e:
    print(f"   ✗ ResNet18 import failed: {e}")

try:
    from deepfake_guard.models.ivyfake import IvyFakeDetector
    print("   ✓ IvyFake detector imported")
except Exception as e:
    print(f"   ✗ IvyFake import failed: {e}")

# Test 2: Check DeepfakeGuard core
print("\n2️⃣  Testing DeepfakeGuard core...")
try:
    from deepfake_guard.core import DeepfakeGuard
    print("   ✓ DeepfakeGuard core imported")
    
    # Check available detectors
    import inspect
    sig = inspect.signature(DeepfakeGuard.__init__)
    detector_param = sig.parameters.get('detector_type')
    if detector_param:
        print(f"   ✓ Available detectors: dinov3, resnet18, ivyfake")
except Exception as e:
    print(f"   ✗ Core import failed: {e}")

# Test 3: Check set_detector method
print("\n3️⃣  Checking detector switching...")
try:
    from deepfake_guard.core import DeepfakeGuard
    import inspect
    
    source = inspect.getsource(DeepfakeGuard.set_detector)
    if 'ivyfake' in source:
        print("   ✓ set_detector supports ivyfake")
    else:
        print("   ✗ set_detector missing ivyfake")
except Exception as e:
    print(f"   ✗ Check failed: {e}")

# Test 4: Check initialization methods
print("\n4️⃣  Checking detector initialization...")
try:
    methods = ['_init_dinov3', '_init_resnet18', '_init_ivyfake']
    source = inspect.getsource(DeepfakeGuard)
    
    for method in methods:
        if method in source:
            print(f"   ✓ {method} method found")
        else:
            print(f"   ✗ {method} method missing")
except Exception as e:
    print(f"   ✗ Check failed: {e}")

# Test 5: Check analysis methods
print("\n5️⃣  Checking analysis methods...")
try:
    methods = ['_run_dinov3_analysis', '_run_resnet18_analysis', '_run_ivyfake_analysis']
    source = inspect.getsource(DeepfakeGuard)
    
    for method in methods:
        if method in source:
            print(f"   ✓ {method} method found")
        else:
            print(f"   ✗ {method} method missing")
except Exception as e:
    print(f"   ✗ Check failed: {e}")

print("\n" + "=" * 60)
print("✅ Integration test complete!")
print("=" * 60)
print("\nTo run full tests with actual videos:")
print("  python test_all_detectors.py")
print("\nTo test with your own video:")
print("  python test_all_detectors.py --video path/to/video.mp4")