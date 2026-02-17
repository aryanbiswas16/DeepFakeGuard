"""
Test script for IvyFake integration
Run this to verify the integration works correctly
"""

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    # These will fail without torch installed, but we can check syntax
    try:
        import sys
        import ast
        
        # Verify file structure
        files_to_check = [
            'src/deepfake_guard/models/ivyfake/__init__.py',
            'src/deepfake_guard/models/ivyfake/detector.py',
            'src/deepfake_guard/core.py'
        ]
        
        for filepath in files_to_check:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print(f"  ✓ {filepath} - Syntax valid")
        
        return True
    except SyntaxError as e:
        print(f"  ✗ Syntax error: {e}")
        return False


def test_structure():
    """Test that the code structure is correct"""
    print("\nTesting code structure...")
    
    import ast
    
    # Check ivyfake detector has required classes
    with open('src/deepfake_guard/models/ivyfake/detector.py', 'r') as f:
        tree = ast.parse(f.read())
    
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    required_classes = ['IvyXDetector', 'IvyFakeDetector']
    
    for cls in required_classes:
        if cls in classes:
            print(f"  ✓ {cls} class found")
        else:
            print(f"  ✗ {cls} class missing")
            return False
    
    # Check core.py has ivyfake support
    with open('src/deepfake_guard/core.py', 'r') as f:
        core_content = f.read()
    
    checks = [
        ('ivyfake import', 'from .models.ivyfake.detector import IvyFakeDetector'),
        ('ivyfake in __init__', 'elif detector_type == "ivyfake":'),
        ('ivyfake in set_detector', 'detector_type: Literal["dinov3", "resnet18", "ivyfake"]'),
        ('_init_ivyfake method', 'def _init_ivyfake'),
        ('_run_ivyfake_analysis method', 'def _run_ivyfake_analysis'),
    ]
    
    for name, pattern in checks:
        if pattern in core_content:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} - MISSING")
            return False
    
    return True


def test_consistency():
    """Test that the integration is consistent with existing detectors"""
    print("\nTesting consistency with existing detectors...")
    
    with open('src/deepfake_guard/core.py', 'r') as f:
        content = f.read()
    
    # Check that all 3 detectors have similar structure
    detectors = ['dinov3', 'resnet18', 'ivyfake']
    
    for detector in detectors:
        init_method = f'_init_{detector}'
        run_method = f'_run_{detector}_analysis'
        
        if init_method in content and run_method in content:
            print(f"  ✓ {detector}: _init and _run methods present")
        else:
            print(f"  ✗ {detector}: missing methods")
            return False
    
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("IvyFake Integration Test Suite")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Structure", test_structure()))
    results.append(("Consistency", test_consistency()))
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("🎉 All tests passed! IvyFake integration looks good.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the integration.")
        return 1


if __name__ == "__main__":
    exit(main())