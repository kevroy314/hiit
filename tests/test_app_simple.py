"""Simple test to check the HIIT Analyzer application without external dependencies."""

import os
import sys
import subprocess
import time
import urllib.request
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_json_serialization_fix():
    """Test that the JSON serialization fix is in place."""
    print("\n=== Testing JSON Serialization Fix ===")
    
    # Check that NumpyEncoder exists in data_io.py
    data_io_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hiit', 'data_io.py')
    
    if os.path.exists(data_io_path):
        with open(data_io_path, 'r') as f:
            content = f.read()
            
        if 'class NumpyEncoder' in content:
            print("✓ NumpyEncoder class found in data_io.py")
            
            # Check that json.dump uses the encoder
            if 'cls=NumpyEncoder' in content:
                print("✓ json.dump calls use NumpyEncoder")
                return True
            else:
                print("✗ json.dump calls don't use NumpyEncoder")
                return False
        else:
            print("✗ NumpyEncoder class not found in data_io.py")
            return False
    else:
        print("✗ data_io.py not found")
        return False


def test_csv_support():
    """Test that CSV files are supported."""
    print("\n=== Testing CSV Support ===")
    
    data_io_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'hiit', 'data_io.py')
    
    if os.path.exists(data_io_path):
        with open(data_io_path, 'r') as f:
            content = f.read()
            
        if '.csv' in content and 'pd.read_csv' in content:
            print("✓ CSV support found in data_io.py")
            return True
        else:
            print("✗ CSV support not found in data_io.py")
            return False
    else:
        print("✗ data_io.py not found")
        return False


def test_app_structure():
    """Test that the app has the correct structure."""
    print("\n=== Testing App Structure ===")
    
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app_dash.py')
    
    if not os.path.exists(app_path):
        print("✗ app_dash.py not found")
        return False
    
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Check for required navigation buttons
    required_navs = [
        'nav-raw',
        'nav-interval', 
        'nav-metrics'
    ]
    
    all_found = True
    for nav in required_navs:
        if nav in content:
            print(f"✓ Found navigation: {nav}")
        else:
            print(f"✗ Missing navigation: {nav}")
            all_found = False
    
    # Check for file selectors
    if 'file-selector-raw' in content:
        print("✓ Found raw data file selector")
    else:
        print("✗ Missing raw data file selector")
        all_found = False
    
    # Check for threshold controls
    if 'threshold-slider' in content or 'threshold-calibration' in content:
        print("✓ Found threshold controls")
    else:
        print("✗ Missing threshold controls")
        all_found = False
    
    return all_found


def test_app_runs():
    """Test that the app can start without crashing."""
    print("\n=== Testing App Startup ===")
    
    app_process = None
    try:
        # Start the application
        print("Starting application...")
        app_process = subprocess.Popen(
            [sys.executable, "app_dash.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        # Give it time to start
        time.sleep(5)
        
        # Check if process is still running
        if app_process.poll() is None:
            print("✓ App process is running")
            
            # Try to connect
            try:
                response = urllib.request.urlopen('http://localhost:8050', timeout=5)
                if response.status == 200:
                    print("✓ App is responding on port 8050")
                    
                    # Read response
                    html = response.read().decode('utf-8')
                    if 'HIIT Analyzer' in html:
                        print("✓ App title found in response")
                        return True
                    else:
                        print("✗ App title not found in response")
                        return False
                else:
                    print(f"✗ App returned status {response.status}")
                    return False
            except Exception as e:
                print(f"✗ Could not connect to app: {e}")
                
                # Check stderr for errors
                stderr = app_process.stderr.read().decode('utf-8')
                if stderr:
                    print(f"App errors:\n{stderr[:500]}")
                return False
        else:
            # Process crashed
            print("✗ App process crashed")
            stderr = app_process.stderr.read().decode('utf-8')
            if stderr:
                print(f"Error output:\n{stderr[:500]}")
            return False
            
    except Exception as e:
        print(f"✗ Failed to start app: {e}")
        return False
    finally:
        # Clean up
        if app_process and app_process.poll() is None:
            app_process.terminate()
            app_process.wait()
            print("App process terminated")


def main():
    """Run all tests."""
    print("=== HIIT Analyzer Simple Tests ===")
    
    results = {
        'JSON Serialization': test_json_serialization_fix(),
        'CSV Support': test_csv_support(),
        'App Structure': test_app_structure(),
        'App Startup': test_app_runs()
    }
    
    print("\n=== Test Summary ===")
    passed = 0
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())