"""End-to-end test using Docker to run the HIIT Analyzer."""

import os
import sys
import subprocess
import time
import urllib.request
import json

def check_docker():
    """Check if Docker is available."""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Docker found: {result.stdout.strip()}")
            return True
        else:
            print("✗ Docker not available")
            return False
    except:
        print("✗ Docker not installed")
        return False


def build_and_run_docker():
    """Build and run the Docker container."""
    print("\n=== Building and Running Docker Container ===")
    
    # First stop any existing containers
    print("Stopping existing containers...")
    subprocess.run(['docker-compose', 'down'], cwd=os.path.dirname(os.path.dirname(__file__)))
    time.sleep(2)
    
    # Build and start the container
    print("Building and starting container...")
    process = subprocess.Popen(
        ['docker-compose', 'up', '--build'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=os.path.dirname(os.path.dirname(__file__))
    )
    
    # Wait for the app to start
    print("Waiting for app to start...")
    app_started = False
    start_time = time.time()
    timeout = 60  # 60 seconds timeout
    
    while time.time() - start_time < timeout:
        line = process.stdout.readline()
        if line:
            print(f"Docker: {line.strip()}")
            
            # Check for successful start
            if 'Running on http://0.0.0.0:8050' in line or 'Listening at: http://0.0.0.0:8050' in line:
                app_started = True
                break
            
            # Check for errors
            if 'Error' in line or 'ERROR' in line or 'Failed' in line:
                print(f"✗ Error detected: {line.strip()}")
    
    return process, app_started


def test_app_endpoints():
    """Test that the app endpoints are working."""
    print("\n=== Testing App Endpoints ===")
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Homepage
    tests_total += 1
    try:
        response = urllib.request.urlopen('http://localhost:8050', timeout=10)
        if response.status == 200:
            html = response.read().decode('utf-8')
            if 'HIIT Analyzer' in html:
                print("✓ Homepage loads successfully")
                tests_passed += 1
            else:
                print("✗ Homepage missing title")
        else:
            print(f"✗ Homepage returned status {response.status}")
    except Exception as e:
        print(f"✗ Failed to load homepage: {e}")
    
    # Test 2: Dash update endpoint
    tests_total += 1
    try:
        # Try to get the layout
        req = urllib.request.Request(
            'http://localhost:8050/_dash-layout',
            headers={'Content-Type': 'application/json'}
        )
        response = urllib.request.urlopen(req, timeout=10)
        if response.status == 200:
            layout = json.loads(response.read().decode('utf-8'))
            if 'props' in layout:
                print("✓ Dash layout endpoint working")
                tests_passed += 1
            else:
                print("✗ Dash layout invalid")
        else:
            print(f"✗ Dash layout returned status {response.status}")
    except Exception as e:
        print(f"✗ Failed to get Dash layout: {e}")
    
    # Test 3: Check for JavaScript errors by examining the page
    tests_total += 1
    try:
        response = urllib.request.urlopen('http://localhost:8050', timeout=10)
        html = response.read().decode('utf-8')
        
        # Check for common error indicators
        error_indicators = ['TypeError:', 'ReferenceError:', 'SyntaxError:', 'Error loading']
        has_errors = any(indicator in html for indicator in error_indicators)
        
        if not has_errors:
            print("✓ No obvious JavaScript errors in page")
            tests_passed += 1
        else:
            print("✗ JavaScript errors detected in page")
    except Exception as e:
        print(f"✗ Failed to check for JavaScript errors: {e}")
    
    return tests_passed, tests_total


def test_data_loading():
    """Test that data can be loaded without errors."""
    print("\n=== Testing Data Loading ===")
    
    # This would require more complex interaction with the Dash app
    # For now, we'll just check that the endpoints respond
    
    try:
        # Try to trigger a callback
        data = json.dumps({
            "output": "page-content.children",
            "outputs": {"id": "page-content", "property": "children"},
            "inputs": [{"id": "current-page-store", "property": "data", "value": "raw"}],
            "changedPropIds": ["current-page-store.data"]
        })
        
        req = urllib.request.Request(
            'http://localhost:8050/_dash-update-component',
            data=data.encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        
        response = urllib.request.urlopen(req, timeout=10)
        if response.status == 200:
            result = json.loads(response.read().decode('utf-8'))
            if 'response' in result:
                print("✓ Dash callbacks responding")
                return True
            else:
                print("✗ Dash callback response invalid")
                return False
        else:
            print(f"✗ Dash callback returned status {response.status}")
            return False
    except Exception as e:
        print(f"✗ Failed to test callbacks: {e}")
        return False


def main():
    """Run all Docker-based tests."""
    print("=== HIIT Analyzer Docker E2E Tests ===")
    
    # Check Docker availability
    if not check_docker():
        print("\n❌ Docker is required for these tests")
        return 1
    
    # Build and run the app
    process, app_started = build_and_run_docker()
    
    if not app_started:
        print("\n✗ App failed to start within timeout")
        if process:
            process.terminate()
        return 1
    
    # Give it a bit more time to stabilize
    time.sleep(5)
    
    # Run tests
    try:
        endpoint_passed, endpoint_total = test_app_endpoints()
        data_loading_passed = test_data_loading()
        
        total_passed = endpoint_passed + (1 if data_loading_passed else 0)
        total_tests = endpoint_total + 1
        
        print(f"\n=== Test Summary ===")
        print(f"Endpoints: {endpoint_passed}/{endpoint_total} passed")
        print(f"Data Loading: {'✓ PASSED' if data_loading_passed else '✗ FAILED'}")
        print(f"\nTotal: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1
            
    finally:
        # Clean up
        print("\nStopping Docker container...")
        if process:
            process.terminate()
            process.wait()
        subprocess.run(['docker-compose', 'down'], cwd=os.path.dirname(os.path.dirname(__file__)))


if __name__ == "__main__":
    sys.exit(main())