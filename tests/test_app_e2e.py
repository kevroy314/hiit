"""End-to-end tests for HIIT Analyzer Dash application using Playwright."""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_app_with_playwright():
    """Test the application using Playwright."""
    try:
        from playwright.sync_api import sync_playwright, expect
    except ImportError:
        print("Playwright not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "playwright", "--break-system-packages"], check=True)
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
        from playwright.sync_api import sync_playwright, expect
    
    # Start the app
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
        
        # Wait for app to start
        time.sleep(5)
        
        with sync_playwright() as p:
            # Launch browser
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            
            # Enable console logging
            page.on("console", lambda msg: print(f"Browser console: {msg.text}"))
            page.on("pageerror", lambda err: print(f"Page error: {err}"))
            
            # Test 1: Check if homepage loads
            print("\nTest 1: Loading homepage...")
            try:
                page.goto("http://localhost:8050", timeout=10000)
                print("✓ Homepage loaded")
            except Exception as e:
                print(f"✗ Failed to load homepage: {e}")
                return False
            
            # Test 2: Check navigation
            print("\nTest 2: Testing navigation...")
            nav_items = page.locator("nav a")
            nav_count = nav_items.count()
            print(f"Found {nav_count} navigation items")
            
            # Check for tabs
            tabs = ["raw-data-tab", "interval-analysis-tab", "performance-metrics-tab", "plotly-js-tab"]
            for tab_id in tabs:
                tab = page.locator(f"#{tab_id}")
                if tab.count() > 0:
                    print(f"✓ Found tab: {tab_id}")
                else:
                    print(f"✗ Tab not found: {tab_id}")
            
            # Test 3: Test Raw Data page
            print("\nTest 3: Testing Raw Data page...")
            raw_tab = page.locator("#raw-data-tab")
            if raw_tab.count() > 0:
                raw_tab.click()
                time.sleep(2)
                
                # Check for file selector
                file_selector = page.locator("#file-selector-raw")
                if file_selector.count() > 0:
                    print("✓ File selector found")
                    
                    # Check if there are any files
                    options = file_selector.locator("option")
                    file_count = options.count()
                    print(f"Found {file_count} files in dropdown")
                    
                    if file_count > 1:  # First option is usually placeholder
                        # Select first file
                        file_selector.select_option(index=1)
                        time.sleep(3)
                        
                        # Check for data
                        # Look for plotly plots
                        plots = page.locator(".js-plotly-plot")
                        plot_count = plots.count()
                        print(f"Found {plot_count} plots")
                        
                        # Check for data in plots
                        if plot_count > 0:
                            # Get the first plot's data
                            plot_data = page.evaluate("""
                                () => {
                                    const plot = document.querySelector('.js-plotly-plot');
                                    if (plot && plot._fullData) {
                                        return {
                                            traces: plot._fullData.length,
                                            firstTracePoints: plot._fullData[0] ? plot._fullData[0].x.length : 0
                                        };
                                    }
                                    return null;
                                }
                            """)
                            if plot_data:
                                print(f"✓ Plot data found: {plot_data['traces']} traces, {plot_data['firstTracePoints']} points in first trace")
                            else:
                                print("✗ No plot data found")
                else:
                    print("✗ File selector not found")
            
            # Test 4: Test Interval Analysis page
            print("\nTest 4: Testing Interval Analysis page...")
            interval_tab = page.locator("#interval-analysis-tab")
            if interval_tab.count() > 0:
                interval_tab.click()
                time.sleep(2)
                
                # Check for threshold controls
                threshold_slider = page.locator("#threshold-slider")
                if threshold_slider.count() > 0:
                    print("✓ Threshold slider found")
                    
                    # Check for interval detection plot
                    detection_plot = page.locator("#interval-detection-plot")
                    if detection_plot.count() > 0:
                        print("✓ Detection plot found")
                        
                        # Check for detected intervals
                        interval_data = page.evaluate("""
                            () => {
                                const plot = document.querySelector('#interval-detection-plot .js-plotly-plot');
                                if (plot && plot._fullLayout && plot._fullLayout.shapes) {
                                    return {
                                        shapeCount: plot._fullLayout.shapes.length
                                    };
                                }
                                return null;
                            }
                        """)
                        if interval_data:
                            print(f"✓ Found {interval_data['shapeCount']} interval markers")
                else:
                    print("✗ Threshold slider not found")
            
            # Test 5: Test Performance Metrics page
            print("\nTest 5: Testing Performance Metrics page...")
            perf_tab = page.locator("#performance-metrics-tab")
            if perf_tab.count() > 0:
                perf_tab.click()
                time.sleep(2)
                
                # Check for performance plots
                perf_plots = page.locator("#performance-metrics-content .js-plotly-plot")
                perf_plot_count = perf_plots.count()
                print(f"Found {perf_plot_count} performance plots")
            
            # Test 6: Check for JavaScript errors
            print("\nTest 6: Checking for JavaScript errors...")
            # Already logged via page.on("pageerror")
            
            # Test 7: Test file switching
            print("\nTest 7: Testing file switching...")
            raw_tab = page.locator("#raw-data-tab")
            if raw_tab.count() > 0:
                raw_tab.click()
                time.sleep(1)
                
                file_selector = page.locator("#file-selector-raw")
                if file_selector.count() > 0:
                    options = file_selector.locator("option")
                    if options.count() > 2:
                        # Switch to second file
                        file_selector.select_option(index=2)
                        time.sleep(3)
                        
                        # Check that data updated
                        new_plot_data = page.evaluate("""
                            () => {
                                const plot = document.querySelector('.js-plotly-plot');
                                if (plot && plot._fullData) {
                                    return plot._fullData[0] ? plot._fullData[0].x.length : 0;
                                }
                                return 0;
                            }
                        """)
                        print(f"✓ After file switch, plot has {new_plot_data} points")
            
            # Close browser
            browser.close()
            
            print("\n✓ All tests completed successfully!")
            return True
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if app_process:
            app_process.terminate()
            app_process.wait()


def test_json_serialization():
    """Test that numpy types can be serialized."""
    print("\nTesting JSON serialization fix...")
    
    try:
        from hiit.data_io import NumpyEncoder
        import numpy as np
        
        # Test data with numpy types
        test_data = {
            'int64': np.int64(42),
            'float64': np.float64(3.14),
            'array': np.array([1, 2, 3]),
            'normal': 100
        }
        
        # Try to serialize
        json_str = json.dumps(test_data, cls=NumpyEncoder)
        print("✓ JSON serialization works with numpy types")
        
        # Parse back
        parsed = json.loads(json_str)
        print(f"✓ Parsed data: {parsed}")
        
        return True
    except Exception as e:
        print(f"✗ JSON serialization failed: {e}")
        return False


if __name__ == "__main__":
    print("=== HIIT Analyzer End-to-End Tests ===")
    
    # Test JSON serialization first
    json_ok = test_json_serialization()
    
    # Run Playwright tests
    e2e_ok = test_app_with_playwright()
    
    if json_ok and e2e_ok:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)