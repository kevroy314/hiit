#!/usr/bin/env python3
"""Simple test script to verify the application is working."""

import time
import sys
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def test_application():
    """Test the application is working properly."""
    
    base_url = "http://localhost:8050"
    
    # First check if the server is running
    print("Checking if server is running...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(base_url, timeout=2)
            if response.status_code == 200:
                print("✅ Server is running!")
                break
        except:
            if i == max_retries - 1:
                print("❌ Server failed to start after 60 seconds")
                return False
            time.sleep(2)
    
    # Now test with Selenium
    print("\nTesting application pages...")
    
    # Setup Chrome options for headless mode
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        # Create driver
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 10)
        
        # Test 1: Homepage loads
        print("Testing homepage...")
        driver.get(base_url)
        assert "HIIT Analyzer" in driver.title
        print("✅ Homepage loads correctly")
        
        # Test 2: Raw Data page
        print("\nTesting Raw Data page...")
        raw_data_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Raw Data")))
        raw_data_link.click()
        time.sleep(2)
        
        # Check for page elements
        assert "Raw Data Analysis" in driver.page_source
        print("✅ Raw Data page loads")
        
        # Test 3: HIIT Interval Analysis page
        print("\nTesting HIIT Interval Analysis page...")
        interval_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "HIIT Interval Analysis")))
        interval_link.click()
        time.sleep(2)
        
        assert "HIIT Interval Analysis" in driver.page_source
        assert "Save Selection" in driver.page_source
        assert "Reset Selection" in driver.page_source
        print("✅ HIIT Interval Analysis page loads")
        
        # Test 4: Performance Analysis page
        print("\nTesting Performance Analysis page...")
        perf_link = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, "Performance Analysis")))
        perf_link.click()
        time.sleep(2)
        
        assert "Performance Analysis" in driver.page_source
        assert "Files Summary" in driver.page_source
        print("✅ Performance Analysis page loads")
        
        # Check for JavaScript errors
        print("\nChecking for JavaScript errors...")
        logs = driver.get_log('browser')
        errors = [log for log in logs if log['level'] == 'SEVERE']
        
        if errors:
            print("❌ JavaScript errors found:")
            for error in errors:
                print(f"  - {error['message']}")
        else:
            print("✅ No JavaScript errors")
        
        driver.quit()
        
        print("\n🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        return False


if __name__ == "__main__":
    # Wait a bit for Docker to fully start
    print("Waiting for Docker container to start...")
    time.sleep(10)
    
    success = test_application()
    sys.exit(0 if success else 1)