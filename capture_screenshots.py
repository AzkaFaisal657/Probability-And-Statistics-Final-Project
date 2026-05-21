"""
Screenshot Capture Script for Dashboard
This script uses Selenium to capture screenshots of each dashboard tab.
"""

import os
import time
from pathlib import Path

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    print("ERROR: Selenium not installed. Install it with: pip install selenium")
    exit(1)

# Screenshot directory
SCREENSHOT_DIR = Path(__file__).parent / "screenshots"
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Dashboard URL
DASHBOARD_URL = "http://localhost:8050/"

# Tab selectors and filenames
TABS = [
    ("Overview Dashboard", "tab1-overview.png"),
    ("Frequency & Distributions", "tab2-frequency.png"),
    ("EDA & Shape of Data", "tab3-eda.png"),
    ("Probability Distributions", "tab4-probability.png"),
    ("Regression & Predictions", "tab5-regression.png"),
]


def capture_screenshots():
    """Capture screenshots of each dashboard tab."""
    
    # Initialize Chrome driver
    try:
        driver = webdriver.Chrome()
    except Exception as e:
        print(f"ERROR: Could not initialize Chrome driver: {e}")
        print("Make sure you have ChromeDriver installed and in your PATH")
        print("Download from: https://chromedriver.chromium.org/")
        return
    
    try:
        print(f"Opening dashboard at {DASHBOARD_URL}")
        driver.get(DASHBOARD_URL)
        
        # Wait for page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "tab"))
        )
        time.sleep(2)  # Additional wait for charts to render
        
        # Capture each tab
        for tab_name, filename in TABS:
            try:
                print(f"Capturing '{tab_name}'...")
                
                # Find and click tab
                tab_element = driver.find_element(
                    By.XPATH, 
                    f"//div[contains(@class, 'tab') and contains(text(), '{tab_name}')]"
                )
                driver.execute_script("arguments[0].click();", tab_element)
                
                # Wait for content to load
                time.sleep(3)
                
                # Scroll to top
                driver.execute_script("window.scrollTo(0, 0);")
                time.sleep(1)
                
                # Save screenshot
                screenshot_path = SCREENSHOT_DIR / filename
                driver.save_screenshot(str(screenshot_path))
                print(f"  ✓ Saved: {screenshot_path}")
                
            except Exception as e:
                print(f"  ✗ Error capturing '{tab_name}': {e}")
        
        print("\nScreenshots captured successfully!")
        print(f"Saved to: {SCREENSHOT_DIR}")
        
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        driver.quit()


if __name__ == "__main__":
    print("Dashboard Screenshot Capture Tool")
    print("=" * 50)
    print("\nMake sure the dashboard is running on http://localhost:8050/")
    print("Start it with: python app.py\n")
    
    input("Press Enter to start capturing screenshots...")
    capture_screenshots()
