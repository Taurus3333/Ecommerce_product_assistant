import os
import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By

def main():
    print("\n🔍 [1/4] Checking environment variable...")
    chromedriver_path = os.getenv("CHROMEDRIVER_PATH")
    if not chromedriver_path:
        raise RuntimeError(
            "❌ Environment variable CHROMEDRIVER_PATH is not set.\n"
            "Please set it to the full path of your ChromeDriver executable.\n"
            "Example (Git Bash): export CHROMEDRIVER_PATH='/c/Program Files/chromedriver-win64/chromedriver-win64/chromedriver.exe'"
        )

    if not os.path.exists(chromedriver_path):
        raise FileNotFoundError(f"❌ ChromeDriver not found at: {chromedriver_path}")

    print(f"✅ CHROMEDRIVER_PATH: {chromedriver_path}")

    print("\n🔍 [2/4] Checking ChromeDriver version...")
    try:
        version_output = os.popen(f'"{chromedriver_path}" --version').read().strip()
        print(f"✅ ChromeDriver version: {version_output}")
    except Exception as e:
        print(f"⚠️ Could not read driver version: {e}")

    print("\n🔍 [3/4] Attempting to launch Chrome via Selenium...")

    options = webdriver.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--headless=new")  # comment this out if you want to see the browser

    service = Service(chromedriver_path)
    try:
        driver = webdriver.Chrome(service=service, options=options)
        print("✅ Chrome session started successfully!")

        driver.get("https://www.google.com")
        time.sleep(2)

        title = driver.title
        print(f"🌐 Opened page title: {title}")

        search_box = driver.find_element(By.NAME, "q")
        search_box.send_keys("Flipkart Laptops")
        print("✅ Located and interacted with the search box successfully!")

    except Exception as e:
        print(f"❌ Selenium test failed: {e}")
        sys.exit(1)

    finally:
        driver.quit()
        print("🧹 Browser closed successfully.")

    print("\n🎉 All checks passed! Your Selenium + Chrome setup is working correctly.")

if __name__ == "__main__":
    main()
