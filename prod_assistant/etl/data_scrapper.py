import csv
import time
import re
import os
import subprocess
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from typing import Optional
import platform

class FlipkartScraper:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------
    # DETERMINISTIC DRIVER/BINARY RESOLUTION (ENV-ONLY)
    # -------------------------
    def _get_env_paths(self) -> (Optional[str], Optional[str]): # type: ignore
        """
        Return tuple (chromedriver_path, chrome_binary_path) from environment variables.
        We require both to be set for deterministic execution.
        """
        chromedriver_path = os.getenv("CHROMEDRIVER_PATH")
        chrome_binary = os.getenv("CHROME_BINARY") or os.getenv("CHROME_BIN")  # allow alternate name
        return chromedriver_path, chrome_binary

    def _get_version_from_executable(self, exe_path: str) -> Optional[str]:
        # Try direct --version
        try:
            out = subprocess.check_output([exe_path, "--version"], stderr=subprocess.STDOUT, text=True)
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass

        # Windows fallback: read FileVersion via PowerShell
        if platform.system() == "Windows" and os.path.exists(exe_path):
            try:
                ps_cmd = [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    f"(Get-Item -Path '{exe_path}').VersionInfo.FileVersion"
                ]
                out = subprocess.check_output(ps_cmd, stderr=subprocess.STDOUT, text=True)
                m = re.search(r"(\d+\.\d+\.\d+\.\d+)", out)
                if m:
                    return m.group(1)
            except Exception:
                pass

        # Final attempt: try basename (if on PATH)
        try:
            basename = os.path.basename(exe_path)
            out = subprocess.check_output([basename, "--version"], stderr=subprocess.STDOUT, text=True)
            m = re.search(r"(\d+\.\d+\.\d+\.\d+)", out)
            if m:
                return m.group(1)
        except Exception:
            pass

        return None


    def _validate_env_and_versions(self, chromedriver_path: str, chrome_binary: str):
        """
        Strict validation:
          - both env vars provided and point to existing files
          - both executables respond to --version
          - major version of chromedriver matches major version of chrome binary
        Raises RuntimeError with clear, actionable message on failure.
        """
        if not chromedriver_path or not os.path.exists(chromedriver_path):
            raise RuntimeError(
                "CHROMEDRIVER_PATH is not set or file does not exist. "
                "Set CHROMEDRIVER_PATH to the exact chromedriver.exe path (e.g. C:\\drivers\\chromedriver.exe)."
            )

        if not chrome_binary or not os.path.exists(chrome_binary):
            raise RuntimeError(
                "CHROME_BINARY is not set or file does not exist. "
                "Set CHROME_BINARY to the exact Chrome executable (e.g. C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe)."
            )

        driver_ver = self._get_version_from_executable(chromedriver_path)
        chrome_ver = self._get_version_from_executable(chrome_binary)

        if not driver_ver:
            raise RuntimeError(f"Failed to run chromedriver at '{chromedriver_path}' with --version. Ensure it is executable.")

        if not chrome_ver:
            raise RuntimeError(f"Failed to run chrome binary at '{chrome_binary}' with --version. Ensure it is executable.")

        driver_major = driver_ver.split(".")[0]
        chrome_major = chrome_ver.split(".")[0]

        if driver_major != chrome_major:
            raise RuntimeError(
                f"Chromedriver/Chrome major version mismatch:\n"
                f"  chromedriver: {driver_ver}\n"
                f"  chrome binary: {chrome_ver}\n\n"
                f"Major versions must match. Download chromedriver for Chrome major {chrome_major} "
                f"and set CHROMEDRIVER_PATH accordingly."
            )

        # all good — print debug for confirmation
        print(f"[FlipkartScraper] Using chromedriver: {chromedriver_path} (v{driver_ver})")
        print(f"[FlipkartScraper] Using chrome binary: {chrome_binary} (v{chrome_ver})")
        return

    def _get_driver_path_or_fail(self) -> str:
        """
        Deterministically return chromedriver path, otherwise raise an error.
        """
        chromedriver_path, chrome_binary = self._get_env_paths()
        if not chromedriver_path or not chrome_binary:
            raise RuntimeError(
                "Environment variables CHROMEDRIVER_PATH and CHROME_BINARY must both be set for scraping to run.\n"
                "Example (Git Bash):\n"
                "  export CHROMEDRIVER_PATH='/c/Program Files/chromedriver-win64/chromedriver-win64/chromedriver.exe'\n"
                "  export CHROME_BINARY='/c/Program Files/Google/Chrome/Application/chrome.exe'\n"
                "Or set them permanently with setx in PowerShell."
            )

        # Validate both and versions
        self._validate_env_and_versions(chromedriver_path, chrome_binary)
        return chromedriver_path

    # -------------------------
    # Scraping methods
    # -------------------------
    def get_top_reviews(self, product_url, count=2):
        """Get the top reviews for a product."""
        options = uc.ChromeOptions()
        # Force chrome binary explicitly (important)
        chrome_binary = os.getenv("CHROME_BINARY")
        if not chrome_binary:
            raise RuntimeError("CHROME_BINARY env var missing. See README.")
        options.binary_location = chrome_binary

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        # options.add_argument("--headless=new")  # optional

        driver_bin = self._get_driver_path_or_fail()
        service = Service(driver_bin)
        driver = uc.Chrome(service=service, options=options, use_subprocess=True)

        if not product_url.startswith("http"):
            driver.quit()
            return "No reviews found"

        try:
            driver.get(product_url)
            time.sleep(4)
            try:
                driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
                time.sleep(1)
            except Exception as e:
                print(f"Error closing popup: {e}")

            for _ in range(4):
                ActionChains(driver).send_keys(Keys.END).perform()
                time.sleep(1.5)

            soup = BeautifulSoup(driver.page_source, "html.parser")
            review_blocks = soup.select("div._27M-vq, div.col.EPCmJX, div._6K-7Co")
            seen = set()
            reviews = []

            for block in review_blocks:
                text = block.get_text(separator=" ", strip=True)
                if text and text not in seen:
                    reviews.append(text)
                    seen.add(text)
                if len(reviews) >= count:
                    break
        except Exception as e:
            print(f"[FlipkartScraper] Exception in get_top_reviews: {e}")
            reviews = []

        driver.quit()
        return " || ".join(reviews) if reviews else "No reviews found"

    def scrape_flipkart_products(self, query, max_products=1, review_count=2):
        """Scrape Flipkart products based on a search query."""
        options = uc.ChromeOptions()
        chrome_binary = os.getenv("CHROME_BINARY")
        if not chrome_binary:
            raise RuntimeError("CHROME_BINARY env var missing. See README.")
        options.binary_location = chrome_binary

        options.add_argument("--no-sandbox")
        options.add_argument("--disable-blink-features=AutomationControlled")
        # options.add_argument("--headless=new")  # optional

        driver_bin = self._get_driver_path_or_fail()
        service = Service(driver_bin)
        driver = uc.Chrome(service=service, options=options, use_subprocess=True)

        search_url = f"https://www.flipkart.com/search?q={query.replace(' ', '+')}"
        driver.get(search_url)
        time.sleep(4)

        try:
            driver.find_element(By.XPATH, "//button[contains(text(), '✕')]").click()
        except Exception as e:
            print(f"Error occurred while closing popup: {e}")

        time.sleep(2)
        products = []

        items = driver.find_elements(By.CSS_SELECTOR, "div[data-id]")[:max_products]
        for item in items:
            try:
                title = item.find_element(By.CSS_SELECTOR, "div.KzDlHZ").text.strip()
                price = item.find_element(By.CSS_SELECTOR, "div.Nx9bqj").text.strip()
                rating = item.find_element(By.CSS_SELECTOR, "div.XQDdHH").text.strip()
                reviews_text = item.find_element(By.CSS_SELECTOR, "span.Wphh3N").text.strip()
                match = re.search(r"\d+(,\d+)?(?=\s+Reviews)", reviews_text)
                total_reviews = match.group(0) if match else "N/A"

                link_el = item.find_element(By.CSS_SELECTOR, "a[href*='/p/']")
                href = link_el.get_attribute("href")
                product_link = href if href.startswith("http") else "https://www.flipkart.com" + href
                match = re.findall(r"/p/(itm[0-9A-Za-z]+)", href)
                product_id = match[0] if match else "N/A"
            except Exception as e:
                print(f"Error occurred while processing item: {e}")
                continue

            top_reviews = self.get_top_reviews(product_link, count=review_count) if "flipkart.com" in product_link else "Invalid product URL"
            products.append([product_id, title, rating, total_reviews, price, top_reviews])

        driver.quit()
        return products

    def save_to_csv(self, data, filename="product_reviews.csv"):
        """Save the scraped product reviews to a CSV file."""
        if os.path.isabs(filename):
            path = filename
        elif os.path.dirname(filename):  # filename includes subfolder like 'data/product_reviews.csv'
            path = filename
            os.makedirs(os.path.dirname(path), exist_ok=True)
        else:
            # plain filename like 'output.csv'
            path = os.path.join(self.output_dir, filename)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["product_id", "product_title", "rating", "total_reviews", "price", "top_reviews"])
            writer.writerows(data)
