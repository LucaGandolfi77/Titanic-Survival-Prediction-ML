# 🕷️ WebScraper Pro — E-commerce Data Extractor

A production-grade Python application that scrapes product data (names, prices,
descriptions, images) from any e-commerce website and exports everything to a
structured Excel file with embedded thumbnails.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Multi-page scraping** | Automatically paginates through product pages (query-param, path, or button) |
| **Image downloading** | Async concurrent image downloader with `aiohttp` + semaphore control |
| **Excel export** | Rich `.xlsx` with embedded thumbnails, auto-width columns, alternating row colors, summary sheet |
| **CSV fallback** | Plain CSV export for quick data access |
| **Anti-bot measures** | User-Agent rotation, randomised delays, `robots.txt` compliance |
| **CAPTCHA detection** | Warns and skips pages that return CAPTCHA / access-denied responses |
| **GUI** | Tkinter desktop app with progress bar, live log, and one-click export |
| **CLI** | Full `argparse` interface with built-in selector presets |
| **Selector presets** | Ready-made selectors for WooCommerce, Shopify, and generic stores |
| **Configurable** | All selectors, headers, rate-limits, and output paths in `config.yaml` |

---

## 🖥️ GUI Preview (ASCII mockup)

```
┌──────────────────────────────────────────────┐
│  🕷️  WebScraper Pro - Product Extractor      │
├──────────────────────────────────────────────┤
│  Target URL: [_____________________________] │
│  Max Pages:  [50]   ☐ Download Images        │
│  [Browse Output Folder]   [Start Scraping]   │
├──────────────────────────────────────────────┤
│  Progress: [████████░░░░░░] 45%  Page 5      │
│  Products found: 127                         │
├──────────────────────────────────────────────┤
│  Log:                                        │
│  [scrollable text area with live logs]       │
│                                              │
├──────────────────────────────────────────────┤
│  [Stop]     [Open Excel]     [Open Folder]   │
└──────────────────────────────────────────────┘
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/webscraper-pro.git
cd webscraper-pro

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- See `requirements.txt` for the full dependency list.

---

## 🚀 Usage

### CLI Mode

```bash
# Basic usage
python main.py --url "https://example.com/shop" --max-pages 10

# Without image downloading
python main.py --url "https://example.com/shop" --max-pages 10 --no-images

# Custom output folder
python main.py --url "https://example.com/shop" --output "./results"

# Use a built-in selector preset
python main.py --url "https://example.com/shop" --selector-preset shopify

# Full example
python main.py \
    --url "https://example.com/shop" \
    --max-pages 20 \
    --output "./results" \
    --selector-preset woocommerce
```

### GUI Mode

```bash
python gui/app.py
```

Or launch from the project root:

```bash
python -m gui.app
```

### Build Standalone EXE (Windows)

```bat
build_exe.bat
```

The executable will be placed in `dist/WebScraperPro.exe`.

---

## 🔎 How to Find CSS Selectors for a New Site

1. **Open the target site** in Google Chrome.
2. **Right-click** on a product name → *Inspect Element*.
3. In the **Elements** panel, identify the HTML tag and class:
   ```html
   <h2 class="product-title">Cool T-Shirt</h2>
   ```
   → Selector: `h2.product-title`
4. Repeat for **price**, **description**, **image**, and the **product container**.
5. Look for the **pagination** link (e.g. "Next →") and note its selector.
6. Update `config/config.yaml` under the `selectors:` section with your values.

> **Tip:** use `document.querySelectorAll("YOUR_SELECTOR")` in the browser console
> to verify your selector matches the expected elements.

---

## ⚙️ Configuration Guide (`config/config.yaml`)

```yaml
output:
  folder: "./output"              # Where Excel/CSV files are saved
  images_folder: "./output/images" # Where downloaded images are stored
  excel_filename: "products.xlsx"
  csv_filename: "products.csv"

scraper:
  timeout: 15                     # HTTP request timeout (seconds)
  max_retries: 3                  # Retries per page on failure
  delay_min: 1.5                  # Min seconds between requests
  delay_max: 3.5                  # Max seconds between requests
  max_pages: 50                   # Page limit to prevent runaway scraping
  user_agents:                    # Rotated randomly on each request
    - "Mozilla/5.0 ..."

selectors:
  product_container: "div.product-item"  # Wrapper around each product
  name: "h2.product-title"               # Product name
  price: "span.price"                    # Price text
  description: "p.product-description"   # Short description
  image: "img.product-image"             # Image tag (uses src attribute)
  next_page: "a.next-page"               # Pagination link to next page
  pagination_type: "query_param"          # query_param | path | button
  page_param: "page"                     # Query-string parameter name
```

---

## 📋 Built-in Selector Presets

| Preset | Container | Name | Price | Image | Next Page |
|---|---|---|---|---|---|
| `woocommerce` | `li.product` | `.woocommerce-loop-product__title` | `span.price` | `img.attachment-woocommerce_thumbnail` | `a.next` |
| `shopify` | `div.product-card` | `.product-card__title` | `span.price-item` | `img.product-featured-media` | `a[rel="next"]` |
| `generic` | `div.product` | `h2.product-title, h3.product-title` | `span.price, div.price` | `img.product-image, img.product-img` | `a.next, a.next-page, a[rel="next"]` |

---

## ⚖️ Legal Notice

> **Use responsibly and ethically.**

- Always check and respect the target site's **`robots.txt`** file.
- Read and comply with the site's **Terms of Service** before scraping.
- This tool enforces **rate limiting** (configurable delays) — never remove it.
- Do **not** use scraped data for purposes that violate copyright or privacy laws.
- The authors are **not liable** for misuse of this software.

---

## 📝 Switching to a JS-rendering Backend

For JavaScript-rendered sites (React, Vue, Angular SPAs) that don't return
product data in the initial HTML, swap `requests` for **Playwright**:

```python
# pip install playwright && python -m playwright install
from playwright.sync_api import sync_playwright

def fetch_page_js(url: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")
        html = page.content()
        browser.close()
        return html
```

Then use `BeautifulSoup(html, "lxml")` as usual.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
