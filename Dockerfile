FROM python:3.11-slim

# System/env hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install runtime deps: chromium + chromedriver for scraper, tini for signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
      chromium chromium-driver ca-certificates tini curl \
      fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

# Point scraper to the browser/driver installed above
ENV CHROME_BINARY=/usr/bin/chromium \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Non-root user
RUN useradd -m appuser
WORKDIR /app

# ---- Python deps (layered for cache) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- App code ----
# Copy just what we need
COPY prod_assistant ./prod_assistant
COPY static ./static
COPY templates ./templates
# Optional utilities if you want to use the scraper UI in the same image
COPY scrapper_ui.py ./scrapper_ui.py
COPY data_scrapper.py ./data_scrapper.py
# Create data folder for CSV outputs (kept writable)
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Tini as entrypoint to handle signals/zombies properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start MCP product_search_server (background) and FastAPI (foreground)
# Provide OPENAI_API_KEY / ASTRA_* via `docker run -e ...`
CMD bash -lc "python prod_assistant/mcp_servers/product_search_server.py & \
              uvicorn prod_assistant.router.main:app --host 0.0.0.0 --port 8000 --workers 2"
