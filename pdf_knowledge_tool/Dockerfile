# Base Image
FROM python:3.12-slim

# Set Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/data/hf_cache

# Set Working Directory
WORKDIR /app

# Install System Dependencies
# libgl1: Required for OpenCV (headless version preferred, but libgl1 covers basic needs)
# libglib2.0-0: Required for OpenCV
# poppler-utils: Required for Docling/PDF processing
# tesseract-ocr: Required for OCR
# curl: Required for Healthcheck
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Application Code
COPY . .

# Expose Streamlit Port
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Create a non-root user and switch to it
# (Optional) For security practices
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the Application
CMD ["streamlit", "run", "search_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
