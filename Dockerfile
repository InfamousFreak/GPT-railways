# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including CA certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    ghostscript \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ca-certificates \
    curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the whole project
COPY . .

# Create necessary directories
RUN mkdir -p ./uploads ./knowledge_pack

# Expose the port
EXPOSE $PORT

# Use environment variable for port, fallback to 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}