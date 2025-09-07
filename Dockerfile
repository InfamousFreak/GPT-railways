# STAGE 1: The "Builder" Stage - For heavy installation
FROM python:3.11-slim as builder

# Install all necessary system dependencies for ML and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# STAGE 2: The "Final" Stage - For a small, clean image
FROM python:3.11-slim

WORKDIR /app

# Copy only the essential system libraries from the builder stage
COPY --from=builder /usr/lib /usr/lib
COPY --from=builder /lib /lib

# Copy only the installed python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# --- THE FIX ---
# Copy the executable scripts (like gunicorn) that were installed by pip
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /app/requirements.txt .

# Copy your application code
COPY . .

# Let Railway handle the port via the PORT environment variable
EXPOSE 8080

# The single, correct command to run the application
CMD gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT app:app

