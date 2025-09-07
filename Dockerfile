# STAGE 1: The "Builder" Stage
# This is where we do all the heavy downloading and installation.
FROM python:3.13-alpine as builder

WORKDIR /app

# Install build essentials that might be needed by some packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential || apk add --no-cache build-base

# Copy only the requirements file
COPY requirements.txt .

# --- FINAL FIX ---
# Install all dependencies in a single step.
# This allows pip's dependency resolver to find a compatible version of everything at once,
# while still using the official index for PyTorch CPU versions.
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu


# STAGE 2: The "Final" Stage
# This is our clean, lightweight final image.
FROM python:3.13-alpine

WORKDIR /app

# Copy ONLY the installed packages from the builder stage.
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/requirements.txt .

# Copy the rest of your application code
COPY . .

# Command to run the application using Gunicorn
CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "app:app"]



