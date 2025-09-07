# STAGE 1: The "Builder" Stage
# This is where we do all the heavy downloading and installation.
FROM python:3.13-slim as builder

WORKDIR /app

# Install build essentials that might be needed by some packages
# --- FIX: Switched from apt-get to apk for Alpine-based builders ---
RUN apk add --no-cache build-base

# Copy only the requirements file
COPY requirements.txt .

# --- CRITICAL FIX ---
# Install PyTorch and its dependencies separately from their official source
# This ensures we get a version compatible with the CPU-based Linux environment
RUN pip install torch torchvision torchaudio --no-cache-dir --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the dependencies from the requirements file
# Pip will see that 'torch' is already installed and skip it.
RUN pip install --no-cache-dir -r requirements.txt


# STAGE 2: The "Final" Stage
# This is our clean, lightweight final image.
FROM python:3.13-slim

WORKDIR /app

# Copy ONLY the installed packages from the builder stage.
# This is the key step that makes our image small.
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/requirements.txt .

# Copy the rest of your application code
COPY . .

# Command to run the application using Gunicorn
CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "app:app"]



