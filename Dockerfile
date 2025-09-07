# STAGE 1: The "Builder" Stage
# This is where we do all the heavy downloading and installation.
FROM python:3.13-alpine as builder

WORKDIR /app

# Install build essentials that might be needed by some packages
RUN apk add --no-cache build-base

# Copy only the requirements file
COPY requirements.txt .

# Install all dependencies. This will create a large layer.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
# STAGE 2: The "Final" Stage
# This is our clean, lightweight final image.
FROM python:3.13-alpine

# Update packages to patch vulnerabilities
RUN apk update && apk upgrade && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy ONLY the installed packages from the builder stage.

# Copy ONLY the installed packages from the builder stage.
# This is the key step that makes our image small.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /app/requirements.txt .

# Copy the rest of your application code
COPY . .

# Command to run the application using Gunicorn
CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8080", "app:app"]


### What to Do Next

