# Base Python image
FROM python:3.12-slim

# Environment settings for Python in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create and set work directory
WORKDIR /app

# Install system dependencies commonly required by ML/DB packages
# - build-essential: for building wheels when needed
# - git: some packages may fetch models/plugins
# - libgl1: for pillow/opencv-related backends
# - libgomp1: required by onnxruntime/scikit-learn in manylinux wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       libgl1 \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Environment variables (defaults; override with -e or --env-file) ---
ENV ENV=prod \
    GOOGLE_API_KEY="" \
    PINECONE_API_KEY="" \
    LANGCHAIN_API_KEY=""

# Install Python dependencies first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose Flask port
EXPOSE 5000

# Run the Flask app (dev server) binding to all interfaces
# The application module is api.app with variable `app`
CMD ["flask", "--app", "api.app", "run", "--host=0.0.0.0", "--port=5000"]
