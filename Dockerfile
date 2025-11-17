# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevents Python from writing .pyc files and enables easier logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies (if you need others later, add them here)
# Note: libgl1 is required by OpenCV (cv2) used transitively by docling's PDF pipeline.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better build cache)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your app runs on
EXPOSE 8000

# Default environment variables (can/should be overridden in the platform UI)
# ENV OPENAI_API_KEY=changeme \
#     MONGODB_URI=mongodb://user:password@host:port \
#     MONGODB_DB_NAME=syllabye \
#     SECRET_KEY=changeme

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
