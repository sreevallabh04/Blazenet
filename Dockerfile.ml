FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    libspatialindex-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

# Set work directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app/ml /app/ml
COPY ./utils /utils

# Create necessary directories
RUN mkdir -p /logs /data

# Default command for training
CMD ["python", "ml/training/train_fire_prediction.py"] 