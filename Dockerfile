FROM python:3.11-slim AS base

WORKDIR /app

# Installing system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    gcc \
    libgl1 \
    curl \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Download and instal facenet model
RUN curl -L https://storage.googleapis.com/hotels-ai-models/facenet.zip -o facenet.zip \
    && unzip facenet.zip -d /app \
    && rm facenet.zip

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt /app

RUN pip install --no-cache-dir -r requirements.txt

FROM base AS final

# Copy the application code
COPY local_verification.py /app
COPY anomaly_handler.py /app
COPY api_notifier.py /app
COPY bounding_box.py /app
COPY face_inference.py /app
COPY facenet.py /app
COPY main.py /app
COPY mtcnn_client.py /app
COPY utils.py /app
COPY video_reader.py /app

CMD ["python", "main.py"]
