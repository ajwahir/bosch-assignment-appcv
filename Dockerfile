FROM python:3.8-slim-bullseye

# Install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first for caching
COPY requirements.txt /workspace/requirements.txt
RUN pip install mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /workspace/requirements.txt

EXPOSE 8501
EXPOSE 5181