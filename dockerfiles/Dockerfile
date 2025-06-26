# Base image
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /resources

# Install system dependencies and Python packages
RUN apt-get update && apt-get install --yes \
    gcc \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-dev \
    libopenslide-dev \
    openslide-tools \
    tmux \
    wget \
    vim && \
    rm -rf /var/lib/apt/lists/*
	
# Install Python dependencies from requirements.txt

RUN pip install --no-cache-dir openslide-python

# Set working directory for nnUNet
WORKDIR /nnUNet

# Copy nnUNet source code into the image
COPY . /nnUNet

# Install nnUNet in editable mode
RUN pip install --no-cache-dir -e .
