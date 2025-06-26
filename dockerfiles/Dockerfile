# Base image
#FROM pytorch/pytorch:2.5.1-cuda12.2-cudnn8-devel
#FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
#FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel 
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /resources

# Copy requirements.txt
# COPY requirements.txt .

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
# RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir openslide-python

# Set working directory for nnUNet
WORKDIR /nnUNet

# Copy nnUNet source code into the image
COPY . /nnUNet

#RUN pip install git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git

# Install nnUNet in editable mode
RUN pip install --no-cache-dir -e .
