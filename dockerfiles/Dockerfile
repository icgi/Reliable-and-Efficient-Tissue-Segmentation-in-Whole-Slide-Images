##############################
# 1. Base image: PyTorch + CUDA
##############################
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Avoid interactive tzdata
ARG DEBIAN_FRONTEND=noninteractive

##############################
# 2. Install build deps
##############################
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git make pkg-config \
    autoconf automake libtool \
    zlib1g-dev libpng-dev libjpeg-dev libtiff-dev \
    libopenjp2-7-dev libgdk-pixbuf2.0-dev libxml2-dev \
    sqlite3 libsqlite3-dev libcairo2-dev libglib2.0-dev \
    ffmpeg libsm6 libxext6 \
    python3-dev \
    tmux wget vim \
    && rm -rf /var/lib/apt/lists/*

##############################
# 3. Build OpenSlide from Git
##############################
WORKDIR /opt

RUN git clone https://github.com/sderaedt/openslide.git

WORKDIR /opt/openslide

# needed when building from Git repo
RUN autoreconf -i

# required in Docker (dependency tracking breaks)
RUN ./configure --disable-dependency-tracking

RUN make -j"$(nproc)" && \
    make install && \
    ldconfig

##############################
# 4. Install nnUNet
##############################
WORKDIR /nnUNet

# copy project files *after* installations → better caching
COPY . /nnUNet

RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir \
        numpy SimpleITK \
        nibabel scikit-image scikit-learn tqdm \
        psutil pandas

##############################
# 5. Clean up (optional)
##############################
RUN rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

##############################
# 6. Set the working directory
##############################
WORKDIR /nnUNet

##############################
# 7. Default command
##############################
CMD ["/bin/bash"]
