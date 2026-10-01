# Use NVIDIA CUDA base image that matches your VM setup
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    zip \
    unzip \
    build-essential \
    libgl1 \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda and accept Terms of Service
RUN conda init bash && \
    conda config --set auto_activate_base false && \
    conda config --set always_yes true && \
    conda config --set show_channel_urls true

# Accept Terms of Service for default channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment with Python 3.11
RUN conda create -n vld python=3.11 -y

# Activate environment and make it default
ENV CONDA_DEFAULT_ENV=vld
ENV PATH=/opt/conda/envs/vld/bin:$PATH

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install basic Python packages
RUN /opt/conda/envs/vld/bin/python -m pip install --upgrade pip && \
    /opt/conda/envs/vld/bin/python -m pip install --no-cache-dir -r requirements.txt

# Install PyTorch 2.6.0
RUN /opt/conda/envs/vld/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

# Set CUDA architecture for RTX 3090
ENV TORCH_CUDA_ARCH_LIST="8.6"
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1

# Install openmim
RUN /opt/conda/envs/vld/bin/python -m pip install -U openmim

# Install mmengine
RUN /opt/conda/envs/vld/bin/mim install mmengine

# Install mmcv==2.1.0 using mim
RUN /opt/conda/envs/vld/bin/mim install "mmcv==2.1.0"

# Verify MMCV installation
RUN /opt/conda/envs/vld/bin/python -c "import mmcv; print(f'MMCV version: {mmcv.__version__}')"

# Clone MMDetection and MMSegmentation repositories
RUN git clone https://github.com/open-mmlab/mmdetection.git && \
    git clone https://github.com/open-mmlab/mmsegmentation.git

# Install MMDetection
WORKDIR /app/mmdetection
RUN /opt/conda/envs/vld/bin/python -m pip install -v -e .

# Install MMSegmentation
WORKDIR /app/mmsegmentation
RUN /opt/conda/envs/vld/bin/python -m pip install -v -e .

# Set working directory back to /app
WORKDIR /app

# Copy source code
COPY . .

# Create a script to activate conda environment and run commands
RUN echo '#!/bin/bash' > /app/run_with_conda.sh && \
    echo 'source /opt/conda/bin/activate vld' >> /app/run_with_conda.sh && \
    echo 'exec "$@"' >> /app/run_with_conda.sh && \
    chmod +x /app/run_with_conda.sh

# Update your shell scripts to use the conda environment
RUN sed -i 's/python3\.11/\/opt\/conda\/envs\/vld\/bin\/python/g' run_preprocess.sh run_train_detection.sh run_train_segmentation.sh run_inference.sh 2>/dev/null || true

# Set the entrypoint to use conda environment
ENTRYPOINT ["/app/run_with_conda.sh"]

# Default command
CMD ["/bin/bash", "-c", "./run_preprocess.sh && ./run_train_detection.sh && ./run_train_segmentation.sh && ./run_inference.sh"]