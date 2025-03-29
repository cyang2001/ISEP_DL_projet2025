FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel
WORKDIR /workspace

# install basic tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# copy project files
COPY requirements.txt .

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# set environment variables
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV LOG_LEVEL=INFO
ENV LOG_DIR=/workspace/logs

# create necessary directories
RUN mkdir -p logs data/raw data/processed saved_models

# start command
CMD ["/bin/bash"] 