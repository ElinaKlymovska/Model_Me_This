FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV A1111_DIR=/workspace/stable-diffusion-webui
ENV PRESET_DIR=/workspace/portrait-enhancer
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tmux \
    unzip \
    python3 \
    python3-venv \
    python3-pip \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -sf $(command -v python3) /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Make bootstrap script executable
RUN chmod +x bootstrap.sh

# Clone A1111
RUN if [ ! -d "$A1111_DIR" ]; then \
        git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui "$A1111_DIR"; \
    fi

# Install extensions
RUN mkdir -p "$A1111_DIR/extensions" && \
    [ -d "$A1111_DIR/extensions/sd-webui-controlnet" ] || git clone https://github.com/Mikubill/sd-webui-controlnet "$A1111_DIR/extensions/sd-webui-controlnet" && \
    [ -d "$A1111_DIR/extensions/adetailer" ] || git clone https://github.com/Bing-su/adetailer "$A1111_DIR/extensions/adetailer"

# Create model directories
RUN mkdir -p "$A1111_DIR/models/Stable-diffusion" "$A1111_DIR/models/ControlNet"

# Install Python dependencies
RUN cd "$PRESET_DIR" && \
    python -m venv .venv && \
    . .venv/bin/activate && \
    pip install -r requirements.txt

# Expose port
EXPOSE 7860

# Set default command
CMD ["/bin/bash", "bootstrap.sh"]
