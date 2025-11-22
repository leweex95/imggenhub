#!/bin/bash
set -e

# Forge UI deployment and setup script for Vast.ai instances
# Installs Forge UI with all required dependencies for image generation

FORGE_DIR="${FORGE_DIR:-/workspace/stable-diffusion-webui-forge}"
MODELS_DIR="${MODELS_DIR:-$FORGE_DIR/models}"

echo "=== Starting Forge UI Deployment ==="
echo "Installation directory: $FORGE_DIR"
echo "Models directory: $MODELS_DIR"

# Install system dependencies
echo "Installing system dependencies..."
apt-get update
apt-get install -y \
    git \
    curl \
    wget \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    libssl-dev \
    libffi-dev

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install pip
echo "Installing pip..."
python3 -m pip install --upgrade pip

# Clone Forge repository
if [ ! -d "$FORGE_DIR" ]; then
    echo "Cloning Forge UI repository..."
    git clone https://github.com/lllyasviel/stable-diffusion-webui-forge "$FORGE_DIR"
    cd "$FORGE_DIR"
else
    echo "Forge UI directory already exists"
    cd "$FORGE_DIR"
    git pull
fi

# Create Python virtual environment (optional, Forge can use system Python)
echo "Setting up environment..."

# Install Forge dependencies
echo "Installing Python dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install xformers transformers diffusers accelerate omegaconf einops pyyaml pillow numpy

# Install additional dependencies from requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

# Create models directory structure
echo "Creating models directory structure..."
mkdir -p "$MODELS_DIR/checkpoints"
mkdir -p "$MODELS_DIR/loras"
mkdir -p "$MODELS_DIR/vae"
mkdir -p "$MODELS_DIR/controlnet"

# Create output directories
mkdir -p "$FORGE_DIR/outputs"

# Print connection information
echo ""
echo "=== Forge UI Deployment Complete ==="
echo "Forge UI location: $FORGE_DIR"
echo "Models directory: $MODELS_DIR"
echo ""
echo "To start Forge UI, run:"
echo "  cd $FORGE_DIR"
echo "  python launch.py --listen 0.0.0.0 --port 7860"
echo ""
