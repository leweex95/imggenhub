#!/bin/bash
set -e

# Setup script for Vast.ai GPU instance
# Installs dependencies and prepares environment for image generation pipeline

echo "=== Starting Vast.ai Instance Setup ==="

# Update package manager
echo "Updating package manager..."
apt-get update
apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
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

# Install pip and poetry
echo "Installing pip and Poetry..."
python3 -m pip install --upgrade pip
python3 -m pip install poetry

# Create working directory
echo "Setting up working directory..."
WORK_DIR="/workspace"
mkdir -p $WORK_DIR
cd $WORK_DIR

echo "=== Setup Complete ==="
echo "Working directory: $WORK_DIR"
echo "Python version: $(python3 --version)"
echo "Poetry version: $(poetry --version)"
