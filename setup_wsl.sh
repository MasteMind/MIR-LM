#!/bin/bash
# ==============================================================================
# MIR-LM Environment Setup — WSL2 + ROCm (ALTERNATIVE PATH)
# ==============================================================================
# This is the WSL2 alternative. For actual pre-training we use native Ubuntu
# dual-boot + ROCm (see setup_ubuntu.sh). WSL2 is acceptable for dev/verification
# and relies on HSA_ENABLE_DXG_DETECTION=1 below to reach the Windows DXG driver.
# For native Ubuntu, do NOT set HSA_ENABLE_DXG_DETECTION — use setup_ubuntu.sh.
# ==============================================================================
set -e

echo "=== MIR-LM WSL2 Environment Setup (alternative path) ==="
echo "Skipping system package updates (already completed by root)..."

# Check if miniconda is already installed
if [ -d "$HOME/miniconda" ]; then
    echo "Miniconda directory already exists. Skipping installation."
else
    echo "Downloading Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    echo "Installing Miniconda..."
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
    rm /tmp/miniconda.sh
fi

# Initialize conda for bash shell
echo "Initializing Conda..."
"$HOME/miniconda/bin/conda" init bash || true

# Source the conda path to enable conda commands in this script execution
source "$HOME/miniconda/etc/profile.d/conda.sh"

echo "Creating conda environment 'mir-lm' with Python 3.10..."
conda create -y -n mir-lm python=3.10

echo "Activating conda environment 'mir-lm'..."
conda activate mir-lm

echo "Installing PyTorch with AMD ROCm 6.0 support + project dependencies..."
# numpy (prepare_data.py, train.py), datasets + tqdm (download_datasets.py, train_tokenizer.py)
pip install torch tokenizers transformers numpy datasets tqdm --index-url https://download.pytorch.org/whl/rocm6.0

echo "Adding GPU environment variables to ~/.bashrc..."
# We append HSA_ENABLE_DXG_DETECTION=1 to ~/.bashrc if not already present
if ! grep -q "HSA_ENABLE_DXG_DETECTION" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "# Enable AMD GPU detection inside WSL2 via DXG translation layer" >> "$HOME/.bashrc"
    echo "export HSA_ENABLE_DXG_DETECTION=1" >> "$HOME/.bashrc"
    echo "Added HSA_ENABLE_DXG_DETECTION=1 to ~/.bashrc"
fi

echo "=== Environment Setup Completed Successfully! ==="
echo "Please reload your shell using 'source ~/.bashrc' or restart your terminal."
