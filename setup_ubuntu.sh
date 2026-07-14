#!/bin/bash
# ==============================================================================
# MIR-LM Environment Setup — NATIVE UBUNTU DUAL-BOOT + ROCm
# ==============================================================================
# For native Ubuntu on metal with an AMD Radeon RX 9070 XT (RDNA4 / gfx1201).
# This is the pre-training path (the WSL2 + DirectML path is dev/verification only).
#
# ┌── PREREQUISITES (manual, not automated — they're OS/sudo/hardware-specific) ──┐
# │ 1. Update the Linux kernel + firmware, then reboot.                           │
# │    sudo apt update && sudo apt upgrade -y && sudo reboot                       │
# │ 2. Install the amdgpu kernel driver for your card (RDNA4 needs a recent      │
# │    kernel; verify with `lspci -nn | grep -i vga` and confirm the 9070 XT).   │
# │ 3. Install AMD ROCm from the official repo matching your Ubuntu release:     │
# │    https://rocm.docs.amd.com/projects/install-on-linux/en/latest/            │
# │    ROCm 6.0 may predate full RDNA4/gfx1201 support — prefer ROCm 6.2+ if the │
# │    wheel below won't see your GPU, or use a system ROCm install.             │
# │ 4. Confirm the GPU is visible to ROCm BEFORE running this script:            │
# │    `rocminfo | grep gfx`      # should list gfx1201                          │
# │    `rocm-smi`                 # should show the 9070 XT                       │
# │ If those fail, stop and fix the driver/ROCm install — this script will not   │
# │ recover from a missing GPU.                                                   │
# └───────────────────────────────────────────────────────────────────────────────┘
set -e

echo "=== MIR-LM Native Ubuntu + ROCm Environment Setup ==="
echo "Assumes amdgpu kernel driver + ROCm are already installed (see header)."

# Sanity check: confirm the GPU is reachable via ROCm before installing anything.
if command -v rocminfo >/dev/null 2>&1; then
    echo "Found rocminfo. Checking for a gfx target..."
    if rocminfo | grep -q "gfx"; then
        echo "ROCm reports a gfx device:"
        rocminfo | grep -i "gfx" | head -n 5
    else
        echo "WARNING: rocminfo found but no gfx device reported."
        echo "The amdgpu driver / ROCm install may be incomplete. Aborting setup."
        exit 1
    fi
else
    echo "WARNING: rocminfo not found on PATH. ROCm may not be installed."
    echo "Install ROCm first (see script header), then re-run this script."
    exit 1
fi

# --- Miniconda ---
if [ -d "$HOME/miniconda" ]; then
    echo "Miniconda directory already exists. Skipping installation."
else
    echo "Downloading Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    echo "Installing Miniconda..."
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda"
    rm /tmp/miniconda.sh
fi

echo "Initializing Conda..."
"$HOME/miniconda/bin/conda" init bash || true
source "$HOME/miniconda/etc/profile.d/conda.sh"

echo "Creating conda environment 'mir-lm' with Python 3.10..."
conda create -y -n mir-lm python=3.10

echo "Activating conda environment 'mir-lm'..."
conda activate mir-lm

echo "Installing PyTorch with AMD ROCm support + project dependencies..."
# ROCm wheel index. NOTE: HSA_ENABLE_DXG_DETECTION is intentionally NOT set here —
# that env var is WSL2-specific (the DXG translation layer). On native Ubuntu the
# amdgpu kernel driver + ROCm userspace handle device discovery directly.
# If torch.cuda.is_available() is False after this, the wheel doesn't cover gfx1201
# (RDNA4); fall back to a newer ROCm wheel index or a system ROCm build.
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
pip install tokenizers transformers numpy datasets tqdm


echo ""
echo "=== Python-side setup complete ==="
echo "Verify the GPU is visible to PyTorch before training:"
echo "    conda activate mir-lm"
echo "    python test_gpu.py"
echo "Expect: 'CUDA (ROCm) available: True' and a GPU matmul. If False, see the"
echo "RDNA4/ROCm-wheel note in the script header."
