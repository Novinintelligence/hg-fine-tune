#!/bin/bash

# RunPod Bootstrap Script for Home Security AI Training
# This script sets up the environment and starts the training process

set -e  # Exit on any error

echo "🚀 Starting RunPod Home Security AI Setup..."

# Navigate to workspace
cd /workspace

# Update system packages
echo "📦 Updating system packages..."
if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update || true
    sudo apt-get install -y git curl wget || true
else
    apt-get update || true
    apt-get install -y git curl wget || true
fi

# Clone the repository (replace with your actual repo)
# You can set REPO_URL env var to override
REPO_URL="${REPO_URL:-https://github.com/Novinintelligence/hg-fine-tune.git}"
HF_REPO="${HF_REPO:-ollieherbert/phi3-homesec-v1}"
echo "📥 Cloning repository from $REPO_URL..."

if [ -d "unified-homesec-ai" ]; then
    echo "Repository exists, pulling latest changes..."
    cd unified-homesec-ai
    git pull origin main
else
    git clone $REPO_URL
    cd unified-homesec-ai
fi

# Install Python dependencies (use CUDA 12.1 wheels for PyTorch on RunPod image)
echo "🐍 Installing Python dependencies..."
python -m pip install --upgrade pip
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
pip install --extra-index-url "$TORCH_INDEX_URL" -r requirements.txt

# Ensure Accelerate default config exists
echo "⚙️  Initializing Accelerate default config..."
python - <<'PY'
from accelerate.utils import write_basic_config
try:
    write_basic_config()
    print("Accelerate default config written")
except Exception as e:
    print(f"Failed to write accelerate default config: {e}")
PY

# Login to Hugging Face (if token provided)
if [ ! -z "$HF_TOKEN" ]; then
    echo "🤗 Logging into Hugging Face..."
    huggingface-cli login --token $HF_TOKEN
else
    echo "⚠️  HF_TOKEN not set. Skipping Hugging Face login."
fi

# Create data directory
mkdir -p data

# Pre-pull model and tokenizer to warm cache (saves GPU time)
echo "⬇️  Pre-pulling base model (this may take several minutes)..."
python - <<'PY'
from transformers import AutoTokenizer, AutoModelForCausalLM
model = 'microsoft/Phi-3-mini-128k-instruct'
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained(model)
print('Downloading model (safetensors shards)...')
AutoModelForCausalLM.from_pretrained(model)
print('Pre-pull complete')
PY

# Run preflight checks (will generate a tiny dataset if missing)
echo "🧪 Running preflight checks..."
python preflight_check.py || { echo "❌ Preflight checks failed"; exit 1; }

# Generate synthetic dataset (full size)
echo "📊 Generating synthetic security dataset..."
python dataset_gen.py --samples 50000 --output data/unified_security_dataset.jsonl

# Verify dataset was created
if [ -f "data/unified_security_dataset.jsonl" ]; then
    echo "✅ Dataset generated successfully!"
    echo "📈 Dataset info:"
    wc -l data/unified_security_dataset.jsonl
else
    echo "❌ Dataset generation failed!"
    exit 1
fi

# Check for GPU
echo "🔍 Checking GPU availability..."
nvidia-smi

# Start training
echo "🧠 Starting Phi-3 fine-tuning..."
echo "Configuration: config/train_config.json"

accelerate launch train_phi3.py --config config/train_config.json

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo "🎉 Training completed successfully!"
    
    # Upload to Hugging Face if token and repo are available
    if [ ! -z "$HF_TOKEN" ] && [ ! -z "$HF_REPO" ]; then
        echo "📤 Uploading final model to Hugging Face repo: $HF_REPO ..."
        # Ensure repo exists
        huggingface-cli repo create "$HF_REPO" --yes --type model || true
        # Upload checkpoint directory
        huggingface-cli upload /workspace/checkpoints "$HF_REPO" --repo-type model --yes || true
    else
        echo "ℹ️  Skipping HF upload (HF_TOKEN or HF_REPO not set)."
    fi
    
    echo "✅ All done! Model saved to /workspace/checkpoints"
else
    echo "❌ Training failed!"
    exit 1
fi

echo "🏁 RunPod setup complete!"
