# Unified Home Security AI

Fine-tuning Phi-3 for intelligent home security event analysis using QLoRA on RunPod.

## 🎯 Project Overview

This project fine-tunes Microsoft's Phi-3-mini-128k-instruct model on synthetic home security data to create an AI that can:
- Analyze security sensor events in real-time
- Provide reasoning for threat assessment
- Recommend appropriate responses
- Distinguish between false positives and genuine threats

## 📁 Project Structure

```
unified-homesec-ai/
├── dataset_gen.py              # Synthetic security dataset generator
├── train_phi3.py               # QLoRA fine-tuning script with checkpoint resume
├── requirements.txt            # Python dependencies
├── runpod_start.sh             # RunPod bootstrap script
├── config/
│   └── train_config.json       # Training configuration
├── data/
│   └── unified_security_dataset.jsonl  # Generated dataset
└── README.md
```

## 🚀 Quick Start on RunPod

### 1. Prepare Your Environment

Before deploying to RunPod:

1. **Fork/Clone this repository locally**
2. **Update the repository URL** in `runpod_start.sh` (line 18)
3. **Push to your GitHub repository**

```bash
git init
git add .
git commit -m "initial commit"
git remote add origin https://github.com/<youruser>/unified-homesec-ai.git
git push -u origin main
```

### 2. Configure RunPod Pod

When creating your RunPod Spot instance:

- **Template**: PyTorch 2.1.0
- **GPU**: H100 PCIe 80GB (recommended) or A100 40GB
- **Storage**: Attach 250GB Network Volume (mount to `/workspace`)
- **Environment Variables**:
  - `HF_TOKEN`: Your Hugging Face access token

### 3. Deploy and Run

1. **Deploy the pod** and wait for it to start
2. **SSH into the pod** via RunPod web interface
3. **Run the bootstrap script**:

```bash
cd /workspace
wget https://raw.githubusercontent.com/<youruser>/unified-homesec-ai/main/runpod_start.sh
chmod +x runpod_start.sh
bash runpod_start.sh
```

### 4. Monitor Training

The script will automatically:
- Generate 50,000 synthetic security samples
- Start QLoRA fine-tuning with checkpointing
- Save checkpoints every 1000 steps to `/workspace/checkpoints`
- Resume from last checkpoint if interrupted
- Upload final model to Hugging Face (if token provided)

## 📊 Dataset Generation

The `dataset_gen.py` script creates realistic security event sequences:

### Event Types
- Motion detection (various rooms)
- Door/window sensors
- Glass break detection
- Environmental sensors (smoke, CO, temperature)
- Camera motion alerts

### Sample Format
```json
{
  "events": [
    {
      "timestamp": "2024-01-15T23:45:12",
      "event_type": "motion_detected",
      "sensor": "living_room_motion",
      "severity": "medium",
      "confidence": 0.85,
      "metadata": {...}
    }
  ],
  "reasoning": "Analyzing security events...\nConclusion: Motion detected in Living Room...",
  "response": "Event logged. No immediate action required. Continued monitoring.",
  "metadata": {
    "sample_id": "SAMPLE_123456",
    "scenario_type": "security_event_analysis"
  }
}
```

## 🧠 Model Training

### Configuration (config/train_config.json)
```json
{
  "base_model": "microsoft/Phi-3-mini-128k-instruct",
  "epochs": 5,
  "learning_rate": 2e-5,
  "batch_size": 16,
  "save_steps": 1000,
  "bits": 4,
  "lora_r": 64,
  "lora_alpha": 32,
  "output_dir": "/workspace/checkpoints"
}
```

### Key Features
- **QLoRA 4-bit quantization** for memory efficiency
- **Automatic checkpoint resume** for Spot instance interruptions
- **Gradient checkpointing** to save GPU memory
- **Hugging Face integration** for model upload
- **Comprehensive logging** and progress tracking

## 🔄 Resume from Checkpoint

If your Spot instance gets terminated, simply:

1. **Deploy a new pod** with same configuration
2. **Re-run the bootstrap script** - it will automatically:
   - Detect the last checkpoint in `/workspace/checkpoints`
   - Resume training from that point
   - Continue with the same configuration

## 📤 Model Deployment

After training completes:

### Export to Hugging Face
```bash
huggingface-cli upload /workspace/checkpoints youruser/phi3-homesec-v1
```

### Local Inference
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "microsoft/Phi-3-mini-128k-instruct"
adapter_path = "/workspace/checkpoints"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(model, adapter_path)
```

## 🧪 Testing and Evaluation

### Bias Testing
Run bias evaluation on Colab or local GPU:
```bash
python evaluate_bias.py --model_path /workspace/checkpoints
```

### Performance Benchmarks
```bash
python benchmark.py --model_path /workspace/checkpoints --test_samples 1000
```

## 📈 Expected Performance

- **Training time**: ~2-4 hours on H100 (50k samples, 5 epochs)
- **Memory usage**: ~24GB GPU memory (4-bit QLoRA)
- **Inference latency**: <500ms on Pi 5 + TPU
- **Model size**: ~2GB (4-bit quantized)

## 🔧 Customization

### Modify Event Types
Edit `SECURITY_EVENTS` in `dataset_gen.py` to add new sensor types.

### Adjust Training Parameters
Modify `config/train_config.json` to change:
- Learning rate and epochs
- LoRA parameters (r, alpha)
- Batch size and sequence length

### Add External Datasets
Merge additional datasets before training:
```bash
python merge_datasets.py --synthetic data/unified_security_dataset.jsonl --external external_data.jsonl --output combined_dataset.jsonl
```

## 🛠️ Troubleshooting

### Common Issues

**Out of Memory Errors**
- Reduce `batch_size` in config
- Increase `gradient_accumulation_steps`
- Ensure 4-bit quantization is enabled

**Training Not Resuming**
- Check `/workspace/checkpoints` directory exists
- Verify checkpoint files are not corrupted
- Ensure `output_dir` matches checkpoint location

**Hugging Face Upload Fails**
- Verify `HF_TOKEN` is valid and has write permissions
- Check repository name doesn't already exist
- Ensure model files are under 50GB (Hugging Face limit)

### Logs and Monitoring

Training logs are saved to:
- Console output (real-time)
- `/workspace/checkpoints/trainer_log.jsonl`
- Tensorboard logs (if enabled)

## 📞 Support

For issues or questions:
1. Check RunPod pod logs
2. Verify dataset format and integrity
3. Ensure all dependencies are installed
4. Test with smaller dataset first

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.
