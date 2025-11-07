#!/usr/bin/env python3
"""
Preflight checks for RunPod environment before starting training.
Validates:
- Workspace mount and write permissions
- GPU availability (CUDA + device name)
- Python dependencies
- Accelerate default config presence
- Hugging Face auth (via HF_TOKEN or cached login)
- Dataset existence and readability (creates small sample if missing)
- Checkpoint directory
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

EXIT_FAIL = 1
EXIT_OK = 0

ROOT = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR", "/workspace"))
CONFIG_PATH = ROOT / "config" / "train_config.json"
DATASET_DEFAULT = ROOT / "data" / "unified_security_dataset.jsonl"
CHECKPOINT_DIR_DEFAULT = WORKSPACE / "checkpoints"


def print_h(level: str, msg: str):
    print(f"[{level}] {msg}")


def run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=check)


def check_workspace() -> bool:
    print_h("INFO", f"Checking workspace mount at {WORKSPACE}...")
    try:
        WORKSPACE.mkdir(parents=True, exist_ok=True)
        test_file = WORKSPACE / ".preflight_write_test"
        test_file.write_text("ok")
        test_file.unlink()
        print_h("OK", "Workspace is writable.")
        return True
    except Exception as e:
        print_h("ERROR", f"Workspace not writable: {e}")
        return False


def check_gpu() -> bool:
    print_h("INFO", "Checking GPU with torch and nvidia-smi...")
    gpu_ok = False
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print_h("OK", f"CUDA available. Device: {name}")
            gpu_ok = True
        else:
            print_h("WARN", "torch.cuda.is_available() = False")
    except Exception as e:
        print_h("ERROR", f"Torch import/CUDA check failed: {e}")

    try:
        res = run(["nvidia-smi"])  # type: ignore[arg-type]
        if res.returncode == 0:
            print_h("OK", "nvidia-smi found:")
            print(res.stdout.splitlines()[0])
            gpu_ok = True or gpu_ok
        else:
            print_h("WARN", "nvidia-smi not available")
    except Exception:
        print_h("WARN", "nvidia-smi not available")

    return gpu_ok


def check_dependencies() -> bool:
    print_h("INFO", "Checking required Python packages...")
    required = [
        "transformers",
        "datasets",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "torch",
        "faker",
        "numpy",
        "huggingface_hub",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except Exception:
            missing.append(pkg)
    if missing:
        print_h("ERROR", f"Missing packages: {', '.join(missing)}")
        return False
    print_h("OK", "All required packages are installed.")
    return True


def ensure_accelerate_default() -> bool:
    print_h("INFO", "Ensuring Accelerate default config exists...")
    try:
        from accelerate.utils import write_basic_config
        write_basic_config()
        print_h("OK", "Accelerate default config written.")
        return True
    except Exception as e:
        print_h("WARN", f"Could not write accelerate default via API: {e}. Trying CLI...")
        try:
            run(["accelerate", "config", "default"], check=True)
            print_h("OK", "Accelerate default config created via CLI.")
            return True
        except Exception as e2:
            print_h("ERROR", f"Failed to create accelerate default config: {e2}")
            return False


def check_hf_login() -> bool:
    print_h("INFO", "Checking Hugging Face authentication...")
    token = os.environ.get("HF_TOKEN")
    try:
        from huggingface_hub import whoami
        if token:
            whoami(token=token)
            print_h("OK", "HF_TOKEN is valid.")
            return True
        else:
            # Try cached login
            whoami()
            print_h("OK", "Found cached Hugging Face login.")
            return True
    except Exception:
        print_h("WARN", "Not logged into Hugging Face (set HF_TOKEN to enable uploads).")
        return True  # Not a hard failure for training


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print_h("ERROR", f"Config not found at {CONFIG_PATH}")
        return {}
    cfg = json.loads(CONFIG_PATH.read_text())
    print_h("OK", "Loaded training config.")
    return cfg


def ensure_dataset(config: dict) -> bool:
    dataset_path = Path(config.get("dataset_path", str(DATASET_DEFAULT)))
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        try:
            with dataset_path.open("r") as f:
                _ = f.readline()
            print_h("OK", f"Dataset present at {dataset_path}")
            return True
        except Exception as e:
            print_h("ERROR", f"Dataset unreadable: {e}")
            return False
    # Try generating a tiny sample
    gen = ROOT / "dataset_gen.py"
    if not gen.exists():
        print_h("ERROR", "dataset_gen.py not found to create dataset")
        return False
    print_h("INFO", f"Dataset not found. Generating a small sample at {dataset_path}...")
    res = run([sys.executable, str(gen), "--samples", "10", "--output", str(dataset_path)])
    print(res.stdout)
    ok = dataset_path.exists()
    print_h("OK" if ok else "ERROR", f"Dataset generation {'succeeded' if ok else 'failed'}.")
    return ok


def ensure_checkpoints(config: dict) -> bool:
    out = Path(config.get("output_dir", str(CHECKPOINT_DIR_DEFAULT)))
    try:
        out.mkdir(parents=True, exist_ok=True)
        print_h("OK", f"Checkpoint directory ready at {out}")
        return True
    except Exception as e:
        print_h("ERROR", f"Failed to create checkpoint dir: {e}")
        return False


def main() -> int:
    ok = True
    ok &= check_workspace()
    ok &= check_gpu()
    ok &= check_dependencies()
    ok &= ensure_accelerate_default()
    ok &= check_hf_login()
    cfg = load_config()
    if not cfg:
        return EXIT_FAIL
    ok &= ensure_dataset(cfg)
    ok &= ensure_checkpoints(cfg)

    print_h("INFO", "Performing trainer dry run...")
    try:
        res = run([sys.executable, str(ROOT / "train_phi3.py"), "--config", str(CONFIG_PATH), "--dry_run"], check=True)
        print(res.stdout)
        print_h("OK", "Dry run succeeded.")
    except subprocess.CalledProcessError as e:
        print_h("ERROR", f"Dry run failed:\n{e.stdout}")
        ok = False

    return EXIT_OK if ok else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
