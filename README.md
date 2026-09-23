# Hdfc-bank-custom-llm-pipeline-project

Fine-tunes `Qwen/Qwen2.5-0.5B` with LoRA on a PII-scrubbed banking FAQ dataset.

## Setup

The project expects **Python 3.14**. Key pins: transformers 5.17.0, torch 2.14.0, peft 0.21.0, trl 1.13.0.

### Windows (WSL2 + NVIDIA GPU)

Run inside WSL (Ubuntu) from the project folder:

    bash setup_wsl.sh
    source ~/hdfc-venv/bin/activate

The script installs everything into `~/hdfc-venv` (outside OneDrive, for speed),
runs `test.py`, and writes `setup_wsl.log` / `test_output.txt` (both gitignored).
Check that `test_output.txt` reports `CUDA available : True`; if not, run
`nvidia-smi` inside WSL to confirm the GPU is visible.

### macOS (Apple Silicon)

    python3.14 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python test.py

`bitsandbytes` is skipped on macOS; the Mac uses the `mac-lora.yaml` profile (no quantization).

## Pipeline

    python download_data.py   # fetch BankFAQs.csv from Kaggle
    python clean_data.py      # scrub PII, write Delta table to data/s3_storage_vault/
    python train.py           # train LoRA adapter -> models/hdfc_lora_adapter/

`train.py` picks the config automatically: `mac-lora.yaml` (MPS), `cuda-qlora.yaml` (CUDA)
or `cpu-demo.yaml` (CPU). Use `QUICK_TEST=1 python train.py` for a 5-step sanity run.

`data/` outputs and `models/` are gitignored; each machine regenerates them.
