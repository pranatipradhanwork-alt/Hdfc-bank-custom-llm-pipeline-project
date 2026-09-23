#!/usr/bin/env bash
# One-shot AI/ML environment setup for WSL (Ubuntu).
# Usage:  bash setup_wsl.sh
# Safe to re-run: pip skips anything already installed, and the log is kept.
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The venv lives in the WSL home folder (much faster than /mnt/c + OneDrive).
VENV_DIR="${VENV_DIR:-$HOME/hdfc-venv}"
LOG_FILE="$PROJECT_DIR/setup_wsl.log"
PIP_OPTS=(--retries 10 --default-timeout 120)

exec > >(tee -a "$LOG_FILE") 2>&1

step() { echo; echo "==================== $* ($(date +%H:%M:%S)) ===================="; }
fail() { echo "[ERROR] $*"; exit 1; }

step "1/7  System packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -y
  sudo apt-get install -y build-essential git curl python3-venv python3-dev python3-pip
else
  echo "[WARN] apt-get not found, skipping system packages"
fi

step "2/7  Choosing Python"
if command -v python3.14 >/dev/null 2>&1; then
  PY=python3.14
else
  PY=python3
  echo "[WARN] python3.14 not found, using $($PY --version). The project expects 3.14."
  echo "       (Install it first if you need exactly 3.14, e.g. via the deadsnakes PPA or uv.)"
fi
echo "Using: $($PY --version) at $(command -v $PY)"

step "3/7  Virtual environment at $VENV_DIR"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  $PY -m venv "$VENV_DIR" || fail "could not create venv"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python --version
python -m pip install "${PIP_OPTS[@]}" --upgrade pip wheel setuptools

step "4/7  Project requirements (pinned: torch, transformers, peft, trl, ...)"
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
  # Large downloads (torch is several GB). Retries make this resilient to drops.
  python -m pip install "${PIP_OPTS[@]}" -r "$PROJECT_DIR/requirements.txt" \
    || echo "[WARN] some pinned requirements failed, see log above; continuing with extras"
else
  echo "[WARN] requirements.txt not found in $PROJECT_DIR, installing core packages unpinned"
  python -m pip install "${PIP_OPTS[@]}" torch transformers datasets accelerate deltalake python-dotenv pyyaml
fi

step "5/7  Extra AI/ML packages"
for pkg in peft trl bitsandbytes sentencepiece huggingface_hub \
           "scikit-learn" pandas numpy matplotlib jupyter ipykernel; do
  echo "--- $pkg"
  python -m pip install "${PIP_OPTS[@]}" "$pkg" || echo "[WARN] failed to install $pkg"
done

step "6/7  Jupyter kernel"
python -m ipykernel install --user --name hdfc-venv --display-name "Python (hdfc-venv)" \
  || echo "[WARN] could not register Jupyter kernel"

step "7/7  GPU check + verification (test.py)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  echo "[INFO] nvidia-smi not found: running CPU-only (fine for tests, slow for training)."
fi

if [ -f "$PROJECT_DIR/test.py" ]; then
  python "$PROJECT_DIR/test.py" | tee "$PROJECT_DIR/test_output.txt"
else
  echo "[WARN] test.py not found next to this script."
fi

echo
echo "DONE. Activate later with:  source $VENV_DIR/bin/activate"
echo "Full log: $LOG_FILE      test.py output: $PROJECT_DIR/test_output.txt"
