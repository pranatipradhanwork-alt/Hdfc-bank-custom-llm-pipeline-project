"""Environment check for the HDFC custom-LLM pipeline.

Run:  python test.py            (fast, no downloads)
      python test.py --model    (also downloads Qwen2.5-0.5B and generates text)
"""
import importlib
import importlib.metadata as md
import platform
import sys
import time
from pathlib import Path

PASS, FAIL = "[ OK ]", "[FAIL]"
failures = []


def check(name, fn):
    t = time.time()
    try:
        result = fn()
        print(f"{PASS} {name}" + (f": {result}" if result else "") + f"  ({time.time() - t:.1f}s)")
    except Exception as e:  # noqa: BLE001
        failures.append(name)
        print(f"{FAIL} {name}: {type(e).__name__}: {e}")


print("=" * 60)
print("Python   :", sys.version.split()[0], "|", platform.platform())
print("Venv     :", sys.prefix)
print("=" * 60)

# 1. Package imports and versions
PACKAGES = [
    "torch", "transformers", "peft", "trl", "bitsandbytes", "sentencepiece",
    "huggingface_hub", "datasets", "accelerate", "deltalake", "pandas",
    "numpy", "sklearn", "matplotlib", "yaml", "jupyter_core", "ipykernel",
]
print("\n-- Packages --")
for pkg in PACKAGES:
    def _imp(p=pkg):
        m = importlib.import_module(p)
        return getattr(m, "__version__", None) or md.version(p)
    check(pkg, _imp)

# 2. Hardware
print("\n-- Hardware --")
try:
    import torch

    print("CUDA available :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU            :", torch.cuda.get_device_name(0))
        print("VRAM (GB)      :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print("MPS available  :", torch.backends.mps.is_available())
    print("CPU threads    :", torch.get_num_threads())
    check("torch tensor math", lambda: f"sum={(torch.ones(3, 3) @ torch.ones(3, 3)).sum().item()}")
except Exception as e:  # noqa: BLE001
    failures.append("torch")
    print(f"{FAIL} torch unavailable: {e}")

# 3. Tiny random Qwen2 + LoRA (validates transformers + peft + torch, no download)
print("\n-- Model + LoRA smoke test (no download) --")


def lora_smoke():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=1000, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(cfg)
    lora = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
                      lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    ids = torch.randint(0, 1000, (2, 16))
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"loss={out.loss.item():.3f}, trainable params {trainable}/{total}"


check("tiny Qwen2 + LoRA forward/backward", lora_smoke)

# 4. Project data (Delta table from clean_data.py)
print("\n-- Project data --")
vault = Path("data/s3_storage_vault/cleaned_banking_table")


def delta_check():
    from deltalake import DeltaTable
    if not vault.exists():
        return "not created yet (run download_data.py then clean_data.py)"
    dt = DeltaTable(str(vault))
    df = dt.to_pandas()
    return f"version {dt.version()}, {len(df)} rows, columns {list(df.columns)}"


check("Delta Lake table", delta_check)

# 5. Optional: real model download + generation
if "--model" in sys.argv:
    print("\n-- Real model test (downloads ~1 GB on first run) --")

    def real_model():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        name = "Qwen/Qwen2.5-0.5B"
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
        inputs = tok("How do I block my debit card?", return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
        return tok.decode(out[0], skip_special_tokens=True)[:120]

    check("Qwen2.5-0.5B generate", real_model)

print("\n" + "=" * 60)
if failures:
    print(f"RESULT: {len(failures)} check(s) failed -> {', '.join(failures)}")
    sys.exit(1)
print("RESULT: all checks passed")
