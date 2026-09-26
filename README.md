# Hdfc-bank-custom-llm-pipeline-project

Fine-tuning a small open model (LoRA / QLoRA) to answer HDFC Bank customer FAQs, tracked with MLflow.

## Pipeline

```bash
python download_data.py      # BankFAQs (+ Banking77) from Kaggle into data/
python clean_data.py         # clean, mask PII, write the Delta table in data/s3_storage_vault/
python train.py              # fine-tune; picks a config in configs/training/ by hardware
python evaluate.py --adapter models/llama_v1   # base vs fine-tuned on the held-out test split
```

`train.py` picks `cuda-qlora.yaml` (Llama 3.2 1B, 4-bit) on an NVIDIA GPU, `mac-lora.yaml` on Apple MPS and
`cpu-demo.yaml` otherwise. Useful environment variables:

| Variable | Purpose |
|---|---|
| `BASE_MODEL` | Override the config's base model, e.g. `Qwen/Qwen2.5-0.5B-Instruct` |
| `LORA_OUTPUT_DIR` | Where the adapter is saved, e.g. `models/llama_v1` (also the MLflow run name) |
| `MAX_STEPS` | Short staged runs (5 -> 150 -> full); `-1` trains for the configured epochs |

Data is split 80/10/10 (train/validation/test) with seed 42. `evaluate.py` rebuilds the same test split from the
Delta version recorded in the adapter's `metrics.json`, so results are reproducible.

## Model selection

**Chosen model: `models/llama_v1`**, a LoRA adapter on `meta-llama/Llama-3.2-1B-Instruct` (QLoRA, 3 epochs,
Delta dataset version 4, 1199 train / 149 validation / 151 test rows).

Both candidates were trained with identical settings and scored on the same 151 held-out questions. "Base" is the
same model with the adapter switched off.

| Adapter | Test loss | ROUGE-L base → tuned | Token F1 base → tuned | Avg words base → tuned | Tuned beats base |
|---|---|---|---|---|---|
| **llama_v1** (Llama 3.2 1B) | **2.22** | 0.107 → **0.209** | 0.166 → **0.257** | 137 → 43 | **77%** |
| qwen_v1 (Qwen2.5 0.5B) | 2.43 | 0.148 → 0.181 | 0.225 → 0.250 | 73 → 49 | 60% |

Reference answers average 56 words. Fine-tuning roughly doubled Llama's overlap with the reference answers and
cut its answer length from rambling (137 words) to on target (43 words).

### Limitations found in manual review

- **Facts are unreliable.** The model learned the support tone and answer length, not the product facts. In a
  20-question sample about 4 answers were correct; the rest were confident but wrong (wrong fees, limits,
  validity periods, coverage). 28 of 151 answers contain numbers, phone numbers or URLs that are not in the
  reference answer, e.g. a Rs. 10,000 balance for a zero-balance account and an invented helpline number.
- **Security:** no answer asks the customer to share an OTP, PIN or CVV. One answer (e-commerce activation)
  invents a registration flow with "enter your CVV / password" steps.
- **Repetition:** a few answers loop on one sentence until the token limit (19 of 151 did not stop on their own).
- **Branding:** the dataset refers to the bank as "M&N Bank" in most answers and "HDFC Bank" in some, and the
  model copies this.

### What comes next

Fine-tuning alone cannot make a 1B model memorise product facts, so the next stage is **RAG**: retrieve the
relevant FAQ answers and put them in the prompt, with `llama_v1` providing the voice. The repetition issue should
be handled in RAG generation settings (`repetition_penalty`, `no_repeat_ngram_size`), and the "M&N Bank" / "HDFC
Bank" naming is best fixed in `clean_data.py` before building the retrieval index.

Reproduce the numbers (runs are also logged to MLflow; `reports/` is local output):

```bash
python evaluate.py --adapter models/llama_v1
python evaluate.py --adapter models/qwen_v1
```
