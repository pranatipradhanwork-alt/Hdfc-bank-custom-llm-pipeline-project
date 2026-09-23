"""Compare a fine-tuned LoRA adapter against its base model on the held-out BankFAQs test split.

Usage:
  python evaluate.py --adapter models/qwen_v1 --chat                                   # type questions, see both answers
  python evaluate.py --adapter models/qwen_v1 --ask "How do I block my debit card?"   # one question, side by side
  python evaluate.py --adapter models/qwen_v1 --limit 20                               # quick scored run
  python evaluate.py --adapter models/qwen_v1                                          # full test split, logged to MLflow

The base model is the same model with the adapter switched off, so the only difference is the fine-tuning.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import mlflow
import pandas as pd
import torch
from deltalake import DeltaTable
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Must match train.py, so prompts and the test split are exactly what training used
SYSTEM_PROMPT = (
    "You are HDFC Bank's customer support assistant. Answer banking questions clearly "
    "and concisely. Never ask for or reveal OTPs, PINs, CVVs, passwords or full account numbers."
)
SEED = 42
LOCAL_S3_VAULT = Path("data/s3_storage_vault/cleaned_banking_table")
REPORTS_DIR = Path("reports")


def load_test_split(dataset_version):
    # Same per-task 80/10/10 split as train.py, on the exact Delta version the adapter was trained on
    df = DeltaTable(str(LOCAL_S3_VAULT), version=dataset_version).to_pandas()
    if "Task" not in df.columns:
        df["Task"] = "faq"
    tests = []
    for _, group in df.groupby("Task"):
        group = group.sample(frac=1, random_state=SEED)
        n_train = int(len(group) * 0.8)
        n_val = int(len(group) * 0.1)
        tests.append(group.iloc[n_train + n_val:])
    test = pd.concat(tests).sample(frac=1, random_state=SEED)
    return test[test["Task"] == "faq"].reset_index(drop=True)


def load_model(base_model, adapter_dir):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    tokenizer.padding_side = "left"  # left padding for batched generation
    if torch.cuda.is_available():
        # Same 4-bit setup as training (cuda-qlora.yaml)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=quantization, device_map="auto")
    else:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype).to(device)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def generate(model, tokenizer, questions, max_new_tokens, batch_size):
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    results = []
    for start in range(0, len(questions), batch_size):
        batch = questions[start:start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in batch
        ]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in generated[:, encoded["input_ids"].shape[1]:].tolist():
            stopped = any(token in eos_ids for token in row)
            results.append((tokenizer.decode(row, skip_special_tokens=True).strip(), stopped))
        print(f"  {min(start + batch_size, len(questions))}/{len(questions)} answered")
    return results


def words(text):
    return re.findall(r"\w+", text.lower())


def rouge_l(prediction, reference):
    # F1 of the longest common word subsequence: rewards answers that follow the reference wording
    pred, ref = words(prediction), words(reference)
    if not pred or not ref:
        return 0.0
    previous = [0] * (len(ref) + 1)
    for p in pred:
        current = [0]
        for j, r in enumerate(ref):
            current.append(previous[j] + 1 if p == r else max(previous[j + 1], current[j]))
        previous = current
    lcs = previous[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(pred), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction, reference):
    # F1 of shared words regardless of order: rewards answers that cover the reference content
    pred, ref = Counter(words(prediction)), Counter(words(reference))
    overlap = sum((pred & ref).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="Adapter folder written by train.py, e.g. models/qwen_v1")
    parser.add_argument("--ask", help="Ask one question and print base vs fine-tuned answers")
    parser.add_argument("--chat", action="store_true", help="Keep asking questions interactively (empty line or 'exit' to quit)")
    parser.add_argument("--limit", type=int, help="Only evaluate the first N test questions")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    train_metrics = json.loads((adapter_dir / "metrics.json").read_text())
    base_model = train_metrics["base_model"]
    print(f"[INFO] Loading {base_model} with adapter {adapter_dir}...")
    tokenizer, model = load_model(base_model, str(adapter_dir))

    def compare(question):
        with model.disable_adapter():
            base_answer, _ = generate(model, tokenizer, [question], args.max_new_tokens, 1)[0]
        tuned_answer, _ = generate(model, tokenizer, [question], args.max_new_tokens, 1)[0]
        print(f"\nQUESTION:\n{question}\n\nBASE MODEL:\n{base_answer}\n\nFINE-TUNED:\n{tuned_answer}\n")

    if args.ask:
        compare(args.ask)
        return
    if args.chat:
        print("\nAsk a banking question (empty line or 'exit' to quit).")
        while True:
            question = input("> ").strip()
            if question.lower() in ("", "exit", "quit"):
                return
            compare(question)

    test = load_test_split(train_metrics["dataset_version"])
    if args.limit:
        test = test.head(args.limit)
    questions = test["User_Query"].tolist()
    print(f"[INFO] Evaluating on {len(questions)} held-out test questions (Delta version {train_metrics['dataset_version']})")

    print("[INFO] Base model (adapter off)...")
    with model.disable_adapter():
        base = generate(model, tokenizer, questions, args.max_new_tokens, args.batch_size)
    print("[INFO] Fine-tuned model (adapter on)...")
    tuned = generate(model, tokenizer, questions, args.max_new_tokens, args.batch_size)

    rows = []
    for (question, reference), (base_answer, base_stopped), (tuned_answer, tuned_stopped) in zip(
        test[["User_Query", "Target_Banking_Response"]].itertuples(index=False), base, tuned
    ):
        rows.append({
            "question": question,
            "reference": reference,
            "base_answer": base_answer,
            "finetuned_answer": tuned_answer,
            "base_rougeL": rouge_l(base_answer, reference),
            "finetuned_rougeL": rouge_l(tuned_answer, reference),
            "base_token_f1": token_f1(base_answer, reference),
            "finetuned_token_f1": token_f1(tuned_answer, reference),
            "base_words": len(words(base_answer)),
            "finetuned_words": len(words(tuned_answer)),
            "base_stopped": base_stopped,
            "finetuned_stopped": tuned_stopped,
        })
    results = pd.DataFrame(rows)

    summary = {
        "adapter": str(adapter_dir),
        "base_model": base_model,
        "dataset_version": train_metrics["dataset_version"],
        "questions": len(results),
        "reference_avg_words": round(results["reference"].map(lambda t: len(words(t))).mean(), 1),
    }
    for prefix in ("base", "finetuned"):
        summary[f"{prefix}_rougeL"] = round(results[f"{prefix}_rougeL"].mean(), 4)
        summary[f"{prefix}_token_f1"] = round(results[f"{prefix}_token_f1"].mean(), 4)
        summary[f"{prefix}_avg_words"] = round(results[f"{prefix}_words"].mean(), 1)
        summary[f"{prefix}_stop_rate"] = round(results[f"{prefix}_stopped"].mean(), 4)
    summary["finetuned_wins_rougeL"] = round((results["finetuned_rougeL"] > results["base_rougeL"]).mean(), 4)

    REPORTS_DIR.mkdir(exist_ok=True)
    name = f"eval_{adapter_dir.name}"
    csv_path = REPORTS_DIR / f"{name}.csv"
    json_path = REPORTS_DIR / f"{name}.json"
    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2))

    # Log next to the training runs, linked to the run that produced the adapter
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("hdfc-bankfaq-lora")
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({
            "run_type": "evaluation",
            "adapter": str(adapter_dir),
            "base_model": base_model,
            "dataset_version": str(train_metrics["dataset_version"]),
            "training_run_id": train_metrics.get("mlflow_run_id", "unknown"),
        })
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(json_path))

    print(f"\n[SUCCESS] Results: {csv_path} (answers side by side), {json_path} (scores)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
