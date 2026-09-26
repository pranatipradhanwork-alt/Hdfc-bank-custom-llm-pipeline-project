"""Compare a fine-tuned LoRA adapter against its base model on the held-out BankFAQs test split.

Usage:
  python evaluate.py --adapter models/qwen_v1 --chat                                   # type questions, see both answers
  python evaluate.py --adapter models/qwen_v1 --ask "How do I block my debit card?"   # one question, side by side
  python evaluate.py --adapter models/qwen_v1 --limit 20                               # quick scored run
  python evaluate.py --adapter models/qwen_v1                                          # full test split, logged to MLflow
  python evaluate.py --adapter models/llama_v1 --rag                                   # also score both models with RAG
  python evaluate.py --adapter models/llama_v1 --rag --reworded                        # customer-style rewordings

The base model is the same model with the adapter switched off, so the only difference is the fine-tuning.
--rag adds base+RAG and fine-tuned+RAG, with retrieved FAQs from rag.py in the prompt, plus retrieval hit rates.
--reworded uses data/rag_reworded_questions.csv: test questions reworded so they do not match the FAQ text.
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList

from guardrails import invented_specifics

# Must match train.py, so prompts and the test split are exactly what training used
SYSTEM_PROMPT = (
    "You are HDFC Bank's customer support assistant. Answer banking questions clearly "
    "and concisely. Never ask for or reveal OTPs, PINs, CVVs, passwords or full account numbers."
)
SEED = 42
NO_REPEAT_NGRAM = 8  # longest phrase an answer may repeat; stops "The payout will be higher..." loops
LOCAL_S3_VAULT = Path("data/s3_storage_vault/cleaned_banking_table")
REPORTS_DIR = Path("reports")
REWORDED_QUESTIONS = Path("data/rag_reworded_questions.csv")


def load_test_split(dataset_version):
    # Same per-task 80/10/10 split as train.py, on the exact Delta version the adapter was trained on
    df = DeltaTable(str(LOCAL_S3_VAULT), version=dataset_version).to_pandas()
    if "Task" not in df.columns:
        df["Task"] = "faq"
    if "Split" in df.columns:
        # Frozen partitions (v7+); same row order train.py uses for its test split
        test = df[df["Split"] == "test"].sample(frac=1, random_state=SEED)
        return test[test["Task"] == "faq"].reset_index(drop=True)
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


class NoRepeatAnswerNGrams(LogitsProcessor):
    # transformers' no_repeat_ngram_size also counts the prompt, which would stop RAG answers copying phrases
    # from the retrieved FAQs; this only blocks an answer from repeating its own n-grams
    def __init__(self, prompt_length, n):
        self.prompt_length, self.n = prompt_length, n

    def __call__(self, input_ids, scores):
        for row, tokens in enumerate(input_ids[:, self.prompt_length:].tolist()):
            if len(tokens) < self.n:
                continue
            prefix = tokens[-(self.n - 1):]
            banned = [tokens[i + self.n - 1] for i in range(len(tokens) - self.n + 1) if tokens[i:i + self.n - 1] == prefix]
            if banned:
                scores[row, banned] = -float("inf")
        return scores


def plain_messages(question):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]


@torch.no_grad()
def generate(model, tokenizer, conversations, max_new_tokens, batch_size):
    eos_ids = model.generation_config.eos_token_id
    eos_ids = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    results = []
    for start in range(0, len(conversations), batch_size):
        batch = conversations[start:start + batch_size]
        prompts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            logits_processor=LogitsProcessorList([NoRepeatAnswerNGrams(encoded["input_ids"].shape[1], NO_REPEAT_NGRAM)]),
        )
        for row in generated[:, encoded["input_ids"].shape[1]:].tolist():
            stopped = any(token in eos_ids for token in row)
            results.append((tokenizer.decode(row, skip_special_tokens=True).strip(), stopped))
        print(f"  {min(start + batch_size, len(conversations))}/{len(conversations)} answered")
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
    parser.add_argument("--rag", action="store_true", help="Also score both models with retrieved FAQs (run rag.py --build first)")
    parser.add_argument("--reworded", action="store_true", help=f"Use the reworded questions in {REWORDED_QUESTIONS}")
    parser.add_argument("--dataset-version", type=int,
                        help="Delta version for questions/references (default: the RAG index version with --rag, else the training version)")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    train_metrics = json.loads((adapter_dir / "metrics.json").read_text())
    base_model = train_metrics["base_model"]
    print(f"[INFO] Loading {base_model} with adapter {adapter_dir}...")
    tokenizer, model = load_model(base_model, str(adapter_dir))

    def compare(question):
        with model.disable_adapter():
            base_answer, _ = generate(model, tokenizer, [plain_messages(question)], args.max_new_tokens, 1)[0]
        tuned_answer, _ = generate(model, tokenizer, [plain_messages(question)], args.max_new_tokens, 1)[0]
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

    index = None
    if args.rag:
        from rag import FaqIndex, rag_messages
        index = FaqIndex()
    default_version = index.meta["dataset_version"] if index else train_metrics["dataset_version"]
    dataset_version = args.dataset_version if args.dataset_version is not None else default_version
    if dataset_version != train_metrics["dataset_version"]:
        # v5/v6 only rewrote text, but v7 re-partitioned the data, so older adapters may have trained on its test rows
        print(f"[WARN] Scoring on Delta v{dataset_version} but the adapter trained on v{train_metrics['dataset_version']}: "
              "held-out status is only guaranteed when both use the same split")
    test = load_test_split(dataset_version)
    if args.reworded:
        reworded = pd.read_csv(REWORDED_QUESTIONS)
        test = test.iloc[reworded["test_index"]].reset_index(drop=True)
        test["User_Query"] = reworded["reworded_question"].tolist()
    if args.limit:
        test = test.head(args.limit)
    questions = test["User_Query"].tolist()
    references = test["Target_Banking_Response"].tolist()
    question_set = "reworded" if args.reworded else "held-out test"
    print(f"[INFO] Evaluating on {len(questions)} {question_set} questions (Delta version {dataset_version})")

    plain = [plain_messages(q) for q in questions]
    conditions = {}
    print("[INFO] Base model (adapter off)...")
    with model.disable_adapter():
        conditions["base"] = generate(model, tokenizer, plain, args.max_new_tokens, args.batch_size)
    print("[INFO] Fine-tuned model (adapter on)...")
    conditions["finetuned"] = generate(model, tokenizer, plain, args.max_new_tokens, args.batch_size)
    if index:
        retrieved = index.search(questions)
        with_faqs = [rag_messages(q, r) for q, r in zip(questions, retrieved)]
        # Prompts are ~4x longer with three FAQs in them, so halve the batch to stay inside GPU memory
        rag_batch = max(1, args.batch_size // 2)
        print("[INFO] Base model + RAG...")
        with model.disable_adapter():
            conditions["base_rag"] = generate(model, tokenizer, with_faqs, args.max_new_tokens, rag_batch)
        print("[INFO] Fine-tuned model + RAG...")
        conditions["finetuned_rag"] = generate(model, tokenizer, with_faqs, args.max_new_tokens, rag_batch)

    rows = []
    for i, (question, reference) in enumerate(zip(questions, references)):
        row = {"question": question, "reference": reference}
        if index:
            # A hit means a retrieved FAQ carries the reference answer (several FAQs share one question wording)
            hits = [index.is_match(faq, reference) for faq, _ in retrieved[i]]
            context = " ".join(faq["Target_Banking_Response"] for faq, _ in retrieved[i])
            row.update({
                "retrieved_questions": " || ".join(faq["User_Query"] for faq, _ in retrieved[i]),
                "retrieval_hit_at_1": hits[0],
                "retrieval_hit_at_k": any(hits),
            })
        for prefix, answers in conditions.items():
            answer, stopped = answers[i]
            invented = invented_specifics(answer, reference)
            row.update({
                f"{prefix}_answer": answer,
                f"{prefix}_rougeL": rouge_l(answer, reference),
                f"{prefix}_token_f1": token_f1(answer, reference),
                f"{prefix}_words": len(words(answer)),
                f"{prefix}_stopped": stopped,
                f"{prefix}_invented": "; ".join(invented),
                f"{prefix}_has_invented": bool(invented),
            })
            if prefix.endswith("_rag"):
                # Numbers copied from a retrieved FAQ are grounded even when that FAQ is not the reference one
                row[f"{prefix}_has_unsupported"] = bool(invented_specifics(answer, reference + " " + context))
        rows.append(row)
    results = pd.DataFrame(rows)

    summary = {
        "adapter": str(adapter_dir),
        "base_model": base_model,
        "dataset_version": dataset_version,
        "question_set": question_set,
        "questions": len(results),
        "reference_avg_words": round(results["reference"].map(lambda t: len(words(t))).mean(), 1),
    }
    if index:
        summary["embed_model"] = index.meta["embed_model"]
        summary["retrieval_hit_at_1"] = round(results["retrieval_hit_at_1"].mean(), 4)
        summary["retrieval_hit_at_k"] = round(results["retrieval_hit_at_k"].mean(), 4)
    for prefix in conditions:
        summary[f"{prefix}_rougeL"] = round(results[f"{prefix}_rougeL"].mean(), 4)
        summary[f"{prefix}_token_f1"] = round(results[f"{prefix}_token_f1"].mean(), 4)
        summary[f"{prefix}_avg_words"] = round(results[f"{prefix}_words"].mean(), 1)
        summary[f"{prefix}_stop_rate"] = round(results[f"{prefix}_stopped"].mean(), 4)
        summary[f"{prefix}_invented_rate"] = round(results[f"{prefix}_has_invented"].mean(), 4)
        if prefix.endswith("_rag"):
            summary[f"{prefix}_unsupported_rate"] = round(results[f"{prefix}_has_unsupported"].mean(), 4)
    summary["finetuned_wins_rougeL"] = round((results["finetuned_rougeL"] > results["base_rougeL"]).mean(), 4)

    REPORTS_DIR.mkdir(exist_ok=True)
    name = f"eval_{adapter_dir.name}" + ("_rag" if index else "") + ("_reworded" if args.reworded else "")
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
            "dataset_version": str(dataset_version),
            "training_dataset_version": str(train_metrics["dataset_version"]),
            "question_set": question_set,
            "rag": str(bool(index)),
            "training_run_id": train_metrics.get("mlflow_run_id", "unknown"),
        })
        if index:
            mlflow.set_tag("embed_model", index.meta["embed_model"])
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(json_path))

    print(f"\n[SUCCESS] Results: {csv_path} (answers side by side), {json_path} (scores)")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
