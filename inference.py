"""One governed answer path: input guardrails -> retrieval -> fine-tuned model -> output guardrails -> typed response.

The gateway (server.py) and the review UI both go through Assistant.answer, so every response carries the same
citations, confidence, escalation decision, policy flags and model identity.
"""
import os
import hashlib
import json
import time
import uuid
from pathlib import Path

from huggingface_hub import snapshot_download

from evaluate import generate, load_model
from guardrails import check_input, check_output, is_injected, search_text
from rag import ANSWER_THRESHOLD, HIGH_CONFIDENCE, FaqIndex, rag_messages
from registry import load_registry
from schemas import Citation, InferenceResponse, ModelInfo

MODELS_DIR = Path("models")
# Private Hugging Face repo holding the released adapters (llama_v1, llama_v2) and the FAQ index (rag_index)
HF_REPO = os.getenv("HF_REPO", "shyam003/hdfc-faq-assistant")
# Optional override; by default the live version from registry.json is served
MODEL_VERSION = os.getenv("MODEL_VERSION")

ESCALATION_MESSAGE = (
    "I don't have verified information to answer that. Please contact HDFC Bank PhoneBanking "
    "or visit your nearest branch for help."
)
REFUSALS = {
    "prompt_injection": "I can only help with questions about HDFC Bank products and services.",
    "transaction_request": (
        "I can't carry out transactions or change your account. Please use NetBanking or MobileBanking, "
        "or contact HDFC Bank PhoneBanking."
    ),
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_local(folder):
    """Return models/<folder>, downloading it from the Hugging Face repo if it is not on this machine."""
    path = MODELS_DIR / folder
    if not path.exists():
        print(f"[INFO] {path} not found locally, downloading it from {HF_REPO}...")
        snapshot_download(repo_id=HF_REPO, allow_patterns=[f"{folder}/*"], local_dir=MODELS_DIR)
    return path


class Assistant:
    def __init__(self, adapter_dir=None, model_version=None, expected_sha256=None):
        # Which version to serve: the argument, else the MODEL_VERSION setting, else the registry's live version
        registry = load_registry()
        version = Path(adapter_dir or MODEL_VERSION or registry["live"]).name  # "models/llama_v2" -> "llama_v2"
        if version in registry["versions"]:
            # A registered version must match the checksum recorded when it was evaluated
            expected_sha256 = expected_sha256 or registry["versions"][version]["adapter_sha256"]
        adapter_dir = ensure_local(version)
        self.adapter_sha256 = file_sha256(adapter_dir / "adapter_model.safetensors")
        if expected_sha256 and self.adapter_sha256 != expected_sha256:
            # Refuse to serve an artifact that differs from the one that was evaluated and registered
            raise ValueError(f"Adapter checksum mismatch for {adapter_dir}: expected {expected_sha256}, got {self.adapter_sha256}")
        train_metrics = json.loads((adapter_dir / "metrics.json").read_text())
        self.tokenizer, self.model = load_model(train_metrics["base_model"], str(adapter_dir))
        self.index = FaqIndex(ensure_local("rag_index"))
        self.info = ModelInfo(
            model_version=model_version or adapter_dir.name,
            base_model=train_metrics["base_model"],
            adapter=str(adapter_dir),
            adapter_sha256=self.adapter_sha256,
            index_dataset_version=self.index.meta["dataset_version"],
            embed_model=self.index.meta["embed_model"],
        )

    def answer(self, question, max_new_tokens=256):
        started = time.perf_counter()
        trace_id = uuid.uuid4().hex
        masked_question, flags = check_input(question)

        def respond(answer, citations=(), confidence="low", escalate=True, missing=None):
            return InferenceResponse(
                trace_id=trace_id, answer=answer, citations=list(citations), confidence=confidence,
                escalation_required=escalate, missing_information=missing, policy_flags=flags, model=self.info,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )

        # Blocked requests never reach the model
        for flag, refusal in REFUSALS.items():
            if flag in flags:
                return respond(refusal, escalate=flag == "transaction_request", missing=f"Request not permitted: {flag}")

        retrieved = self.index.search([search_text(masked_question)])[0]
        # Source content is untrusted too: drop any FAQ carrying injected instructions
        clean = [(faq, score) for faq, score in retrieved if not is_injected(faq["Target_Banking_Response"])]
        if len(clean) < len(retrieved):
            flags.append("context_injection_removed")
        citations = [
            Citation(faq_id=faq["faq_id"], question=faq["User_Query"], score=round(score, 4),
                     dataset_version=self.index.meta["dataset_version"])
            for faq, score in clean
        ]
        top_score = clean[0][1] if clean else 0.0
        if top_score < ANSWER_THRESHOLD:
            flags.append("out_of_scope")
            return respond(ESCALATION_MESSAGE, missing="No approved FAQ matches this question")
        confidence = "high" if top_score >= HIGH_CONFIDENCE else "medium"

        answer, _ = generate(self.model, self.tokenizer, [rag_messages(masked_question, clean)], max_new_tokens, 1)[0]
        context = " ".join(faq["Target_Banking_Response"] for faq, _ in clean)
        answer, output_flags, unsupported = check_output(answer, context)
        flags.extend(output_flags)
        # Never return an answer that asks for credentials or states numbers the sources do not contain
        if "asks_for_credentials" in output_flags or unsupported:
            reason = "Answer asked for credentials" if "asks_for_credentials" in output_flags else (
                f"Answer contained figures not in the cited FAQs: {', '.join(unsupported)}")
            return respond(ESCALATION_MESSAGE, citations, confidence, missing=reason)
        return respond(answer, citations, confidence, escalate=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ask the governed assistant one question and print the JSON response")
    parser.add_argument("question")
    parser.add_argument("--adapter", help="Adapter to serve, e.g. llama_v1 (default: the live version in registry.json)")
    parser.add_argument("--expected-sha256", default=os.getenv("ADAPTER_SHA256"), help="Refuse to start if the adapter differs")
    args = parser.parse_args()
    print(Assistant(args.adapter, expected_sha256=args.expected_sha256).answer(args.question).model_dump_json(indent=2))
