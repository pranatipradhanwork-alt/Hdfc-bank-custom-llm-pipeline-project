# Model card: HDFC FAQ assistant (`llama_v2`)

## Summary

| | |
|---|---|
| **What it is** | A LoRA adapter on `meta-llama/Llama-3.2-1B-Instruct` that answers HDFC Bank customer FAQs, used together with FAQ retrieval (RAG) and guardrails |
| **Live version** | `llama_v2` (see [`registry.json`](registry.json)); previous version `llama_v1` is kept for rollback |
| **Artifacts** | Private Hugging Face repo `shyam003/hdfc-faq-assistant` (`llama_v2/`, `llama_v1/`, `rag_index/`) |
| **Adapter SHA-256** | `16e4545e77faa44b4631a5979cba62205c183c849cfd0e5be766512072e0a0e6` (checked at startup; the service refuses to start on a mismatch) |
| **Owners** | HDFC GenAI capstone team (AlmaBetter) |

## Intended use

- **For:** answering general questions about HDFC Bank products and services (cards, accounts, loans, deposits,
  insurance, NetBanking) from the bank's published FAQs, with citations.
- **Not for:** account-specific questions (balances, transactions), carrying out transactions, financial or
  investment advice, or anything outside HDFC Bank's FAQs. These are refused or escalated to PhoneBanking.
- **Users:** customers on web or mobile, and support staff checking answers.

## Training data

- **Source:** Kaggle BankFAQs dataset (1,773 question–answer pairs).
- **Cleaning (Delta table version 7):**
  - Masked account/card numbers, phone numbers, emails and OTP/PIN/CVV values.
  - Restored "HDFC Bank" where the dataset had anonymised it as "M&N Bank".
  - Removed 295 exact and 73 near-duplicate FAQs, leaving 1,405.
- **Split:** 1,124 train / 141 validation / 140 test. Questions that mean the same thing are kept in the same split.
  No test question is ≥ 0.95 similar to a training question (report: `data/quality_reports/cleaned_banking_table_v7.json`).

## Training

- QLoRA (4-bit base model + LoRA adapter), 3 epochs, seed 42, on one RTX 4050 (6 GB).
- Config: `configs/training/cuda-qlora.yaml`. Code commit `35cd560`. MLflow run `d150a6f636ae472c832e3a33b9d5bf02`.

## How it answers

1. **Input guardrails:** mask personal data; refuse prompt injection and transaction requests.
2. **Retrieval:** find the 3 closest FAQs with `BAAI/bge-small-en-v1.5` embeddings. If the best match scores
   below 0.70, the question is treated as out of scope and escalated.
3. **Generation:** the fine-tuned model answers using only those FAQs.
4. **Output guardrails:** if the answer asks for credentials or states a figure not in the cited FAQs,
   it is replaced with an escalation message.
5. **Response:** answer, citations (FAQ id, question, score, dataset version), confidence, escalation flag,
   policy flags, and the exact model version and checksum.

## Evaluation

Held-out test split (140 questions) and 30 hand-reworded questions. "Base" is Llama 3.2 1B without the adapter.

| Setup | ROUGE-L (test) | ROUGE-L (reworded) | Answers with invented figures (test) |
|---|---|---|---|
| Base | 0.11 | 0.11 | 31% |
| Fine-tuned | 0.22 | 0.24 | 19% |
| Base + RAG | 0.47 | 0.49 | 6% |
| **Fine-tuned + RAG (served)** | **0.82** | **0.51** | 7% |

- Retrieval finds the right FAQ in the top 3 for 94% of test questions (93% reworded).
- **Quote the reworded score (0.51).** The 0.82 on the test split is inflated: the fine-tuned model has seen
  near-copies of those answers during training.
- After output guardrails, 0% of served answers contain figures missing from the cited FAQs.

## Limitations

- **It can answer from the wrong FAQ.** Even when retrieval finds the right FAQ, the 1B model sometimes writes
  a sentence from a different one (for example, "What if I forget my ATM PIN?" answered with a PIN delivery
  sentence). The output guardrail catches invented figures, not wrong wording. Always show the citations.
- **Confidence measures retrieval, not correctness.** "High" means the question matches an FAQ closely,
  not that the generated answer is right.
- **FAQ content is a snapshot.** Answers reflect the dataset, not current HDFC policy, fees or rates.
- **English only**, single-turn questions.
- **Speed:** about 40 seconds per answer on a laptop CPU; a GPU is needed for interactive use.

## Safety

- Never asks for or reveals OTPs, PINs, CVVs, passwords or full account numbers (enforced by the output guardrail).
- Refuses prompt injection and transaction requests; FAQ text is treated as data, not instructions.
- Personal data typed by a customer is masked before it reaches retrieval or the model.
- Guardrail and registry behaviour is covered by `pytest tests` (15 tests).

## Versions and rollback

| Version | Data | Status | Notes |
|---|---|---|---|
| `llama_v2` | Delta v7 | **live** | De-duplicated data, frozen grouped splits |
| `llama_v1` | Delta v4 | approved (rollback target) | Trained before de-duplication |

Switch versions with `python registry.py --promote <version>` or `python registry.py --rollback`. Every change is
recorded with a timestamp in `registry.json`, and the service verifies the checksum of whichever version it loads.
