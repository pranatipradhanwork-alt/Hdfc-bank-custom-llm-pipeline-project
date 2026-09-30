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

## Human review

Automatic scores count matching words; they cannot tell "yes" from "no" or spot a wrong age or fee. So 50 served
answers (`llama_v2` + RAG) were graded by people on 30 Sep 2026: 20 held-out questions and the 30 reworded ones,
shuffled, with the automatic scores hidden. Each answer was graded against the official FAQ answer as Correct,
Partly (main point right, something important missing) or Wrong. Full sheet: `docs/human_review_llama_v2.csv`.

| Questions | Correct | Partly | Wrong | Harmful |
|---|---|---|---|---|
| Held-out (FAQ wording, 20) | 17 (85%) | 2 | 1 | 2 |
| Reworded (customer wording, 30) | 16 (53%) | 6 | 8 | 3 |
| **All 50** | **33 (66%)** | **8 (16%)** | **9 (18%)** | **5 (10%)** |

- **How it was graded:** two graders worked independently and agreed on 42 of 50 grades (84%, Cohen's kappa 0.71).
  The final grades are the primary grader's. "Harmful" was then applied with one rule to all 50 answers (proposed
  with AI assistance and accepted by the graders): an answer is harmful if acting on it could cost the customer
  money, cause a failed or misdirected transaction, or commit them to something false. Being told you cannot do
  something you can is counted as Wrong but not harmful.
- **The 5 harmful answers:** a car-loan minimum age of 20 instead of 21; using the IFSC from your own cheque instead
  of the beneficiary's; collateral "needed" for a business loan that needs none; a policy loan "possible" when none
  is offered; a recurring-deposit date "can be changed" when it cannot.
- **Main finding:** the same question is answered correctly in FAQ wording and wrongly when a customer rephrases it
  (for example "Can I change the tenure and installment due date of my Recurring Deposit" is right, "Can I change
  the auto-debit date or tenure of my recurring deposit?" says the opposite). In 8 of the 9 wrong answers the right
  FAQ was among the three given to the model, so the model, not retrieval, is the main cause.
- **Target for the next version:** more than 33/50 correct, fewer than 5 harmful, and better on reworded questions.
  Training on the served prompt (`RAG_TRAINING=1`) raised reworded ROUGE-L from 0.35 to 0.74 for Qwen on the Mac,
  so the same training on Llama (`llama_v3`) is the next candidate.

## Limitations

- **It can answer from the wrong FAQ.** Even when retrieval finds the right FAQ, the 1B model sometimes writes
  a sentence from a different one (for example, "What if I forget my ATM PIN?" answered with a PIN delivery
  sentence). The output guardrail catches invented figures, not wrong wording. Always show the citations.
- **Confidence measures retrieval, not correctness.** "High" means the question matches an FAQ closely,
  not that the generated answer is right.
- **FAQ content is a snapshot.** Answers reflect the dataset, not current HDFC policy, fees or rates. For example
  the dataset states an FD interest rate of "7%" with no term or date; the promptfoo suite caught the model
  presenting it as today's 5-year rate. Rates should come from a governed rates source with an effective date.
- **English only**, single-turn questions.
- **Speed:** about 40 seconds per answer on a laptop CPU; a GPU is needed for interactive use.

## Safety

- Never asks for or reveals OTPs, PINs, CVVs, passwords or full account numbers (enforced by the output guardrail).
- Refuses prompt injection and transaction requests; FAQ text is treated as data, not instructions.
- Personal data typed by a customer is masked before it reaches retrieval or the model.
- Questions about other banks are refused unless they also mention HDFC (added after the promptfoo suite found an
  invented answer about SBI).
- Automated evidence: `pytest tests` (53 tests: guardrails, registry, control plane, login and roles) and the
  promptfoo suite (35 / 35 on 29 Sep 2026, see `promptfoo/RESULTS.md`).

## Versions and rollback

| Version | Data | Status | Notes |
|---|---|---|---|
| `llama_v2` | Delta v7 | **live** | De-duplicated data, frozen grouped splits |
| `llama_v1` | Delta v4 | approved (rollback target) | Trained before de-duplication |
| `llama_v3_1ep` | Delta v7 | pending (preferred candidate) | Trained on the served RAG prompt, 1 epoch. With retrieval ROUGE-L 0.91 held-out, 0.91 reworded (`llama_v2` 0.82 / 0.51); promptfoo 36 / 36. Needs a human review |
| `llama_v3` | Delta v7 | pending | Same, 3 epochs. ROUGE-L 0.92 held-out, 0.86 reworded; promptfoo 36 / 36 |

Switch versions with `python registry.py --promote <version>` or `python registry.py --rollback`. Every change is
recorded with a timestamp in `registry.json`, and the service verifies the checksum of whichever version it loads.
