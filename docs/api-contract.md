# Model gateway API contract (v1)

The review UI and the promptfoo suite build against this contract. The typed source of truth is
[`schemas.py`](../schemas.py); change it only with a version bump.

## `POST /v1/inference`

Answers one customer question through the governed path:
input guardrails → FAQ retrieval → fine-tuned model → output guardrails.

### Request

```json
{
  "question": "I forgot my ATM PIN, how can I reset it?",
  "channel": "web",
  "purpose": "customer_faq",
  "max_new_tokens": 256
}
```

| Field | Type | Notes |
|---|---|---|
| `question` | string, 1–1000 chars | Required. Identifiers and secrets are masked before use. |
| `channel` | `web` \| `mobile` \| `branch` \| `internal` | Default `web`. |
| `purpose` | `customer_faq` | Only approved purpose in this release. |
| `max_new_tokens` | int, 16–512 | Default 256. |

### Response

```json
{
  "trace_id": "5f0c2d7e9a8b4c1d8e2f3a4b5c6d7e8f",
  "answer": "You can reset your ATM PIN by logging in to Prepaid NetBanking using your IPIN ...",
  "citations": [
    {"faq_id": "faq-3b9e1c0a7d", "question": "What if I forget my ATM PIN", "score": 0.8712, "dataset_version": 7}
  ],
  "confidence": "high",
  "escalation_required": false,
  "missing_information": null,
  "policy_flags": [],
  "model": {
    "model_version": "hdfc-faq-assistant/2",
    "base_model": "meta-llama/Llama-3.2-1B-Instruct",
    "adapter": "models/llama_v2",
    "adapter_sha256": "…",
    "index_dataset_version": 7,
    "embed_model": "BAAI/bge-small-en-v1.5"
  },
  "latency_ms": 1840
}
```

### Behaviour the UI and tests can rely on

| Situation | `answer` | `escalation_required` | `policy_flags` includes |
|---|---|---|---|
| Normal banking question | Grounded answer | `false` | — |
| Off-topic (weather, stock tips, other banks) | Contact PhoneBanking / branch message | `true` | `out_of_scope` |
| Prompt injection ("ignore previous instructions…") | "I can only help with questions about HDFC Bank…" | `false` | `prompt_injection` |
| Transaction request ("transfer Rs 5000 for me") | "I can't carry out transactions…" | `true` | `transaction_request` |
| Customer typed an OTP / card / phone / email | Answered normally; value masked before use | as normal | `input_contains_masked_*` |
| Model asked for credentials | Replaced with escalation message | `true` | `asks_for_credentials` |
| Model stated a figure not in the cited FAQs | Replaced with escalation message | `true` | `unsupported_specifics` |
| A retrieved FAQ contained injected instructions | That FAQ is dropped from the context | as normal | `context_injection_removed` |

`confidence` comes from retrieval strength: `high` (top FAQ score ≥ 0.80), `medium` (0.70–0.80),
`low` (below 0.70, answer withheld). It says the question is in scope, not that the answer is correct.

## Other endpoints (planned for Day 3)

| Endpoint | Purpose |
|---|---|
| `GET /v1/health` | Liveness plus the model version currently served |
| `GET /v1/models` | Registered versions, their evaluation scores and which one is live |
| `POST /v1/models/{version}/promote` | Serve a version that passed the release gate |
| `POST /v1/models/rollback` | Return to the previously served version |
| `GET /v1/evaluations` | Latest evaluation summaries (base vs fine-tuned, with and without RAG) |
