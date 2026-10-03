# HDFC Bank: Custom LLM Development Pipeline

A governed pipeline that turns approved banking data into a fine-tuned, retrieval-grounded assistant for HDFC Bank
customer FAQs, and proves its quality and safety before it is served: LoRA / QLoRA fine-tuning, RAG, guardrails,
automatic and human evaluation, a red-team suite, a model registry with approvals and rollback, a FastAPI gateway,
a role-based web platform and monitoring.

| | |
|---|---|
| **Live model** | `llama_v3_1ep`: Llama 3.2 1B Instruct + LoRA adapter, trained on the served RAG prompt (since 1 Oct 2026) |
| **Rollback target** | `llama_v2` |
| **Deployed app** | https://shyam003-hdfc-ai-platform.hf.space (Hugging Face Space, CPU; sign in with the demo accounts provided with the submission) |
| **Demo video** | _link to be added_ |
| **Presentation** | [`docs/presentation.pdf`](docs/presentation.pdf) (20 slides) |
| **Model card** | [`MODEL_CARD.md`](MODEL_CARD.md) |
| **API contract** | [`docs/api-contract.md`](docs/api-contract.md) |

## Results

Same questions for every model: 140 held-out test questions in FAQ wording, 30 hand-reworded customer-style
questions, and 50 served answers graded by people.

| | `llama_v2` (previous) | **`llama_v3_1ep` (live)** |
|---|---|---|
| ROUGE-L with retrieval, FAQ wording (140) | 0.82 | **0.91** |
| ROUGE-L with retrieval, customer wording (30) | 0.51 | **0.91** |
| Human review: fully correct (50) | 31 (62%) | **41 (82%)** |
| Human review: customer-worded correct (30) | 16 (53%) | **25 (83%)** |
| Harmful answers (50) | 5 | **3** |
| promptfoo red-team suite | 35 / 35 | **36 / 36** |

- **What changed:** human review showed the previous model answered correctly in FAQ wording but often wrongly when
  a customer rephrased the question, although the right FAQ was in its prompt in 8 of its 9 wrong answers. It had
  been trained on question → answer only. `llama_v3_1ep` is trained on the served prompt (`RAG_TRAINING=1`: the
  question with its own FAQ and two look-alike FAQs, shuffled), so it learns to read the FAQs.
- **Model selection:** on the same data and settings, Llama 3.2 1B beat Qwen 2.5 0.5B on customer-worded questions
  (0.91 vs 0.74 with RAG-aware training). One epoch beat three (0.91 vs 0.86): validation loss stopped improving after
  about 0.7 epochs.
- **Reproduced on a second machine:** trained on an RTX 4050 (WSL); re-evaluated on a MacBook Air from the Hugging
  Face artifact (checksum verified): 0.91 held-out, 0.88 reworded, no unsupported figures.
- **Live monitoring** (MacBook Air, no GPU; 36 red-team + 50 customer requests): 100% success (86 / 86),
  p50 1.7 s, p95 6.9 s, all 50 customer questions answered.

### Model versions

| Version | Training | Status | Result (ROUGE-L with retrieval: FAQ / customer wording) |
|---|---|---|---|
| `llama_v1` | Llama 3.2 1B, plain fine-tuning, 3 epochs, data before de-duplication | approved | 0.60 / 0.46 |
| `llama_v2` | Llama 3.2 1B, plain fine-tuning, 3 epochs, de-duplicated data (v7) | approved (rollback target) | 0.82 / 0.51; human review 31/50 correct, 5 harmful |
| `llama_v3` | Llama 3.2 1B, **RAG-aware training**, 3 epochs | rejected: lower on customer wording than 1 epoch | 0.92 / 0.86 |
| **`llama_v3_1ep`** | Llama 3.2 1B, **RAG-aware training**, 1 epoch (8 min on an RTX 4050) | **live** since 1 Oct 2026 | **0.91 / 0.91**; human review **41/50 correct, 3 harmful**; promptfoo 36/36 |
| `qwen_mac_rag` | Qwen 2.5 0.5B, RAG-aware training, 1 epoch, trained on a Mac | pending (lightweight backup; adapter only on the Mac) | 0.92 / 0.74 |
| `qwen_mac_v2`, `qwen_mac_v3` | Qwen 2.5 0.5B, plain fine-tuning, 1 and 3 epochs, Mac | rejected | 0.62 / 0.35 and 0.54 / 0.27 |

The registry (`registry.json`) holds each version's checksum, data version, MLflow run, evaluation, human review and
approval history; the dashboard's Models and Evaluations pages show the same.

Full evaluation, the human review method and known limitations are in [`MODEL_CARD.md`](MODEL_CARD.md); graded
sheets are in [`docs/`](docs/).

## How a question is answered

```
question ─► input guardrails ─► retrieval ─► fine-tuned model ─► output guardrails ─► typed response
            mask PII; block      top-3 FAQs    Llama 3.2 1B +      block unsupported     answer, citations,
            injection,           (bge-small);  LoRA, answers only  figures and           confidence, escalation,
            transactions,        < 0.70 →      from those FAQs     credential requests   model version + checksum,
            other banks          escalate                                                trace ID
```

Facts come from the governed FAQ index at answer time, not from model weights, so FAQ updates need no retraining.

## Repository layout

| Path | What it is |
|---|---|
| `download_data.py`, `clean_data.py` | Download BankFAQs / Banking77; clean, mask PII, deduplicate, split, write the versioned Delta table |
| `train.py`, `configs/training/` | LoRA / QLoRA fine-tuning with MLflow tracking; config chosen by hardware |
| `rag.py` | FAQ embedding index and retrieval |
| `evaluate.py` | Base vs fine-tuned, with and without retrieval, held-out and reworded questions |
| `guardrails.py` | Input and output controls (PII masking, injection, transactions, other banks, unsupported figures) |
| `inference.py` | Governed answer path: guardrails → retrieval → model → output checks |
| `server.py`, `schemas.py`, `auth.py` | FastAPI gateway, typed contracts, login, roles and rate limits |
| `registry.py`, `registry.json` | Model registry: versions, checksums, evidence, approvals, promotion and rollback |
| `control_plane.py`, `control/` | Datasets, training runs, assistants, audit log, monitoring and SLOs |
| `frontend/index.html` | Web platform (Admin, AI Engineer, Employee) |
| `promptfoo/` | Red-team and quality suite against the live API, and its results |
| `tests/` | Unit tests (guardrails, registry, control plane, login and roles); run in CI |
| `infrastructure/`, `Dockerfile`, `docker-compose.yml`, `deploy/` | Prometheus, Grafana, Kubernetes manifests, container and Hugging Face Space |
| `docs/` | API contract and human review sheets |

## Setup

Python 3.12+ (3.13 used on the Mac). Model downloads need a Hugging Face read token for an account that has
accepted the Llama 3.2 licence and can read the private repo `hdfc-capstone/hdfc-faq-assistant`.

```bash
python -m venv hdfc_env && source hdfc_env/bin/activate
pip install -r requirements.txt          # training and evaluation (requirements-serve.txt for serving only)
cp .env.example .env                     # then fill in HF_TOKEN, APP_KEY and METRICS_TOKEN
```

Never commit `.env`.

## Pipeline

```bash
python download_data.py                  # BankFAQs and Banking77 into data/
python clean_data.py                     # mask PII, deduplicate, split, write a new Delta version + quality report
python rag.py --build                    # embed every FAQ into models/rag_index

RAG_TRAINING=1 NUM_EPOCHS=1 LORA_OUTPUT_DIR=models/llama_v3_1ep python train.py
python evaluate.py --adapter models/llama_v3_1ep --rag              # held-out questions, with and without retrieval
python evaluate.py --adapter models/llama_v3_1ep --rag --reworded   # customer-worded questions
```

`train.py` uses `cuda-qlora.yaml` (Llama 3.2 1B, 4-bit) on an NVIDIA GPU, `mac-lora.yaml` (Qwen 2.5 0.5B) on Apple
MPS and `cpu-demo.yaml` otherwise. Every run is logged to MLflow (`sqlite:///mlflow.db`) with base model, data
version, config, seed, code commit and platform.

| Variable | Purpose |
|---|---|
| `BASE_MODEL` | Override the config's base model, e.g. `Qwen/Qwen2.5-0.5B-Instruct` |
| `LORA_OUTPUT_DIR` | Where the adapter is saved (also the MLflow run name) |
| `RAG_TRAINING` | `1` trains on the served prompt (question + its FAQ + two retrieved look-alike FAQs) |
| `NUM_EPOCHS` | Override the config's epochs for one run |
| `MAX_STEPS` | Short staged runs (5 → 150 → full); `-1` trains for the configured epochs |

`evaluate.py` options: `--rag`, `--reworded`, `--limit N`, `--dataset-version N` (score against a specific Delta
version), `--index PATH` (a RAG index built from another version). Reports are written to `reports/` (local) and
MLflow.

## Run the platform

```bash
set -a && source .env && set +a
uvicorn server:app --port 7860           # web platform at http://localhost:7860, API at /v1, docs at /docs
```

On start the server loads the registry's live model from Hugging Face and refuses to serve it if its checksum differs
from the registered one. With Docker, `docker compose up --build` starts the platform with Prometheus
(http://localhost:9090) and Grafana (http://localhost:3000).

| Role | Can do |
|---|---|
| Admin | Everything, including approving models, datasets and assistants, promote, rollback, users |
| AI Engineer | Register and prepare datasets, request runs, register models; cannot approve |
| Employee | Use the assistants assigned to them |

### Deployment

The review app runs as a Docker Space on Hugging Face: https://shyam003-hdfc-ai-platform.hf.space. The Space is
built from this repository's `Dockerfile` (CPU-only PyTorch, port 7860) with `deploy/space-README.md` as its README.
At start-up it downloads the base model and the live adapter from the private repo `hdfc-capstone/hdfc-faq-assistant`
using the Space secret `HF_TOKEN` (a read token), and verifies the adapter checksum against `registry.json`. To
redeploy, upload the files of `main` to the Space (`git archive main`, with `deploy/space-README.md` copied over
`README.md`); the Space rebuilds automatically. The workflow `.github/workflows/keep-space-awake.yml` calls
`/v1/health` once a day so the Space is not paused between review visits.

The Space runs on CPU, so answers are slower than on a GPU. Team
assistants created at runtime keep their knowledge index on the Space's disk, so they last until the next restart.

## API

All endpoints are under `/v1`; the full contract is in [`docs/api-contract.md`](docs/api-contract.md).

| Endpoint | Purpose |
|---|---|
| `POST /v1/inference` | Answer a question through the governed path (API key in `x-api-key`) |
| `POST /v1/feedback` | Rate an answer, linked to its trace ID |
| `POST /v1/auth/login` · `/logout` · `GET /v1/auth/me` | Session login and current user |
| `GET/POST /v1/datasets` · `/{id}/prepare` · `/{id}/approve` | Register, prepare and approve dataset versions |
| `GET/POST /v1/runs` · `GET /v1/runs/{id}` · `GET /v1/base-models` | Training runs and approved base models |
| `GET /v1/models` · `POST /v1/models/register` · `/{version}/review` · `/{version}/promote` | Registry, review and promotion |
| `POST /v1/deployments/production/rollback` | Return to the previous approved version |
| `GET /v1/evaluations` · `GET /v1/monitoring` · `GET /metrics` | Evaluation summaries, SLOs and Prometheus metrics |
| `GET/POST /v1/assistants` · `/{id}/review` · `GET /v1/users` · `/v1/audit` | Assistants, users and the audit log |
| `GET /v1/health` | Liveness and the model version being served |

## Testing and safety

```bash
python -m pytest -q tests                # 55 unit tests; also run by GitHub Actions on every push
```

The promptfoo suite (36 cases: prompt injection, transactions, personal data, credentials, privacy, off-topic,
other banks, invented figures, bias, robustness) calls the running API like a banking application; see
[`promptfoo/README.md`](promptfoo/README.md) and [`promptfoo/RESULTS.md`](promptfoo/RESULTS.md).

## Release process

1. An engineer trains and evaluates a candidate, uploads the adapter to the Hugging Face repo and registers it as
   **pending** with its checksum and evidence.
2. A second person re-runs the evaluation, runs promptfoo and grades the 50-question human review.
3. The release criteria must all hold against the live model: customer-worded ROUGE-L higher, held-out not lower,
   promptfoo passing, more answers correct and fewer harmful in the human review.
4. An admin **approves** and **promotes**; the server reloads the new version, and the previous one stays available
   for one-click **rollback**. Every step is recorded in `registry.json` and the audit log.

## Limitations

- FAQ content is a snapshot: rates and fees can go stale; they should come from a governed live source.
- Three harmful answers remain in the human review (self-employed car-loan age 20 instead of 21 on two questions,
  collateral described for a business loan that needs none).
- Retrieval can miss the right FAQ for some wordings; "confidence" measures retrieval similarity, not correctness.
- The human review covers 50 questions, so its accuracy is an estimate.
- English only and single-turn. Not yet implemented: artifact signing, canary releases, GPU serving with vLLM / TGI.
