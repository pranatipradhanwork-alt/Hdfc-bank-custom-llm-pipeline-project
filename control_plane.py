"""Control plane records: datasets, training runs, served requests and reviewer feedback.

Heavy jobs run on the GPU worker (clean_data.py, train.py). This file keeps the records that the API and
dashboard show, and the rules that gate each step: only prepared data can be approved, only approved data
can be trained on, and only approved base models can be used.

Worker commands (run after the heavy job finishes):
  python control_plane.py --record-prepared ds-001   # attach the newest quality report after clean_data.py
  python control_plane.py --sync-runs                # copy finished training runs from MLflow
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

CONTROL_DIR = Path("control")
DATASETS_FILE = CONTROL_DIR / "datasets.json"
RUNS_FILE = CONTROL_DIR / "runs.json"
REQUESTS_FILE = CONTROL_DIR / "requests.jsonl"
FEEDBACK_FILE = CONTROL_DIR / "feedback.jsonl"
AUDIT_FILE = CONTROL_DIR / "audit.jsonl"
ASSISTANTS_FILE = CONTROL_DIR / "assistants.json"
QUALITY_REPORTS_DIR = Path("data/quality_reports")

APPROVED_BASE_MODELS = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def append_line(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def change_status(record, status, note):
    record["status"] = status
    record["history"].append({"time": now(), "status": status, "note": note})


# ---------- Datasets: registered -> preparing -> prepared -> approved ----------

def list_datasets():
    return read_json(DATASETS_FILE, [])


def get_dataset(dataset_id):
    for dataset in list_datasets():
        if dataset["id"] == dataset_id:
            return dataset
    raise KeyError(f"Unknown dataset: {dataset_id}")


def save_dataset(updated):
    datasets = [updated if d["id"] == updated["id"] else d for d in list_datasets()]
    write_json(DATASETS_FILE, datasets)


MIN_TEAM_FAQS = 10  # a team knowledge index needs at least this many FAQs to be useful


def register_dataset(name, source, owner, purpose, classification, permission_basis, retention,
                     parent_id=None, keywords=None):
    """Register a data source. A team dataset sets parent_id + keywords: it selects part of an approved dataset."""
    if classification == "restricted":
        raise ValueError("Restricted data cannot be registered for fine-tuning")
    if parent_id:
        if get_dataset(parent_id)["status"] != "approved":
            raise ValueError(f"Parent dataset {parent_id} is not approved")
        if not keywords:
            raise ValueError("A team dataset needs keywords that select its FAQs")
    datasets = list_datasets()
    dataset = {
        "id": f"ds-{len(datasets) + 1:03d}",
        "name": name, "source": source, "owner": owner, "purpose": purpose,
        "classification": classification, "permission_basis": permission_basis, "retention": retention,
        "parent_id": parent_id, "keywords": keywords or [],
        "status": "registered", "delta_version": None, "quality_report": None, "approved_by": None,
        "history": [{"time": now(), "status": "registered", "note": f"Registered by {owner}"}],
    }
    write_json(DATASETS_FILE, datasets + [dataset])
    return dataset


def team_index_dir(dataset_id):
    return Path("models") / f"rag_index_{dataset_id}"


def request_preparation(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset["status"] == "approved":
        raise ValueError("Approved datasets are frozen; register a new dataset to change the data")
    if dataset.get("parent_id"):
        return prepare_team_dataset(dataset)
    change_status(dataset, "preparing", "Preparation requested; run `python clean_data.py` on the worker")
    save_dataset(dataset)
    return dataset


def prepare_team_dataset(dataset):
    """Team datasets reuse already-cleaned, approved FAQs, so they are prepared right away (no GPU needed)."""
    from rag import INDEX_DIR, build_team_index

    parent = get_dataset(dataset["parent_id"])
    meta = build_team_index(dataset["keywords"], team_index_dir(dataset["id"]), INDEX_DIR)
    dataset["delta_version"] = parent["delta_version"]
    dataset["quality_report"] = {
        "final_rows": meta["faqs"], "derived_from": parent["id"], "keywords": dataset["keywords"],
        "delta_version": parent["delta_version"], "pii_masked_upstream": True, "duplicates_removed_upstream": True,
    }
    change_status(dataset, "prepared",
                  f"Built a knowledge index of {meta['faqs']} FAQs from {parent['id']} (Delta v{parent['delta_version']})")
    save_dataset(dataset)
    return dataset


def record_prepared(dataset_id):
    """Attach the newest quality report written by clean_data.py."""
    dataset = get_dataset(dataset_id)
    reports = sorted(QUALITY_REPORTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not reports:
        raise ValueError("No quality report found; run clean_data.py first")
    report = json.loads(reports[-1].read_text())
    dataset["quality_report"] = report
    dataset["delta_version"] = report["delta_version"]
    change_status(dataset, "prepared", f"Prepared as Delta version {report['delta_version']} ({reports[-1].name})")
    save_dataset(dataset)
    return dataset


def approve_dataset(dataset_id, approved_by):
    dataset = get_dataset(dataset_id)
    if dataset["status"] != "prepared":
        raise ValueError(f"Only prepared datasets can be approved (status is {dataset['status']})")
    if dataset.get("parent_id"):
        # Quality gate for team datasets: enough FAQs to answer from
        rows = dataset["quality_report"]["final_rows"]
        if rows < MIN_TEAM_FAQS:
            raise ValueError(f"Only {rows} FAQs matched the keywords; at least {MIN_TEAM_FAQS} are needed")
    else:
        # Quality gate for training data: the test split must not contain near-copies of training questions
        leaked = dataset["quality_report"]["contamination"]["test_rows_at_or_above_0.95"]
        if leaked > 0:
            raise ValueError(f"{leaked} test rows are near-copies of training rows; fix the split before approval")
    dataset["approved_by"] = approved_by
    change_status(dataset, "approved", f"Approved and frozen at Delta version {dataset['delta_version']} by {approved_by}")
    save_dataset(dataset)
    return dataset


# ---------- Assistants: pending -> approved / rejected ----------
# An assistant = the shared fine-tuned model + one approved dataset's knowledge index.

def list_assistants():
    return read_json(ASSISTANTS_FILE, {})


def get_assistant(assistant_id):
    assistants = list_assistants()
    if assistant_id not in assistants:
        raise KeyError(f"Unknown assistant: {assistant_id}")
    return assistants[assistant_id]


def save_assistant(assistant_id, assistant):
    assistants = list_assistants()
    assistants[assistant_id] = assistant
    write_json(ASSISTANTS_FILE, assistants)


def create_assistant(assistant_id, name, description, dataset_id, created_by):
    if assistant_id in list_assistants():
        raise ValueError(f"Assistant {assistant_id} already exists")
    dataset = get_dataset(dataset_id)
    if dataset["status"] != "approved":
        raise ValueError(f"Dataset {dataset_id} is not approved")
    index_dir = team_index_dir(dataset_id) if dataset.get("parent_id") else Path("models/rag_index")
    assistant = {
        "name": name, "description": description, "purpose": assistant_id, "owner": dataset["owner"],
        "dataset_id": dataset_id, "index_dir": str(index_dir).replace("\\", "/"),
        "faqs": dataset["quality_report"]["final_rows"], "status": "pending",
        "created_by": created_by, "approved_by": None,
        "history": [{"time": now(), "status": "pending", "note": f"Created by {created_by} from {dataset_id}"}],
    }
    save_assistant(assistant_id, assistant)
    return assistant


def review_assistant(assistant_id, status, reviewer):
    if status not in ("approved", "rejected"):
        raise ValueError("Status must be 'approved' or 'rejected'")
    assistant = get_assistant(assistant_id)
    if assistant["status"] != "pending":
        raise ValueError(f"Assistant {assistant_id} is already {assistant['status']}")
    assistant["approved_by"] = reviewer if status == "approved" else None
    change_status(assistant, status, f"{status.capitalize()} by {reviewer}")
    save_assistant(assistant_id, assistant)
    return assistant


# ---------- Training runs ----------

def list_runs():
    return read_json(RUNS_FILE, [])


def get_run(run_id):
    for run in list_runs():
        if run["id"] == run_id:
            return run
    raise KeyError(f"Unknown run: {run_id}")


def request_run(name, dataset_id, base_model, config, seed=42):
    """Validate a fine-tuning request and queue it for the GPU worker."""
    dataset = get_dataset(dataset_id)
    if dataset["status"] != "approved":
        raise ValueError(f"Dataset {dataset_id} is not approved for fine-tuning")
    if base_model not in APPROVED_BASE_MODELS:
        raise ValueError(f"Base model {base_model} is not on the approved list")
    if not Path(config).exists():
        raise ValueError(f"Config {config} does not exist")
    runs = list_runs()
    if any(run["id"] == name for run in runs):
        raise ValueError(f"A run named {name} already exists")
    run = {
        "id": name, "status": "queued", "base_model": base_model, "dataset_id": dataset_id,
        "dataset_version": dataset["delta_version"], "config": config, "seed": seed,
        "command": f"BASE_MODEL={base_model} SEED={seed} LORA_OUTPUT_DIR=models/{name} python train.py",
        "requested_at": now(),
    }
    write_json(RUNS_FILE, runs + [run])
    return run


def sync_runs_from_mlflow(tracking_uri="sqlite:///mlflow.db", experiment="hdfc-bankfaq-lora"):
    """Copy finished training runs from MLflow into runs.json (evaluation-only runs are skipped)."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    found = mlflow.search_runs(experiment_names=[experiment], output_format="list")
    runs = {run["id"]: run for run in list_runs()}
    for mlflow_run in found:
        tags, metrics = mlflow_run.data.tags, mlflow_run.data.metrics
        if "output_dir" not in tags:
            continue
        name = Path(tags["output_dir"]).name
        config = tags["config_profile"]
        runs[name] = {
            "id": name,
            "status": "completed" if mlflow_run.info.status == "FINISHED" else mlflow_run.info.status.lower(),
            "base_model": tags["base_model"],
            "dataset_version": int(tags["dataset_version"]),
            "config": config,
            "lora": yaml.safe_load(Path(config).read_text()).get("lora"),
            "seed": int(tags["seed"]),
            "max_steps": int(tags["max_steps"]),
            "git_commit": tags["git_commit"],
            "device": tags.get("device"),
            "train_loss": round(metrics["train_loss"], 4) if "train_loss" in metrics else None,
            "test_loss": round(metrics["test_loss"], 4) if "test_loss" in metrics else None,
            "started_at": datetime.fromtimestamp(mlflow_run.info.start_time / 1000, timezone.utc).isoformat(timespec="seconds"),
            "minutes": round((mlflow_run.info.end_time - mlflow_run.info.start_time) / 60000, 1) if mlflow_run.info.end_time else None,
            "mlflow_run_id": mlflow_run.info.run_id,
            "output_dir": tags["output_dir"],
        }
    ordered = sorted(runs.values(), key=lambda run: run.get("started_at") or run.get("requested_at"))
    write_json(RUNS_FILE, ordered)
    return ordered


# ---------- Audit log ----------

def audit(user, action, resource, result, detail=""):
    """Record who did what, to which resource, and whether it succeeded."""
    append_line(AUDIT_FILE, {"time": now(), "user": user, "action": action, "resource": resource,
                             "result": result, "detail": detail})


def audit_log():
    return list(reversed(read_lines(AUDIT_FILE)))


# ---------- Served requests and feedback ----------

def log_request(response, user, assistant_id):
    # Identifiers and outcomes only: the customer's question is not stored
    append_line(REQUESTS_FILE, {
        "trace_id": response.trace_id, "time": now(), "user": user, "assistant": assistant_id, "ok": True,
        "model_version": response.model.model_version,
        "adapter_sha256": response.model.adapter_sha256, "confidence": response.confidence,
        "escalation_required": response.escalation_required, "policy_flags": response.policy_flags,
        "latency_ms": response.latency_ms,
    })


def log_failed_request(trace_id, user, assistant_id, model_version, latency_ms, error):
    """A request the gateway could not answer (a server error), so the success-rate SLO counts it."""
    append_line(REQUESTS_FILE, {
        "trace_id": trace_id, "time": now(), "user": user, "assistant": assistant_id, "ok": False,
        "model_version": model_version, "adapter_sha256": None, "confidence": None, "escalation_required": True,
        "policy_flags": [], "latency_ms": latency_ms, "error": type(error).__name__,
    })


def find_request(trace_id):
    for request in read_lines(REQUESTS_FILE):
        if request["trace_id"] == trace_id:
            return request
    raise KeyError(f"Unknown trace id: {trace_id}")


# Service level objectives: the targets the gateway is held to (checked over the last 24 hours)
SLO_WINDOW_HOURS = 24
SLO_SUCCESS_RATE = 0.99                                            # requests answered without a server error
SLO_P95_LATENCY_MS = int(os.getenv("SLO_P95_LATENCY_MS", "30000"))  # 95% of answers within this time (GPU target)
SLO_MAX_BAD_FEEDBACK = 0.20                                        # share of rated answers marked bad


def percentile(values, share):
    """Value below which `share` of the sorted values fall (nearest-rank method)."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, round(share * len(ordered)) - 1)]


def within_window(records, hours=SLO_WINDOW_HOURS):
    cutoff = datetime.now(timezone.utc).timestamp() - hours * 3600
    return [r for r in records if datetime.fromisoformat(r["time"]).timestamp() >= cutoff]


def check_slos(requests, feedback):
    """Each SLO as {name, target, actual, status}; status is met, breached, or no_data."""
    ok = [r for r in requests if r.get("ok", True)]
    success_rate = len(ok) / len(requests) if requests else None
    p95 = percentile([r["latency_ms"] for r in ok], 0.95)
    rated = len(feedback)
    bad_share = sum(1 for f in feedback if f["rating"] == "bad") / rated if rated else None

    def status(actual, is_met):
        return "no_data" if actual is None else ("met" if is_met else "breached")

    return [
        {"name": "Success rate", "target": f">= {SLO_SUCCESS_RATE:.0%}",
         "actual": None if success_rate is None else f"{success_rate:.1%}",
         "status": status(success_rate, success_rate is not None and success_rate >= SLO_SUCCESS_RATE)},
        {"name": "p95 answer time", "target": f"<= {SLO_P95_LATENCY_MS / 1000:.0f} s",
         "actual": None if p95 is None else f"{p95 / 1000:.1f} s",
         "status": status(p95, p95 is not None and p95 <= SLO_P95_LATENCY_MS)},
        {"name": "Bad feedback", "target": f"<= {SLO_MAX_BAD_FEEDBACK:.0%} of rated answers",
         "actual": None if bad_share is None else f"{bad_share:.0%} of {rated}",
         "status": status(bad_share, bad_share is not None and bad_share <= SLO_MAX_BAD_FEEDBACK)},
    ]


def monitoring_summary():
    """Counts, latency percentiles and SLO status from the request log, for the Monitoring page."""
    requests = read_lines(REQUESTS_FILE)
    feedback = read_lines(FEEDBACK_FILE)
    ok = [r for r in requests if r.get("ok", True)]
    answered = [r for r in ok if not r["escalation_required"]]
    flags = {}
    for request in requests:
        for flag in request["policy_flags"]:
            flags[flag] = flags.get(flag, 0) + 1
    latencies = [r["latency_ms"] for r in ok]
    return {
        "requests": len(requests),
        "failed": len(requests) - len(ok),
        "answered": len(answered),
        "escalated": len(ok) - len(answered),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "policy_flags": flags,
        "feedback_good": sum(1 for f in feedback if f["rating"] == "good"),
        "feedback_bad": sum(1 for f in feedback if f["rating"] == "bad"),
        "slo_window_hours": SLO_WINDOW_HOURS,
        "slos": check_slos(within_window(requests), within_window(feedback)),
        "recent": list(reversed(requests[-50:])),
    }


def prometheus_metrics(live_model):
    """The same numbers in Prometheus text format, so a Prometheus server can scrape /metrics."""
    requests = read_lines(REQUESTS_FILE)
    summary = monitoring_summary()
    lines = ["# HELP hdfc_requests_total Gateway requests by assistant and outcome.",
             "# TYPE hdfc_requests_total counter"]
    counts = {}
    for r in requests:
        outcome = "failed" if not r.get("ok", True) else ("escalated" if r["escalation_required"] else "answered")
        key = (r.get("assistant", "customer_faq"), outcome)
        counts[key] = counts.get(key, 0) + 1
    for (assistant, outcome), value in sorted(counts.items()):
        lines.append(f'hdfc_requests_total{{assistant="{assistant}",outcome="{outcome}"}} {value}')
    lines += ["# HELP hdfc_policy_flags_total Guardrail events by type.", "# TYPE hdfc_policy_flags_total counter"]
    lines += [f'hdfc_policy_flags_total{{flag="{flag}"}} {n}' for flag, n in sorted(summary["policy_flags"].items())]
    lines += ["# HELP hdfc_answer_latency_seconds Answer time percentiles.", "# TYPE hdfc_answer_latency_seconds gauge"]
    for name, quantile in (("p50", "0.5"), ("p95", "0.95")):
        value = summary[f"{name}_latency_ms"]
        if value is not None:
            lines.append(f'hdfc_answer_latency_seconds{{quantile="{quantile}"}} {value / 1000:.3f}')
    lines += ["# HELP hdfc_slo_met 1 when the SLO is met, 0 when breached (absent when there is no data).",
              "# TYPE hdfc_slo_met gauge"]
    for slo in summary["slos"]:
        if slo["status"] != "no_data":
            label = slo["name"].lower().replace(" ", "_")
            lines.append(f'hdfc_slo_met{{slo="{label}"}} {1 if slo["status"] == "met" else 0}')
    lines += ["# HELP hdfc_live_model Model version currently served.", "# TYPE hdfc_live_model gauge",
              f'hdfc_live_model{{version="{live_model}"}} 1']
    return "\n".join(lines) + "\n"


def add_feedback(trace_id, rating, comment="", reviewer="reviewer"):
    request = find_request(trace_id)
    feedback = {"time": now(), "trace_id": trace_id, "model_version": request["model_version"],
                "rating": rating, "comment": comment, "reviewer": reviewer}
    append_line(FEEDBACK_FILE, feedback)
    return feedback


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record-prepared", metavar="DATASET_ID")
    parser.add_argument("--sync-runs", action="store_true")
    args = parser.parse_args()
    if args.record_prepared:
        print(json.dumps(record_prepared(args.record_prepared), indent=2))
    if args.sync_runs:
        for run in sync_runs_from_mlflow():
            print(f"{run['id']:22} {run['status']:10} {run['base_model']:35} data v{run['dataset_version']}  test loss {run.get('test_loss')}")
