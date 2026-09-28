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
from datetime import datetime, timezone
from pathlib import Path

import yaml

CONTROL_DIR = Path("control")
DATASETS_FILE = CONTROL_DIR / "datasets.json"
RUNS_FILE = CONTROL_DIR / "runs.json"
REQUESTS_FILE = CONTROL_DIR / "requests.jsonl"
FEEDBACK_FILE = CONTROL_DIR / "feedback.jsonl"
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


def register_dataset(name, source, owner, purpose, classification, permission_basis, retention):
    if classification == "restricted":
        raise ValueError("Restricted data cannot be registered for fine-tuning")
    datasets = list_datasets()
    dataset = {
        "id": f"ds-{len(datasets) + 1:03d}",
        "name": name, "source": source, "owner": owner, "purpose": purpose,
        "classification": classification, "permission_basis": permission_basis, "retention": retention,
        "status": "registered", "delta_version": None, "quality_report": None, "approved_by": None,
        "history": [{"time": now(), "status": "registered", "note": f"Registered by {owner}"}],
    }
    write_json(DATASETS_FILE, datasets + [dataset])
    return dataset


def request_preparation(dataset_id):
    dataset = get_dataset(dataset_id)
    if dataset["status"] == "approved":
        raise ValueError("Approved datasets are frozen; register a new dataset to change the data")
    change_status(dataset, "preparing", "Preparation requested; run `python clean_data.py` on the worker")
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
    # Quality gate: the test split must not contain near-copies of training questions
    leaked = dataset["quality_report"]["contamination"]["test_rows_at_or_above_0.95"]
    if leaked > 0:
        raise ValueError(f"{leaked} test rows are near-copies of training rows; fix the split before approval")
    dataset["approved_by"] = approved_by
    change_status(dataset, "approved", f"Approved and frozen at Delta version {dataset['delta_version']} by {approved_by}")
    save_dataset(dataset)
    return dataset


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


# ---------- Served requests and feedback ----------

def log_request(response):
    # Identifiers and outcomes only: the customer's question is not stored
    append_line(REQUESTS_FILE, {
        "trace_id": response.trace_id, "time": now(), "model_version": response.model.model_version,
        "adapter_sha256": response.model.adapter_sha256, "confidence": response.confidence,
        "escalation_required": response.escalation_required, "policy_flags": response.policy_flags,
        "latency_ms": response.latency_ms,
    })


def find_request(trace_id):
    for request in read_lines(REQUESTS_FILE):
        if request["trace_id"] == trace_id:
            return request
    raise KeyError(f"Unknown trace id: {trace_id}")


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
