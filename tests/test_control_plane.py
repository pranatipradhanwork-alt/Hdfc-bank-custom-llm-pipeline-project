import json

import pytest

import control_plane


@pytest.fixture(autouse=True)
def temporary_files(tmp_path, monkeypatch):
    # Point every control-plane file at a temporary folder so tests never change the real records
    monkeypatch.setattr(control_plane, "DATASETS_FILE", tmp_path / "datasets.json")
    monkeypatch.setattr(control_plane, "RUNS_FILE", tmp_path / "runs.json")
    monkeypatch.setattr(control_plane, "REQUESTS_FILE", tmp_path / "requests.jsonl")
    monkeypatch.setattr(control_plane, "FEEDBACK_FILE", tmp_path / "feedback.jsonl")
    reports = tmp_path / "quality_reports"
    reports.mkdir()
    monkeypatch.setattr(control_plane, "QUALITY_REPORTS_DIR", reports)
    return reports


def register():
    return control_plane.register_dataset("BankFAQs", "data/BankFAQs.csv", "data-owner", "customer_faq",
                                          "internal", "Public FAQ content", "2 years")


def write_report(folder, leaked=0):
    report = {"delta_version": 7, "contamination": {"test_rows_at_or_above_0.95": leaked}}
    (folder / "report_v7.json").write_text(json.dumps(report))


def prepared_dataset(folder, leaked=0):
    dataset = register()
    control_plane.request_preparation(dataset["id"])
    write_report(folder, leaked)
    return control_plane.record_prepared(dataset["id"])


def test_register_gives_new_dataset_an_id():
    assert register()["id"] == "ds-001"
    assert register()["id"] == "ds-002"


def test_restricted_data_is_refused():
    with pytest.raises(ValueError):
        control_plane.register_dataset("Chats", "crm", "owner", "customer_faq", "restricted", "none", "1 year")


def test_cannot_approve_before_preparation():
    dataset = register()
    with pytest.raises(ValueError):
        control_plane.approve_dataset(dataset["id"], "risk-lead")


def test_prepare_then_approve(temporary_files):
    dataset = prepared_dataset(temporary_files)
    assert dataset["delta_version"] == 7
    approved = control_plane.approve_dataset(dataset["id"], "risk-lead")
    assert approved["status"] == "approved"
    assert [step["status"] for step in approved["history"]] == ["registered", "preparing", "prepared", "approved"]


def test_contaminated_split_blocks_approval(temporary_files):
    dataset = prepared_dataset(temporary_files, leaked=3)
    with pytest.raises(ValueError):
        control_plane.approve_dataset(dataset["id"], "risk-lead")


def test_approved_dataset_is_frozen(temporary_files):
    dataset = prepared_dataset(temporary_files)
    control_plane.approve_dataset(dataset["id"], "risk-lead")
    with pytest.raises(ValueError):
        control_plane.request_preparation(dataset["id"])


def test_run_needs_approved_dataset(temporary_files):
    dataset = prepared_dataset(temporary_files)
    with pytest.raises(ValueError):
        control_plane.request_run("llama_v3", dataset["id"], "meta-llama/Llama-3.2-1B-Instruct",
                                  "configs/training/cuda-qlora.yaml")


def test_run_needs_approved_base_model(temporary_files):
    dataset = prepared_dataset(temporary_files)
    control_plane.approve_dataset(dataset["id"], "risk-lead")
    with pytest.raises(ValueError):
        control_plane.request_run("gpt_v1", dataset["id"], "some/unreviewed-model", "configs/training/cuda-qlora.yaml")


def test_valid_run_is_queued(temporary_files):
    dataset = prepared_dataset(temporary_files)
    control_plane.approve_dataset(dataset["id"], "risk-lead")
    run = control_plane.request_run("llama_v3", dataset["id"], "meta-llama/Llama-3.2-1B-Instruct",
                                    "configs/training/cuda-qlora.yaml")
    assert run["status"] == "queued"
    assert run["dataset_version"] == 7
    assert control_plane.get_run("llama_v3")["id"] == "llama_v3"


@pytest.fixture
def team_setup(tmp_path, monkeypatch, temporary_files):
    """An approved parent dataset plus a tiny FAQ index to select team FAQs from."""
    import numpy as np
    import pandas as pd
    import rag

    source = tmp_path / "rag_index"
    source.mkdir()
    questions = [f"How do I open a fixed deposit {i}" for i in range(12)] + ["How do I block my credit card"]
    pd.DataFrame({"faq_id": [f"faq-{i}" for i in range(13)], "User_Query": questions,
                  "Target_Banking_Response": ["Answer."] * 13}).to_parquet(source / "faqs.parquet")
    np.save(source / "embeddings.npy", np.eye(13, dtype=np.float32))
    (source / "meta.json").write_text(json.dumps({"dataset_version": 7, "embed_model": "test", "faqs": 13}))
    monkeypatch.setattr(rag, "INDEX_DIR", source)
    monkeypatch.setattr(control_plane, "ASSISTANTS_FILE", tmp_path / "assistants.json")
    monkeypatch.setattr(control_plane, "team_index_dir", lambda dataset_id: tmp_path / f"rag_index_{dataset_id}")

    parent = prepared_dataset(temporary_files)
    control_plane.approve_dataset(parent["id"], "risk-lead")
    return parent


def register_team(parent_id, keywords):
    return control_plane.register_dataset("FD FAQs", "BankFAQs subset", "fd-team", "fd_assistant", "public",
                                          "Approved FAQ content", "1 year", parent_id, keywords)


def test_team_dataset_needs_approved_parent(temporary_files):
    parent = prepared_dataset(temporary_files)  # prepared, not approved
    with pytest.raises(ValueError):
        register_team(parent["id"], ["fixed deposit"])


def test_team_dataset_is_prepared_right_away(team_setup, tmp_path):
    team = register_team(team_setup["id"], ["fixed deposit"])
    prepared = control_plane.request_preparation(team["id"])
    assert prepared["status"] == "prepared"
    assert prepared["quality_report"]["final_rows"] == 12  # the credit card FAQ is left out
    assert (tmp_path / f"rag_index_{team['id']}" / "faqs.parquet").exists()


def test_team_dataset_with_too_few_faqs_is_not_approved(team_setup):
    team = register_team(team_setup["id"], ["credit card"])  # matches 1 FAQ
    control_plane.request_preparation(team["id"])
    with pytest.raises(ValueError):
        control_plane.approve_dataset(team["id"], "risk-lead")


def test_assistant_lifecycle(team_setup):
    team = register_team(team_setup["id"], ["fixed deposit"])
    control_plane.request_preparation(team["id"])
    with pytest.raises(ValueError):  # dataset not approved yet
        control_plane.create_assistant("fd_assistant", "FD Assistant", "", team["id"], "engineer")
    control_plane.approve_dataset(team["id"], "risk-lead")
    created = control_plane.create_assistant("fd_assistant", "FD Assistant", "", team["id"], "engineer")
    assert created["status"] == "pending" and created["faqs"] == 12
    approved = control_plane.review_assistant("fd_assistant", "approved", "admin")
    assert approved["status"] == "approved" and approved["approved_by"] == "admin"
    with pytest.raises(ValueError):  # decisions are final
        control_plane.review_assistant("fd_assistant", "rejected", "admin")


def request(latency_ms, ok=True, escalated=False, flags=()):
    return {"trace_id": "t", "time": control_plane.now(), "user": "u", "assistant": "customer_faq", "ok": ok,
            "model_version": "llama_v2", "escalation_required": escalated, "policy_flags": list(flags),
            "latency_ms": latency_ms, "confidence": "high"}


def test_percentile_nearest_rank():
    assert control_plane.percentile([10, 20, 30, 40], 0.5) == 20
    assert control_plane.percentile(list(range(1, 101)), 0.95) == 95
    assert control_plane.percentile([], 0.95) is None


def test_slos_met():
    slos = control_plane.check_slos([request(5000) for _ in range(100)], [{"rating": "good"}] * 5)
    assert [s["status"] for s in slos] == ["met", "met", "met"]


def test_slos_breached():
    requests = [request(60000) for _ in range(95)] + [request(1000, ok=False) for _ in range(5)]
    feedback = [{"rating": "bad"}] * 3 + [{"rating": "good"}]
    assert [s["status"] for s in control_plane.check_slos(requests, feedback)] == ["breached", "breached", "breached"]


def test_slos_without_data():
    assert [s["status"] for s in control_plane.check_slos([], [])] == ["no_data", "no_data", "no_data"]


def test_failed_requests_count_in_summary_and_metrics():
    control_plane.append_line(control_plane.REQUESTS_FILE, request(4000, flags=["prompt_injection"]))
    control_plane.log_failed_request("x1", "u", "customer_faq", "llama_v2", 120, RuntimeError("boom"))
    summary = control_plane.monitoring_summary()
    assert summary["requests"] == 2 and summary["failed"] == 1
    metrics = control_plane.prometheus_metrics("llama_v2")
    assert 'hdfc_requests_total{assistant="customer_faq",outcome="failed"} 1' in metrics
    assert 'hdfc_policy_flags_total{flag="prompt_injection"} 1' in metrics
    assert 'hdfc_live_model{version="llama_v2"} 1' in metrics


def test_feedback_links_to_model_version():
    control_plane.append_line(control_plane.REQUESTS_FILE, {"trace_id": "abc123", "model_version": "llama_v2"})
    feedback = control_plane.add_feedback("abc123", "bad", "Answered from the wrong FAQ")
    assert feedback["model_version"] == "llama_v2"


def test_feedback_for_unknown_trace_is_refused():
    with pytest.raises(KeyError):
        control_plane.add_feedback("does-not-exist", "good")
