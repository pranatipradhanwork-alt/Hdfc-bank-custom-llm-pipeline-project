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


def test_feedback_links_to_model_version():
    control_plane.append_line(control_plane.REQUESTS_FILE, {"trace_id": "abc123", "model_version": "llama_v2"})
    feedback = control_plane.add_feedback("abc123", "bad", "Answered from the wrong FAQ")
    assert feedback["model_version"] == "llama_v2"


def test_feedback_for_unknown_trace_is_refused():
    with pytest.raises(KeyError):
        control_plane.add_feedback("does-not-exist", "good")
