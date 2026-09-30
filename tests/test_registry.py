import shutil

import pytest

from registry import load_registry, promote, register_model, rollback, save_registry, set_status


@pytest.fixture
def registry_file(tmp_path):
    # Work on a copy so the tests never change the real registry.json
    path = tmp_path / "registry.json"
    shutil.copy("registry.json", path)
    return path


def test_rollback_switches_to_previous_version(registry_file):
    before = load_registry(registry_file)
    after = rollback(registry_file)
    assert after["live"] == before["previous"]
    assert after["previous"] == before["live"]
    assert after["history"][-1]["action"] == "rollback"


def test_second_rollback_restores_original(registry_file):
    original = load_registry(registry_file)["live"]
    rollback(registry_file)
    assert rollback(registry_file)["live"] == original


def test_promote_makes_version_live(registry_file):
    before = load_registry(registry_file)
    rollback(registry_file)  # the previous version is live again
    registry = promote(before["live"], registry_file)
    assert registry["live"] == before["live"]
    assert registry["previous"] == before["previous"]


def test_promote_rejects_unknown_version(registry_file):
    with pytest.raises(ValueError):
        promote("llama_v99", registry_file)


def test_registered_model_starts_pending_and_cannot_go_live(registry_file):
    entry = register_model("test_candidate", "test_candidate", "a" * 64, {"rougeL_finetuned_with_rag": 0.5}, path=registry_file)
    assert entry["status"] == "pending"
    with pytest.raises(ValueError):
        promote("test_candidate", registry_file)


def test_approved_model_can_be_promoted(registry_file):
    register_model("test_candidate", "test_candidate", "a" * 64, {}, path=registry_file)
    set_status("test_candidate", "approved", "model-risk-lead", registry_file)
    assert promote("test_candidate", registry_file)["live"] == "test_candidate"


def test_live_model_cannot_be_rejected(registry_file):
    live = load_registry(registry_file)["live"]
    with pytest.raises(ValueError):
        set_status(live, "rejected", "model-risk-lead", registry_file)


def test_promote_rejects_unapproved_version(registry_file):
    registry = load_registry(registry_file)
    registry["versions"]["llama_v1"]["status"] = "rejected"
    save_registry(registry, registry_file)
    with pytest.raises(ValueError):
        promote("llama_v1", registry_file)
    assert load_registry(registry_file)["live"] == registry["live"]  # nothing changed
