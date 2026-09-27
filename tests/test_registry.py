import shutil

import pytest

from registry import load_registry, promote, rollback, save_registry


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
    rollback(registry_file)  # llama_v1 live
    registry = promote("llama_v2", registry_file)
    assert registry["live"] == "llama_v2"
    assert registry["previous"] == "llama_v1"


def test_promote_rejects_unknown_version(registry_file):
    with pytest.raises(ValueError):
        promote("llama_v99", registry_file)


def test_promote_rejects_unapproved_version(registry_file):
    registry = load_registry(registry_file)
    registry["versions"]["llama_v1"]["status"] = "rejected"
    save_registry(registry, registry_file)
    with pytest.raises(ValueError):
        promote("llama_v1", registry_file)
    assert load_registry(registry_file)["live"] == registry["live"]  # nothing changed
