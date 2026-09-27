"""Model registry: which adapter versions exist, which one is live, and a history of every change.

The registry is a plain JSON file (registry.json) kept in git, so every release decision is reviewable.

Usage:
  python registry.py                    # show the live version and all registered versions
  python registry.py --promote llama_v2 # make an approved version live
  python registry.py --rollback         # go back to the previously live version
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path("registry.json")


def load_registry(path=REGISTRY_PATH):
    return json.loads(Path(path).read_text())


def save_registry(registry, path=REGISTRY_PATH):
    Path(path).write_text(json.dumps(registry, indent=2) + "\n")


def live_version(path=REGISTRY_PATH):
    registry = load_registry(path)
    return registry["live"], registry["versions"][registry["live"]]


def promote(version, path=REGISTRY_PATH):
    registry = load_registry(path)
    if version not in registry["versions"]:
        raise ValueError(f"Unknown model version: {version}")
    # Release gate: only versions that passed evaluation and safety review can go live
    if registry["versions"][version]["status"] != "approved":
        raise ValueError(f"{version} is not approved for release")
    if version == registry["live"]:
        return registry
    registry["previous"], registry["live"] = registry["live"], version
    record(registry, "promote", registry["previous"], version)
    save_registry(registry, path)
    return registry


def rollback(path=REGISTRY_PATH):
    registry = load_registry(path)
    if not registry.get("previous"):
        raise ValueError("No previous version to roll back to")
    # Swap live and previous, so a second rollback undoes the first
    registry["live"], registry["previous"] = registry["previous"], registry["live"]
    record(registry, "rollback", registry["previous"], registry["live"])
    save_registry(registry, path)
    return registry


def record(registry, action, from_version, to_version):
    registry["history"].append({
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "action": action,
        "from": from_version,
        "to": to_version,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--promote", metavar="VERSION")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    if args.promote:
        promote(args.promote)
    elif args.rollback:
        rollback()
    registry = load_registry()
    print(f"Live: {registry['live']}   Previous: {registry['previous']}")
    for name, entry in registry["versions"].items():
        scores = entry["evaluation"]
        print(f"  {name:10} {entry['status']:9} data v{entry['training']['dataset_version']}  "
              f"ROUGE-L with RAG {scores['rougeL_finetuned_with_rag']} (reworded {scores['rougeL_finetuned_with_rag_reworded']})")
