"""Explicit bounded metadata fetch across public LeRobot policy families; no weights."""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORIES = (
    "lerobot/act_aloha_sim_transfer_cube_human",
    "lerobot/diffusion_pusht",
    "lerobot/smolvla_libero",
)


def fetch(url, limit=8 * 1024 * 1024):
    with urllib.request.urlopen(url, timeout=30) as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError("Artifact exceeds the metadata/statistics download limit")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    records = []
    for repo in REPOSITORIES:
        info = json.loads(fetch(f"https://huggingface.co/api/models/{repo}"))
        revision = info["sha"]
        available = {entry["rfilename"] for entry in info.get("siblings", [])}
        names = ["config.json"]
        names.extend(name for name in ("policy_preprocessor.json", "policy_postprocessor.json") if name in available)
        root = args.destination / repo.split("/")[-1]
        root.mkdir(parents=True, exist_ok=True)
        base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        entries = []
        for name in names:
            content = fetch(base + name)
            (root / name).write_bytes(content)
            data = json.loads(content)
            entries.append({"path": name, "sha256": hashlib.sha256(content).hexdigest(), "url": base + name, "bytes": len(content)})
            if name.endswith("processor.json"):
                for step in data.get("steps", []):
                    state = step.get("state_file")
                    if state and state not in names:
                        if Path(state).is_absolute() or ".." in Path(state).parts or not state.endswith(".safetensors"):
                            raise ValueError("Invalid processor state reference")
                        binary = fetch(base + state)
                        (root / state).parent.mkdir(parents=True, exist_ok=True)
                        (root / state).write_bytes(binary)
                        entries.append({"path": state, "sha256": hashlib.sha256(binary).hexdigest(), "url": base + state, "bytes": len(binary)})
        records.append({"repository": repo, "revision": revision, "license": (info.get("cardData") or {}).get("license"),
                        "processors_available": "policy_postprocessor.json" in available, "files": entries})
        print(repo, revision, len(entries), "artifacts")
    (args.destination / "provenance.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
