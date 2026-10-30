"""Download bounded public metadata and one Parquet shard at pinned Hub revisions.

This script is an explicit development operation. The MCP/diagnostic engine has no downloader.
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def get(url, cap=8 * 1024 * 1024):
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(cap + 1)
    if len(data) > cap:
        raise ValueError(f"Download exceeds {cap} byte cap")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    entries = []
    for repo, kind, local in (("lerobot/smolvla_base", "models", "smolvla_base"),
                              ("lerobot/pusht", "datasets", "pusht")):
        metadata = json.loads(get(f"https://huggingface.co/api/{kind}/{repo}"))
        revision = metadata["sha"]
        prefix = "datasets/" if kind == "datasets" else ""
        files = ["meta/info.json", "meta/stats.json", "data/chunk-000/file-000.parquet"] if kind == "datasets" else ["config.json", "policy_preprocessor.json", "policy_postprocessor.json"]
        base = f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/"
        if kind == "models":
            for filename in files[1:].copy():
                config = json.loads(get(base + filename))
                files += [step["state_file"] for step in config["steps"] if step.get("state_file")]
        for filename in files:
            data = get(base + filename)
            path = args.destination / local / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            entries.append({"path": str(Path(local) / filename), "url": base + filename,
                            "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data), "hub_revision": revision})
            print(f"Downloaded {local}/{filename}: {len(data)} bytes")
    (args.destination / "provenance.json").write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    main()
