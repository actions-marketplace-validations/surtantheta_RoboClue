"""Explicit developer fetch of one hash-pinned upstream C++ file, not an MCP tool."""
import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from policy_doctor.trajectory import REFERENCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New source file')
    args = parser.parse_args()
    url = f"https://raw.githubusercontent.com/ros-controls/ros2_controllers/{REFERENCE['revision']}/{REFERENCE['file']}"
    with urllib.request.urlopen(url, timeout=30) as response:
        raw = response.read(128001)
    if len(raw) > 128000 or hashlib.sha256(raw).hexdigest() != REFERENCE['sha256']:
        raise SystemExit('Pinned upstream source digest/size mismatch')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open('xb') as stream:
        stream.write(raw)
    print(json.dumps({'path': str(args.destination), 'url': url, 'sha256': REFERENCE['sha256']}))


if __name__ == '__main__':
    main()
