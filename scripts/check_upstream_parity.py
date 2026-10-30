"""Compare CPU arithmetic to functions extracted from the pinned upstream source.

Development-only: explicitly executes two functions from the trusted pinned checkout.
The installed diagnostic engine never imports or executes checkpoint/source code.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from policy_doctor.trace import transform

PIN = "146d8474932c8b566717eb2738ba459337ab71c4"


def extract(path, name, namespace):
    tree = ast.parse(path.read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def main():
    args = argparse.ArgumentParser()
    args.add_argument("upstream", type=Path)
    args = args.parse_args()
    revision = subprocess.check_output(["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise SystemExit("Refusing to execute unpinned upstream source")
    root = args.upstream / "src/lerobot/processor"
    modes = {name: name for name in ("IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES", "QUANTILE10")}
    namespace = {"torch": torch, "NormalizationMode": SimpleNamespace(**modes)}
    upstream_normalize = extract(root / "normalize_processor.py", "_apply_transform", namespace)
    upstream_absolute = extract(root / "relative_action_processor.py", "to_absolute_actions", {"torch": torch})
    random = np.random.default_rng(42)
    cases, max_error = 0, 0.
    for mode in modes:
        for degenerate in (False, True):
            stats = {"mean": [2., -3.], "std": [0. if degenerate else 2., 3.],
                     "min": [1., 2.], "max": [1. if degenerate else 4., 8.],
                     "q01": [1., 2.], "q99": [1. if degenerate else 4., 8.],
                     "q10": [1., 2.], "q90": [1. if degenerate else 4., 8.]}
            tensors = {k: torch.tensor(v, dtype=torch.float64, device="cpu") for k, v in stats.items()}
            owner = SimpleNamespace(norm_map={"ACTION": mode}, _tensor_stats={"action": tensors}, eps=1e-8)
            for inverse in (False, True):
                values = random.normal(size=(7, 2))
                expected = upstream_normalize(owner, torch.tensor(values, dtype=torch.float64), "action", "ACTION", inverse=inverse).numpy()
                actual = transform(values, stats, mode, 1e-8, inverse)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                max_error = max(max_error, float(np.max(np.abs(expected - actual))))
                cases += 1
    for mask in ([True, True], [True, False], [False, True]):
        values, anchor = random.normal(size=(1, 7, 2)), random.normal(size=(1, 2))
        expected = upstream_absolute(torch.tensor(values), torch.tensor(anchor), mask).numpy()
        actual = values + anchor[:, None, :] * np.array(mask)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        cases += 1
    result = {"upstream_commit": revision, "cpu_arithmetic_cases": cases, "status": "pass",
              "normalization_max_absolute_difference": max_error, "torch_version": torch.__version__,
              "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (root / "normalize_processor.py", root / "relative_action_processor.py")},
              "scope": "Extracted upstream CPU arithmetic only; no policy factory, dtype/GPU equivalence, model inference or hardware validation."}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
