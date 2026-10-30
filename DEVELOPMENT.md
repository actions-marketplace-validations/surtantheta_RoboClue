# Development

Use Python 3.11 or 3.12 in an isolated environment:

```sh
python -m pip install -e '.[dev]' build
python -m ruff check src tests examples scripts
python scripts/test_project.py --python "$(command -v python)" -- -q
python scripts/validate_demo.py
python -m build
```

`constraints/minimum.txt` supplies the supported minimum dependency profile.
The CI matrix covers current Python 3.11/3.12, minimum Python 3.11 and a
restrictive umask. It builds fresh artifacts, installs outside the source tree
and exercises CLI plus actual MCP stdio. Native robotics frameworks/models are
not dependencies of the default suite; their execution profiles are separately
documented in [evidence](docs/EVIDENCE.md).

## Structure

| Directory | Purpose |
| --- | --- |
| `src/policy_doctor/` | Diagnostic engines, native adapters, CLI and MCP |
| `tests/` | Regression tests and explicitly licensed inert source fixtures |
| `examples/` | Generated examples and application integration recipes |
| `scripts/` | Validation, installed smoke and upstream reference helpers |
| `docs/` | User guides, compatibility and evidence |
| `constraints/` | Dependency profiles |

Preserve native behavior, explicit input authority and bounded reads. A new
semantic check needs an intended criterion, healthy/counterexample controls and
an UNKNOWN path for missing evidence. Keep generated controls separate from
observed incidents. Never regenerate a reference after a fault merely to make
its correction pass.
