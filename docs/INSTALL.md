# Install RoboClue

Use Python 3.11 or newer. From a downloaded or cloned copy of this repository,
open its directory and create a separate environment:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
roboclue --help
```

This installs the core (NumPy, safetensors and packaging). Add dataset or agent dependencies
when you need those workflows. Windows PowerShell activation is
`.venv\Scripts\Activate.ps1`; the install command is the same. Automated
qualification currently covers Linux Python 3.11/3.12; Windows and macOS are not
qualified profiles.

Installation from the checkout is the supported first-public route. We have not
published a RoboClue-named PyPI package. The distribution, public Python import
and main command are `roboclue`. `python -m roboclue` uses the same entry point.
For example, `from roboclue import Gr00tTrainingGuard` imports the existing adapter.
Start with `roboclue check`, `demo`, `guard`, `mcp` or `advanced`.

Use a fresh environment for the first public installation. Existing scripts and
retained report identities remain compatible; [compatibility details](FAQ.md#are-older-scripts-and-reports-still-supported)
explain the historical implementation labels. New workflows use the five public
verbs above, with specialist options under `roboclue advanced`.

## Choose dependencies

| Install from the checkout | Purpose |
| --- | --- |
| `python -m pip install .` | Lightweight core; NumPy, safetensors and packaging |
| `python -m pip install '.[dataset]'` | Parquet/HDF5/YAML dataset workflows |
| `python -m pip install '.[mcp]'` | MCP server for your agent |
| `python -m pip install '.[dataset,mcp]'` | Dataset workflows plus the synthetic CLI/MCP demo |
| `python -m pip install -e '.[dev]' build` | Development and the default regression suite |

## If installation keeps retrying a package index

Repeated DNS, connection or certificate warnings identify an installation/network
problem before RoboClue runs. Pip can consult a primary index and extra indexes;
changing `--index-url` alone does not remove a configured extra index. Note which
host is failing and whether other downloads progress. Inspect your pip settings
locally if needed; do not paste raw environment/configuration output into an
issue. Index URLs, user information and query strings can contain credentials.

Use your organization's approved mirror, proxy and certificate settings when
required. If direct public PyPI is permitted, this **optional Linux/macOS command**
tests a single install in the new environment above without changing saved pip
configuration:

```sh
env -u PIP_EXTRA_INDEX_URL PIP_CONFIG_FILE=/dev/null \
  PIP_INDEX_URL=https://pypi.org/simple \
  python -m pip install --index-url https://pypi.org/simple .
```

It removes the extra-index environment variable and disables pip configuration
files for this command and its build subprocesses. The explicit index environment
variable also reaches build dependency installation. It leaves other environment
settings in place; it is not a universal network reset. Disabling configuration
files can omit a required corporate mirror or certificate, so use this only when
that direct route is appropriate. No global/user configuration is edited. Pip's
`--isolated` alone ignores environment/user settings, not every global or site
configuration. See pip's [configuration](https://pip.pypa.io/en/stable/topics/configuration/)
and [isolated-mode documentation](https://pip.pypa.io/en/stable/cli/pip/#cmdoption-isolated).

If the direct route also fails, keep the error and resolve network access or use
an approved offline wheelhouse. Avoid reinstalling a working robotics framework
to debug an index. Before sharing logs, remove credentials and token-bearing URLs;
share the failing hostname, error type, Python/pip versions and attempted command.

## Attach to an existing training environment

Build/install RoboClue using that framework's compatible environment and
constraints. The adapters do not install Torch, GR00T, LeRobot or OpenPI for you.
Review the proposed dependency changes before installing into a working trainer.
Use a separate demo environment for first use.

Retained criteria bind the verifier and their NumPy runtime. Our GR00T and OpenPI
proofs use NumPy 1.26.4; LeRobot uses 2.2.6. A new reference records its own
runtime. A mismatched saved reference is refused rather than silently reinterpreted.
[Qualified profiles](SCOPE.md#training-integrations).

Choose a [workflow matching your existing evidence](CHOOSE_A_WORKFLOW.md), or
run the core-only [offline demo](QUICKSTART.md):

```sh
roboclue demo
```

Use a new output directory when a
command creates artifacts; existing files are not overwritten.
