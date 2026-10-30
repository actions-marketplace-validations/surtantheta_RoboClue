# First five minutes

From a downloaded or cloned checkout, with Python 3.11 or newer:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[dataset]'
roboclue check hf://OWNER/DATASET --fetch
```

Replace `OWNER/DATASET` with the public dataset you plan to use, or pass your
local dataset directory. This reads bounded public metadata/Parquet with explicit
download permission, not videos or model weights. The response names a supported
contradiction or the evidence still missing. Network checks can take minutes;
the default byte cap is 64 MiB and incomplete coverage is not approval.
[Hub revisions and budgets](CHECK.md#public-datasets-are-explicit-downloads).

For an offline installation/control example, run `roboclue demo` instead.
The offline demo creates three synthetic source rows. It catches incorrect
absolute targets in a field declared to hold relative targets, computes the
correction from the unchanged source into a separate copy, and verifies it
against the original criterion. The source, incorrect capture, criterion and
pin remain unchanged. No native trainer, GPU, robot or network is involved.
[Demo evidence](DEMO.md).

Read the three-line answer, then copy its `Next:` command to verify the corrected
capture yourself. The covered target/mask check passes. Other inputs, training
quality and robot behavior remain unassessed. Use a new `--output` directory to
repeat the demo; existing output is never overwritten.

## Check your data

```sh
python -m pip install '.[dataset]'
roboclue check /path/to/dataset
roboclue check hf://OWNER/DATASET --fetch
```

The answer explains what passed, the main contradiction or coverage gap, and a
next command. `PARTIAL` means incomplete assessment, not proof of broken data.
Use `--explain` to see full evidence or `--json` for the unchanged machine report.
[Supported artifacts and exit codes](CHECK.md).

## Prepare for the next training change

```sh
roboclue guard init --framework groot --output ./my-guard
```

Choose `lerobot` or `openpi` when appropriate. The default creates instructions
and templates, not a reference. Supply the reviewed native setup in your existing
training environment; an explicit capture invocation then saves the reference
and prints its pin for independent retention. Capture before changing the
producer. A suspect current run cannot establish its own historical intent.
[Complete guided path](GUARD_INIT.md).

## Connect an agent

```sh
python -m pip install '.[mcp]'
roboclue mcp setup --root /absolute/path/to/artifacts
```

Setup prints configuration; it edits no client files. The default catalog has
eight read-only tools, including simulator, units/configuration and recording
recovery checks. Ask the agent to check your local dataset and explain the
first problem and what remains unassessed. [MCP](MCP.md).

For specialist workflows, run `roboclue advanced --help`. The attributed public
native captures remain available with
`roboclue advanced demo-native --output ./saved-native-demo`; that replays saved
arrays rather than executing a new native trainer. [Native replay](NATIVE_REPLAY.md).
