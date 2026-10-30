<h1 align="center">RoboClue</h1>
<p align="center"><strong>Catch silent robot-learning pipeline errors. Verify the correction.</strong></p>
<p align="center">
  <a href="#try-it">Try it</a>&nbsp;&nbsp;
  <a href="#what-you-can-do">Features</a>&nbsp;&nbsp;
  <a href="docs/MCP.md">Agent setup</a>&nbsp;&nbsp;
  <a href="docs/EVIDENCE.md">Evidence</a>
</p>

Your dataset loads. The batch has the right shape. Training runs.
**The targets can still be wrong.**

RoboClue checks robot-learning datasets, saved training inputs and software
recordings. It explains the first supported problem, preserves the evidence and
lets you verify a correction against the same declared criterion.

<p align="center">
  <img src="assets/launch/roboclue-square.gif?v=e8180734" alt="25-second RoboClue demonstration, animated preview" width="100%">
</p>

## What you can do

| What can go wrong | How RoboClue helps |
| --- | --- |
| [Training runs on unintended targets](docs/GUARD_INIT.md) | Compare covered startup inputs with a reference reviewed before the change; block mismatches. |
| [Fine-tuning starts from scratch](docs/PIPELINE_CONTRACTS.md) | Check that supported OpenPI code selects the pretrained-load path you requested. |
| [Bad data wastes a training run](docs/HEALTH_MONITOR.md) | Check metadata, numeric rows and episode membership; track results across your dataset inventory. |
| [An interrupted recording risks your demonstrations](docs/FEATURES.md) | Identify verified episode prefixes for recovery, and compare supported conversions with their source. |
| [Published evaluations don't reproduce](docs/REGISTER_CATALOGUE_EVIDENCE.md) | Compare declared versions and evaluation settings with twelve curated, source-backed checks. |
| [A missing camera shifts the policy's inputs](docs/PIPELINE_CONTRACTS.md#lerobot-stop-a-missing-camera-before-it-changes-another-cameras-slot) | Guard camera-slot mapping at supported native boundaries before image reads. |
| [FastWAM receives incorrectly processed images](docs/rules/RC009.md) | Compare numeric image preprocessing with the supported FastWAM model adapter. |
| [A failed run leaves you guessing](docs/RUNTIME_RECORDER.md) | Record software API requests and returns, then trace findings to individual events. |
| [An upgrade changes a working setup](docs/UPGRADE_WATCH.md) | Track saved artifacts and compare supported reference and candidate evidence when upstream code changes. |
| [An instruction isn't in the checkpoint's task catalog](docs/rules/RC013.md) | Compare the instruction with the selected checkpoint's linked task catalog. |

Each check has supported inputs and limits. A result does not certify a whole
pipeline or predict robot success. [Coverage](docs/SCOPE.md).

The first public version is **0.1.0**. Its qualified wheel and reports are in
[the release folder](release/0.1.0/README.md). Source installation is available
below; this GitHub release does not imply a PyPI upload.

## Try it

From a downloaded or cloned checkout, with Python 3.11 or newer:

```sh
git clone https://github.com/surtantheta/RoboClue.git
cd RoboClue
python -m venv .venv
. .venv/bin/activate
python -m pip install '.[dataset]'
roboclue demo
```

The offline CPU demo creates a tiny **synthetic** dataset, catches incorrect
relative-action targets, computes a correction from the source into a separate
copy, and verifies it against the original criterion. It needs no robot, GPU,
framework, account or download. Existing output directories are refused; use
`--output ./another-demo` to run it again. [What the demo proves](docs/DEMO.md).

Then check your own data:

```sh
roboclue check /path/to/dataset
roboclue check hf://OWNER/DATASET --fetch
```

Replace the local path or `OWNER/DATASET` with your dataset. This checks bounded
metadata, numeric rows and available saved statistics. The answer names the first
supported problem or missing evidence, then the next command. Hub reads require
explicit `--fetch`, pin the resolved revision, and download at most 64 MiB of
metadata/Parquet by default. No videos or credentials are downloaded. Network
scans can take minutes; partial coverage is reported.
[Hub examples and limits](docs/CHECK.md#public-datasets-are-explicit-downloads).

For a reproducible public example, this pinned revision declares zero frames
although its Parquet rows contain data:

```sh
roboclue check hf://SO-101-robot-learning/so101_pick_place --fetch \
  --revision c56fed5bc39d966c25120293b2f6dc3eb5ead5c7 --max-bytes 8388608
```

The supported episode-total contradiction exits 1. This observation applies to
that revision; it does not label every revision or every check as broken.

Default results use three lines: **verdict, problem, next command**. Raw evidence
and coverage details are available with `--json` or `--explain`.

### Native training evidence

In our controlled public-data GR00T proof, the original native trainer accepts
an incorrect relative-action transformation. The table shows model forwards
and optimizer steps:

| Run | Forwards / steps |
| --- | ---: |
| Native trainer, wrong target transformation | Runs / 2 |
| Same fault with RoboClue | **0 / 0** |
| Corrected transformation, original criterion | Runs / 2 |

RoboClue identifies **1,400 differing target values** and a separate mask-dtype
contradiction. Correction and replay pass the original pre-fault criterion.
This native training proof uses public simulation-derived data, a pretrained
Qwen backbone and a newly initialized GR00T head. It does not measure policy
quality or robot success. The quick demo above is a separate synthetic example.
[Exact evidence and scope](docs/EVIDENCE.md)

[Replay the saved native captures](docs/NATIVE_REPLAY.md).

### Check other artifacts

```sh
roboclue check /path/to/checkpoint
roboclue check /path/to/flight.jsonl
```

Ask the question you need answered; these selectors run the existing specialist
checks through the same command:

| Question | Command |
| --- | --- |
| Do these units match my robot? | `roboclue check DATASET --robot-command 'YOUR_DEPLOY_COMMAND' --lerobot-version YOUR_ACTUAL_VERSION` |
| Did a simulator conversion preserve the source conventions? | `roboclue check source.hdf5 --converted merged.hdf5` |
| Is an interrupted recording safe to resume? | `roboclue check DATASET --question recovery` |
| Can I apply the supported recovery plan to a separate copy? | `roboclue check DATASET --write-copy NEWDIR` |
| Does this dataset match my GR00T configuration? | `roboclue check DATASET --groot-version n1.5 --groot-config-name YOUR_CONFIG` |
| Is the OpenPI delta-transform choice consistent? | `roboclue check DATASET --openpi-config pi0_libero --openpi-source-root ./openpi` |

HDF5 inputs reach simulator checks automatically. A local `meta/modality.json`
reaches the GR00T contract, and a damaged v3 recording reaches recovery checks.
An OpenPI transform match is separate from the raw-key check. The report also
names the dataset format and the consumer's pinned reader; installed-loader support
and training remain unverified. A dataset-only units result names the inferred modes
and says that no robot was compared.
Camera declarations compare key names; omitted settings use the qualified native
defaults, including an empty camera mapping for SO robots. Size, rate and policy
mapping remain separate. Supply the actual `--lerobot-version`: tested units/default
profiles cover published 0.6.1 and recorded 0.6.2 sources for SO-100/101, Koch, OMX,
bimanual SO and the LeKiwi arm. A version label does not verify installed files.
Missing or unsupported versions leave the comparison incomplete.
[Exact consumer coverage](docs/CONSUMER_VERSION.md).

The command recognizes supported metadata and capture formats. A configuration
can be compared with `--reference /path/to/reviewed-config.json`; a saved training
capture additionally needs the independently retained criterion pin. Ambiguous
inputs need `--kind`; arbitrary model weights or Python training scripts are
never executed to guess what they contain.

**INCOMPLETE means more evidence is needed.** The answer names checks that
passed and evidence still missing. Missing video or statistics do not establish
that the dataset is broken. The machine status remains `unknown` (exit 3), so
incomplete checks cannot silently approve training. [Results and exit codes](docs/CHECK.md).

### Guard the next training change

```sh
roboclue guard init --framework groot --output ./my-guard
```

This prepares capture and attachment templates for your existing training
project. It creates no reference and runs no training. After you supply the
reviewed native setup, one explicit capture command saves the startup reference
and prints its pin for independent retention. GR00T, LeRobot and OpenPI use their
existing qualified adapters. A reference created from a suspect batch cannot
prove that batch correct. [Guided capture and attachment](docs/GUARD_INIT.md).

For an interrupted recording, use the explicit resume gate before handing
control to LeRobot:

```sh
roboclue guard resume ./recording --dry-run -- \
  lerobot-record --dataset.root=./recording --resume=true
```

The dry run checks the existing recording without starting the recorder. Remove
`--dry-run` to run the declared command only after the required checks clear.
The gate never repairs the original dataset. [Resume gate](docs/RESUME_GUARD.md).

### Use your MCP agent

```sh
python -m pip install '.[mcp]'
roboclue mcp setup --root /absolute/path/to/artifacts
```

Setup prints configuration for review. The default MCP catalog has **eight
read-only tools** covering workflow choice, consumer preflight, dataset health,
simulator conventions, interrupted recordings, saved training targets,
runtime recordings and upgrade manifests. Other specialist tools are available with
`roboclue mcp serve --advanced`. Live training guards attach inside your Python
process. [Agent setup](docs/MCP.md).

### Five commands to start

| Command | Purpose |
| --- | --- |
| `roboclue check ARTIFACT` | Recognize an artifact and run supported checks |
| `roboclue demo` | Follow a small offline fault → correction → verification loop |
| `roboclue guard init` | Prepare or capture intended startup inputs before a change |
| `roboclue mcp` | Configure or run your agent's read-only tools |
| `roboclue advanced` | Access specialist diagnostics, monitoring and retained workflows |

The distribution, public import and command are `roboclue`. Legacy imports and
launchers remain compatibility interfaces. [Installation](docs/INSTALL.md)

[Five-minute guide](docs/QUICKSTART.md).

Browse the [numbered rule catalog](docs/rules/README.md) for upstream evidence,
tested versions and a verification path for each failure. Check the framework
metadata visible to RoboClue with `roboclue advanced versions`; the recorder or
trainer may run in another environment. An untested installed version is not a
compatibility approval. [Version context](docs/FRAMEWORK_VERSIONS.md)

[Run checks in CI or pre-commit](docs/AUTOMATION.md)

[Contribute a rule](CONTRIBUTING.md).

## Works with your stack

| Integration | Demonstrated boundary |
| --- | --- |
| GR00T N1.5 / N1.7 | Version-specific static metadata/config contracts; N1.7 additionally has the original native trainer, collation and actual loss-input proof |
| LeRobot ACT | Original reader, preprocessing and update function; selected state/camera witnesses; two workers |
| OpenPI PyTorch | Original transforms/loader and a model-input hook with a controlled Torch consumer; Pi0 training unassessed |
| LeRobot pi0 / pi05 / SmolVLA | Source-bound camera-slot checks and native preprocessing hooks; whole-policy competence unassessed |
| LeRobot FastWAM | Numeric image-map comparison for qualified saved or fresh-training source profiles; resume unassessed |
| Robot-style applications | Synchronous software request/return recording; physical behavior unassessed |
| Isaac Lab / robomimic artifacts | Supported saved-data conventions and bound comparisons |

Adapters qualify explicit source revisions and execution settings. A separate
two-rank CPU/Gloo LeRobot test verifies cross-rank refusal in a custom loop.
This does not establish arbitrary DDP, JAX or framework-version support.
[Compatibility and coverage](docs/SCOPE.md).

Use the [operations guide](docs/PRODUCTION_OPERATIONS.md) for strict CI gates,
installed MCP setup, retained evidence and rollback.

## Why RoboClue?

RoboClue brings dataset monitoring, consumer-boundary checks, runtime triage and
upgrade evidence into one diagnostic workflow. Its strongest demonstrated
capability is the native **accepted fault → refusal → verified correction** loop.
Existing linters, curation tools, native validators and visualization platforms
remain useful alongside it. [Comparison and opportunity](docs/COMPARISON.md).

Results retain their coverage. A passing startup cohort does not certify later
batches. Missing evidence remains UNKNOWN, and a mistaken reference can preserve
a mistaken transformation. [How to interpret a result](docs/SCOPE.md#read-the-result).

## Documentation and contributing

Start with the [documentation index](docs/README.md), [FAQ](docs/FAQ.md) and
[evidence](docs/EVIDENCE.md). Report a wrong finding, confusing result or unsupported
workflow through the repository's issue templates; small healthy controls help.

[Contributing](CONTRIBUTING.md) · [Development](DEVELOPMENT.md)

[Roadmap](docs/ROADMAP.md) · [Security](SECURITY.md)

[License](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md)
