# Frequently asked questions

## Does RoboClue know the correct target transformation?

You declare intended behavior and retain the reference before a change. RoboClue
can detect supported contradictions against it. A mistaken reference can preserve
a mistaken transformation; the tool does not infer universal physical truth.

## What can I do without a retained reference?

Run `roboclue advanced start --goal training --framework groot` (or use the MCP
`choose_workflow` tool) for a readiness plan. Supported local dataset/configuration
checks can still identify contradictions. Review the intended recipe and retain
native inputs before the next change. A reference made from the suspect candidate
cannot recover missing historical intent. [Choose a workflow](CHOOSE_A_WORKFLOW.md).

## Does the quick start run a real policy?

No. `roboclue demo` creates three synthetic source rows, catches incorrect
relative-action targets, derives a separate corrected candidate from the source
and verifies it against the same retained criterion. It runs no framework,
model forward or optimizer step. The attributed saved-native replay is a
separate example under `roboclue advanced demo-native`.
[Demo boundary](DEMO.md) · [Native replay](NATIVE_REPLAY.md) · [Original native training proofs](EVIDENCE.md).

## Is OpenPI supported as fully as GR00T?

The profiles differ. OpenPI has original transforms/loader and a model-input hook
with a controlled Torch consumer. Pi0 forward/optimizer and JAX training have not
been qualified. [Compatibility table](SCOPE.md#training-integrations).

## Why does a healthy case say UNKNOWN?

A selected numeric check can pass while another requested component lacks
evidence. Read the individual findings and coverage. UNKNOWN does not mean the
dataset is broken; PASS does not clear unassessed behavior.

## Does a generic health card check GR00T compatibility?

No. Generic cards inspect supported dataset artifacts without the GR00T dataset
contract. `--consumer` is a freeform label, not a compatibility selection. Use
`--groot-contract` (MCP `groot_contract=true`) or explicit GR00T recipe/version
configuration to enable that contract. Even a passing card does not certify
native training compatibility. Use `roboclue check DATASET --question groot`
with the intended configuration/version for that question. The detailed
`roboclue advanced preflight` interface retains its automatic GR00T checks.

## Does the guard check the whole run?

The qualified adapters check a declared bounded startup cohort. Later batches
are explicitly unassessed. Keep cohort selection deterministic and preserve the
required source IDs through preprocessing.

## Will it fix my data automatically?

The native guards diagnose/refuse and verify a correction to ordinary framework
configuration. Some existing recovery workflows propose bounded outputs or a
plan for a separate copy. There is no general automatic repair engine.

## Are older scripts and reports still supported?

The distribution, public import and command are `roboclue`. Older
`policy-doctor` launchers and `policy_doctor` implementation imports remain
compatible. Retained JSON can contain the historical `policy-deployment-doctor`
engine label: rewriting it would change report/card identities. It is provenance,
not another product to install. Do not co-install the old distribution and the
renamed one in the same environment because they share implementation files.
Use the [checkout install](INSTALL.md); no PyPI publication is implied.

## Can I use MCP to control a robot or launch training?

The read-only MCP tools inspect saved artifacts. Training and runtime attachment
use explicit Python integration in your own application. [MCP boundaries](MCP.md).
