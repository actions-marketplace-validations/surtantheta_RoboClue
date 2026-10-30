# Linked task catalog comparison

`roboclue check hf://lerobot/smolvla_robocasa --kind checkpoint --fetch --revision f7af588d11e30b0069f200792fb285904b74c80b --instruction 'Close the fridge doors.'`
compares a declared instruction with the checkpoint's linked public task catalog.
With an `hf://` source, explicit `--instruction`/`--task`,
`--instruction-file`, `--language-catalog` or `--launch-command` selects the
checkpoint role automatically. `--kind checkpoint` remains an explicit option.
An explicit `--kind dataset` is never overridden. Mixed checkpoint and dataset
questions still require separate calls, and retrieval still requires `--fetch`.
For explicit local inputs, use
`roboclue check CHECKPOINT --instruction 'Close the fridge doors.' --language-catalog LOCAL_DATASET`.
The saved
`train_config.json` identifies the dataset repository and revision. An explicit
local catalog supplies bytes for comparison but does not authenticate that
repository or its historical training revision. Public intake requires the
existing explicit Hub permissions; saved absolute `dataset.root` hints grant
no filesystem authority and are never followed.

The report distinguishes exact text membership, lexical-fold correspondence,
and no match. The text-only `lexical-fold-v1` profile lowercases, collapses
whitespace, replaces underscores with spaces and removes trailing ASCII
punctuation. It is not the normalization of every native processor. A
lexical-only match therefore retains UNKNOWN native token-path equivalence;
it never silently certifies a pipeline. Exact membership also does not prove
historical training coverage, native processing or physical capability.

OUT-OF-CATALOG emits a WARNING with a bounded nearest lexical string, literal
words absent from the catalog, and any literal sequence/multiple-action-word
marker not present in the catalog. The suggestion is not a semantic rewrite.
Preserve the intended task; use a catalog string only after confirming that
it describes that same task. No instruction is changed or executed.

Null or mutable saved revisions are resolved once to a current immutable Hub
commit when public intake is allowed. The report displays that revision and
the unresolved historical provenance even for exact matches. A saved episode
subset does not authenticate which whole-catalog labels that subset used.
One or two catalog labels are a catalog-size observation, not evidence that
the policy ignores changes to instructions.

`--task` aliases `--instruction`. `--launch-command` reads literal `--task`,
`--dataset.single_task` or `--env.task_description` values without executing
the command. `--env.task` alone remains UNKNOWN: it can identify an environment
rather than the instruction actually sent. Compound shell commands,
expansions and conflicting instruction sources remain UNKNOWN.
`--instruction-file` accepts a bounded JSON string list (or an object with an
`instructions` list) or one literal Python `TASK_INSTRUCTIONS`, `INSTRUCTIONS`
or `task_instructions` assignment. Computed Python inputs are not executed.

The default MCP `preflight_check` accepts `instruction`, `launch_command`,
`instruction_file` and `language_catalog`. With a checkpoint source, it augments
the existing checkpoint audit. With a dataset source and explicit checkpoint,
it augments the existing dataset preflight. Independent errors remain present.
An empty selector set preserves existing behavior. Model intake needs
`--allow-hub`; linked remote catalog intake additionally needs
`--allow-hub-datasets`. All explicit local inputs must be inside read roots.

Bounds: 2,048 catalog rows, 8 MiB compressed and expanded catalog bytes,
64 instructions, 4,096 UTF-8 bytes per instruction, 2,048 bytes per catalog
label, 64 Parquet row groups and eight columns. Row-group allocation counts
are checked before Arrow reads. Nearest ranking uses token overlap to select
at most eight candidates and a character ratio on at most 256 characters.
This is lexical similarity, not a semantic or task-success score. Missing,
malformed, unsupported, unreachable or over-budget inputs remain UNKNOWN.

Public LFS task catalogs require the pinned Hub API's exact repository,
revision, task path, content hash and size before retrieval. A catalog-only
anonymous transport permits one HTTPS redirect to the evidenced
`us.aws.cdn.hf.co` delivery route. It verifies the downloaded bytes against
the API's LFS SHA256 and size; a CDN ETag is not substituted for that identity.
Private DNS addresses, other hosts, additional redirects and credentialed
fallbacks are refused. TCP connects to a vetted public address with the
original hostname retained for TLS verification. A worker bounds the complete
DNS/HTTP operation to 35 seconds. Unsupported delivery routes remain UNKNOWN;
this does not expand network permissions for other artifact types.

The content-identity association follows the publisher's pinned
[`huggingface_hub/file_download.py:1675–1688`](https://github.com/huggingface/huggingface_hub/blob/4a993357d524028204216b4307ed4d97b8111818/src/huggingface_hub/file_download.py#L1675)
metadata fields and its
[`LFS identity for Xet files`](https://github.com/huggingface/huggingface_hub/blob/4a993357d524028204216b4307ed4d97b8111818/src/huggingface_hub/file_download.py#L1904).
No model weights, videos or native policy code are downloaded or executed.

Evidence: [LeRobot #4548](https://github.com/huggingface/lerobot/issues/4548)
and [#4045](https://github.com/huggingface/lerobot/issues/4045) report different
training and evaluation strings. Pinned LeRobot
`72e1d9e5e76ee719bacfc78bc34d817bee5fb521`
[`planner.py:283–302`](https://github.com/huggingface/lerobot/blob/72e1d9e5e76ee719bacfc78bc34d817bee5fb521/src/lerobot/rollout/planner.py#L283)
already loads a linked vocabulary for its planner, calling it an inference
rather than a contract. At the same pin,
[`lerobot_eval.py:298–304`](https://github.com/huggingface/lerobot/blob/72e1d9e5e76ee719bacfc78bc34d817bee5fb521/src/lerobot/scripts/lerobot_eval.py#L298)
selects the environment's description before its task identifier. This check
makes no new policy-performance claim.
