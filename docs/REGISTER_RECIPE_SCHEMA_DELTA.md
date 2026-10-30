# Register recipe successor schema delta

## Combined successor: evidenced presence and absence

`config_key`, `processor_key` and `launch_flag` predicates may use
`{"op":"exists","value":false}` to ask for evidenced absence. The optional
`value` must be a strict JSON boolean; omission means `true` for existing entries.
There is no general negation or Boolean expression language.

For example, an applicability predicate can be
`{"launch_flag":{"flag":"--pytorch_weight_path","op":"exists","value":false}}`.
A complete single supported selected command without that flag establishes this
predicate. An explicit CLI override excludes it. A missing/ambiguous command
remains UNKNOWN; UNKNOWN is never negated into an evidenced absence. An absent
config/processor final key requires a loaded unambiguous parent object. A missing
parent/document or duplicate processor registry instance remains UNKNOWN. Present
JSON null is still a present key.

If all entries are excluded or none can be fully evaluated, the aggregate is
UNKNOWN/unassessed, not a blanket PASS. A known consistent applicable entry can
remain consistent; an independent ERROR continues to dominate.

Concise witnesses retain the typed decisive observation, expected value, operator
and phase. Large values/context use explicit type, preview, byte length and SHA256;
they are never substituted as an actual JSON value. The full detailed report keeps
the original bounded value and a streaming digest binds each concise report to it.

This successor preserves the external-review source02 candidate. The final
catalog contains [twelve independently reviewed entries](REGISTER_CATALOGUE_EVIDENCE.md).
Adding another entry requires independent validation of its exact adapted bytes
and fixtures; unreviewed drafts are never loaded by production.

## Publisher evidence and fixed comparison

Evidence rows may add `basis: user_report | maintainer_statement | publisher_recipe`. Legacy non-recipe evidence without a basis retains its original user-report classification; this is structural compatibility, not verification of the report. Publisher-only evidence cannot support a version trap or contract. Every recipe evidence row must declare its basis explicitly.

A `kind: recipe` entry requires one `recipe_field`:

```json
{
  "field": "n_action_steps",
  "checkpoint": "publisher/checkpoint",
  "status": "published",
  "published_value": 8,
  "source_index": 0
}
```

`source_index` selects exactly one pinned `publisher_recipe` evidence row. `applies_when` must explicitly bind the reference checkpoint in `checkpoint_id_in`; no package or stack alone establishes a recipe's checkpoint identity. If an independently validated card applies to additional explicitly listed checkpoints, each needs its own validation and the message names the actual declared checkpoint selected by those predicates. The recipe has exactly one violation predicate, e.g. `{"config_key":{"path":"n_action_steps","op":"ne","value":8}}`. Its field and typed value must equal `recipe_field`. All recipe findings are capped at WARNING even if the supplied severity is ERROR or the evidence also includes user reports. The engine replaces authored titles/remedies with:

`<field> = <actual> differs from the published evaluation recipe for <checkpoint> (<published>, <repo>@<sha>:<path>:<line>). Results are not comparable to the published numbers.`

First fix: `set <field> to <published> to reproduce the published setup`.

This is a comparison with a declared published setup, not an execution or failure claim. A missing actual value remains UNKNOWN; it does not become a mismatch.

An unpublished field instead uses `{"field":"simulator_version","checkpoint":"publisher/checkpoint","status":"unknown","reason":"Not published in the pinned sources"}`. A conflicting field uses `status:"conflict"`, a reason and `source_indices:[0,1]` referring to 2..8 unique pinned evidence rows. Both require `violation_when:[]`, invent no value and appear only in the full report's entries. They produce no finding and do not bury another entry's evidenced ERROR/WARNING. Use separate entries for separate fields; no `any_of` has been added.

## Explicit selected stacks

Input manifests declare `"stacks":["openpi","libero"]`. This is an author-declared **complete selection**, not a list inferred from every installed package. An empty list explicitly selects no stack. A declared selection disjoint from an entry's stacks excludes that entry before predicates are read. Every entry requires this declaration; its omission remains UNKNOWN. The combined integration successor removes the source04 legacy-entry exception.

Identifiers are canonical lowercase tokens (`[a-z0-9][a-z0-9_-]{0,99}`), not display names. Entry `stacks` is an eligibility union, not an arbitrary discovery-tag list. Shared benchmark tags can admit unrelated model stacks; adapted recipes must use intended primary stack identifiers and retain benchmark/model restrictions in explicit applicability predicates. There is no benchmark-to-framework inference.

Package presence can provide observed metadata but does not identify the selected trainer or launcher. This successor does not infer stack absence across Python environments, checkpoint training versions, local cache/GCS aliases or resolved TrainConfigs.

## One applicable command

Replace `@script` predicates with an optional entry-level selector list, e.g.:

```json
{"launch_target":["script:scripts/train_pytorch.py","module:openpi.training.train_pytorch","entrypoint:lerobot-eval"]}
```

Each selector is exact and typed. Script selectors are canonical relative POSIX paths; no basename or first-arbitrary-`.py` heuristic is used. Entry points match the named executable; module selectors match Python `-m`. Supported common launchers include simple Python commands, `python -m`, `torchrun` and `python -m torch.distributed.run`, plus `uv run` around supported launchers. Only tested launcher options are parsed; unsupported options remain UNKNOWN.

Unquoted `;`, `&&` and `||` separate simple commands. Quoted or escaped semicolons remain argument content. Substitutions, control flow, redirections, background/pipeline syntax, malformed quotes, ambiguous duplicate targets and multiple commands without a target remain UNKNOWN. Only the selected command's flags are considered. Nothing is executed and conditional command text does not attest historical execution.

For one selected applicable command, an absent flag has `exists:false`. Absence is not JSON null. Other comparisons remain UNKNOWN unless the source-backed default below applies. A missing command never supplies launch defaults.

## Source-backed absence defaults

A value predicate may add:

```json
{
  "default_when_absent": {
    "value": false,
    "source": {"repo":"owner/repo","sha":"<40 lowercase hex>","path":"src/package/config.py","line":12},
    "sha256":"<64 lowercase hex>"
  }
}
```

The exact default source hash must also appear for that path in a `source_files` applicability profile. The selected stack must match, the authorized runtime source bytes must match, and the actual selected context must exist: a loaded config/processor parent object with only its final key absent, or an explicitly targeted selected script with the flag absent. Launch defaults additionally require input `command_directory`, the explicit actual command cwd; resolving the selected script against it must stay inside `runtime_source_root`, and that script's own bytes must match a `source_files` applicability profile. Module/entrypoint selection supports explicit flags, but their absent defaults remain UNKNOWN until actual source resolution has its own bounded input contract. Missing/unreadable config, missing parents, duplicate processor instances, missing commands, unrecognized source/version or explicit null never invokes the default. Source mismatch remains UNKNOWN. A file matching somewhere else is insufficient.

Launch defaults are additionally restricted to one direct invocation. Preparations such as `cd nested; python ...`, `source activate; ...`, `uv run`, `env` and leading environment assignments retain explicit flag selection but cannot authorize absence defaults: their effective cwd/environment is not reconstructed. Use a captured actual resolved config for those contexts. Quoted semicolons inside a single command remain argument content and do not disable an otherwise source-bound default.

## Scalar values, numeric predicates and versions

Present JSON null is an evidenced scalar value: `eq` a string is false; `ne` a string is true. Scalar equality preserves types and never treats booleans as integers. Missing keys stay UNKNOWN unless a verified absence default applies. Container/type ambiguity remains UNKNOWN. Numeric `lt/lte/gt/gte` predicates are supported for finite config/processor numbers, excluding booleans; launch flag text is not silently converted to numbers.

PEP 440 versions use the maintained [`packaging.version.Version`](https://packaging.pypa.io/en/stable/version.html) implementation (required dependency `packaging>=24`), including prereleases such as `3.0.0b1`, epochs, development/post releases and local versions. Unparseable versions remain UNKNOWN; comparisons never order Git SHAs or dates.

## Processor registry selectors

Direct dotted object paths remain supported. For a saved processor containing a `steps` array, `<registry_name>.config.<key>` selects exactly one step whose literal `registry_name` matches, then traverses its actual saved object. Example: `normalizer_processor.config.normalization_mapping.VISUAL`. There is no implicit `.config` insertion, class-name alias, training-version inference or first-step fallback. Missing or duplicate registry instances, a colliding top-level key and malformed/oversized registry tables remain UNKNOWN. The full witness records the selected step index, saved file hash and selector.

`dataset_key`, cross-artifact flag-vs-processor comparison, `train_config_key`, source-byte negative membership and generic boolean/`any_of` expressions are not added. Each needs its own concrete input/provenance contract and qualification before adoption.

## Path and literal-absence corrections

Nested paths and API read roots reject other-user `~username` aliases as UNKNOWN rather than attempting account lookup. An unresolved current-user home alias is also normalized to a bounded input error; current-user `~`/`~/...` and direct authorized paths remain supported. The combined root owns equivalent top-level CLI normalization before register intake.

`source_contains` now permits `occurrences:{eq:0}` as an explicit bounded literal-absence hint. It never establishes behavior or authenticates a changed implementation; a source-behavior entry supported only by literal hints remains capped at WARNING. `gte:0`, negative counts, booleans and non-finite/oversized bounds are rejected. This supports the curator's PT-HINT correction without adding negative source-hash membership.
