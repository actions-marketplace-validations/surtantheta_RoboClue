# Version and provenance register

The register evaluates curated facts through `roboclue check INPUT.json` or the existing default MCP `preflight_check(source=...)`. `choose_workflow(..., manifest=...)` recognizes this input role. It does not introduce another command or default tool.

The publisher-recipe amendment, complete selected-stack declarations, exact command selectors, source-backed defaults and unique processor-registry syntax are specified in [the successor schema delta](REGISTER_RECIPE_SCHEMA_DELTA.md).

No content is inferred from a package version. Installed metadata and lockfiles declare versions. Exact source-file hashes are a separate applicability predicate, required for entries whose claim depends on source behavior. There is no Git/date ordering and no framework import, network request, dynamic expression, include directive or command execution.

## Input v1

```json
{
  "schema_version": 1,
  "kind": "provenance_register_input",
  "stacks": ["lerobot"],
  "collect_installed": false,
  "metadata_paths": ["environment/lerobot-0.6.1.dist-info"],
  "lockfiles": ["requirements.txt", "uv.lock"],
  "config": "checkpoint/config.json",
  "processor": "checkpoint/policy_preprocessor.json",
  "checkpoint_id": "explicit/published-checkpoint-identity",
  "launch_command": "python train.py --env LIBERO",
  "runtime_source_root": "source-checkout"
}
```

Only `schema_version` and `kind` are mandatory. Omitted evidence remains UNKNOWN. Paths are relative to the manifest directory or absolute. Every nested path obeys MCP read roots. All path components must be materialized, and only bounded regular files are read. Input JSON is limited to 256 KiB, 4096 nodes and depth 16. At most 16 metadata directories and 16 lockfiles are read. The catalogue is limited to 128 entries and 64 distinct packages. Source profiles have at most 32 files, 2 MiB each, 16 MiB total; one evaluation additionally caps source reads at 256 files and 64 MiB. Budget exhaustion remains UNKNOWN. Large observed values retain an explicitly truncated preview and full digest. A manifest selects exactly one config and one processor document; it does not implicitly inspect an entire checkpoint directory.

`collect_installed=true` selects metadata in the RoboClue Python environment. It does not establish the environment of another trainer process. Equivalent PEP 440 versions are coalesced; disagreement between pins, unresolved constraints, malformed metadata or multiple conflicting pins remains UNKNOWN. uv/Poetry lockfiles are parsed as bounded TOML, including wheel-rich metadata; requirements/pip-freeze text supports exact pins, a UTF-8 BOM, comments, hash continuations and inert index declarations. Index locations and hashes do not authenticate an installed artifact. Includes, local/editable/VCS/direct sources, package markers and constraints remain unresolved; no environment marker, installer, URL or supplied code is executed. Unrelated recognized package rows do not override selected pins. PEP 440 pre/dev/post/local versions and epochs are compared with `packaging.version.Version`; invalid strings remain UNKNOWN.

Lock input has its own 2 MiB byte bound, 100,000 parsed nodes, depth 24 and at most 2,048 TOML package rows. Requirements retain a 4,096-line bound and a 16,384-character logical-line bound. At most 64 observations per selected package are retained; overflow is explicitly UNKNOWN, never silently truncated into a consistent result. Rule/config JSON retains its smaller existing bounds.

Every entry needs an explicit complete selected-stack declaration to be evaluated;
omitting `stacks` remains UNKNOWN. See the [twelve packaged entries and their
limits](REGISTER_CATALOGUE_EVIDENCE.md) for a supported comparison.

## Curated entry v1

The authoritative executable schema is `src/policy_doctor/register/schema.py:validate_entry`; `src/policy_doctor/register/schema-v1.json` supplies the machine-readable interchange schema. Store one JSON object per `register/entries/<ID>.json`. A `.yaml` suffix is accepted only when its contents are valid JSON, the JSON subset of YAML. The core has no YAML dependency.

Each entry has these required fields; the successor additionally permits `launch_target` and requires `recipe_field` for `kind: recipe`:

```json
{
  "schema_version": 1,
  "id": "EXAMPLE-CONTRACT",
  "kind": "contract",
  "title": "Describe the independently validated contradiction",
  "stacks": ["example"],
  "claim_basis": "release_metadata",
  "applies_when": [
    {"package_version": {"package": "example", "version": {"gte": "1.2", "lt": "2.0"}}}
  ],
  "violation_when": [
    {"config_key": {"path": "settings.mode", "op": "eq", "value": "bad"}}
  ],
  "severity": "ERROR",
  "evidence": [{
    "url": "https://example.org/public-report",
    "quote": "An independently machine-checked verbatim public report excerpt",
    "source": {"repo": "owner/repo", "sha": "0000000000000000000000000000000000000000", "path": "source/file.py", "line": 1}
  }],
  "healthy_controls": ["Documented exception", "Corrected setup"],
  "first_fix": "One evidence-supported next action",
  "unknown_when": "Identify the unresolved evidence needed",
  "last_verified": "2026-10-29",
  "validation": {"status": "confirmed", "record": "Independent validator receipt identifier"}
}
```

This is a schema illustration, not a registered fact. The placeholder evidence is not validated and must never be imported. Entry validation checks structure and bounds; it cannot establish scientific truth or verify a public quote offline. Maintainers must independently check the pinned source, quote, version boundary and fixtures before adding an entry. The validation record names that review. Production entry loading never reads draft curation directories.

All predicates within each list are AND. Supported predicates:

| Predicate | Body |
|---|---|
| `package_version` | `{"package":"name","version":{"eq":"1.2"}}`, or PEP 440 `gt`/`gte` and `lt`/`lte` bounds |
| `checkpoint_id_in` | Nonempty list of explicit checkpoint identity strings |
| `config_key`, `processor_key` | `{"path":"nested.key","op":"eq","value":literal}`; `ne`, `in` and `exists` also supported |
| `launch_flag` | `{"flag":"--env","op":"eq","value":"LIBERO"}`; explicit single long flags only, no effective-default inference |
| `source_files` | `{"files":[{"path":"src/package/module.py","sha256":"<64 lowercase hex>"}]}` |
| `source_contains` | `{"path":"src/package/module.py","literal":"bounded exact text","occurrences":{"gte":1}}`; applicability only, weak hint |

Use `claim_basis=source_bytes` for implementation behavior; this requires `source_files` or `source_contains` in `applies_when`. A source hash mismatch gives UNKNOWN rather than declaring a different checkout healthy. All listed file profiles must match. A literal-only source rule is capped at WARNING even when the entry requests ERROR: comments, strings, dead branches and later overrides can contain the text without implementing the behavior. Literal hints support `gte` or `eq` occurrence counts, defaulting to `gte:1`; they are forbidden in `violation_when`. An unresolved file remains UNKNOWN. The source profile covers the claimed path only, not a whole installed environment. Exact commit metadata is retained as a declaration and cannot substitute for source hashes. `commit_range`, arbitrary regex/predicates and inherited/default expressions are rejected.

For independently verified historical alternatives, each source-file row may contain `sha256_in:["<hash one>","<hash two>"]` instead of `sha256`. This is an explicit OR over 2..32 unique verified byte hashes for that one path. Different file rows remain AND, and repeated paths are invalid. Every alternative must have its own independent source/validation evidence; an array is never inferred from commit ancestry, dates or a version label.

Within applicable entries, any unresolved applicability predicate yields UNKNOWN, including when another applicability predicate is false. When all applicability inputs are resolved, an evidenced exclusion yields `not_applicable`. For an applicable entry, any unresolved violation predicate yields UNKNOWN. All applicability and violation predicates true yield the declared severity (recipe findings are always capped at WARNING); a wholly resolved false violation yields `consistent`. Missing keys remain UNKNOWN unless a verified absence default applies. Present scalar values compare without coercion: null is unequal to a string, and `true` does not equal `1`. Container/type ambiguity remains UNKNOWN. A selected complete command can establish flag `exists:false`; an absent flag is never implicitly null. Complete author-declared stack selection and exact command targets exclude unrelated entries before their predicates. Publication-CONFLICT/UNKNOWN recipe fields are full-report metadata only. A package match alone does not imply a violation. The result includes each predicate, observed value, file/hash/selector, public source citation, validation record and first fix. PASS is scoped only to the loaded register entries, never to compatibility or task performance. An empty or partially invalid catalogue remains UNKNOWN. The existing `--fail-on error` gate fails only on evidenced ERROR; UNKNOWN/WARNING keep that gate at exit 0.

Each production entry requires a positive, two healthy controls, UNKNOWN, parent discrimination, malformed-input checks and an independently reviewed mutation probe (at least 80% killed). Engine fixture entries under tests are explicitly synthetic and do not qualify real public incidents.
