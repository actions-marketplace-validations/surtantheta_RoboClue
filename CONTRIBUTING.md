# Contributing

Start with a concrete robotics decision: the command or pipeline, its intended
behavior, what the native tool does, and the evidence available. Healthy and
ambiguous examples matter as much as failures. You can contribute a small
reproducer, a source/version update, a clearer rule page or a new supported check.

## Report an incorrect or confusing result

Use the issue templates. Include environment/package versions, exact arguments,
component verdicts and a small reproducer or pinned public artifact. Redact
credentials and private data; full datasets or weights are usually unnecessary.
Name the rule ID or diagnostic code when present. A missing video is not by
itself a broken dataset, and a component PASS is not whole-pipeline approval.

## Add or extend a rule

The [rule catalog](docs/rules/README.md) gives each supported failure family a
stable public ID while preserving existing machine diagnostic codes. Start with
the [worked RC006 contribution](docs/rules/CONTRIBUTION_EXAMPLE.md): it runs the
actual detector on tiny generated controls without our server, GPU or robot stack.

1. State the decision and criterion. Explain the native failure and link primary
   upstream source or an issue, including the tested revision. Do not invent an
   upstream issue or infer compatibility for every release.
2. Identify an existing engine in `src/policy_doctor/`. Extend it where the
   semantics match; isolate version-specific declaration readers from reusable
   comparison logic. Return evidence, exact scope, a supported remedy and an
   UNKNOWN path for missing, ambiguous or unsupported inputs.
3. Add the metadata record in `src/policy_doctor/rule_catalog.py`, using the next
   unused RC ID and the actual machine codes. Create `docs/rules/RCxxx.md`, link
   it from the index and list its upstream evidence, tested profiles, trigger,
   limits, correction/recheck and regression paths. Keep IDs stable; do not
   renumber existing rules or rename old report codes for presentation.
4. Add broken, healthy, corrected and missing/unsupported controls. Prove the
   correction clears the **same** original criterion and preserves originals.
   Include the documented CLI and default MCP route so a correct detector cannot
   remain hidden behind an unrelated entry point.
5. Run focused tests and the appropriate [development checks](DEVELOPMENT.md).
   Keep native execution evidence distinct from generated controls. A mock or
   static configuration cannot establish model, simulator or physical behavior.

For catalog changes:

```sh
python -m pip install -e '.[dev]'
python -m pytest -q tests/test_rule_catalog.py
python examples/rules/contribution_walkthrough.py
python -m ruff check src tests examples scripts
```

`rule_catalog()`, `get_rule()` and `rules_for_code()` are metadata discovery APIs;
`validate_catalog()` finds structural contribution errors. The catalog does not
load executable plugins from user artifacts. Adding a new metadata entry does
not make its detector run: route and engine changes need review and tests.

## Updating framework support

A pin records the behavior we tested. Add a new profile with upstream evidence,
healthy/negative controls and the relevant native execution when the claim needs
it. Explain changed defaults or semantics rather than broadening a version range
by name alone. Preserve old profiles where their data remains in use. Do not
weaken a check, drop a condition or regenerate a suspect reference to obtain PASS.

## Documentation and pull requests

Make the first useful decision clear. State supported profiles and unresolved
coverage; do not interpret UNKNOWN as a data defect. Preserve upstream notices,
fixture provenance and credit existing validators. PR descriptions should state
the user-visible problem, resulting behavior, validation and material limits.
