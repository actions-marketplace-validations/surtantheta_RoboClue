# Worked contribution: RC006 statistics consistency

This example follows an existing rule from problem statement to detector,
metadata, healthy/ambiguous controls and public discovery. It creates tiny
statistics documents in a temporary directory, uses RoboClue's real detector and
checks original bytes afterward. No server paths, GPU, framework or dataset
account are required.

From a fresh checkout with Python 3.11 or 3.12:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python examples/rules/contribution_walkthrough.py
roboclue advanced rules RC006
python -m pytest -q tests/test_rule_catalog.py tests/test_brand_and_preview_feedback.py
```

The generated controls are `broken`, `healthy`, `corrected`, `missing` and
`unsupported_shape`. Only the broken same-cohort summaries produce the
`dataset.statistics_inconsistent` ERROR. Missing and non-vector statistics stay
UNKNOWN. Healthy/corrected controls are internally consistent; that phrase is
not a whole-dataset PASS. The correction is source-declared fixture intent,
not a claim to reconstruct an unknown real cohort from its saved summaries.

## Follow the implementation

1. Read [RC006](RC006.md). Its narrow proposition is arithmetic self-consistency,
   not "the dataset's statistics match its rows".
2. Find `dataset_statistics_findings` in `src/policy_doctor/diagnostics.py`.
   It returns existing codes and explicit evidence; the rule catalog does not
   execute or replace it.
3. Find RC006 in `src/policy_doctor/rule_catalog.py`. It maps exact diagnostic
   codes to the public ID, upstream context, scope, routes and tests.
4. Read `tests/test_brand_and_preview_feedback.py` for rounding-scale and
   legitimate fixed-feature controls. Read `tests/test_rule_catalog.py` for
   metadata/link/code consistency and actual CLI/MCP discovery controls.
5. Follow `roboclue check DATASET --question dataset --json` to the dataset health
   card. Inspect `dataset_statistics`; unrelated missing row/video evidence can
   keep the overall result partial. Never rewrite an overall UNKNOWN to PASS
   just because this rule clears.

## Add a different check

Start with a small failing native/public artifact or a clearly labeled generated
control. Write the intended criterion and a healthy case first. Extend the
existing detector when possible; for a different consumer, separate its
version-specific declaration reader from reusable comparison logic. Keep
unsupported expressions, revisions or missing declarations UNKNOWN. A version
pin documents evidence; deleting it is not a compatibility fix.

Add one metadata record to `_RULES`, assign the next unused RC ID, create its
page under `docs/rules/`, and add a row to the catalog index. Preserve old IDs and
machine codes. Include primary upstream links at tested commits and the focused
test paths. `validate_catalog()` checks structure; repository tests check links
and code mappings. Neither establishes the scientific claim by itself.

Add meaningful broken, healthy, corrected and missing/unsupported controls,
including the documented `roboclue check` route and default MCP route when the
rule is offered there. A new semantic check also needs tests proving its
specific decision; cloning the catalog test does not validate new detector logic.
If native execution is needed, keep that qualification separate from generated
unit controls and record its framework versions and original criterion.

For a candidate metadata record, build a new in-memory `rule_catalog()` result,
append the record and update `count`, then call `validate_catalog(candidate)`.
This does not install or execute a rule; detector registration and routing are
reviewed source changes. External artifacts cannot supply executable plugins.

## Before the pull request

Run the focused checks above, then the [development checks](../../DEVELOPMENT.md)
required for your change. Explain the user decision, before/after behavior,
supported version, evidence limitations and how the correction was rechecked.
Retain original inputs and licenses. Do not claim support for another framework
release merely because the same fixture passes RoboClue's own comparison.
