# Combined launch integration draft

This tree assembles W4 recovery, W1 FastWAM, W2 register, W7 instruction
catalogue/transport and W3 initialization/camera contracts from the isolated
component commits. Component passes do not qualify this combined source.
The catalogue remains infrastructure-only for RC012: no curated entries have
been imported. The default MCP surface remains eight tools.

Conflicts were resolved by retaining both checks and validating their artifact
roles before dispatch. Pipeline JSON, provenance JSON and checkpoint directories
reach the existing `check` and default `preflight_check`. Checkpoint image and
instruction observations can coexist; an independent ERROR remains an ERROR.
The chooser retains separate checkpoint, pipeline-contract and register-manifest
plans and refuses incompatible selections.

There is one CLI `--launch-command` and one MCP `launch_command` observation.
For a provenance input, a captured and supplied declaration must agree; disagreement
is UNKNOWN without modifying the manifest. Source authorization precedes reads.
Unknown source versions and missing source files do not establish corrected behavior.

The source02 independent register review's H1–H4 checklist remains explicit:

- Select exact supported launch targets without merging flags across commands;
  substitutions, ambiguous targets and absent targets in multiple commands stay UNKNOWN.
- Only explicitly recognized checkpoint identities establish applicability. A
  trailing path delimiter is normalized; case aliases and cache/URI mappings are
  not inferred. Unrecognized IDs remain UNKNOWN.
- Every `source_contains` applicability hint caps a result at WARNING, including
  when `claim_basis` is `release_metadata` or exact source rows are also present.
- Every entry requires author-declared complete `stacks`. UNKNOWN first actions
  request missing/conflicting evidence rather than recommend an unproven repair.

Concise output prioritizes the decisive violation witness. Nested JSON equality
uses strict types, including booleans versus integers. Recipe wording and first
fixes remain complete in CLI/default MCP. Displayed citations use file:line;
the structured result retains the full pinned citation. If retaining complete
bounded required fields exceeds the usual 8 KiB preview target, the result marks
that explicitly rather than discarding those fields. Unresolved publication
fields remain full-only metadata and prevent an overall PASS without burying an
independent finding.

Open qualification work includes the exact combined six-profile matrix, fresh
installed wheel, clean dataset sweep, broad independent mutation/discrimination,
native source gates and external review. The source02 review's realistic large
lockfile/directive intake issue remains disclosed; the current conservative
bounds can return UNKNOWN on those inputs. The successor integrates root's
native-sidecar snapshot fix from `f5a2ac30d2cf74519389e7ffcdbdb46cc7d766f2`;
independent repeated validation remains separate. This draft has no release or
public-action approval.
