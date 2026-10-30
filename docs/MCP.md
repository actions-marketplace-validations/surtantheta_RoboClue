# Use RoboClue with an MCP agent

From the checkout in your installed environment, add the optional `mcp` extra,
then ask setup to print registration configuration:

```sh
python -m pip install '.[mcp]'
roboclue mcp setup --root /absolute/path/to/artifacts
```

Use an existing directory containing the artifacts you want inspected. Setup
prints configuration for review; it does not silently modify your agent settings.
Copy the returned server definition into your client's MCP configuration and
restart/reconnect the client. The absolute server executable must belong to your
installed environment, and every local input/reference must be under permitted
read roots. Run `roboclue mcp serve --help` for server options.

Generated Python launchers use `-I -B` to load the installed server rather than
a same-named module in the client's working directory or its `PYTHONPATH`.
Install the package in that interpreter's environment before registering it;
an ad-hoc source-only `PYTHONPATH` is deliberately not used by this launcher.

## Choose the evidence tool

| Question | Tool |
| --- | --- |
| What can I check with the evidence I already have? | `choose_workflow` |
| Inspect a supported local/public dataset | `dataset_health_card` |
| Compare saved native targets with retained intent | `check_training_targets` |
| Inspect recorded command/feedback evidence | `triage_flight_recording` |
| Assess supplied upgrade comparisons | `assess_upgrade_impact` |
| Inspect detailed dataset, robot-units or GR00T configuration questions | `preflight_check` |
| Inspect simulator HDF5 conventions and explicit conversions | `check_sim_provenance` |
| Inspect interrupted local recordings and recovery plans | `check_recording_salvage` |

The default server advertises these **eight read-only tools**. Ask the client to list their schemas for
the precise arguments. Concise responses retain findings and coverage; detailed
responses remain bounded. [Exact four-workflow arguments](BIG_BETS_USAGE.md).

Read the `roboclue://rules` resource for the numbered rule catalog: diagnostic
code mappings, upstream evidence, tested profiles, remedies and rule pages.
This is read-only discovery metadata, not another tool or an executable plugin.
Brief deployment findings include the public rule IDs associated with their
exact diagnostic code; an unmapped code is not assigned a guessed rule.

Use `roboclue mcp setup --root /absolute/artifacts --advanced` to configure the
full 29-tool read-only specialist catalog, or pass `--advanced` when serving.
The default catalog includes the specialist checks needed for the main supported
questions; users do not need advanced mode to reach them. Existing legacy server
entry points keep their original catalog. No new case-writing or native-execution permission is enabled by
`--advanced`; separately authorized specialist options remain under
`roboclue advanced serve`.

`dataset_health_card` omits the GR00T dataset contract by default. Its `consumer`
field is a freeform label, not a semantic binding. Set `groot_contract=true` or
select a recipe where its tool schema supports it. The compact health-card tool
does not expose a GR00T-version selector: use `preflight_check`,
or `roboclue check DATASET --groot-version n1.7` at the terminal for that option.
Generic card results do not establish native compatibility. `preflight_check`
retains its existing automatic GR00T checks. The CLI route
`roboclue check DATASET --question groot` focuses on the selected recipe question;
`roboclue advanced preflight` exposes the complete preflight interface.

### A short deployment answer

With the default server, ask `preflight_check` with the dataset, your actual
`robot_command` or `robot_config`, and `lerobot_version` from that robot's environment.
Every argument in the default catalog has a description in the advertised schema;
the original strict types and bounds still apply. For a simple recorded-units comparison, its
default answer keeps structured JSON within 2 KiB and the complete SDK result
within 2.5 KiB:
the original overall status, scoped units status, named recorded/configured
modes, decisive evidence references and an exact recheck command. A units PASS
does not approve deployment; policy processing, calibration and physical
behavior remain unassessed. Unrelated checks retain their original overall
status and remain available in the detailed report.

If the command/config explicitly declares camera inputs, RoboClue compares their
feature-key sets with local `observation.images.*` declarations even without a
checkpoint. Disjoint names produce a scoped declaration mismatch; partial or
empty view selections remain advisory because they may be intentional. A
matched key set does not compare image size, frame rate/fps, joint order, decoded
images or physical cameras. The brief names these unassessed interfaces even
when no camera mapping was supplied. For a qualified source profile whose native
default is an empty camera mapping, an omitted mapping is compared as empty and
reported as a warning when the dataset declares images. An unsupported version
cannot establish that default. `camera_keys.feature_prefix` applies to
the shortened names in that view. Its metadata digest identifies
`meta/info.json`; the declaration digest identifies the supplied command text or
config bytes. Declared names reflect an explicit supported rename map, with raw
names retained separately when different. If combined witnesses cannot fit, the answer
retains the camera result and requests details rather than clipping conditions
or claiming deployment approval.

Concise `resolved_paths` fields are display previews capped at 240 characters.
Clipped fields carry `<field>_truncated`, `<field>_length` and
`full_<field>_sha256` for the complete UTF-8 value. Detailed responses retain
complete paths; the concise view retains its full-report digest and read-root
authorization result. Preview strings do not establish a complete path identity.

Repeat the same tool request with `response_format="detailed"` for the original
preflight evidence. This performs a new read; the returned report hash can help
detect changed evidence. `$dataset` in a brief evidence path means the dataset
supplied in this request. Text is always readable prose, including requests with
no camera mapping. It names the units, robot and assessed
camera keys while keeping full conditions and the exact next action in structured
content. Exact detector enums, source revisions and hashes remain in the
structured evidence. Framework context reports an installed distribution version
only when one is observed in RoboClue's Python process; it does not establish the recorder/trainer environment or renew
native qualification. Mixed checkpoint, conversion
or consumer comparisons keep the existing summary bound rather than discard
their conditions. If a required command/witness cannot fit, the short answer
requires details and gives no concise approval; a known ERROR remains ERROR.
Legacy servers retain their existing transport; canonical advanced views retain
the larger report bounds.

Canonical product tools avoid duplicating a detailed JSON report in text and
structured content: text lists readable findings and artifact references, while
`structuredContent` retains the returned full JSON evidence. Concise dataset
cards similarly use a short text recap plus their original structured view.
Dataset-card text distinguishes saved-statistics self-consistency from a complete
comparison of supported numeric statistics with the covered rows. That numeric
comparison excludes image statistics and quantiles; incomplete cohorts stay
unassessed. The original separate component results remain in structured evidence.
These generic cards and mixed-consumer reports have their own larger bounds;
the 2.5 KiB limit applies to the scoped deployment brief, not every tool result.
Explicit legacy launchers keep their original transport representation.

### Declare the robot environment's LeRobot version

Set `lerobot_version="0.6.2"` alongside the actual `robot_command` or
`robot_config` when that is the consumer you intend to check. The label selects
a tested robot/version source profile for defaults and normalization, as listed
in the rule catalog; it
does not attest the fleet's installed code or qualify the whole release. In
that source profile `lerobot-record` is collection-only; policy deployment uses
`lerobot-rollout`. The compact `consumer` context names the declared version,
selected profile and command role, with `fleet_verified=false`. Full source
basis and scope remain in the detailed units component.

An unqualified explicit version leaves consumer units UNKNOWN rather than
borrowing the old default or recommending a mode change. Recorded-data inference
and unrelated dataset errors are retained. Omitting the version also leaves the
canonical product's consumer comparison UNKNOWN and asks for the actual version.
Explicit historical APIs retain their conditional compatibility behavior.
Canonical CLI preflight and MCP use the same engine qualification before report
size reduction; `roboclue advanced preflight` accepts `--lerobot-version` for the
actual consumer. A missing version cannot acquire an active units comparison
from an older conditional report during presentation.
Tool-Python distribution metadata remains a
separate observation; it is never used as the fleet version. A correction must
repeat the same request and version, changing only the intended setting.

Both `roboclue mcp setup --advanced` and `roboclue advanced setup` emit canonical
RoboClue launch commands. Their servers identify as `RoboClue`; explicit
historical launchers remain compatibility paths. [Compatibility notes](FAQ.md).

Example instruction to your agent:

> First use choose_workflow with goal="training" and framework="groot". I have
> no retained reference. Explain what I can inspect now and how to retain
> reviewed evidence for my next pipeline change.

The read-only chooser accepts `goal` (`dataset`, `training`, `runtime`, `upgrade`,
`conversion`, `deploy`, `recovery` or `sim`), optional `framework`, and paths for evidence you have.
For deployment, include `lerobot_version` with the actual robot command/config;
the chooser carries it to the suggested preflight request. The chooser text names
the next tool without repeating the structured JSON. It checks
readiness within authorized read roots and returns `assessment_status="not_run"`,
missing evidence and suggested commands/requests. It does not validate semantics,
fetch inputs, create a reference, run training or execute the suggested tools.
`readiness="inputs_present"` means required inputs were supplied, not a passing
assessment. [Choose by your existing evidence](CHOOSE_A_WORKFLOW.md).

Once you have the required evidence:

> Use RoboClue's check_training_targets tool with my saved contract, independently
> retained criterion pin and native capture. Show the first source-linked
> mismatch and distinguish assessed targets from unknown additional inputs.

## What runs where

MCP reads saved artifacts. Live training guards attach through Python inside the
training process. Runtime recording also attaches through explicit application
code. The four evidence tools do not send robot commands, launch arbitrary
training, write repairs or expand their own read roots.

Public Hub access requires both operator `--allow-hub-datasets` and request
`fetch=true` with an explicit `hf://` source. Otherwise local requests remain
offline. Some existing advanced tools need separate operator opt-ins to create
evidence cases; they are not enabled by the four read-only evidence routes.

An agent's explanation is an interpretation of the returned evidence. Verify
the correction under the retained criterion rather than treating prose as proof.

The repository also includes an optional [agent investigation guide](../.agents/skills/robot-incident-investigation/SKILL.md).
It describes existing evidence routes and retained cases; it does not register
the MCP server or expand its permissions. Follow your client's skill-loading
instructions if you want to use that guide.
