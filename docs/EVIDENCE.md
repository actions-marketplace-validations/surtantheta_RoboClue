# Evidence behind RoboClue

We separate generated controls, original native execution, numerical review and
package checks. A passing test count alone does not establish user benefit.

## Native execution

| Study | Demonstrated result | Limit |
| --- | --- | --- |
| GR00T original native trainer | Wrong relative targets accepted for two optimizer steps; guard catches 1,400 target differences plus mask dtype before zero forwards/steps; correction/replay pass the original pre-fault criterion | Public simulation-derived data; pretrained Qwen backbone and newly initialized native GR00T head; recorded active parameter-slice equality, not trained-policy competence |
| LeRobot original update function | Wrong action/state normalization and swapped selected cameras blocked; corrected/healthy/replay take two steps; two-worker case and independent source/selected-parameter audit | Native ACT/update-function profile; named-parameter hashes exclude buffers |
| LeRobot distributed control | Two real CPU/Gloo ranks refuse before forward when one rank is faulty or unresolved | Separate custom optimizer loop, not the original distributed trainer CLI |
| OpenPI boundary | Original native transforms/loader plus actual Torch input hook; changed targets refused before the controlled model's forward | Controlled Torch consumer; no Pi0 optimizer/model execution |

References were retained before faults and were not regenerated to make a
correction pass. Independent numerical checks rederived covered source values
and selected optimizer updates without importing product/native training code.

## Dataset, runtime and upgrade workflows

- A pinned public Berkeley UR5 scan assessed 97,939 rows and 5,537,181
  metadata/Parquet bytes. It retained video-window contradictions. Six monitor
  cycles cover caching, configuration/revision changes and offline behavior.
  This is an existing incident, not a new discovery or Hub-wide prevalence study.
- Runtime recording replay used 32 public recorded numeric rows and 32 decoded
  frames through a software interface. An injected return discrepancy was caught
  and corrected under the same criterion; physical associations remained UNKNOWN.
- Upgrade watch observed a real public upstream revision and assessed bound
  affected/unaffected/unknown controls. Isaac Lab end-effector findings concern
  explicit caller conventions: raw old/new end-effector values can be unchanged,
  and a legal mixed field-specific convention passes. This is not an intrinsic
  upstream-rotation-bug claim.

## Package qualification

Source tests, fresh installed CLI/MCP checks, native framework experiments and
archive verification are separate gates. A native experiment proves its stated
profile; it does not certify every later package or every framework version.
Optional native/Torch cases have explicit native evidence; skipped default-suite
cases remain skips. Current checks also cover bounded numeric-statistics
comparison, native camera defaults and separate OpenPI raw-key availability.

The recorded native video has its own [qualification receipt](../assets/native-qualification.json)
and [scope](VIDEO.md). That historical receipt does not certify the current
package. Source and installed-artifact qualification must bind the exact build;
earlier internal CI cannot certify a later public commit. The repository's
workflow will run against the public head after the owner supplies the destination.

## What this evidence does not establish

There is no independent human installation trial, physical robot qualification,
measured GPU-time saving, comparative practitioner-effort result or community
adoption measurement. The native demonstrations use controlled faults and known
criteria. They establish supported refusal/correction behavior, not all-failure
detection or task success.

To reproduce the lightweight saved-artifact workflow, run the [quick start](QUICKSTART.md).
For native reference creation and execution settings, use the [GR00T](GR00T_TRAINING_GUARD.md),
[LeRobot](LEROBOT_TRAINING_GUARD.md) and [OpenPI](OPENPI_TRAINING_GUARD.md) guides.
