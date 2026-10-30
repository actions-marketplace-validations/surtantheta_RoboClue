# Compatibility and coverage

RoboClue qualifies named profiles. Familiar filenames and a recent framework
version do not imply support. Custom or missing evidence stays UNKNOWN.

## Training integrations

| Profile | Qualified behavior | Unassessed |
| --- | --- | --- |
| GR00T N1.7 `944fe70c4fd54e3fcb80a7d7258a328f9c061176` | Original native trainer, collation/loss input; one process, zero workers, documented dataset settings; portable offline installation and fault/correction/replay | Other versions/settings, later batches and policy competence |
| LeRobot `146d8474932c8b566717eb2738ba459337ab71c4` ACT | Original reader/preprocessor/update function and genuine Accelerator; selected deterministic state/camera witnesses; two workers | Full training CLI, arbitrary policies/augmentations/framework revisions |
| Separate LeRobot CPU/Gloo profile | Two ranks, native ACT in a custom optimizer loop; one faulty/unresolved rank blocks both before forward | Original distributed CLI, NCCL/FSDP, failed-rank recovery |
| OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479` PyTorch | Original transforms/loader and Observation conversion; actual model-input hook with a controlled Torch consumer | Pi0 forward/optimizer, JAX, workers and DDP |

The demonstrated retained banks use NumPy 1.26.4 for GR00T/OpenPI and 2.2.6 for
LeRobot. Preserve the bank's runtime rather than rebuilding a reference to bypass
a mismatch. The common lightweight package also passed independent core/full
installed profiles. Native frameworks remain separate dependencies.

## Other workflows

- Health monitoring scans explicit inventories. It is scheduling-ready local
  software, not an operated public Hub service or universal quality badge.
- Generic health cards omit GR00T compatibility checks. `consumer` is a label;
  explicit `groot_contract` selection or GR00T recipe/version configuration
  enables that dataset contract. A card does not certify native compatibility.
  `roboclue check DATASET --question groot` selects the GR00T question explicitly;
  detailed preflight options remain available under `roboclue advanced preflight`.
- Runtime recording covers synchronous software calls. Physical feedback,
  sensor-clock alignment, motor delivery and task success need separate evidence.
- Upgrade watch has a supported catalog. A version difference alone cannot
  establish that an owned asset is affected.
- Simulator-order inference can be wrong for ceiling/wall mounts; confirm the
  declared mounting/version assumptions before acting on an inferred verdict.

## Read the result

| Result | Meaning |
| --- | --- |
| PASS / healthy | The named bounded criterion held for assessed inputs. |
| WARNING | Review the supported advisory finding. |
| ERROR / changed | A supported contradiction or retained-criterion violation was observed. |
| UNKNOWN | Required evidence or supported coverage is missing. |

A component pass does not clear another UNKNOWN. A finding can remain ERROR even
when another component is unknown. Invalid input is separate from an assessment.
For example, "selected targets equal" means those covered values matched the
retained criterion. Overall UNKNOWN can still mean that state/camera inputs,
later batches or source evidence were not assessed. Keep the useful comparison
and inspect the named coverage gap before choosing a next action.
The main evidence commands use exits 0 (pass/advisory), 1 (error/changed), 2
(invalid invocation/input) and 3 (unknown); monitor locking adds exit 75.
`start` exits 0 for a produced readiness plan, and `demo-native` exits 0 when its
expected controls complete. Neither exit means that all evidence was assessed
or that the overall research question passed.

**Intent must be independently justified.** A mistaken reference can preserve a
mistaken transformation. A passing startup cohort does not certify later data,
physical calibration, a safe robot or successful behavior. Hashes bind observed
bytes; they do not authenticate history or independently supplied physical truth.
Without a reviewed retained reference, supported artifact checks remain useful,
but a new baseline from the suspect candidate cannot establish its past
correctness. [Choose immediate checks or prospective retention](CHOOSE_A_WORKFLOW.md).
