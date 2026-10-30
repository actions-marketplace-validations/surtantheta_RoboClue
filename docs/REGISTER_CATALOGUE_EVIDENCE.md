# What the version and recipe register checks

The register contains twelve bounded entries. Supply an explicit complete set of
selected stacks and the requested version/config/command evidence; RoboClue
does not infer the effective environment of another process. Missing evidence
is UNKNOWN. A scoped match is not whole-pipeline compatibility.

| Entry | Supported comparison | Result when the predicate is met |
| --- | --- | --- |
| `MUJOCO-310-ROBOSUITE-140-CRASH` | Exact released robosuite1.4.0 and MuJoCo3.10.0 mass-matrix API pair | ERROR for that pair; other/future releases remain unassessed |
| `MUJOCO-333-LIBERO-RENDER` | Declared LIBERO-Object setup crosses the reported MuJoCo3.3.3 rendering boundary | WARNING; the threshold does not establish all-version reproduction |
| `MUJOCO-GE340-LIBERO-SPATIAL-INIT-OPENVLA` | Supported single-suite OpenVLA command crosses the measured MuJoCo3.4.0 initial-state boundary | WARNING about evaluation comparability |
| `ROBOTWIN-SAPIEN-COMMIT` | Declared LingBot-VA RoboTwin checkpoint uses a version at or above3.0.0, including later-version prereleases, rather than publisher3.0.0b1 | Prospective WARNING; the cited recovery changed both SAPIEN and RoboTwin commit |
| `RECIPE-LIBERO-GROOT-N17` | Explicit resolved `n_action_steps` differs from publisher value8 | Publisher-recipe WARNING |
| `RECIPE-LIBERO-OPENVLA-OFT` | Explicit resolved `num_open_loop_steps` differs from publisher value8 | Publisher-recipe WARNING |
| `RECIPE-LIBERO-XVLA` | Explicit resolved `init_seed` differs from publisher value42 | Publisher-recipe WARNING |
| `LEROBOT-RECIPE-FASTWAM-LIBERO-NSTEPS` | Exact `lerobot/fastwam_libero_uncond_2cam224`, explicit typed `n_action_steps` versus10 | Publisher-recipe WARNING |
| `LEROBOT-RECIPE-PI05-LIBERO-NSTEPS` | Exact `lerobot/pi05_libero_finetuned`, explicit typed `n_action_steps` versus10 | Publisher-recipe WARNING |
| `RECIPE-LIBERO-OPENVLA` | Exact `openvla/openvla-7b-finetuned-libero-spatial`, typed `center_crop` versus true | Publisher-recipe WARNING |
| `RECIPE-SIMPLERENV-BRIDGE-GROOT-N17` | Exact `nvidia/GR00T-N1.7-SimplerEnv-Bridge`, typed `n_action_steps` versus4 | Publisher-recipe WARNING |
| `RECIPE-SIMPLERENV-FRACTAL-GROOT-N17` | Exact `nvidia/GR00T-N1.7-SimplerEnv-Fractal`, typed `n_action_steps` versus1 | Publisher-recipe WARNING |

The source/report citations and pinned revisions are retained in each packaged
entry under `src/policy_doctor/register/entries/`. Version declarations do not
authenticate installed source bytes. The crash ERROR is deliberately limited
to the exact independently checked release pair; it makes no prediction about
other versions or locally patched installations.

The rendering and initial-state warnings describe risk beyond their pinned
public measurements. They do not claim every newer release was reproduced.
Recipe warnings mean the declared setting differs from the published evaluation
setup; results cannot be compared on that basis. Deliberate alternative recipes
are allowed. No warning predicts physical failure or a particular success rate.

Read the [input schema and limits](PROVENANCE_REGISTER.md) and
[publisher-recipe semantics](REGISTER_RECIPE_SCHEMA_DELTA.md). Inputs use the
existing `roboclue check INPUT.json` and default MCP `preflight_check` routes.

Independent content validation on29October2026 covered71fixtures with actual
CLI/default-MCP gates,34valid semantic mutants and184bounded malformed/control
probes. These are content-check results, not simulator-performance or practitioner
benefit measurements. Final package qualification is recorded separately.

The five additional publisher cards have a separate
[independent acceptance record](REGISTER_PRODUCTION_ACCEPTANCE.md). The original
35-item draft inventory is not a shipped coverage claim: twelve entries are
loaded, and the remaining proposals require more evidence, are redundant with
existing checks, or cannot support a reliable observable predicate. Missing
source/config facts remain UNKNOWN rather than being inferred from a model name.

The register does not ship the proposed inferred-intent OpenPI loader rule:
identical files can describe deliberately unused settings or unintended scratch
initialization. Use the [explicit-intent initialization guard](PIPELINE_CONTRACTS.md)
for that supported native boundary instead.
