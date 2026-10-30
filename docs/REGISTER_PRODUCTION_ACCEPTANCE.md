# Publisher-card acceptance, 29 October 2026

Five cards were authored and locally qualified separately from the integrating
reviewer. The reviewer independently read each pinned publisher's command and
checkpoint context, fetched all four source files afresh and verified every
quoted line. CPU job141125 independently checked30 controls: published value,
deliberate alternative, missing field, unsupported container, unrecognized
checkpoint and explicitly unrelated complete stack selection for each card.
The resulting confirmed bytes are the imported files; pending author files
and the first review-harness failure remain preserved outside the package.
The first harness expected an unknown checkpoint alias to be excluded; the
engine correctly retained UNKNOWN. The corrected harness tests that behavior
and separately tests an explicitly unrelated stack as not applicable.

The author's earlier CPU141073 covered49 fixture and49 actual CLI/default-MCP
cases,24 semantic mutations and120 malformed/value probes. CPU141083 replayed
all71 original catalogue cases and preserved a mixed existing ERROR/new WARNING
through CLI and MCP. These receipts establish candidate content behavior;
the final assembled source and wheel require their own qualification.

## LEROBOT-RECIPE-FASTWAM-LIBERO-NSTEPS

`huggingface/lerobot@72e1d9e5e76ee719bacfc78bc34d817bee5fb521`,
`docs/source/fastwam.mdx`, binds the exact checkpoint and horizon in its
reproduction command. Compare explicit typed `n_action_steps` with10.

## LEROBOT-RECIPE-PI05-LIBERO-NSTEPS

The same LeRobot pin, `docs/source/libero.mdx`, names the exact finetuned
checkpoint in the evaluation command and sets `n_action_steps` to10.

## RECIPE-LIBERO-OPENVLA

`openvla/openvla@c8f03f48af692657d3060c19588038c7220e9af9`, `README.md`,
names the Spatial checkpoint in the Spatial command and explicitly enables
center cropping. Compare the resolved boolean `center_crop` with true.

## RECIPE-SIMPLERENV-BRIDGE-GROOT-N17

`NVIDIA/Isaac-GR00T@c56f54128d92f743889b7e4cecfc1ceafd88739e`,
`examples/SimplerEnv/README.md`, pairs its named Bridge checkpoint server
example with a client using `n_action_steps`4.

## RECIPE-SIMPLERENV-FRACTAL-GROOT-N17

The same NVIDIA pin and file pairs its named Fractal checkpoint example with
a client using `n_action_steps`1.

All five findings are capped at WARNING. They compare one declared actual
setting with one pinned published recipe. They do not verify that a server
actually loaded the declared checkpoint, infer absent flags, measure policy
scores, predict robot failure, or attest framework compatibility. Deliberate
alternative recipes remain valid experiments. Full external source URLs and
line selectors are embedded in each packaged entry.
