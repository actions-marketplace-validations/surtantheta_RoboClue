# A silent label shift before training

This example generates **entirely synthetic** numeric LeRobot v2.1 folders and GR00T N1.7 G1 saved-processor JSON. A finite one-frame-forward action-label shift looks structurally valid. Compare it with the analytic reference before using those labels for training.

Install this checkout into your active environment with its `dataset` and `mcp` extras, then run:

```sh
python -m pip install '.[dataset,mcp]'
python examples/silent_pipeline/run.py --output /tmp/roboclue-silent-pipeline
```

Choose a new output directory each time. The script makes four actual installed CLI calls and four actual stdio MCP calls. It prints the returned selected-value effects, component and overall statuses, CLI exits, source witnesses and report paths. It does not substitute expected verdicts for tool responses.

The generated folders are:

| Folder | Synthetic input change | Expected selected control |
| --- | --- | --- |
| `healthy` | Same analytic data and saved arithmetic as the reference. | `EQUAL`; comparison `pass`. |
| `shifted_labels` | `action[frame, coordinate]` uses `0.02 * (frame + 1) + 0.001 * coordinate`; states and auxiliary commands are unchanged. | `CHANGED`; comparison `error` under explicit equality. |
| `inactive_statistic` | One saved `q01` entry changes while `use_percentiles` is false. | Changed saved identity, `EQUAL` selected outputs. |
| `incomplete_processor` | The candidate processor's `statistics.json` is deliberately absent. | `UNKNOWN`. |

All requests select this example's N1.7 consumer explicitly (`--groot-version n1.7` in CLI commands, `"groot_version": "n1.7"` in MCP requests) and require equality for stored episode 72, physical anchor 0 and its 50-step action window. This criterion makes a supported observed difference an equality error. A comparison `pass` covers this selected arithmetic; the overall report can remain `unknown` and CLI exit 3 is retained. Missing evidence must remain `UNKNOWN`.

The example's expected controls are checked separately from the actual installed CLI/MCP responses. See the [quick start](../../docs/QUICKSTART.md) and [evidence](../../docs/EVIDENCE.md) for interpretation and qualification scope.

Each folder contains `commands.sh` for its real CLI invocation and `mcp_request.json` for `preflight_check`. `MCP_SETUP.txt` gives the read-only server command. To generate only these inputs and commands, use `--prepare-only`.

`results.json` links the actual reports. `reports/*.mcp.response.json` retains the SDK envelopes; the CLI JSON and invocations are also retained. `demo_validation.json` separately compares the demonstrated controls with the expectations above; the script exits nonzero if they are not demonstrated. Expected values never replace product responses. `input_manifest.json` binds the generated numeric inputs. A source witness identifies current physical rows and declared slices, not physical truth or historical recording fidelity.

No camera images, robot recordings, weights or third-party datasets are included. This example does not run training, the full native GR00T processor or a robot. Its 43 state coordinates, 35 selected action coordinates and supported saved arithmetic follow this repository's generated G1 fixture layout. Example code uses the repository's MIT license; dependencies retain their own licenses.
