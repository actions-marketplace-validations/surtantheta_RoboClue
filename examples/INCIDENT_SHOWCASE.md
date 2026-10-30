# Same saved outputs, different commands

Run `python examples/create_incident_showcase.py /tmp/policy-doctor-incident` from the checkout after installation with the dataset extra. The destination must be new. The script writes only there and checks its expected results. It uses trusted in-memory fixtures implementing the capture API, not a trained model or an actual LeRobot runtime.

| Question | Evidence generated | Expected observation |
| --- | --- | --- |
| What is in the run? | `reports/investigation.md` and JSON | One captured processor pipeline and one native dataset; contract audit passes |
| Did the saved model outputs change? | `reports/recordings.md`, native Parquet `model_output` | No difference across three rows |
| Did decoded actions change? | Native `action` with explicit `frame_index` | Row 0, coordinate 0: 11 versus 13 |
| What changed in the captured processing? | `reports/parameters.md`, hash-bound safetensors | Pre/post action mean changed from 1 to 3 in coordinate 0 |
| Does the candidate reproduce expected commands? | `candidate/trace.json` | Saved replay fails, maximum command difference 2 |
| Does the proposed correction restore parity? | `corrected/trace.json`, `reports/corrected.md` | Fresh corrected capture passes saved replay and numeric recording comparison |
| Can unexplained state drift be diagnosed? | `reports/unknown_cause.md` | Matching command arrays, first state difference row 4; causal conclusion unknown |

The sample's shoulder decoder is `model_output * std + mean + reference_state`. For the first row: `0 * 2 + 1 + 10 = 11`; changing the mean gives `0 * 2 + 3 + 10 = 13`. The gripper is excluded from relative anchoring. These independently specified numbers establish the fixture's expected arithmetic.

The Parquet recordings retain `episode_index`, `frame_index`, `index`, `task_index`, `timestamp`, `action` and `observation.state`. An explicit extra `model_output` field stores supplied outputs. Different episode/global IDs remain visible as bookkeeping differences and do not replace the first signal discrepancy. The generated metadata is a tiny numeric format fixture, not a complete collected training dataset or evidence of episode-level completeness.

Matching sample IDs establish correspondence of supplied row labels only. The native `action` field in arbitrary datasets may be a target, demonstration or command at a different phase. This fixture explicitly stores decoded commands there; the reader never assigns that meaning automatically.

Reports identify exact parsed recording bytes, selected episodes, array coordinates and values. Runtime comparison verifies both capture manifests before and after comparing normalized values. Declared deployment semantics remain declarations. Dtype, model weights, hidden controller memory, queues and anchor freshness are outside equality scope.

The corrected result proves saved arithmetic parity in this fixture. It does not prove a live robot repair, a shared checkpoint identity, policy inference equivalence or diagnosis-time savings. The negative control guards against promoting first divergence to a physical cause.

Markdown is portable human evidence, with source paths and hashes. Copy the underlying captures/recordings as appropriate for independent reproduction; a report alone does not contain those artifacts. Inspect reports for local paths and sensitive values before sharing. The existing `bundle` command packages its documented metadata/statistics diagnostic scope; it does not bundle every recording automatically.
