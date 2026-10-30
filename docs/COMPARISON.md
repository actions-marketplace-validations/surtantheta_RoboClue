# Where RoboClue fits

RoboClue's strongest demonstrated capability is a supported native pipeline
accepting a plausible fault, followed by refusal before consumption and a
correction passing the unchanged retained criterion. It combines this with
dataset monitoring, software runtime triage and owned-asset upgrade monitoring.

Primary documentation reviewed on 26 October 2026:

| Existing tool | Capabilities to credit | Our opportunity |
| --- | --- | --- |
| [LeRobot](https://github.com/huggingface/lerobot), [GR00T](https://github.com/NVIDIA/Isaac-GR00T), [OpenPI](https://github.com/Physical-Intelligence/openpi) | Native processing, training/inference and stack-specific validation | A common diagnostic workflow around supported native boundaries and retained intent |
| [lerobot-doctor](https://github.com/jashshah999/lerobot-doctor) | Dataset diagnostics, policy-type gates, repair, CI and merge checks | Check covered post-preprocessing inputs actually reaching the model |
| [trajlens](https://github.com/Kunal-Somani/trajlens) | Linting, fixes, dashboard, CI baselines and live watch | Consumer-boundary diagnosis and same-criterion correction across supported stacks |
| [Calibra](https://github.com/Calibra-Robotics/Calibra) | Multi-format integrity/quality/coverage analysis, coresets, cards/watch and training-data integrations | Verify whether a specific supported consumer preserved intended transformations |
| [Foxglove MCP](https://docs.foxglove.dev/docs/agents/mcp-server) | Agent access to robotics recordings, layouts, scripts and playback | Supply learning-specific contradiction and correction evidence alongside visual investigation |
| [quackd](https://github.com/rokbenko/quackd) | Robot/policy interface checks and deployment integration | Dataset-to-training evidence and supported cross-stack diagnosis |
| [ART-Embodied](https://github.com/wandb/ART-Embodied) | Training/evaluation integration, traces and artifacts | A focused input-transformation diagnostic layer alongside experiments |
| [lerobot-lancedb](https://github.com/lancedb/lerobot-lancedb) | Conversion, defect auditing and native-item parity checks | Reusable source/consumer criteria for our supported incident workflows |

Static checkpoint/config linters also exist: [LeRobot #4517](https://github.com/huggingface/lerobot/issues/4517)
documents a reporter using one to find contradictory normalizer dimensions.
Testing, reference comparisons, dataset linting and robotics MCP access are
established ideas; those individual features are not exclusivity claims.

Among the tools reviewed, we did not identify another documented product with
RoboClue's exact combination. This is a dated documentation comparison, not an
exhaustive search or head-to-head benchmark. Users can build equivalent checks
with native tools and custom code. Our practical case rests on useful reusable
workflows, evidence quality and integration effort.

Public reports motivate the problem: [OpenPI #849](https://github.com/Physical-Intelligence/openpi/issues/849)
discusses normalization changes affecting a checkpoint, while [GR00T #489](https://github.com/NVIDIA/Isaac-GR00T/issues/489)
records a corrected example camera mapping. These are historical reports, not
claims that current upstream releases remain broken or that RoboClue solves
every reported symptom. [Our demonstrated scope](SCOPE.md).
