# RoboClue documentation

Start with one of these five guides. Most users need only the first two.

| Guide | Use it when |
| --- | --- |
| [Install and try it](INSTALL.md) | You are setting up RoboClue |
| [Check an artifact](CHECK.md) | You have a dataset, checkpoint, config or recording |
| [Offline demo](DEMO.md) | You want to see a fault corrected and verified in seconds |
| [Guard setup](GUARD_INIT.md) | You want to preserve intended inputs before a training change |
| [MCP](MCP.md) | You want your agent to use the eight read-only question tools |

[First five minutes](QUICKSTART.md) · [Features](FEATURES.md) ·
[Compatibility](SCOPE.md) · [Evidence](EVIDENCE.md) · [FAQ](FAQ.md).

Run at the point of use: [before recording resume](RESUME_GUARD.md),
[GitHub Action and pre-commit](AUTOMATION.md). For maintainers:
[numbered rules](rules/README.md), [framework versions and revalidation](FRAMEWORK_VERSIONS.md).

<details>
<summary>Specialist integrations and retained evidence</summary>

- Training: [GR00T](GR00T_TRAINING_GUARD.md), [LeRobot](LEROBOT_TRAINING_GUARD.md), [OpenPI](OPENPI_TRAINING_GUARD.md).
- Datasets: [health cards](HEALTH_CARDS.md), [recurring monitoring](HEALTH_MONITOR.md), [preflight](PREFLIGHT.md), [GR00T dataset contracts](GROOT_DATASET_CONTRACT.md).
- Runtime: [application recorder](RUNTIME_RECORDER.md), [event format and triage](FLIGHT_RECORDER.md).
- Upgrades: [owned-asset watch](UPGRADE_WATCH.md), [impact evidence](UPGRADE_IMPACT.md).
- Data interpretation: [action conventions](ACTION_CONVENTIONS.md), [units/frames/timebase](UNITS_MANIFEST.md), [declared robot units/cameras](ROBOT_DECLARATIONS.md), [simulator provenance](SIM_DATA_PROVENANCE.md).
- Existing artifacts: [recording salvage](RECORDING_SALVAGE.md), [apply a recovery copy](RECOVERY_COPY.md), [conversion criteria](CONVERSION_REFERENCE.md), [numeric consumer comparison](CONSUMER_SAMPLE_COMPARISON.md), [episode windows](EPISODE_WINDOWS.md).
- Other evidence: [saved native replay](NATIVE_REPLAY.md), [evidence readiness](CHOOSE_A_WORKFLOW.md), [tool comparison](COMPARISON.md).

Detailed validation receipts are retained for audit; they are not a required
reading sequence. Run `roboclue advanced --help` for specialist commands.

</details>

For contributors: [development](../DEVELOPMENT.md),
[contributing](../CONTRIBUTING.md), [roadmap](ROADMAP.md).
