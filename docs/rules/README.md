# Rule catalog

Find a supported failure by the decision you need to make. Rule IDs are stable
public names; diagnostic codes in existing reports remain unchanged.

```sh
roboclue advanced rules
roboclue advanced rules RC001
roboclue advanced rules RC005 --json
```

| Rule | Decision | CLI entry point |
| --- | --- | --- |
| [RC001](RC001.md) | Do recorded motor units match my declared robot? | `roboclue check DATASET --robot-command 'YOUR_DEPLOY_COMMAND'` |
| [RC002](RC002.md) | Did an Isaac Lab conversion preserve quaternion conventions? | `roboclue check SOURCE.hdf5 --converted EXPORT.hdf5` |
| [RC003](RC003.md) | Does my dataset match the selected GR00T metadata/config profile? | `roboclue check DATASET --groot-version n1.5 --groot-config-name CONFIG` |
| [RC004](RC004.md) | Is the declared delta transformation consistent? | `roboclue check DATASET --openpi-config NAME --openpi-source-root OPENPI_SOURCE` |
| [RC005](RC005.md) | Is this interrupted recording safe to resume as-is? | `roboclue check DATASET --question recovery` |
| [RC006](RC006.md) | Can these saved statistics describe one finite cohort? | `roboclue check DATASET --question dataset` |
| [RC007](RC007.md) | Do explicitly declared camera names correspond to dataset identities? | `roboclue check DATASET --robot-command 'YOUR_COMMAND_WITH_CAMERAS'` |
| [RC008](RC008.md) | Does the selected OpenPI adapter find its raw input keys? | `roboclue check DATASET --openpi-config NAME --openpi-source-root OPENPI_SOURCE` |
| [RC009](RC009.md) | Does supported FastWAM image processing preserve the model's input map? | `roboclue check CHECKPOINT --runtime-source-root LEROBOT_SOURCE` |
| [RC010](RC010.md) | Is the requested pretrained-load branch selected? | `roboclue check actual-initialization.json --fail-on error` |
| [RC011](RC011.md) | Do actual camera inputs retain their declared slots? | `roboclue check actual-camera-inputs.json --fail-on error` |
| [RC012](RC012.md) | Does declared environment/recipe evidence cross a supported boundary? | `roboclue check provenance-input.json` |
| [RC013](RC013.md) | Is the instruction in the explicitly selected task catalog? | `roboclue check CHECKPOINT --instruction 'YOUR_INSTRUCTION'` |

These thirteen entries group supported findings; they do not enumerate every
low-level check. A rule can emit ERROR, WARNING, UNKNOWN or a scoped PASS. Its
presence in this catalog is not a compatibility verdict. An unlisted diagnostic
code remains valid and keeps its original meaning.

Each page links primary upstream source and exact tested revisions rather than
inventing an issue number. Where an upstream issue is later available, add it
alongside the retained source evidence. Untested versions are not implied by a
familiar framework name.

Python discovery is read-only and does not execute checks or third-party code:

```python
from roboclue import get_rule, rule_catalog, rules_for_code

catalog = rule_catalog()
rule = get_rule("RC001")
assert rules_for_code("units.joint_mode_mismatch")[0]["id"] == "RC001"
```

[Add or extend a rule](../../CONTRIBUTING.md#add-or-extend-a-rule) ·
[Worked statistics contribution](CONTRIBUTION_EXAMPLE.md) ·
[Robot declarations and camera keys](../ROBOT_DECLARATIONS.md)
