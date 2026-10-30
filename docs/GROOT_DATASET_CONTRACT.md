# GR00T dataset contract

RoboClue's static preflight component checks selected dataset and training-config behavior against two pinned Isaac-GR00T releases: N1.5 `4af2b62` and N1.7 `944fe70`. These releases read LeRobot v2 metadata differently. Findings identify their version, source citation and affected path: fine-tuning, direct loaders, or both. This component does not execute training, a model or supplied config code.

Both profiles are included. Static dataset/configuration checks have a different scope from the native training guards; see [evidence](EVIDENCE.md).

## Choose the version and training config

```sh
roboclue check ./my_groot_dataset --question groot --json --output ./groot-report.json
roboclue check ./n17_dataset \
  --groot-version n1.7 --groot-config ./so100_config.py
roboclue check ./n17_dataset --groot-version n1.7 \
  --groot-config ./Isaac-GR00T/gr00t/configs/data/embodiment_configs.py --groot-config-name libero_sim
roboclue check ./n15_dataset \
  --groot-version n1.5 --groot-config-name so100_dualcam
roboclue check ./n15_dataset --groot-version n1.5 \
  --groot-config ./custom_data_config.py --groot-config-name MyArmConfig
```

`--groot-version` accepts `n1.5` or `n1.7`; omit it for the defaults below. There is no `auto` or `both` CLI value.

| Inputs | Profiles checked |
| --- | --- |
| Explicit version | That one profile |
| No version, unambiguous training-config file | The profile identified by its form |
| No version, known N1.5 builtin config name | N1.5 |
| No version, ambiguous/unsupported config form or unknown name without a file | UNKNOWN, no profile's rules applied; provide a supported config and explicit version |
| No version and no config/name | Both N1.5 and N1.7, with separate verdicts |

N1.5 config files define `BaseDataConfig` subclasses. `--groot-config-name` selects a class or known map key; a known builtin alone needs no file. A custom `module:Class` needs that module's actual `.py` file; a matching builtin class name does not establish what another module defines.

N1.7 config files contain `ModalityConfig` dicts with bare state/action/video group names. `--groot-config-name` selects a variable or `MODALITY_CONFIGS` key in the file; without a name, the supported registered dict is used. Explicit N1.7 plus a config name requires the file too. Text is parsed statically, not imported or executed.

`roboclue check DATASET --question groot` selects this question, as do the
explicit GR00T recipe/version flags. Passing `meta/modality.json` directly also
routes to the dataset's GR00T contract. A generic dataset card does not assume
GR00T compatibility. The detailed `roboclue advanced preflight` interface runs
the component automatically when `meta/modality.json` exists; its
`--no-groot-contract` (MCP `groot_contract=false`) skips that automatic component.
Do not combine an opt-out with explicit GR00T selectors.

## Read the version-specific result

**An ERROR from any checked version makes the combined contract ERROR.** The headline, `fails_on`, `versions` and each finding's `groot_version` identify which consumer fails and how. An N1.5-only error does not claim N1.7 fails; an N1.7-only error does not claim N1.5 fails.

The three-line verdict names every checked profile. An unversioned inspection
retains the conservative combined exit status; select your actual
`--groot-version` before using it as that consumer's CI gate. `coverage.complete`
means the selected static metadata checks completed, not that the separate
`not_checked` items (native rows, decoded cameras, model execution or training)
were verified.

For example, a cached statistic with an incompatible one-element `count` can fail when N1.5 opens the dataset, while N1.7 recomputes that cache. Conversely, a state/action column declared `object` can be omitted from N1.7 fine-tuning statistics and fail when a training config reads it. These are different paths and conditions. Check the intended version/config and retain the other profile's stated scope.

Without a training config, config keys, camera pairing and dependent processing checks remain incomplete. N1.5 opens every state/action/video modality entry, so opening faults can hold regardless of selected keys. N1.7 entry-level findings without a config are conditional on reading the cited entries; dataset-wide required-file faults apply to any config. A profile with no error and omitted config coverage is not a training-compatibility certificate. Reports retain the per-version `statement`, `applies_to` and `coverage.not_checked`.

Normal preflight exit codes are 1 for ERROR, 3 for unresolved checks and 2 for invalid input. Other preflight components retain their own verdicts. `--fail-on` changes the automation threshold, not what a version-specific finding establishes.

## Profile-specific checks

| Area | N1.5 `4af2b62` | N1.7 `944fe70` |
| --- | --- | --- |
| Slices and metadata | Validates modality schema and opens every state/action/video entry. Out-of-range NumPy index arrays can raise; negative indices can wrap silently. | Checks declared vector widths/slices and the entries selected by a supported config; over-wide slices can truncate silently. |
| Cameras | Pairs config keys to metadata by name, without positional fallback; checks supported original-key associations. | Checks the pinned loader's name/position pairing and supported original-key associations. |
| Statistics | Valid cached statistics can be retained without a fingerprint, including incorrectly wide statistics from another source; malformed caches can be recomputed. | Checks cached widths/fingerprints and relative-action keys, distinguishing fine-tuning regeneration from direct-loader reuse. |
| Config and processing | Reads supported class attributes and selected literal transforms: key/normalization/rotation constraints, concat omissions, model widths and horizons. Runtime modality metadata supports state/action/video; normalization statistics support state/action only. | Reads supported literal ModalityConfig dicts, registration/config keys, language and relative-action declarations. |
| Language/metadata | Checks supported annotation resolution and the pinned loader's task/episode parsing. Annotation entries in modality JSON do not make them valid StateActionTransform runtime metadata. | Checks supported annotation resolution, tasks and episodes under its own loader rules. |

N1.5 custom hooks or class-body-assigned methods, instance-set attributes, decorated classes and dynamic expressions can leave dependent checks UNKNOWN. A filtered or mutated `super().transform()` is not assumed to preserve its parent's transforms. For StateActionTransform on video, a plainly returned literal transform sequence can establish the required tensor input; an unproved ordering stays UNKNOWN. A no-op after `VideoToTensor` is not the same as applying it to raw NumPy frames.

## Limits and bounded reports

- This contract reads metadata/config text, not Parquet contents or video frames/codecs. Native decoding, physical camera identity, complete model/embodiment targets, historical training consumption and task success require separate evidence.
- Static parsing is intentionally limited. Rebound or dynamically assembled N1.7 dicts, custom transform helpers and arbitrary runtime behavior are not inferred from familiar names alone. Read each profile's `coverage.not_checked`.
- N1.5 transform checks do not establish all native transform semantics. Rotation representation versus actual slice width, full wrapper/model processing and decoded video are separate limits stated in coverage.
- Hub sources do not run this local contract automatically. Explicit GR00T selectors on a Hub source give an UNKNOWN naming the local input needed; download the dataset and pass its local directory.
- Findings are bounded examples, with a normal target of 50 examples/128 KiB. Error and unknown examples are retained by version/code/status; `counts`, `total_findings`, `finding_codes` and `versions` describe more than the listed examples when `findings_truncated` is true. Evidence may be trimmed. A shorter preview is not a complete list of affected entries.
- At the overall preflight report bound, details can be reduced, stubbed or left out with their status/counts and reason in coverage. Version identities and failure status remain material. Smaller `--max-files` can reduce competing row-report detail; it does not promise an unbounded metadata finding list. Read any truncation/refusal remedy and rerun on smaller inputs as appropriate.

Hugging Face cache metadata symlinked into the same repository's `blobs/` is supported by the contained metadata rules. Other escapes are refused. Under MCP, every resolved input/config path must also remain within current configured read roots.

## MCP and Python

Use the existing `preflight_check` tool:

```json
{
  "source": "/absolute/path/to/artifacts/n15_dataset",
  "groot_version": "n1.5",
  "groot_config_name": "so100_dualcam"
}
```

For N1.7, supply `groot_version: "n1.7"` and `groot_modality_config` with the training file; `groot_config_name` optionally selects its dict. Omitted/null `groot_version` keeps config detection or the no-config combined default. `groot_contract: false` skips the automatic component. Concise/detailed views must preserve the version-specific verdict and affected path.

For a cache snapshot, authorize the repository cache directory containing both `snapshots/` and `blobs/`, rather than only the snapshot. Saved reports grant no additional read authority.

The Python combined API is `check_groot_dataset_versions(..., groot_version=None)`; it accepts `None`, `"n1.5"` and `"n1.7"`. The legacy single-profile `check_groot_dataset` retains its N1.7 default and separate detect-only `"auto"` behavior. Those defaults are intentionally different; use the combined API for the preflight selection behavior described here.
