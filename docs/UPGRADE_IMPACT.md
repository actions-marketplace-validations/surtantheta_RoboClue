# Native upgrade impact for your assets

`assess_upgrade_impact(manifest_path)` answers which named recording or training
recipe violates the criterion you retained before a change. It verifies actual
saved input/configuration files, native source files and boundary captures by
SHA256, checks both captures against that same criterion, and names the affected
asset and criterion-violating field. Snapshots are supplied by the caller and
can include explicitly declared reference adapters. The check performs no
upstream code execution or network work and establishes no causal attribution
to a dependency upgrade.

A package/version difference alone is insufficient. A changed inactive file
with preserved native values is HEALTHY. An unsupported consumer, revision,
source pin, sample or field cohort is UNKNOWN. A failing reference is UNKNOWN:
the checker cannot establish a new criterion violation against an already
failing baseline. The aggregate preserves counts of all three outcomes, even
when a retained criterion violation makes the overall result CHANGED.

## Supported evidence

| Profile | Qualified source | Criterion/capture |
| --- | --- | --- |
| `isaaclab-hdf5-xyzw` | IsaacLab v2.3.2 `37ddf626871758333d6ed89cf64ad702aef127d0` and v3.0.0 `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4`; exact original loader/base/EpisodeData and new math pins | Explicit named numeric fields, including declared quaternion ordering; `BoundaryContract` and `save_boundary_capture` |
| `gr00t-n1.7-unitree-g1` | GR00T `944fe70c4fd54e3fcb80a7d7258a328f9c061176`; exact original processor/state-action/collator pins | Existing `TargetContract` and native training target/mask captures |

The source hashes are explicit in `upgrade_impact.ISAACLAB_SOURCES` and
`QUALIFIED_SOURCES`. Other versions remain UNKNOWN, including a future version
that happens to preserve a supplied capture. These are coverage pins, not
version defaults or an inferred migration database. GR00T same-revision recipe
comparisons are labeled `configuration_comparison`; they are not upstream
upgrade evidence. The `isaaclab-hdf5-xyzw` profile name identifies qualified
source coverage; the caller's retained field-specific intent decides which
quaternion order is required at each field.

## Retain intent before changing a dependency

For training targets, reuse the original `TargetContract` and its independently
retained SHA256. Do not rebuild the reference from a failing candidate.

For explicitly named native fields:

```python
from roboclue import BoundaryContract, save_boundary_capture

# Each value is a finite numeric NumPy array [rows, channels]. These come
# from your observed original native boundary under explicitly declared intent.
contract = BoundaryContract.from_reference(
    reference_fields,
    intent={"quaternion_order": "XYZW", "episode": "demo_0"},
)
contract.save("criterion.json")             # Requires a new file.
print(contract.criterion_sha256)             # Retain separately before changing anything.
save_boundary_capture(reference_fields, "before.json")
save_boundary_capture(candidate_fields, "after.json")
```

Equality is exact, including dtype and shape. Order conversion, field mappings
and tolerances are not guessed. For a legacy WXYZ recording used with an XYZW
destination, an explicit reference adapter must be declared and reviewed before
testing the new loader. The example below uses such a declared adapter, not an
orientation heuristic.

## Bind the asset and compare

The manifest has exactly these fields. Paths may be absolute or relative to
the manifest's directory. `source_files` and `configs` map logical names to
actual files and retained SHA256 identities. `configs` can also bind a small
recording or other user asset consumed by the declared run; it is not parsed
to infer a native configuration.

```json
{
  "schema_version": 1,
  "kind": "native_upgrade_impact",
  "contract": {"path": "criterion.json", "sha256": "<independently retained SHA256>"},
  "assets": [{
    "asset_id": "my-recording/demo_0",
    "reference": {
      "consumer": "isaaclab-hdf5-xyzw",
      "revision": "37ddf626871758333d6ed89cf64ad702aef127d0",
      "source_files": {"datasets/hdf5_dataset_file_handler.py": {"path": "old/hdf5_dataset_file_handler.py", "sha256": "<SHA256>"}},
      "configs": {"recording.hdf5": {"path": "recording.hdf5", "sha256": "<SHA256>"}},
      "capture": {"path": "before.json", "sha256": "<SHA256>"}
    },
    "candidate": {
      "consumer": "isaaclab-hdf5-xyzw",
      "revision": "e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4",
      "source_files": {"datasets/hdf5_dataset_file_handler.py": {"path": "new/hdf5_dataset_file_handler.py", "sha256": "<SHA256>"}},
      "configs": {"recording.hdf5": {"path": "recording.hdf5", "sha256": "<SHA256>"}},
      "capture": {"path": "after.json", "sha256": "<SHA256>"}
    }
  }]
}
```

The abbreviated `source_files` maps above show the schema. For covered results,
include **every** required name/hash from `ISAACLAB_SOURCES[revision]`; omitting
base/EpisodeData/new math produces UNKNOWN. Hashes must describe the real files,
not placeholder values. Captures and criteria are bounded to 4 MiB, 128 fields
and one million numeric values; manifests cover at most 32 assets. Bound input
files are each limited to 4 MiB. Symlinks, FIFOs, duplicate JSON keys, changed
pins and changed inputs during assessment are refused. Large recordings need a
separately retained bounded selection; its identity and scope must be explicit.

```python
from roboclue import assess_upgrade_impact, write_upgrade_alerts

report = assess_upgrade_impact("impact-manifest.json")
receipt = write_upgrade_alerts(report, "new-ci-results")
raise SystemExit(receipt["exit_code"])
```

The new output directory contains deterministic `impact.json` and `alerts.md`.
Exit policy is 0 for HEALTHY, 1 for a retained criterion violation, and 3 for UNKNOWN.
Invalid input raises `InputError` separately. No issue, email, repository change
or external alert is sent. A CI runner can retain the files as its own artifacts.
Existing output directories are refused.

`affected_asset_ids` names actual assessed user assets. Each asset retains both
capture/config/source identities and its reference/candidate checks. Both the
full report and alerts retain `criterion_intent` and `comparison_basis`, also
present per asset. The neutral code `retained_boundary_criterion_violated`
identifies a mismatch against that intent. `native_boundary_delta` states its
comparison basis explicitly: an adapted reference can make its differences
different from raw old-versus-new native values.

`associated_config_changes` lists real file hash
changes associated by the caller with the run; it does not claim that each edit
caused the criterion violation. An unchanged recording can violate a declared
destination intent during an upgrade assessment. A changed file can be harmless
under the retained criterion. Neither outcome alone proves an intrinsic upstream
bug or an upstream-caused regression.

## Genuine pinned upstream transition

The local qualification uses the original public
[NVlabs/RoboLab bundled recording](https://media.githubusercontent.com/media/NVlabs/RoboLab/b6cfcf6ccd2113c4cd1216d35b33024a7d5691da/examples/recorded_data/RubiksCubeAndBananaTask/data.hdf5),
393,616 bytes, SHA256
`edf09c3fa8e0773694e3d4ea2174e10f7410571f29e874e1562570470eba064a`.
It executes intact original HDF5 loader/base/EpisodeData modules from
[IsaacLab v2.3.2](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py)
and [v3.0.0](https://github.com/isaac-sim/IsaacLab/blob/e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4/source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py),
plus the original new math module. Package namespaces avoid starting the
simulator; no loader implementation, function body or dependency is replaced
by a dummy implementation.

The retained intent requires both selected fields in XYZW. The old native
WXYZ fields are explicitly canonicalized by the declared reference adapter
before freezing criterion
`651976f11eb9c647f03cc9d4d83fba3da19af462e2cf7d533f6b5333d7309c5f`.
The new loader runs afterwards on the unchanged original recording. Its raw
`ee_pose/orientation` values are **identical** to the old loader's raw values.
The 2,592 differences below are against the caller's manually canonicalized
all-XYZW reference, not 2,592 native old-to-new value changes.

A second, separately declared legal criterion requires root-pose XYZW and
end-effector WXYZ. Its old reference applies only an explicit root quaternion
adapter and preserves the raw end-effector values. That criterion is frozen
before a fresh new-loader invocation. The unchanged public recording passes
this field-specific intent. It does not replace or clear the original all-XYZW
case; that original criterion and its violation/correction/replay remain intact.

| Assessed asset | Observed result under that criterion |
| --- | --- |
| Original RoboLab `demo_0` recording under v3.0.0, caller's all-XYZW intent | CHANGED under `retained_boundary_criterion_violated`: native `root_pose` becomes XYZW, while unchanged raw `/data/demo_0/ee_pose/orientation` stays WXYZ; 2,592 values differ from the manually adapted reference across 648 rows |
| New copy with both explicitly selected source fields converted to XYZW and `format_version=1`, plus an inactive render-file edit | HEALTHY under the same retained criterion |
| Unchanged original recording, separately frozen legal root-XYZW/end-effector-WXYZ intent | HEALTHY; raw end-effector preservation is legal under this explicitly field-specific contract |
| Future unqualified loader revision | UNKNOWN; version coverage is not inferred from equal values |

The correction is a supplied new recording for those explicitly declared
fields. It is not an automatic whole-file migration recommendation. The
criterion covers two named fields of one original episode. Other cameras,
objects, episodes and arbitrary quaternion conventions are outside its coverage.
Repair and repeat assessment preserve the original criterion bytes and pin.

The qualification is a CPU native **HDF5 loader boundary** test, with original
module source hashes and source-input postguards. It is not a whole-IsaacLab
installation, simulator startup, native physics replay or training-quality
result. Its Torch/NumPy runtime and read-only h5py dependency overlay are
recorded with the local receipt. The installed checker consumes the saved
criterion/captures and does not need those native runtime dependencies.
An initial missing-h5py attempt is retained separately from the passing run.
No independent practitioner completion, adoption, time saved, GPU savings or
physical benefit is established.
