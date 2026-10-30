# Simulator data provenance

`roboclue check RECORDING.hdf5` (MCP `check_sim_provenance`) checks the conventions that decide how a simulator demonstration file is read back: the quaternion order of an Isaac Lab HDF5, the controller that robomimic builds for a robosuite file, and LeRobot features named as quaternions. Each of these can make training data wrong without any error. The check is offline and read-only: no simulator, converter, network or model runs.

## What it checks

| Convention | Pinned upstream | What goes wrong silently |
| --- | --- | --- |
| Isaac Lab quaternion order | Isaac Lab v3.0.0 `e3f7d9ff` (byte-identical to main `c7d50a6` for the cited files) | The root `format_version` attribute alone declares the order (absent or below 1 = legacy WXYZ, 1 = XYZW). Isaac Lab 3.0 reorders only datasets named `root_pose` when it loads or converts a legacy file, and `merge_hdf5_datasets.py` and `mp4_to_hdf5.py` drop the attribute. A merged XYZW recording can silently rotate loaded poses (180 degrees in the identity-root reproduction); a converted file can retain other quaternions (end-effector, cameras, objects) in WXYZ. |
| robomimic controller versus actions | robomimic v0.5.0 `ae5799f`, robosuite 1.4.1 `c848ca8` and 1.5.1 (`a071383`, PyPI 1.5.1 `osc.py` byte-equal) | robomimic builds the controller from the file's `env_args`. robosuite 1.4 flat configs use `control_delta` (default true); robosuite 1.5 composite configs ignore it, and `input_type` (default `delta`) and `input_ref_frame` (default `base`) decide. Absolute actions under a delta controller replay as nonsense (0/10 successes natively on the Diffusion Policy `low_dim_abs` files). |
| LeRobot declared quaternions | LeRobot `146d847` | A feature whose names declare quaternion components (`qx, qy, qz, qw`, `rx, ry, rz, rw`, ...) should hold unit quaternions. `lerobot/libero_10` names its state that way, but 0% of 101,469 rows have unit norm: the values are axis-angle plus a second finger position. |

Quaternion checks require the reported mounting and provenance assumptions. Wall- or ceiling-mounted data can evade the residual heuristic, including an overall PASS; PASS does not establish the original mounting or quaternion convention. Mixed source-version evidence requires the per-version original recordings. Do not stamp or reorder a mixed merged file as a whole.

Inputs:

- `SOURCE`: a robomimic or Isaac Lab HDF5 file, or a LeRobot dataset directory (or its `meta/info.json`) when no HDF5 exists.
- `--converted PATH` (optional): an HDF5 derived from `SOURCE` (merge, convert or augment output), or the LeRobot dataset converted from it. Pairs are compared per dataset and demo: the cyclic reorder that maps each source quaternion to the converted one, a row and its negation counting as the same rotation. A robomimic HDF5 with a LeRobot conversion is traced to the action stream (`actions` or `actions_abs`) that the converted actions equal.
- `--max-demos N`: by default every demo is inspected (up to 4096). A cap selects N demos evenly spaced over the file. A check never passes on partial coverage.
- `--max-rows N`: LeRobot rows read (default 200,000, at most 5,000,000).

## What it cannot do

The limits below are the engine's own statement (`report['scope']`), restated:

- A quaternion order is not inferable from values in general. Content evidence is used per demo, and only when a robot articulation root reads upright under exactly one order. Robots are assumed upright at reset unless mounted tilted; robots mounted on a ceiling or wall read upright under the other order, so remedies that rest on the roots state the upright-mount condition. Undecidable checks are UNKNOWN or conditional, never PASS, and no remedy prescribes reordering data on undecidable evidence.
- The robomimic guard needs `obs/robot0_eef_pos` and covers single-arm robosuite OSC controllers (`OSC_POSE`, `OSC_POSITION`). Multi-arm, other controllers and variable impedance are UNKNOWN.
- Source-to-converted checks exist only for Isaac Lab HDF5 to Isaac Lab HDF5 (quaternion order) and robomimic robosuite HDF5 to LeRobot (action stream and controller convention). Any other pair is reported as `conversion.not_compared` (UNKNOWN), with each input checked on its own.
- A converter that transforms actions (scaling, filtering no-op steps, reordering) cannot be traced.
- Gripper sign and range, and the contract of a downstream consumer, are out of scope.
- Reads are bounded: HDF5 files up to 64 GiB are hashed in full and read through the same handle; soft and external links, virtual and external datasets and non-builtin filters are refused without being opened; 4 million values per dataset and 40 million in total. An exhausted budget is reported as `sim.read_budget_exhausted` (UNKNOWN).

## Run it

```sh
roboclue check merged.hdf5
roboclue check recording.hdf5 --converted merged.hdf5
roboclue advanced check-sim-data low_dim_abs.hdf5 --fail-on error
roboclue advanced check-sim-data demos.hdf5 --converted ./lerobot-dataset --json --output provenance.json
roboclue advanced check-sim-data ./lerobot-dataset --markdown --output provenance.md
```

Exit codes follow the rest of the CLI: 0 for PASS or WARNING, 1 for ERROR, 3 for UNKNOWN, and 2 when no report was produced (a refused input, a missing optional extra, an unreadable file). `--fail-on error|warning|unknown` exits 1 when any finding has that status (warning also counts errors; unknown counts every non-pass), otherwise 0.

The terminal view prints every finding with its full message, remedy and first evidence items. `--json` prints the complete report. `--markdown` creates a portable report with every remedy list written out. `--output FILE` writes to a new file only (an existing file is never replaced, and a file inside a checked LeRobot dataset is refused); the file always holds the complete report, including every remedy list.

The HDF5 checks need the dataset extra (`pip install ".[dataset]"`, which brings h5py and pyarrow). Without it the command exits 2 and names the extra.

## Remedy lists

A remedy that applies to many datasets refers to `report['remedy_lists'][key]`, for example `pair.fixes` or `source.non_root_quaternion_datasets`. Each list holds groups (relation, `np.roll` shift, quaternion columns, demos). Dataset and demo name lists are stored once in `remedy_lists['names']` and referenced by id. In Python:

```python
from roboclue import check_sim_provenance, remedy_list

report = check_sim_provenance('recording.hdf5', converted='merged.hdf5')
for entry in remedy_list(report, 'pair.fixes'):  # one entry per dataset, names resolved
    print(entry['dataset'], entry['quaternion_columns'], entry['roll'], entry['demos'])
```

A partial conversion of a few hundred demos with many cameras produces lists of several hundred kilobytes. The terminal view abbreviates them past 40 entries and says so. `--json`, `--markdown` and `--output FILE` always contain them in full.

## MCP

`check_sim_provenance(source, converted=None, max_demos=None, max_rows=200000, response_format='concise', remedy_list=None, offset=None, limit=None)`. Both paths must lie inside the server's read roots. The tool is read-only and part of the default tool set.

Response-size limits below measure UTF-8 compact JSON for the report object. They are not transport-byte limits: an MCP client or SDK can also serialize an indented text representation and protocol envelopes, which add bytes.

- **Concise** (default, at most 8 KiB of compact JSON): the verdict and counts, the findings that need attention (the most severe first), the codes of passed checks, the inputs with their sha256, and a digest of the remedy lists. A shown remedy is kept whole. If it cannot fit, the whole remedy is explicitly withheld with its SHA-256 and a route to the full CLI report; conditions are never cut away from a shown instruction. Per list, the digest gives the number of groups and entries, the first groups and names, and `list_sha256`.
- **Detailed**: the complete report when it fits 192 KiB. A larger report (a partial conversion of 400 demos with 200 cameras is about 525 KB) keeps every finding and the coverage, sets `remedy_lists` to null, names the omitted lists in `remedy_lists_omitted` and adds `remedy_lists_digest`, `truncated_from_bytes` and `detailed_bounded: true`. The lists are never dropped silently.
- **Complete lists in bounded pages**: call the tool again with the same inputs and `remedy_list=<key>`. Each page returns `entries` (a slice of `remedy_list(report, key)`), `offset`, `returned`, `entries_total`, `next_offset` (null on the last page), `complete`, `list_sha256` and `remedy_lists_sha256`. A page stops at `limit` entries (default 500, at most 5000) or at the 192 KiB bound, whichever comes first (`byte_bounded` says which). To verify, concatenate the `entries` of every page in offset order and compute

  ```python
  hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
  ```

  It equals `list_sha256` of the digest and of every page. `remedy_lists_sha256` is the sha256 of the canonical lists (the same encoding over `report['remedy_lists']`). Pages with different values come from different reads; restart at offset 0.
- **Reuse**: repeated calls on complete HDF5 file inputs with the same bounds reuse the server's last report (up to four are kept) only while every input file has the same device, inode, size, modification time and change time as when it was read. Otherwise the call reads again. `served_from` says `fresh_read` or `cached_report_same_file_identity`. LeRobot inputs are always read again, including the `meta/info.json` spelling; the identity of that one file does not establish that the recording is unchanged.
- Calls run off the server's event loop, so a long check does not delay later requests.
- `offset` or `limit` without `remedy_list`, an unknown list key, an offset past the end, booleans or strings for integers and unknown argument names are refused.

## Finding codes

`isaaclab.format_version`, `isaaclab.declared_order_vs_content`, `isaaclab.mixed_order_across_demos`, `isaaclab.quaternions_outside_root_pose`, `isaaclab.conversion_quaternion_order`, `robomimic.env_args`, `robomimic.controller_vs_actions`, `conversion.action_source`, `conversion.not_compared`, `lerobot.declared_quaternion`, `sim.read_budget_exhausted`, `sim.no_applicable_checks`. Every evidence item carries the file path, its sha256, an HDF5 or JSON pointer and the value read.
