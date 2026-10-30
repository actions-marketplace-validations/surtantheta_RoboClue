# Check which episode an action window actually reads

Version 0.8.0 checks native LeRobot v3 episode ranges against the dataset's numeric rows, then shows the action indices and padding selected by an explicit query. It can expose an incorrect metadata boundary even when two datasets have identical action/state files. Keep the native JSON and Parquet files; no new capture or dataset conversion is required.

Install the dataset reader, and MCP when using an agent:

```sh
python -m pip install '.[dataset,mcp]'
roboclue advanced audit-windows ./native-dataset
roboclue advanced audit-windows ./native-dataset --episode-index 0 --anchor-index 29 --delta-indices 0 1 2 3 4
```

`anchor-index` is the **absolute dataset index**, not an episode-relative `frame_index`. Query options are supplied together. Delta indices are explicit integer row offsets; no FPS, timestamp or clock alignment is inferred. `--json` returns the complete bounded report; `--markdown --output window-report.md` writes an explicitly requested new report file.

## A wrong boundary selects another episode

Consider a numeric reconstruction with episode 0 at global indices `[0,30)` and episode 1 at `[30,60)`. The data rows are unchanged, but episode 0's metadata says `dataset_to_index=33`. At anchor 29, the pinned native reader's clipping/padding rule gives:

| Delta | Requested index | Declared selected index | Declared `action_is_pad` | Actual selected episode | Corrected selected index | Corrected `action_is_pad` |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 29 | 29 | false | 0 | 29 | false |
| 1 | 30 | 30 | false | 1 | 29 | true |
| 2 | 31 | 31 | false | 1 | 29 | true |
| 3 | 32 | 32 | false | 1 | 29 | true |
| 4 | 33 | 32 | true | 1 | 29 | true |

The first declared window has **three unpadded actions from episode 1**. The final selected action is also from episode 1, but its padding flag is true, so it is listed separately from the unpadded witnesses. With the boundary corrected to 30, the five actions all select index 29 and the four future positions are padded.

The report identifies the exact metadata Parquet file, original-byte hash, physical row, column, observed value 33 and actual-span expectation 30. Its correction candidate changes that metadata cell only. The declared and corrected windows retain selected indices, padding flags, actual episode IDs, queried action values and native row locators.

The corrected window is a **what-if**, not an applied repair. Produce a separate corrected dataset copy using your ordinary native tooling, retain before/after metadata and data-shard hashes, and audit that copy:

```sh
roboclue advanced audit-windows ./corrected-dataset --episode-index 0 --anchor-index 29 --delta-indices 0 1 2 3 4
```

The diagnostic never edits or renames the original dataset. Correcting this numeric inconsistency does not establish better policy performance or a physical robot repair.

## Use the same check through MCP

Authorize the dataset's root directory:

```sh
roboclue advanced serve --root /absolute/path/to/native-dataset
```

Call the read-only `audit_dataset_windows` tool:

```json
{
  "dataset": "/absolute/path/to/native-dataset",
  "episode_index": 0,
  "anchor_index": 29,
  "delta_indices": [0, 1, 2, 3, 4]
}
```

An example request is: “Check this native dataset's episode ranges. Show the five-action window at absolute index 29 of episode 0, including padding and actual episode IDs. Identify any exact metadata contradiction and show a corrected-window what-if with source hashes. Keep policy and physical effects unknown.”

CLI, Python `policy_doctor.windows.audit_episode_windows`, and MCP call the same engine. MCP performs no writes, downloads, shell execution or robot commands. CLI report output is an explicit write to a new destination.

For a separate numeric recording comparison, use the discovered **Parquet file paths** with `compare_rollout_recordings`; that tool does not accept dataset directories. A payload comparison alone does not check the episode metadata used by the window audit. A fresh installed operator independently completed the diagnosis, explicit copy correction and relocated verification; its two initially refused directory comparisons remain recorded.

## What is verified before suggesting a change

The supported input is native `v3.*` metadata with the default/declared Parquet backend and standard `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` template. The checker reads `meta/info.json`, canonical `meta/episodes` shards and all bounded canonical data shards. An episode may continue into another data file; a start-shard pointer alone is not taken as complete coverage. Shared data files are read once for row identity.

Physical file/row order must agree with the absolute `index`, metadata physical row position must agree with positional episode lookup, and each episode's frames and global indices must be contiguous. Declared totals must establish complete coverage. The engine preserves native fields such as `state`; it does not rename them to `observation.state` or substitute them for `action`.

Only requested action rows are returned. Qualifying values require a native floating `features.action` declaration, its exact vector width and finite floating values. Missing or incompatible action evidence leaves values unknown. Source hashes are rechecked after reading; files changed during evaluation are refused.

## Controls, unknowns and limits

Healthy controls include episode 0 at index 10, where offsets 0–4 select indices 10–14 without padding; episode 1 at index 59, where future offsets pad its own last row; negative offsets at an episode start; a valid episode spanning data files; and a separately corrected metadata copy with no unpadded leakage.

Missing shards/metadata, insufficient budgets, duplicate/gapped/reordered global IDs, ambiguous episode ordering, unsupported backends or custom path templates do not establish a correction. They return unknown or refuse malformed inputs. An anchor that actually belongs to another episode is refused. A pass covers the named numeric checks only.

Defaults are 100,000 data rows and 64 Parquet files; supported maxima are 200,000 rows and 128 files. Footer, compressed/expanded byte and value budgets also apply. Queries allow up to 64 integer offsets. Declared plus corrected action values are limited to 4,096 returned cells; reduce the offsets if that limit is exceeded. Reports remain within 192 KiB, retaining at most 16 findings, metadata-patch and episode previews with exact counts. Larger or partial inputs are not silently certified.

The reference is pinned to LeRobot `53feaef37b67e8f4c23b6953e880ff535e39d6fe` for `_get_query_indices` / `delta_window`. Validation also runs the unmodified full native package: real metadata construction, reader activation and item/batch reads in a separate CPU overlay, without stubs or source edits. All 60 anchors are exercised for healthy/bad/manually corrected numeric snapshots. The actual bad reader produces nine unpadded cross-episode action-vector slots across four windows; healthy and corrected readers produce zero. Nine installed MCP window checks agree exactly with the full native reader, and their corrected what-ifs agree with the reader of the corrected copy. Exact scope and final artifact receipt (see [qualification summary](EVIDENCE.md)).

These inputs reconstruct a reported numeric mechanism; they are not original field data. The overlay reuses an unchanged CPU Torch installation and is separate from ordinary product installation. No independent practitioner savings, community adoption, video correctness, learned policy competence or physical cause is claimed. See the independent adversarial audit and versioned validation (see [qualification summary](EVIDENCE.md)).
