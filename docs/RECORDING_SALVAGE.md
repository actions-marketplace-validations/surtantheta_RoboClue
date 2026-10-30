# Recording salvage

`roboclue check DATASET --question recovery` (MCP `check_recording_salvage`) describes the retained prefix of an interrupted local LeRobot v3 recording and the evidence behind its exclusions. The check reads bytes only. With the explicit CLI option `--write-copy NEWDIR`, RoboClue can apply a supported prefix plan to a new directory, rebuild statistics from complete retained episode summaries, and recheck that copy. The original stays unchanged. See [Recovery copies](RECOVERY_COPY.md) for the supported operations and limits.

The reference is the pinned LeRobot writer and loader, `146d847` (`lerobot-record`, `dataset_writer.py`, `dataset_metadata.py`). Each finding cites the upstream lines it rests on.

## Why

An interrupted `lerobot-record` session (a hard kill, Ctrl-C or an exception while encoding video, a full disk) leaves states that the pinned LeRobot either refuses with an unrelated error, or loads without any error:

- Rows written before their metadata: `save_episode()` writes the data rows before the episode metadata, so an interruption in between leaves rows that no metadata row describes. LeRobot loads silently, and a later `--resume` serves stale rows (observed natively: 14 of 20 resumed rows never served).
- Footer-less Parquet and unflushed metadata after a hard kill: LeRobot fails with a Hub request, `ArrowInvalid` or `DatasetGenerationError`, never naming the file.
- A destructive resume: after a hard kill of a multi-file recording, `--resume` succeeds and replaces the data and video files of committed episodes whose metadata was never flushed.

## What it checks

Info and stats JSON validity, Parquet footers, the task table size against the largest `task_index` of the rows, the episode-metadata rows, the data-row identity columns (`index`, `episode_index`, `frame_index`), staging and encoder leftovers (`images/`, `tmp*` directories) and, by default, the MP4 header boxes of local videos. The report holds:

- `episodes`: the episodes to keep (always a prefix `0..K-1`, because LeRobot looks metadata up by row position), each dropped episode with its reasons and evidence, episodes that need renumbering, committed episodes that cannot be verified, and staging-only episodes.
- `plan`: the strategy (`none_needed`, `none_needed_rows_only`, `keep_as_is`, `metadata_copy`, `rebuild_required`, `not_metadata_only` or `unknown`), the corrected `info.json` fields, the files to copy unchanged, rewrite or omit, pointer corrections, and whether a later `--resume` on the copy is safe, with its blockers.
- `findings`: `salvage.file_unreadable`, `salvage.episode_metadata_missing`, `salvage.episodes_dropped`, `salvage.episodes_need_renumbering`, `salvage.feature_unlocatable`, `salvage.metadata_plan`, `salvage.not_metadata_only`, `salvage.resume_hazard`, `salvage.leftovers`, `salvage.nothing_committed`, `salvage.stats_missing`, `salvage.tasks_unusable`, `salvage.video_file_truncated`, `salvage.video_unchecked` and `salvage.coverage_unknown`. Each finding cites the existing detector codes that report the same condition (`detected_by`).

## What it cannot do

The limits below are the engine's own statement (`report['scope']`), restated:

- No Parquet page is parsed without a footer: rows inside a footer-less file are committed-but-not-verifiable, not recovered. Recovering them needs a byte-level footer rebuild on a separate copy, or re-recording.
- No video is decoded, and no row values other than the identity columns and `task_index` are read. No recording intent is inferred. With identical identity values, the data-file pointer decides (evidence, not proof).
- Recoverability of episodes without metadata rests on encoder temp-file names with their MP4 header sample counts, on staging frame-file counts, or, for committed episodes beside an existing `videos/` directory, on the recorder writing unbatched video before metadata. Their frames are not located in the video files; verify they decode before relying on a recovery.
- Orphan rows are excluded from the default retained metadata prefix. A complete, bounded healthy row cohort is compared with supported saved numeric mean/std/min/max; partial cohorts and image/quantile values remain explicitly unassessed. Numeric saved counts must match the declared frame total. The explicit copy operation can trim a leading data prefix and aggregate complete retained summaries. The separate `--include-candidates` opt-in can regenerate missing metadata only for intact rows and decoded RGB videos in a source-anchored native rollover layout. Footer-less rows are never recovered.
- Only native v3 metadata with the standard `data_path` template is supported. Absent videos leave video coverage UNKNOWN, never PASS.
- A symlink in the inspected dataset trees stops the check with the link and target named. A bounded inventory names all links there without traversing their targets. Use only the reported link-specific remedy: an eligible target is copied with `cp -rP`, preserving inner links for review on the next check. Recursive link-following of the entire dataset is not prescribed. Following links into separately authorized directories is unsupported.
- Bounds: `--max-rows` (default 200,000, at most 1,000,000) and `--max-files` (default 1024, at most 4096). An exceeded budget makes the ledger UNKNOWN; no episode is declared complete or droppable from a partial reading.

A physically evidenced disagreement between an episode span and its readable
episode run remains ERROR even when another defect prevents a complete repair.
`--no-video-headers` can withhold repair permission without changing that error.
Stored `index` values that disagree with positions, including noncontiguous
values, do not justify dropping intact episode runs: the report preserves the
verified prefix as evidence and refuses an automatic exclusion copy. Plan and
resume decisions remain separate from the severity of each observed defect.

Parquet intake validates each row-group row count and their sum against the
row budget and file-level count before reading pages. An inconsistent or
over-budget footer yields a bounded UNKNOWN with no copy plan; an allocation
failure is a named input refusal.
The default preflight episode audit shares these intake guards with its
consistency fallback. It retains independent ERRORs for impossible span bounds
and stored indices that contradict physical row positions, even when the
strict audit cannot derive a metadata correction.

## Run it

```sh
roboclue check ./my-recording --question recovery
roboclue check ./my-recording --question recovery --json --output salvage.json
roboclue advanced check-salvage ./my-recording/meta/info.json --markdown --output salvage.md
roboclue advanced check-salvage ./my-recording --no-video-headers --fail-on error
```

Exit codes: 0 for PASS or WARNING, 1 for ERROR, 3 for UNKNOWN, and 2 when no report was produced (a refused input, a missing optional extra). `--fail-on error` exits 1 for errors; `--fail-on warning` also counts errors; `--fail-on unknown` counts every non-PASS finding. Otherwise the selected gate exits 0.

The compact `check` answer focuses on recovery and current-directory resume
safety. The detailed specialist terminal view gives episode decisions, the plan
strategy and findings. `--json` prints the complete report; the specialist
`--markdown` option writes the complete plan and episode ledger. `--output FILE`
writes a new file only; a file inside the checked dataset is refused so the
recording stays byte-identical.

Inputs that are not a LeRobot dataset are refused by name: a folder without `meta/info.json`, a single data file, an HDF5 file (use `check-sim-data`), a FIFO or device, or an `hf://` path (the check is local only). The check needs the dataset extra (`pip install ".[dataset]"`); without it the command exits 2 and names the extra.

### Applying a plan

Use the explicit copy command for a supported retained prefix:

```sh
roboclue check ./interrupted --question recovery --write-copy ./recovered
roboclue check ./recovered --question recovery
```

The destination must not exist. RoboClue recomputes the plan from current source
files; an exported or edited plan is not an input. It copies planned unchanged
files as independent regular files, rewrites the listed metadata, performs
supported leading data-row trims, and rechecks the candidate. All excluded
files stay in the original. `episodes.drop` describes an exclusion from this
copy and never instructs deletion from the original.

The writer rebuilds `meta/stats.json` from complete retained native episode
summaries, including sampled image summaries without decoding images. Quantiles
use the pinned native conservative envelopes; they are not exact pooled
quantiles. If summaries are missing or unsupported, the copy omits stale global
statistics and reports UNKNOWN. The complete copied numeric rows are also
compared against saved mean/std/min/max. A healthy structural check on its own does not
verify normalization statistics or approve training. See the copy receipt's
separate `verification` fields and [full limits](RECOVERY_COPY.md).

`episodes.recovery_candidates` identifies complete row runs with surviving
frame evidence. The read-only report does not claim they were decoded or
reconstructed. `--write-copy NEWDIR --include-candidates` explicitly opts into
the bounded source-supported regeneration described in [Recovery copies](RECOVERY_COPY.md).
All retained and included image summaries are then recomputed from actual decoded
frames; a neighboring summary is never borrowed. If `plan.required_data_rewrite` names physical row
ranges, the copy command supports leading-prefix trims while preserving Arrow
schema metadata. Other required rebuilds are refused before creation.
Recheck the candidate and load it with the
intended native reader.

`plan.files.lists_complete` describes the scoped trees and current bounded
listing, not a recipe for deleting files from a full copy. All listed video
files must be accounted for as copy or omission. The plan's
`plan.validity.resume_safe` is a conditional prediction for the assembled copy
after every operation. It never approves the current interrupted directory.
Read `resume_safe_applies_to`, `copy_verification_required` and
`current_directory_resume_status`, then rerun
`roboclue check CANDIDATE --question recovery` and a native loader smoke test
before considering resume on the candidate.

### Resolve a named link before reading

The refusal identifies the link and its resolved target. A missing/looping target, a non-regular target, or a target containing the link/dataset itself does not get a copy command: restore the intended data instead. A regular file or non-recursive directory target can get a command that replaces only that named link.

For example, if the report names `/datasets/interrupted/videos` and its resolved directory target `/datasets/archive/videos`, the generated command has this form:

```sh
test ! -e /datasets/interrupted/videos.tmp && test ! -L /datasets/interrupted/videos.tmp && \
  cp -rP -- /datasets/archive/videos /datasets/interrupted/videos.tmp && \
  rm -- /datasets/interrupted/videos && \
  mv -T -- /datasets/interrupted/videos.tmp /datasets/interrupted/videos
```

Use the actual command returned for your named paths; it quotes paths as needed. Both temporary-path tests are required: a stale directory/file or dangling temporary symlink must stop the command before copying or removing the original link. Inspect and move previous temporary output aside before retrying; do not copy into it or silently overwrite it.

`cp -rP` copies this resolved target while preserving any links inside it. It does not recursively follow them. The next check inventories those newly visible links and refuses them for review, including upward/recursive links. An external target's contents are not inventoried through its outer link before copying. This command resolves a named intake blocker; it does not apply a salvage plan, certify the whole target, recompute statistics or establish resume safety. Rerun the checker after each reviewed replacement.

## MCP

`check_recording_salvage(dataset, video_headers=True, max_rows=200000, max_files=1024, response_format='concise')`. The dataset must lie inside the server's read roots. The tool is read-only and part of the default tool set; calls run off the server's event loop, so a slow input does not delay later requests.

- **Concise** (default, at most 8 KiB): the verdict and finding counts, the episodes to keep (ranges and count), the first dropped episodes with their reasons, the plan strategy, kept episodes and frames, corrected `info.json` fields, file counts and resume safety (with `applied: false`), and the findings that need attention.
- **Detailed**: the complete report. The engine bounds it to 192 KiB (long drop lists are cut and marked `drop_truncated`).
- Booleans and integers are strict (`video_headers="yes"` and `max_files=true` are refused), and unknown argument names are refused.
