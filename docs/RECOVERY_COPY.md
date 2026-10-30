# Recover a separate recording prefix

An interrupted recording can have readable episode rows, missing metadata,
unclosed Parquet files and stale dataset statistics at the same time. A file
copy alone can preserve those problems. The explicit recovery writer applies
the existing supported prefix plan and checks the result:

```sh
roboclue check ./interrupted --question recovery --write-copy ./recovered
roboclue check ./recovered --question recovery
```

`./recovered` must not exist, even as an empty directory or dangling link. The
original is retained byte for byte. The default check and all MCP tools remain
read only; this write is authorized only by the CLI flag or an explicit Python
call.

## Supported operations

The writer recomputes a plan from the current local LeRobot v3 source. It accepts
no plan file. A complete bounded file list and a nonempty retained episode
prefix are required. It can:

- Copy the plan's unchanged data, metadata, task table and referenced videos
  into new regular files, without links to the original.
- Update the listed info totals and train split, retain leading episode-metadata
  rows, and apply the plan's supported data-file pointer corrections.
- Trim the explicitly listed leading physical data rows, preserving Arrow
  schema metadata. Metadata range endpoints are inclusive; data trim endpoints
  are exclusive. The backend handles those different plan conventions.
- Rebuild global statistics from complete retained per-episode summaries,
  then compare supported numeric mean, standard deviation, minimum and maximum
  against the complete copied rows within the scan budget.
- With the separate opt-in below, regenerate missing metadata from intact rows
  and decoded video in a supported native rollover layout. Footer-less data
  stays excluded; the writer never repairs missing bytes.

Files outside the plan, including staging frames and encoder temporary output,
stay with the original. They are not copied or deleted. Complete but excluded
episodes may be candidates for the explicit extension below.

## Include intact episodes whose metadata was not flushed

```sh
pip install 'roboclue[recovery]'
roboclue check ./interrupted --question recovery --write-copy ./recovered --include-candidates
```

The default copy retains the already-verifiable metadata prefix. This opt-in
can append a contiguous run of complete candidate episodes. It currently
requires the standard LeRobot v3 path templates, one episode per data/video
file, persisted rollover size thresholds that account for that layout, one
episode-metadata file, and matching existing metadata anchors. Candidate rows
must have contiguous episode, frame and global indices, the declared frame
rate, and tasks found in the original task table. Each camera needs its own
matching complete video with the expected dimensions, frame count and timestamps.
Ambiguous mappings, custom splits, unsupported metadata fields, missing or
truncated files and exceeded budgets are refused before creating the output.

No neighboring image statistics are reused. The operation regenerates numeric
statistics from every retained and included row and image statistics from every
decoded RGB frame in that same prefix. Image channel summaries cover all pixels,
with exact linear per-episode histogram quantiles. These replace the original
sampled/precompression summaries; they do not reproduce those original values.
Global quantiles still use the native conservative episode envelopes. The
receipt records this changed method, included episodes and decoded byte count.
The opt-in decoder has a fixed 128 MiB decoded RGB budget, in addition to the
source, numeric-row and decoded-table budgets. It requires PyAV; the ordinary
prefix copy does not.

## Statistics have their own result

Copying the original `meta/stats.json` after excluding episodes can silently
retain their normalization values. The writer always replaces that aggregate
with a fresh aggregation of the retained episode summaries when their feature
families, shapes, finite values and counts are supported and complete.

The aggregation follows the pinned LeRobot
[native statistics function](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/datasets/compute_stats.py):
count-weighted mean and variance, minimum/maximum bounds, and native conservative
quantile envelopes. Those quantiles are bounds across episode estimates, not
exact quantiles over all frames. Image summaries keep their native sampling
semantics; no image is decoded to regenerate them.

`statistics.aggregation_status` describes the aggregation separately from
`statistics.row_comparison_status`. `statistics.status: pass` requires both
supported aggregation and the complete numeric-row comparison to pass. The
default copy does not decode image pixels or independently validate sampled
image summaries. If summaries are missing, malformed or
unsupported, the copy omits stale `stats.json` and returns UNKNOWN. Missing
statistics also appear in the candidate structural report. Restore or recompute
them using the intended native tooling before training qualification.

Numeric `stats.count` is checked against `info.total_frames`; sampled image and
video counts are excluded from that rule. A count contradiction blocks health,
recovery and resume checks. Changing only the count does not repair stale values.
A bounded or incomplete row cohort never becomes evidence that saved mean/std
are correct. Read `statistics_comparison` and
`coverage.observed_row_statistics_compared` for that distinction.
Some native float32 mean/std calculations lose precision, particularly when a
large offset hides a small spread. If native rounding can explain a difference,
that metric is UNKNOWN rather than a stale-statistics error or a widened PASS.
`unassessed_metrics` and `precision_unassessed` identify these cases. Independent
supported differences still fail the comparison. Recomputing statistics with
higher precision can resolve the uncertainty.

The report separates `recording_integrity`, `statistics`, `native_reload` and
`training_qualification`. Native reload is `not_run`; training qualification is
`not_assessed`. A structural PASS does not certify a policy's normalization or
authorize recorder resume.

## Copy safety and limits

The source and destination must be separate local trees. Existing outputs,
symbolic links in their traversed paths, and source symlinks, FIFOs, sockets or
devices are refused. Source files are hashed before planning and after copying;
unchanged copies are also hashed as written. No source file is opened for
writing, no hardlink is created, and existing output files are never overwritten.
A cooperating RoboClue process holds an advisory source-directory lock. Other
recorders do not necessarily honor it: stop recording before recovery. Detected
changes abort the operation; these checks are not a transactional snapshot
against an actively hostile concurrent writer.

Default bounds are 200,000 data rows, 1,024 total source files and 1 GiB total
source bytes, including files excluded from the candidate. Parquet rewrites
also require at most 64 MiB cumulative decoded metadata and rewrite tables and
bounded footer parsing.
Oversized or incomplete plans are refused before creating a destination.
Python callers can explicitly choose larger source bounds within the documented
function limits; the decoded-table bound remains fixed.

During creation the destination holds `.roboclue-copy-incomplete.json`.
Interrupted or failed operations leave this marker. Its mere presence, including
a link, FIFO or fabricated success contents, makes subsequent public salvage
and resume checks refuse the candidate. Return to the original and choose a
new output directory; do not erase the marker to assert success. After a full
copy, source-stability check and candidate recheck, RoboClue writes the sibling
`./recovered.roboclue-recovery.json` and removes the marker last. A receipt
collision also prevents output creation. The receipt includes source
hashes, copied/rewritten/omitted lists, result scopes and output hashes. A receipt
is evidence of that operation, not an attestation that files remain unchanged
forever; subsequent checks read the actual files. It contains local source
paths and stays outside the dataset root so a native dataset upload does not
automatically publish those paths. Review it before sharing it separately.

No recorder, robot, policy, network service or dataset upload is started by this
operation.

## Python

```python
from roboclue import write_recovery_copy

report = write_recovery_copy("./interrupted", "./recovered")
# Optional bounded regeneration of source-supported complete candidates:
# report = write_recovery_copy("./interrupted", "./recovered", include_candidates=True)
print(report["status"], report["verification"])
```

Check `status` and the separate verification fields. An UNKNOWN statistics result
does not become a training PASS because the row/header recheck succeeded.
