# Recheck an export under the same criterion

This workflow extends the supported Arena G1 original-recording check. It preserves an explicit criterion for later source/export comparisons. See the [qualification summary and limits](EVIDENCE.md).

Save a full preflight report while supplying the actual original and converter/config:

```sh
roboclue advanced preflight /workspace/robot-job/export-before \
  --original-recording /workspace/robot-job/recordings/run.hdf5 \
  --converter-source /workspace/robot-job/Arena \
  --converter-config /workspace/robot-job/actual_config.yaml \
  --json --output conversion-before.json

roboclue advanced preflight /workspace/robot-job/export-after \
  --conversion-reference conversion-before.json \
  --json --output conversion-after.json
```

Relative recipe references require an explicit `--converter-directory` on the first call. The saved record preserves that choice. Output files must be new; the command refuses existing files, including aliases of the baseline.

The later call restores only original/recipe selectors. It revalidates actual files under current read authority, recomputes the criterion and compares the later export. A known wrong initial export can establish the source criterion when the source basis is complete; a corrected export must then satisfy that same criterion. An incomplete or unsupported source cannot establish a reusable criterion. Other dataset warnings and unknowns remain visible.

The report separates criterion equality from candidate correctness. A matching criterion can accompany a numeric ERROR. Proven criterion drift prevents expected-criterion acceptance, while preserving separately labeled current-criterion observations. A refused or unestablished check cannot silently become a replacement baseline. To choose a new criterion deliberately, retain the old report and run a new check with explicit original/recipe selectors.

For an additional CI pin, supply `--expected-conversion-criterion SHA256`, using the digest already returned by the baseline. Replacing the saved file cannot silently replace that separately supplied pin. A self-consistent saved JSON record is not authenticated history; every recheck still needs the actual original and recipe.

## Through MCP

Call the existing `preflight_check` with the original selectors and `response_format="conversion_reference"`. This returns a generated reference view containing the complete criterion and selectors, without the large row-evidence records. It is explicitly a reference view, not a full preflight report. The client saves the returned JSON using its existing file capability; the MCP server remains read-only.

Then call:

```json
{
  "source": "/workspace/robot-job/export-after",
  "conversion_reference": "/workspace/robot-job/conversion-before.json",
  "expected_conversion_criterion": "<criterion_sha256 returned by the first call>"
}
```

Both the reference and all restored paths must fit the server's current read roots. Saved records cannot enable network, writes or native execution. Default concise results show the binding verdict, expected/current digests and numeric statuses. An unavailable or oversized reference view reports its availability as UNKNOWN rather than truncating the criterion; any known source ERROR remains visible. The normal detailed response can fall back to a summary when the full report exceeds 192 KiB; save the explicit reference view or use CLI `--json --output FILE` instead.

## What must stay fixed

This version binds original bytes and source selection, actual recipe/config bytes and paths, numeric interpretation, evaluator code, Python and dependencies. Candidate export paths/bytes may change. Original or recipe relocation, YAML comment edits and environment/verifier upgrades can cause drift; establish a new baseline explicitly when those changes are intended. Older reports without the generated reuse context are not upgraded by guessing missing inputs.

The small reference record does not contain the original recording or reproduce the earlier candidate. Source-to-export numeric agreement cannot establish source camera timing, historical production, task success or model/robot behavior. See the original association evidence and limits (see [qualification summary](EVIDENCE.md)).
