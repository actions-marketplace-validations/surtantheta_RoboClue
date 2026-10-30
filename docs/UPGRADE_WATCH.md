# Watch upstream changes for owned assets

`roboclue advanced upgrade-watch scan-once` inventories explicitly named
local assets, reads their saved version declarations, observes a public GitHub
release or source ref, and saves local alerts. It is one explicit operation;
your own CI or scheduler can invoke it again. It installs no background service
and sends no issue, message or publication.

The three outcomes are **affected**, **unaffected** and **UNKNOWN**. A release
or version difference alone stays advisory/UNKNOWN. A configured native
comparison can establish whether your asset violates or preserves the exact
criterion you retained, including its field-specific intent and reference
adapter. This is not an inferred migration rule or proof of an upstream bug.

## Start with your owned configuration

First ask what evidence you retained. Owned files and version declarations let
you inventory bytes and observe upstream changes now. They do not determine
whether yesterday's evaluation assumptions were right. Impact needs the supported
native before/after comparison and independently retained criterion described
below. Without them, retain reviewed commits, configuration, source identities and
native captures during the next ordinary run; do not manufacture a retrospective
reference from the suspect current files. [Workflow chooser](CHOOSE_A_WORKFLOW.md).

Create a watch document like this:

```json
{
  "schema_version": 1,
  "kind": "owned_upgrade_watch",
  "watches": [{
    "watch_id": "isaaclab",
    "repository": "isaac-sim/IsaacLab",
    "kind": "release",
    "ref": "latest"
  }],
  "assets": [{
    "asset_id": "my-recording/demo_0",
    "consumer": "isaaclab-hdf5-xyzw",
    "watch_id": "isaaclab",
    "files": {"recording.hdf5": "owned/recording.hdf5"},
    "version": {
      "path": "owned/consumer.json",
      "format": "json",
      "selector": "/revision"
    }
  }]
}
```

Paths resolve beside this document. The owned `consumer.json` might contain
`{"revision":"v2.3.2"}`. The scanner records its real file SHA256 and selector
alongside the owned recording's path, hash and size. These are saved declarations,
not inferred loaded runtime versions. TOML supports the same explicit pointer
syntax, for example `/tool/consumer/revision`. A `requirements` version source
uses a package name as its selector and accepts only one exact `package==version`
line. A range such as `isaaclab>=2` does not identify a loaded version.

Use `kind: "commit"` with an explicit commit SHA, tag or branch for a source
watch. A `release` watch accepts an explicit release tag or the explicitly
requested value `latest`; the scanner resolves the observed tag to a complete
immutable commit. It retains the public API URL, observation time, exact response
body/hash and release metadata. A publication date is not mistaken for the
freshness of the current observation.

```bash
roboclue advanced upgrade-watch catalog
roboclue advanced upgrade-watch scan-once \
  --config watch.json --state-dir .upgrade-watch --json
```

The explicit catalog lists qualified source revisions and rule scope for the
existing IsaacLab field and GR00T target checks. The scanner does not guess rules
for another consumer, future release or unqualified source version. The known
IsaacLab v2.3.2/v3.0.0 aliases resolve to their pinned revisions. Other version
strings have no implicit compatibility default.

## Add retained native evidence

Add this optional object to an asset:

```json
"native_comparison": {
  "manifest": "native-comparison.json",
  "manifest_sha256": "<independently retained manifest SHA256>",
  "asset_id": "the-selected-asset-in-that-manifest",
  "criterion_sha256": "<the-original-retained-criterion-SHA256>"
}
```

The comparison uses the [upgrade-impact manifest](UPGRADE_IMPACT.md). Its
candidate consumer/revision must match the watched public upstream revision.
Every owned file must match both the **path and bytes** in one complete supplied
reference or candidate snapshot. A mixture assembled from unrelated before/after
files is refused. The comparison manifest and criterion pins are never silently
updated. Changed current files, changed evidence pins, unsupported source
coverage or a different watched revision produce UNKNOWN with a concrete next
step: capture the actual current pipeline and review a new comparison against
the original retained intent.

An affected alert identifies the owned asset and intended-field evidence,
preserves `criterion_intent` and `comparison_basis`, and asks you to review that
intent/adapter and replay a supplied correction under the **same** criterion.
An unaffected alert means that bounded criterion was preserved. It does not
certify the entire recording, model, robot or dependency upgrade.

In particular, IsaacLab's raw `ee_pose/orientation` can stay identical WXYZ
between the qualified old and new loaders. A manually canonicalized all-XYZW
reference flags a mismatch with that destination intent. A separately retained
legal root-XYZW/end-effector-WXYZ criterion can pass the unchanged recording.
Neither case establishes an intrinsic upstream defect. The watch preserves
this distinction in its reports and readable alerts.

## Repeated scans and offline behavior

Each completed scan writes immutable `scan-<id>.json` and `.md` files plus an
atomically replaced `state.json` index. The state directory is reserved for
scanner output. It retains the latest eight completed scans and removes older
managed reports. A process lock prevents concurrent writers and releases on
process exit. Existing report hashes are checked before continuing; tampering
requires preserving that directory and starting a new state location.

Every online scan makes a fresh upstream request. Earlier results are retained
as history and are never promoted into a new scan's observation. No conditional
ETag cache is used. Network failure, rate limiting, malformed/oversized metadata
or a failed refresh produces UNKNOWN even if the previous scan was unaffected.

```bash
roboclue advanced upgrade-watch scan-once \
  --config watch.json --state-dir .upgrade-watch --offline --json
```

Offline mode makes no network request and explicitly reports the upstream as
unobserved/UNKNOWN. It can inventory local files but does not certify that an
older release receipt is current. Run another explicit online scan when the
upstream is reachable.

The operation supports at most four watches, two public API requests per watch,
32 assets and 16 owned files per asset. Files/configuration/evidence and the
result are bounded to 4 MiB; each API response is limited to 256 KiB with a
five-second socket timeout. API redirects cannot switch away from HTTPS
`api.github.com`. No GitHub credentials or arbitrary API hosts are accepted.
Source/config/file postguards refuse changes during the scan. Large artifacts
need a separately declared bounded selection and suitable retained evidence.

Exit codes are 0 for wholly unaffected, 1 for any retained native criterion
impact, 3 for incomplete/UNKNOWN coverage, and 2 for invalid configuration/state.
Counts preserve all outcomes when an affected asset is present. The report and
alerts remain local; the scanner does not repair source, code, configurations
or data and does not schedule itself.

## Qualification scope

Automated controls cover affected and legal unaffected intent, unqualified
consumers/versions, saved configuration changes, requirements ranges, stale
native pins/files, failed refresh after a previous result, offline behavior,
release-to-commit resolution, bounded public responses, concurrency and bounded
history. A separate recorded CPU proof uses a new locally owned copy of the
original public RoboLab recording, the intact pinned new IsaacLab HDF5 loader,
fresh boundary captures and a real public GitHub commit observation. It retains
both original all-XYZW and separate legal field-specific criteria, with the raw
end-effector identity caveat above. This is a source/native-boundary automation
demonstration, not independent practitioner adoption, a whole-simulator trial,
a physical result or measured time saved.
