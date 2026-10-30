# Dataset health cards

A health card packages RoboClue's existing bounded dataset preflight into a JSON report, a scoped SVG badge, and a short CI summary. It binds the reported result to the source revision or observed file manifest, caller-supplied consumer label, effective preflight configuration, coverage and evidence. It checks explicit local inputs or a list of at most 16 public Hugging Face repositories. It does not discover repositories, run a hosted monitoring service, or publish anything.

The compact health view displays otherwise passing video checks as **N/A** when coverage explicitly says `not_applicable` with zero declared video features. It also preserves the raw component status. A stored video-index `pass` in that case means the check did not apply; no video content was validated. Missing or incomplete coverage is not converted to N/A, and errors or unknowns remain visible.

The badge reads **bounded checks: pass / warning / unknown / error**. A pass applies to those checks and their documented coverage. The consumer name is a label supplied by the caller, not evidence that the named trainer or loader has been run. Use the preflight component reports to see which consumer contracts were actually requested and assessed.

## Python and local CI

The default path is offline and leaves the source unchanged. Native row checks require the dataset extra (`pyarrow`); installing `.[dataset]` provides it. Missing dependencies and unreadable/unsupported inputs remain unassessed or produce an input error rather than certifying the dataset.

```python
from roboclue import health_card, write_health_card

card = health_card(
    "./dataset",
    consumer="my-training-job",
    config={"checkpoint": "./checkpoint", "max_rows": 200000},
)
paths = write_health_card(card, "./ci-artifacts/dataset-health")
print(card["status"], paths)
# An example strict local CI policy; the original verdict stays in the report.
raise SystemExit(0 if card["status"] == "pass" else 1)
```

`write_health_card` creates a new directory containing `health-card.json`, `badge.svg` and `summary.txt`. It refuses an existing output path. Retain these as CI artifacts; displaying the SVG does not automatically refresh it. A badge's accessible title includes the source, consumer, revision, card ID and limitations. No remote badge service or tracking URL is used.

`config` accepts these existing preflight options:

| Purpose | Options |
| --- | --- |
| Intake and scan bounds | `max_bytes`, `max_rows`, `max_files`, `video_headers` |
| Checkpoint/deployment declarations | `checkpoint`, `checkpoint_revision`, `deployment` |
| GR00T metadata contract | `groot_modality_config`, `groot_config_name`, `groot_version`, `groot_contract` |
| Additional recorded-data checks | `units`, `commanded_reach`, `openpi_config`, `openpi_source_root`, `groot_embodiment` |

Every accepted option's effective default is recorded. Generic health cards default to `groot_contract=False`: the presence of `meta/modality.json` does not select a GR00T dataset-format requirement. Request `config={"groot_contract": True}` explicitly to include that contract. Supplying `groot_modality_config`, `groot_config_name` or `groot_version` also enables it when the Boolean is omitted, since those options explicitly select the contract. Explicit `groot_contract=False` together with one of those selectors remains an input error. The separate `preflight` command keeps its existing automatic GR00T behavior.

The CLI also accepts `roboclue advanced health-card DATASET --groot-version n1.5` (or `n1.7`) to assess the actual intended consumer version.

Other options retain [preflight](PREFLIGHT.md) semantics and limitations. Supplying a consumer label never selects a contract, even when that label names a framework. A selected checkpoint or embodiment retains its established interface/action-convention checks without automatically selecting the GR00T dataset-format contract. Disabling an optional check does not establish it, and `video_headers=False` leaves declared video timing unassessed. Missing statistics or unsupported episode coverage can still make a generic card UNKNOWN; numerical or video contradictions still make it ERROR. `fetch` and `revision` are separate API arguments and cannot be smuggled through `config`. Robot execution and model inference are not performed.

In the compact view, `groot_contract.enabled` records the card's effective Boolean setting, while `recipe_selectors_supplied` records whether a modality config, config name or version was supplied. The legacy embedded preflight field `coverage.groot_contract_requested` means those selectors were supplied. It can be false while the contract is enabled through `groot_contract=True`; it does not mean the contract was disabled. Missing historical selection metadata remains unknown. These display fields leave the original report, counts, card ID and verification behavior unchanged.

## Explicit public-HF batch

Network intake requires `fetch=True`; only the existing public, ungated, pinned Hub intake is used. It does not use credentials, execute repository code, or download model weights. A mutable branch such as `main` is resolved once for each source and the actual 40-character commit is retained. Supply an immutable revision for reproducible CI. Content byte limits cover the existing metadata/Parquet intake; optional video headers have the separate bounds reported in preflight provenance.

```python
from roboclue import scan_health_cards

batch = scan_health_cards(
    [
        {
            "source": "hf://lerobot/berkeley_autolab_ur5",
            "revision": "c4e26a697fc4c04776b0558f83e563d14be0109f",
            "consumer": "dataset-input-review",
            "config": {"max_bytes": 16 * 1024 * 1024},
        },
        {"source": "./dataset", "consumer": "local-input-review"},
    ],
    fetch=True,
    output_dir="./ci-artifacts/health-batch",
)
print(batch["counts"])
```

Each entry is a source string or a dictionary with `source` and optional `revision`, `consumer`, `config`. Top-level `consumer` and `config` supply defaults; an entry's config replaces the batch config. Sources are checked sequentially, with a hard limit of 16 entries. There is no whole-Hub enumeration or aggregate healthy-dataset verdict. Local entries can be scanned alongside explicit Hub entries, but their own network mode is recorded as batch-authorized when `fetch=True`.

The output directory contains a compact `index.json` and one numbered subdirectory per source with the same three artifacts as the single-source API. Invalid request structures fail before any scans. Checker/input errors for individual entries produce an `unknown` card with an `intake_error` and allow later entries to run. The single-source API raises `InputError` for invalid invocation. Transport failures represented by preflight retain their independent `dataset_intake` findings and coverage.

## Binding and interpretation

- `source_revision.kind == "hub_commit"` records the exact public dataset commit that was resolved and read. An unresolved or mismatched revision prevents a pass.
- Local `observed_artifacts_sha256` hashes the checker's actual file manifest, including paths and file hashes. It is not a complete directory-tree digest. A local source does not accept a caller-invented Hub revision.
- `card_id` hashes the whole envelope, including the request, effective configuration, source identity, scope, limitations, verdict and complete embedded preflight report. Changing the consumer, scan bounds, evidence bytes or optional checks changes that ID.
- `verify_health_card(card)` checks the envelope's consistency and derived verdict. `render_badge(card)` and artifact writing refuse inconsistent cards. Verification does not reread current source files: regenerate the card to detect source changes.
- Previously retained cards keep their recorded effective configuration, including `groot_contract=True`. Verification and rendering do not reinterpret them under a newer default. A new generic scan has a different effective scope and card ID; retain the old evidence when comparing results.
- Missing required components, unsupported statuses, partial coverage, timeouts and missing source identity cannot produce green. A supported error remains `error` even when another component is incomplete. A download limit or transport failure is `unknown`, not an intrinsic data defect.

These hashes detect accidental modification. They are not signatures, independent public-origin authentication, or protection against someone constructing and resealing fabricated evidence. A card cannot establish whole-dataset quality, semantic labels, video decoding, native trainer compatibility, training success, robot health, physical safety or task success. The embedded preflight report retains the precise limits of each completed component. Local paths and supplied consumer/configuration details are included in local artifacts; review those before sharing.

## Validation scope

The test module exercises real local Parquet inputs, a recorded numeric defect, partial scans, missing statistics, unsupported folders, output preservation, consumer/config/content binding, SVG escaping, malformed-card rejection, explicit-network gates, a pinned-intake control, transport timeout, missing/unsupported components and batch error continuation. These controls test this feature; they are not new public incidents or measured community benefit.

A selected live public scan and a deliberate download-budget control are retained privately with the release validation receipts. Their actual status and inspected row count are recorded in those receipts. This bounded demonstration does not represent a Hub census, a deployed public service, adoption, saved training cost or a physical-system experiment.

The selected live check of `lerobot/berkeley_autolab_ur5` at `c4e26a697fc4c04776b0558f83e563d14be0109f` read **97,939 rows** and downloaded **5,537,181 metadata/Parquet bytes**. Saved statistics and numeric row checks passed; episode-window and video-index checks reported `video.interval_past_end` errors, so the retained badge is red **bounded checks: error**. A separate one-byte download-budget control remained **unknown**. These are existing recorded dataset boundary errors surfaced through the new card API, not newly discovered incidents or a repair claim. MP4 header response bytes use their separate reported budget and are not included in the metadata/Parquet byte count.
