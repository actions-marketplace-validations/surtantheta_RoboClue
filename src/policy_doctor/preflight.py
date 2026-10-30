"""One bounded dataset check with independent, explicit coverage for each consumer contract."""

from __future__ import annotations

import copy
import hashlib
import json
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

from . import hub
from .artifacts import (
    LEROBOT_REVISION,
    MAX_JSON_BYTES,
    InputError,
    contained,
    dataset_root,
    load_dataset,
    load_deployment,
    read_json,
    sha256,
)
from .dataset import scan_dataset
from .diagnostics import _vector, audit, dataset_statistics_findings, kind
from .windows import audit_episode_windows

PRIORITY = ("error", "unknown", "warning", "pass")
SCOPE = (
    "Bounded native dataset rows, saved dataset statistics, episode references and optional MP4 header timing, plus declared "
    "checkpoint and deployment interfaces when supplied. No video decoding, model inference, training, "
    "robot execution, physical calibration or task success was tested. A pass applies only to the "
    "reported checks and coverage."
)
# Every file the GR00T dataset contract may open; it also follows Hugging Face cache snapshot symlinks.
GROOT_METADATA = ("meta/info.json", "meta/modality.json", "meta/stats.json", "meta/relative_stats.json",
                  "meta/episodes.jsonl", "meta/tasks.jsonl")
TITLES = {"groot_contract": "GR00T dataset contract", "commanded_reach": "Recorded-position residual triage",
          "units": "Units, frames and timebase", "action_conventions": "Action convention",
          "openpi_context": "OpenPI raw input keys", "camera_keys": "Camera input names"}
RESIDUAL_STATEMENTS = {
    "pass": "No residual pattern was found in the read rows.",
    "warning": "A residual pattern between recorded action and recorded observation.state was found in the read "
               "rows; see the cited joints.",
    "unknown": "Residual triage did not cover every joint or row; see the cited reasons.",
}
RESIDUAL_LIMITS = ("Recorded values only: what the motors received is not recorded, and neither a residual nor its "
                   "absence establishes calibration or robot health.")
GROOT_WITH_CONFIG = "Dataset metadata was checked against the supplied training modality config."
GROOT_WITHOUT_CONFIG = ("Dataset metadata only: no training modality config was supplied, so config keys, camera "
                        "pairing, language and relative actions were not checked. This is not a full GR00T "
                        "compatibility check.")
GROOT_NOT_READ = "Parquet contents, video frames, embodiment metadata and training were not checked."
GROOT_INCOMPLETE = "The GR00T dataset contract did not complete; see the cited reason."
SMALLER_INPUT = ("Rerun with a smaller --max-files (MCP max_files): row checks then cover fewer files, while the GR00T "
                 "contract reads only meta/ files and stays complete")
GROOT_REDUCED = ("Reduced form: with every finding this report would exceed its 768 KiB bound, so one shortened example "
                 "of each finding code and status is kept; status, counts and finding_codes cover every finding. "
                 + SMALLER_INPUT + " and lists every finding.")
GROOT_STUB = ("Status stub at the 768 KiB report bound: the status and counts cover every contract finding, but "
              "no finding is listed.")
GROOT_OMITTED = ("Even in reduced form this report would exceed its 768 KiB bound. " + SMALLER_INPUT
                 + " and lists every finding, or skip the contract with --no-groot-contract (MCP groot_contract=false).")
REDUCED = ("Reduced form: with every detail this report would exceed its 768 KiB bound, so one shortened example of "
           "each finding code and status is kept; status, counts and finding_codes cover every finding. "
           "Rerun with a smaller --max-files (MCP max_files) for the full component.")
OMITTED = "Even in reduced form this report would exceed its 768 KiB bound. Rerun with a smaller --max-files (MCP max_files)."
OPT_OUT = {"groot_contract": "--no-groot-contract (MCP groot_contract=false)", "units": "--no-units (MCP units=false)",
           "action_conventions": "a separate preflight call selecting only the intended consumer"}
REPORT_REFUSED = ("Preflight report exceeds its 768 KiB bounded transport budget. " + SMALLER_INPUT
                  + ", or request fewer optional checks.")
GROOT_MAX_FINDINGS, GROOT_MAX_BYTES = 50, 128 * 1024  # detailed contract findings kept, one example per code first
UNITS_MAX_BYTES, UNITS_LISTED = 128 * 1024, 16  # units component bound; coordinate records kept per non-motor feature
# Per-finding text bounds (JSON-escaped bytes): one finding can never take a component's whole budget.
TEXT_BOUNDS = {"message": 1536, "remedy": 768, "path": 768, "pointer": 768, "value": 768}
REPORT_LIMIT = 768 * 1024
UNITS_LIMITS = ("Units are inferred from the recorded values and coordinate names; LeRobot records none. A use_degrees "
                f"left unset takes the default of LeRobot {LEROBOT_REVISION[:7]}; other LeRobot versions may default "
                "differently. The policy (--policy.path) and its processors are not read, and calibration files and "
                "the arm are not inspected.")
CONVENTION_LIMITS = ("Recorded action/state numbers and names only; no model, consumer code or robot is run. Delta and "
                     "velocity differ only by a scale, and 'absolute' allows a +-15% scale tolerance, so it does not "
                     "prove identical units.")
# Dataset-level fingerprint findings; every other action-convention finding is about the named consumer.
FINGERPRINT_CODES = {"convention.fingerprint", "convention.action_copies_state", "convention.declared_conflict",
                     "convention.rotation_declared_conflict"}


def _status(values):
    return next((s for s in PRIORITY if s in values), "unknown")


def _unknown(component, reason, *, artifacts=None):
    return {"status": "unknown", "coverage": {"complete": False}, "artifacts": artifacts or [],
            "findings": [{"code": f"preflight.{component}_unassessed", "status": "unknown",
                          "message": str(reason), "evidence": []}]}


def _run(component, operation):
    from .fs_guard import DatasetLayoutError

    try:
        report = copy.deepcopy(operation())
    except DatasetLayoutError as exc:
        if exc.tree == 'meta':
            # Cache metadata aliases may remain unassessed by this stricter native
            # row reader without erasing independently authorized metadata facts.
            return _unknown(component, exc)
        raise
    except Exception as exc:
        return _unknown(component, exc)
    if not isinstance(report, dict) or report.get("status") not in PRIORITY:
        return _unknown(component, "The checker did not return a supported assessment status.")
    return report


def _dataset_statistics(source):
    """Assess the declared statistics conditionally; missing/uninterpretable values remain unknown."""
    root, observed = dataset_root(source), []
    try:
        # Preserve bounded source identities even when load_dataset refuses malformed JSON.
        for relative, role in (("meta/info.json", "dataset_info"), ("meta/stats.json", "dataset_stats")):
            path = contained(root, relative)
            if path.is_file():
                with path.open("rb") as stream:
                    raw = stream.read(MAX_JSON_BYTES + 1)
                if len(raw) > MAX_JSON_BYTES:
                    raise InputError("Dataset statistics intake exceeds the native JSON byte bound")
                observed.append({"path": str(path), "role": role, "sha256": hashlib.sha256(raw).hexdigest(),
                                 "size_bytes": len(raw)})
        snapshot = load_dataset(root)
        artifacts = [asdict(a) for a in snapshot.artifacts]
        if artifacts != observed:
            raise InputError("Dataset metadata changed while statistics were read")
        if not any(a["role"] == "dataset_stats" for a in artifacts):
            return _unknown("dataset_statistics", "meta/stats.json is absent; saved dataset statistics were not assessed.",
                            artifacts=artifacts)
        stats = snapshot.config["stats"]
        findings = dataset_statistics_findings(snapshot)
        considered, available, unresolved = [], [], []
        for key, feature in snapshot.features.items():
            if not key.startswith(("observation", "action")) or kind(feature, key) == "VISUAL":
                continue
            considered.append(key)
            entry = stats.get(key)
            if not isinstance(entry, dict):
                unresolved.append(key + ": missing or unsupported statistics entry")
                continue
            # Use the corrected interpreter's exact numeric acceptance rule, without duplicating its math.
            vectors = [_vector(entry[name]) for name in ("mean", "std", "min", "max", "q01", "q99", "q10", "q90")
                       if name in entry]
            paired = any(lo in entry and hi in entry for lo, hi in
                         (("mean", "std"), ("min", "max"), ("q01", "q99"), ("q10", "q90")))
            widths = {len(v) for v in vectors if v is not None}
            expected = feature.get("shape")
            wrong_width = isinstance(expected, list) and len(expected) == 1 and widths != {expected[0]}
            if not vectors or not paired or any(v is None for v in vectors) or len(widths) != 1 or wrong_width:
                unresolved.append(key + ": unsupported or inconsistent numeric statistic vectors")
            else:
                available.append(key)
        if unresolved:
            findings.extend(_unknown("dataset_statistics", "; ".join(unresolved))['findings'])
        return {"status": _status([f["status"] for f in findings] or ["pass"]), "findings": findings,
                "artifacts": artifacts, "coverage": {"complete": not unresolved,
                    "features_considered": considered, "features_with_statistics": available,
                    "unassessed": unresolved, "observed_row_statistics_compared": False,
                    "normalization_mode": "conditional; no policy mode inferred"},
                "scope": "Saved native dataset statistics and conditional normalization consequences; not recomputed row statistics."}
    except Exception as exc:
        return _unknown("dataset_statistics", exc, artifacts=observed)


def _input(value, name, fetch, *, must_exist=True):
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a nonempty local path or hf://ORG/REPO")
    if "://" in value:
        hub._repo(value)
        if fetch is not True:
            raise InputError("Hub network is disabled; pass --fetch (MCP: enable the appropriate Hub operator permission) for preflight")
    else:
        local = Path(value).expanduser()
        try:
            exists = local.exists()
        except (OSError, RuntimeError) as exc:
            raise InputError(f"Cannot resolve {name.lower()} {str(local)!r}: {exc}") from exc
        if not exists and must_exist:
            raise InputError(f"{name} does not exist: {str(local)!r}. Check the path and rerun.")
        from .fs_guard import refuse_special_file

        refuse_special_file(local, name.lower())


def _inside(path, read_roots):
    return read_roots is None or any(Path(path).is_relative_to(Path(root).expanduser().resolve())
                                     for root in read_roots)


def _within(path, read_roots, label):
    """Refuse a resolved local path outside explicit read roots before it is opened."""
    if not _inside(path, read_roots):
        raise InputError(f"{label} is outside the configured read roots")


def _groot_config(value, read_roots, base_directory):
    """Authorize the GR00T modality config path before any intake; it is parsed later, never executed."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise InputError("The GR00T modality config must be an explicit local .py path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_directory or Path.cwd()) / path
    path = path.resolve()
    _within(path, read_roots, "The GR00T modality config")
    if not path.is_file():  # refused rather than dropped, even where the contract would stop before reading it
        raise InputError(f"The GR00T modality config is not a file: {path}")
    return str(path)


def _groot_layout(source):
    try:
        return (dataset_root(source) / "meta/modality.json").is_file()
    except OSError:
        return False


def _config_identity(config):
    from .groot_contract import MAX_CONFIG_BYTES

    path = Path(config)
    if not path.is_file() or path.stat().st_size > MAX_CONFIG_BYTES:
        return None  # The contract refuses it with the exact reason.
    return {"path": str(path), "role": "groot_modality_config", "sha256": sha256(path),
            "size_bytes": path.stat().st_size}


def _groot_contract(source, config, config_name, read_roots, groot_version=None):
    from .groot_contract import check_groot_dataset_versions

    root = dataset_root(source)
    for name in ("", *GROOT_METADATA):
        target = (root / name).resolve()
        if not _inside(target, read_roots):  # never echo where an outside link points
            raise InputError(f"GR00T dataset metadata {name or 'directory'} resolves outside the configured read "
                             "roots. For a Hugging Face cache snapshot, authorize the repository cache directory that "
                             "holds both snapshots/ and blobs/; access is not broadened automatically.")
    identity = _config_identity(config) if config is not None else None
    report = check_groot_dataset_versions(root, config, config_name=config_name, groot_version=groot_version)
    if identity is not None and _config_identity(config) != identity:
        raise InputError("The GR00T modality config changed while it was read")
    report["artifacts"] = [{"path": str((root / name).resolve()), "role": "groot_metadata", "sha256": digest}
                           for name, digest in sorted(report["files"].items())] + ([identity] if identity else [])
    report["coverage"].update({"complete": report["counts"]["unknown"] == 0,
                          "modality_config_name": report["modality_config_name"],
                          "metadata_files_read": sorted(report["files"]),
                          "parquet_rows_read": False, "video_frames_read": False,
                          "preflight_not_checked": ["Parquet contents", "video frames", "embodiment metadata", "training"]})
    # Keep the dispatcher's complete totals when bounding its already selected examples.
    # Preflight also bounds huge free-text keys/messages that can exceed the module cap.
    return _bound_contract(report)


def _rank(finding):
    status = finding.get("status")
    return PRIORITY.index(status) if status in PRIORITY else PRIORITY.index("unknown")


def _example_key(finding):
    """One example is kept per code and status (and per consumer version when a finding names one)."""
    return (str(finding.get("code")), str(finding.get("status")),
            str(finding.get("groot_version", finding.get("version", ""))))


def _tally(findings):
    """Every finding, by code and status: survives any cap, and next actions are derived from it."""
    tally = {}
    for finding in findings:
        by_status = tally.setdefault(str(finding.get("code")), {})
        status = str(finding.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
    return dict(sorted(tally.items()))


def _short(finding, bounds=TEXT_BOUNDS):
    """A copy whose free text fits the bounds; each cut keeps its prefix and the full text's SHA-256."""
    from .consumer_preflight import _preview, _text_digest

    item, cut = dict(finding), []
    for key in ("message", "remedy"):
        if isinstance(item.get(key), str):
            text = item[key]
            item[key], clipped = _preview(text, bounds[key])
            if clipped:
                cut.append(key)
                item[key + "_sha256"] = _text_digest(text)
    evidence = item.get("evidence")
    if isinstance(evidence, (list, dict)):
        listed = [evidence] if isinstance(evidence, dict) else evidence
        shortened = []
        for index, entry in enumerate(listed):
            if not isinstance(entry, dict):
                shortened.append(entry)
                continue
            entry = dict(entry)
            for key in ("path", "pointer", "value"):
                if key not in entry or entry[key] is None:
                    continue
                text = entry[key] if isinstance(entry[key], str) else json.dumps(entry[key], allow_nan=False, default=str)
                preview, clipped = _preview(text, bounds[key])
                if clipped:
                    entry[key] = preview
                    entry[f"full_{key}_sha256"] = _text_digest(text)
                    cut.append(f"evidence/{index}/{key}")
            shortened.append(entry)
        item["evidence"] = shortened[0] if isinstance(evidence, dict) else shortened
    if cut:
        item["text_truncated"] = cut
    return item


def _bound_findings(report, max_findings, max_bytes):
    """Keep at most max_findings findings within max_bytes: first one example of every non-passing code and status
    (its first instance), then the rest most severe first. counts, finding_codes and total_findings still cover
    every finding, so no counted code is left without a cited example while the cap allows."""
    findings = [_short(f) for f in report["findings"]]
    ranked = sorted(range(len(findings)), key=lambda i: _rank(findings[i]))
    examples, rest, seen = [], [], set()
    for i in ranked:
        key = _example_key(findings[i])
        if key in seen or findings[i].get("status") == "pass":
            rest.append(i)
        else:
            examples.append(i)
            seen.add(key)
    kept = (examples + rest)[:max_findings]  # the byte loop below drops the severity fill before any example
    report.setdefault("total_findings", len(findings))
    report.setdefault("finding_codes", _tally(findings))
    while True:
        report["findings"] = [findings[i] for i in sorted(kept)]
        report["findings_truncated"] = len(kept) < report["total_findings"]
        if len(kept) <= 1 or len(json.dumps(report, allow_nan=False).encode()) <= max_bytes:
            break
        kept.pop()
    expected = set(seen)
    for version, codes in report.get('finding_codes_by_version', {}).items():
        expected.update((code, status, version) for code, by_status in codes.items()
                        for status in by_status if status != 'pass')
    missing = expected - {_example_key(findings[i]) for i in kept}
    report["codes_without_example"] = sorted({f"{version + '/' if version else ''}{code}/{status}"
                                               for code, status, version in missing})
    return report


def _bound_contract(report):
    """Keep at most GROOT_MAX_FINDINGS detailed contract findings within GROOT_MAX_BYTES (see _bound_findings), so a
    large modality.json cannot push preflight past its report bound."""
    return _bound_findings(report, GROOT_MAX_FINDINGS, GROOT_MAX_BYTES)


def _reduced(name, part):
    """The smallest component that keeps the verdict: status, counts, the per-code tally and one shortened example per
    code and status. Used only for an automatic component that would push the report past its bound."""
    findings = part.get("findings", [])
    examples, seen = [], set()
    for finding in sorted(findings, key=_rank):
        key = _example_key(finding)
        if finding.get("status") != "pass" and key not in seen:
            seen.add(key)
            short = _short(finding, {"message": 320, "remedy": 200, "path": 256, "pointer": 256, "value": 120})
            evidence = short.get("evidence")
            short["evidence"] = (evidence[:1] if isinstance(evidence, list) else [evidence] if evidence else [])
            examples.append(short)
    reduced = {key: part[key] for key in ("kind", "title", "consumer", "modality_config_name", "units_limits",
                                          "compared_with", "convention_limits", "verdicts", "artifacts",
                                          "headline", "remedy", "mode", "groot_version", "versions_run", "fails_on", "versions")
               if key in part}
    reduced.update(status=part.get("status", "unknown"), reduced=True, findings=examples, findings_truncated=True,
                   counts=part.get("counts") or {s: sum(f.get("status") == s for f in findings) for s in PRIORITY},
                   total_findings=part.get("total_findings", len(findings)),
                   finding_codes=part.get("finding_codes") or _tally(findings),
                   coverage={**part.get("coverage", {}), "complete": False, "reduced_at_report_bound": True},
                   reduced_statement=GROOT_REDUCED if name == "groot_contract" else REDUCED)
    if name == "groot_contract":
        reduced["contract_statement"] = GROOT_REDUCED
    return reduced


def _omitted(name, part):
    """A short stub that still carries the component's status and counts: an ERROR stays an ERROR (and fails
    --fail-on error) even when not one example fits."""
    counts = part.get("counts") or {s: sum(f.get("status") == s for f in part.get("findings", [])) for s in PRIORITY}
    status = part.get("status", "unknown")
    text = GROOT_OMITTED if name == "groot_contract" else OMITTED + f" To skip it, use {OPT_OUT[name]}."
    tally = ", ".join(f"{counts.get(s, 0)} {s}" for s in ("error", "warning", "unknown", "pass"))
    omitted = {"status": status, "title": TITLES[name], "omitted_at_report_bound": True, "counts": counts,
               "total_findings": part.get("total_findings", len(part.get("findings", []))),
               "findings_truncated": True, "artifacts": part.get("artifacts", []),
               "coverage": {"complete": False, "omitted_at_report_bound": True},
               "findings": [{"code": f"preflight.{name}_details_omitted", "status": status, "evidence": [],
                             "message": f"{TITLES[name]}: {status} ({tally}); its findings are not listed. " + text}]}
    if name == "groot_contract":
        omitted["contract_statement"] = GROOT_STUB
        for key in ("headline", "remedy", "mode", "groot_version", "versions_run", "fails_on", "versions"):
            if key in part:
                omitted[key] = part[key]
        omitted["coverage"].update({k: v for k, v in part.get("coverage", {}).items() if k != "complete"})
    for key in ("units_limits", "convention_limits", "kind"):
        if key in part:
            omitted[key] = part[key]
    return omitted


def _commanded_reach(source, max_rows, max_files, read_roots, recorded):
    from .commanded_reach import check_commanded_reach

    root = dataset_root(source)
    _within(root, read_roots, "The dataset")  # every file it reads is contained in this root
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise InputError("Install policy-deployment-doctor[dataset] for recorded-position residual triage; "
                         "it reads Parquet rows") from exc
    report = check_commanded_reach(root, max_rows=max_rows, max_files=max_files)
    coverage = dict(report.get("coverage") or {})
    receipts = coverage.pop("data_files", [])
    read = {item["path"]: item["sha256"] for finding in report["findings"] for item in finding.get("evidence", [])
            if item.get("path") and item.get("sha256")}
    read.update({item["path"]: item["sha256"] for item in receipts})
    listed = [{"path": str(contained(root, name)), "sha256": digest,
               "role": "dataset_info" if name == "meta/info.json" else "recorded_position_rows"}
              for name, digest in read.items()]
    # A file another component already recorded with the same hash is not listed twice, which keeps large
    # datasets inside the report bound; any differing hash is listed so input_stability still reports it.
    report["artifacts"] = [item for item in listed if recorded.get(item["path"]) != item["sha256"]]
    report["coverage"] = {**coverage, "complete": bool(coverage) and report["counts"]["unknown"] == 0,
                          "data_files_read": len(receipts), "max_rows": max_rows, "max_files": max_files,
                          "files_with_matching_hash_in_other_components": len(listed) - len(report["artifacts"])}
    return report


def _units(source, robot_command, robot_config, read_roots, base_directory, max_rows, max_files, recorded,
           *, lerobot_version=None, require_consumer_version=False):
    from .units_manifest import units_manifest

    root = dataset_root(source)
    _within(root, read_roots, "The dataset")  # every file the manifest reads is contained in this root
    if robot_config is not None:
        from .readiness_workflow import _local

        robot_config = _local(robot_config, read_roots, base_directory, explicit="robot config")  # readiness authorization
    report = units_manifest(root, robot_command=robot_command, robot_config=robot_config,
                            max_rows=max_rows, max_files=max_files)
    coverage = report["coverage"]
    files = coverage.pop("files", [])
    info = str(contained(root, "meta/info.json"))
    read = {item["path"]: item["sha256"] for item in files}
    read.update({e["path"]: e["sha256"] for f in report["findings"] for e in f.get("evidence", [])
                 if isinstance(e, dict) and e.get("path") == info and e.get("sha256")})
    listed = [{"path": path, "sha256": digest, "role": "dataset_info" if path == info else "units_rows"}
              for path, digest in read.items()]
    # As for the residual reader: files another component already hashed identically are not listed twice.
    report["artifacts"] = [item for item in listed if recorded.get(item["path"]) != item["sha256"]]
    coverage["files_with_matching_hash_in_other_components"] = len(listed) - len(report["artifacts"])
    coverage.update(max_rows=max_rows, max_files=max_files)
    if lerobot_version is not None or require_consumer_version:
        from .consumer_version import apply_consumer_version

        report = apply_consumer_version(report, lerobot_version, require_version=require_consumer_version)
    return _bound_units(report)


def _summarize_coordinates(entry, listed):
    """Keep the motor-joint records (the ones compared with the robot) and the first `listed` others; the rest are
    counted by status and unit, so a wide non-motor feature (environment state, embeddings) stays small."""
    coordinates = entry.get("coordinates")
    if not isinstance(coordinates, list):
        return
    motors = [c for c in coordinates if isinstance(c, dict) and c.get("motor") is not None]
    others = [c for c in coordinates if not (isinstance(c, dict) and c.get("motor") is not None)]
    if len(others) <= listed:
        return
    kept = {id(c) for c in motors + others[:listed]}
    hidden = others[listed:]
    count = {}
    for coordinate in hidden:
        key = (str(coordinate.get("status")), str(coordinate.get("unit"))) if isinstance(coordinate, dict) else ("?", "?")
        count[key] = count.get(key, 0) + 1
    entry["coordinates"] = [c for c in coordinates if id(c) in kept]
    entry["coordinates_summary"] = {
        "total": len(coordinates), "listed": len(entry["coordinates"]), "not_listed": len(hidden),
        "not_listed_by_status_and_unit": [{"status": s, "unit": u, "count": n} for (s, u), n in sorted(count.items())],
        "not_listed_indexes": [hidden[0].get("index"), hidden[-1].get("index")] if isinstance(hidden[0], dict) else None}
    names = entry.get("names")
    if isinstance(names, list) and len(names) > 4 * listed:
        entry["names_total"] = len(names)
        entry["names_sha256"] = hashlib.sha256(json.dumps(names, allow_nan=False).encode()).hexdigest()
        entry["names"] = names[:4 * listed]


def _bound_units(report):
    """The units component stays within UNITS_MAX_BYTES: every finding is kept (shortened), motor joints keep their
    coordinate records, and wide non-motor features are summarized, then capped, before anything else is dropped."""
    features = ((report.get("manifest") or {}).get("features") or {}) if isinstance(report.get("manifest"), dict) else {}
    originals = {key: (entry.get("coordinates"), entry.get("names")) for key, entry in features.items()
                 if isinstance(entry, dict)}
    for entry in features.values():
        if isinstance(entry, dict):
            _summarize_coordinates(entry, UNITS_LISTED)

    def size():
        return len(json.dumps(report, allow_nan=False, default=str).encode())

    if size() > UNITS_MAX_BYTES:
        report["findings"] = [_short(f) for f in report.get("findings", [])]
    for listed in (4, 0):  # still too large: fewer non-motor records, then none, each from the full list
        if size() <= UNITS_MAX_BYTES:
            break
        for key, (coordinates, names) in originals.items():
            entry = features[key]
            for extra in ("coordinates_summary", "names_total", "names_sha256"):
                entry.pop(extra, None)
            entry["coordinates"], entry["names"] = coordinates, names
            _summarize_coordinates(entry, listed)
    if size() > UNITS_MAX_BYTES:
        # Many motor joints or many features: keep each feature's verdict-bearing parts, list the rest by name only.
        for entry in features.values():
            if isinstance(entry, dict) and isinstance(entry.get("coordinates"), list) and len(entry["coordinates"]) > 64:
                entry["coordinates_not_listed"] = len(entry["coordinates"]) - 64
                entry["coordinates"] = entry["coordinates"][:64]
    report.setdefault("coverage", {})["manifest_bounded"] = size() > UNITS_MAX_BYTES or any(
        isinstance(e, dict) and ("coordinates_summary" in e or "coordinates_not_listed" in e) for e in features.values())
    return report


def _openpi_root(value, read_roots, base_directory):
    """Authorize the pinned openpi checkout before any intake; its files are then read statically, never imported."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise InputError("The openpi source root must be an explicit local directory")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base_directory or Path.cwd()) / path
    path = path.resolve()
    _within(path, read_roots, "The openpi source root")
    if not path.is_dir():
        raise InputError(f"The openpi source root is not a directory: {path}")
    return str(path)


def _convention_consumer(checkpoint):
    """The action consumer a checkpoint folder declares that the fingerprint models, or None (read-only). A
    processor_config.json that links out of the checkpoint folder is not opened: (None, reason)."""
    from .artifacts import load_checkpoint, step_kind

    path = Path(checkpoint).expanduser().resolve()
    if path.is_file() and path.name == "config.json":  # the checkpoint loader accepts its config.json too
        path = path.parent
    processor = path / "processor_config.json"
    if processor.is_file() or processor.is_symlink():
        try:
            processor = contained(path, "processor_config.json")  # the checkpoint loader's own containment rule
        except InputError:
            return None, ("The checkpoint's processor_config.json links outside the checkpoint folder, so it was not "
                          "read. Copy the file into the checkpoint folder and rerun.")
        if not processor.is_file():
            return None
        document = read_json(processor)
        kwargs = document.get("processor_kwargs", document) if isinstance(document, dict) else None
        if isinstance(kwargs, dict) and isinstance(kwargs.get("modality_configs"), dict):
            return {"groot_config": str(processor)}, "GR00T processor_config.json"
        return None
    if not (path / "config.json").is_file():
        return None
    cp = load_checkpoint(path)
    relative = (cp.config.get("use_relative_actions") is True
                or any(step_kind(s["name"]) == "relative" and s["config"].get("enabled") for s in cp.pre)
                or any(s["config"].get("use_relative_action") is True or s["config"].get("use_relative_actions") is True
                       for s in cp.pre + cp.post))
    return ({"checkpoint": str(path)}, "LeRobot relative-action processor") if relative else None


def _action_conventions(source, consumers, read_roots, max_rows, max_files, recorded):
    from .action_conventions import action_conventions

    root = dataset_root(source)
    _within(root, read_roots, "The dataset")
    for key in ("checkpoint", "groot_config"):
        if consumers.get(key):
            _within(Path(consumers[key]).resolve(), read_roots, "The checkpoint")
    report = action_conventions(root, max_rows=max_rows, max_files=max_files, **consumers)
    listed = [{"path": item["path"], "sha256": item["sha256"], "role": "convention_rows"}
              for item in report.get("coverage", {}).get("files", []) if item.get("path") and item.get("sha256")]
    report["artifacts"] = [item for item in listed if recorded.get(item["path"]) != item["sha256"]]
    return report


def _convention_verdicts(component, labels):
    """Per-consumer display: the consumer comparison and the dataset-level fingerprint are separate verdicts."""
    findings = component.get("findings", [])

    def worst(values):
        return next((s for s in ("error", "warning", "unknown", "pass") if s in values), None)

    consumer = worst([f.get("status") for f in findings if f.get("code") not in FINGERPRINT_CODES])
    fingerprint = {f.get("status") for f in findings if f.get("code") in FINGERPRINT_CODES}
    coordinates = [c for g in (component.get("fingerprint") or {}).get("groups", []) for c in g.get("coordinates", [])]
    decided = sum(c.get("convention") not in (None, "unknown") for c in coordinates)
    # Undecided coordinates and a names-vs-numbers warning are separate facts; show both rather than the worst.
    label = ", ".join([("partly undecided" if decided else "undecided")] * ("unknown" in fingerprint)
                      + [s for s in ("error", "warning") if s in fingerprint]) or worst(fingerprint) or "not assessed"
    per_consumer = {name: worst([f.get('status') for f in findings if f.get('consumer') == name]) or
                   (consumer if len(labels) == 1 else None) or 'not compared' for name in labels}
    return {"consumers": labels, "per_consumer": per_consumer,
            "consumer": consumer or "not compared", "dataset_fingerprint": label,
            "decided_coordinates": decided, "coordinates": len(coordinates)}


def _provenance(view):
    # Reuse the established bounded file manifest and content-ID provenance format.
    return hub.portable_dataset_report({"status": "unknown", "artifacts": []}, view)["provenance"]


def _checkpoint(root, dataset, deployment, view):
    if view is not None and not isinstance(read_json(Path(root) / "config.json").get("type"), str):
        raise hub.HubUnavailable("Checkpoint metadata is outside the supported LeRobot policy-config format")
    try:
        report = audit(str(root), str(dataset) if dataset is not None else None, deployment)
    except InputError as exc:
        if dataset is None:
            raise
        # Unsupported dataset metadata must not hide independently provable checkpoint/contract errors.
        report = audit(str(root), None, deployment)
        report["findings"].append({"code": "preflight.dataset_interface_unassessed", "status": "unknown",
            "message": "Dataset/checkpoint interfaces could not be compared: " + str(exc), "evidence": []})
    hub._withheld_unknown(report, view)
    hub._metadata_scope(report)
    report = hub._finish(report, [], [], scope="deployment" if deployment else "metadata")
    if deployment:
        report["assessment_scope"] = "declared_deployment_contract"
        report["deployment_status"] = report["status"]
        report["scope"] = (
            "Declared checkpoint, dataset and supplied deployment contract; no inference, saved runtime "
            "observations, command execution or physical behavior was tested."
        )
    return report


def _video_component(window):
    coverage = copy.deepcopy(window.get("coverage", {}).get("video_index_details", {}))
    findings = [copy.deepcopy(f) for f in window.get("findings", []) if f.get("code", "").startswith("video.")]
    if not coverage:
        return _unknown("video_index", "Structured video-index coverage was not available.")
    states = [f.get("status", "unknown") for f in findings]
    counts = coverage.get("finding_counts", {})
    states.extend(s for s in PRIORITY if counts.get(s, 0))
    if coverage.get("status") not in ("complete", "not_applicable"):
        states.append("unknown")
    elif not states:
        states.append("pass")
    return {"status": _status(states), "coverage": coverage, "findings": findings,
            "total_findings": max(sum(counts.values()), len(findings)) if counts else len(findings),
            "findings_truncated": sum(counts.values()) > len(findings) if counts else False,
            "scope": "Declared video indexes, file presence and supported MP4 header timing; no frame decoding."}


NOT_A_DATASET = "Not a LeRobot dataset (no meta/info.json); preflight checks LeRobot v2.1/v3 datasets."
NOT_A_DATASET_ACTION = ("Point preflight at a LeRobot v2.1/v3 dataset folder (the one that holds meta/info.json). For a "
                        "folder of other robot files (recordings, captures, logs), run policy-doctor investigate "
                        "(MCP investigate_run_directory) to see what RoboClue recognizes there.")
UNKNOWN_ACTION_PREFIX = "Read each unknown component's coverage or finding."


def _not_a_dataset(source):
    """The plain statement for a local path that holds no LeRobot dataset, or None."""
    try:
        root = dataset_root(source)
        if (root / "meta" / "info.json").is_file():
            return None
        if not root.exists():
            return "The path does not exist. " + NOT_A_DATASET
        if (root / "config.json").is_file():
            return NOT_A_DATASET + (" This folder holds config.json, so it may be a policy checkpoint: pass it with "
                                    "--policy beside a dataset.")
    except OSError:
        return None
    return NOT_A_DATASET


def _next_actions(components):
    actions = []
    # Every finding counts, including those a capped component keeps only in its per-code tally.
    codes = {f.get("code") for c in components.values() for f in c.get("findings", [])}
    codes |= {code for c in components.values() if isinstance(c.get("finding_codes"), dict) for code in c["finding_codes"]}
    conversion = components.get("conversion_fidelity")
    if conversion and conversion.get("numeric_association_status") == "error":
        actions.append("Inspect the cited original joint/action and export row. Produce a separate export with the "
                       "intended converter and rerun the same source comparison; do not relabel the original recording.")
    if "video.interval_past_end" in codes:
        actions.append("Inspect the cited episode/camera intervals. Use recover-video-reference only with the "
                       "original clip and explicit episode to test the suggested neighboring-file reference.")
    if codes & {"video.feature_without_index", "video.file_missing"}:
        actions.append("Restore the cited native camera index or missing video from its source; do not invent frames.")
    window = components.get('episode_windows', {})
    if any('Only native v3 metadata' in str(f.get('reason', '')) for f in window.get('findings', [])):
        actions.append("episode_windows: this check requires native LeRobot v3 episode metadata and its standard "
                       "data_path. A v2 recording remains unassessed here; check it with its intended native loader. "
                       "Do not relabel its codebase_version to make this check pass.")
    video = components.get('video_index', {})
    if video.get('status') == 'unknown':
        actions.append("video_index: restore the original clips and native episode/camera indexes named in coverage. "
                       "If headers were disabled, rerun without --no-video-headers (MCP video_headers=true). "
                       "If a read cap was reached, select a smaller explicit recording or set --max-files/--max-rows "
                       "within their documented limits; this does not verify unselected data.")
    groot = components.get("groot_contract", {})
    if (any(f.get("code") == "groot.config" and f.get("status") == "unknown" for f in groot.get("findings", []))
            or ((groot.get("finding_codes") or {}).get("groot.config") or {}).get("unknown")):
        actions.append("Supply the GR00T training modality config .py (parsed, never executed) to also check config "
                       "keys, camera pairing, language and relative actions: --groot-config PATH "
                       "(MCP groot_modality_config); --groot-version n1.5 or n1.7 selects the intended consumer."
                       + (" This report is at its 768 KiB bound: add a smaller --max-files (MCP max_files) to that "
                          "rerun." if groot.get("reduced") else ""))
    if codes & {"reach.held_at_an_end", "reach.steady_offset"}:
        actions.append("Check the cited joints' travel and homing on the recording robot before training on these "
                       "actions. The residual is between recorded values; what the motors received is not recorded.")
    if "reach.not_applicable" in codes:
        actions.append("--commanded-reach (MCP commanded_reach=true) requires original action and observation.state "
                       "motor-position recordings with matching <joint>.pos names. No flag can infer that relationship "
                       "for other action spaces; leave this check unassessed or supply a compatible original recording.")
    if "units.joint_mode_mismatch" in codes:
        actions.append("Deploy with the cited use_degrees mode the data was recorded in, or retrain on data recorded in "
                       "the robot's mode. This assumes the policy was trained on these values; the policy is not read.")
    if any(c.get("status") == "error" for c in components.values()):
        actions.append("Resolve the cited error findings and rerun preflight on the corrected, separately retained input.")
    if any(c.get("status") == "unknown" for c in components.values()):
        actions.append("Read each unknown component's coverage or finding. Supply its missing input, install the "
                       "requested extra, or raise the stated budget explicitly, then rerun preflight.")
    if any(c.get("status") == "warning" for c in components.values()):
        actions.append("Review the warnings against the intended consumer; a warning alone does not prove a training failure.")
    if not actions:
        actions.append("Run a small native loader and consumer smoke check for the intended job; preflight does not test training or policy success.")
    return actions


def preflight(source: str, *, checkpoint: str | None = None, deployment: str | None = None,
              fetch: bool = False, revision: str = "main", checkpoint_revision: str = "main",
              video_headers: bool = True, max_bytes: int = hub.MAX_CONTENT_BYTES,
              max_rows: int = 200000, max_files: int = 1024,
              robot_command: str | None = None, robot_config: str | None = None,
              processor_state: bool = False, readiness_fetch: bool | None = None,
              read_roots=None, base_directory=None,
              original_recording: str | None = None, converter_source: str | None = None,
              converter_config: str | None = None, converter_directory: str | None = None,
              conversion_reference: str | None = None,
              expected_conversion_criterion: str | None = None,
              consumer_reference_dataset: str | None = None,
              consumer_reference_checkpoint: str | None = None,
              consumer_samples: list | None = None,
              consumer_require_equal: bool = False,
              groot_modality_config: str | None = None,
              groot_config_name: str | None = None,
              groot_version: str | None = None,
              commanded_reach: bool = False,
              units: bool | None = None,
              groot_contract: bool = True,
              openpi_config: str | None = None,
              openpi_source_root: str | None = None,
              groot_embodiment: str | None = None,
              lerobot_version: str | None = None,
              require_consumer_version: bool = False) -> dict:
    """Check one explicit local/Hub dataset without launching a policy or writing to the source.

    Hub rows and optional video headers require ``fetch=True``. Each Hub repository is resolved once;
    dataset rows are shared by every component. A supplied deployment contract is always assessed or
    explicitly unknown, and never discarded by metadata-only filtering. The GR00T dataset contract runs
    for a local dataset with meta/modality.json or a supplied modality config; recorded-position residual
    triage runs only when ``commanded_reach=True``. The units manifest runs with a robot command or config unless
    ``units=False``, and on its own when ``units=True``. ``groot_contract=False`` skips the automatic GR00T dataset contract. The action-convention
    fingerprint runs when the inputs name a consumer it models: a local LeRobot checkpoint with a relative-action
    processor, a GR00T processor_config.json, or ``openpi_config``.
    """
    if readiness_fetch is not None and type(readiness_fetch) is not bool:
        raise InputError("readiness_fetch must be a strict boolean when supplied")
    if type(require_consumer_version) is not bool:
        raise InputError('require_consumer_version must be a strict boolean')
    robot_requested = robot_command is not None or robot_config is not None
    if (lerobot_version is not None or require_consumer_version) and not robot_requested:
        raise InputError('lerobot_version/required consumer comparison needs the actual robot_command or robot_config')
    if lerobot_version is not None:
        from .consumer_version import consumer_version_context

        consumer_version_context(lerobot_version)  # Validate caller labels before any input reads.
    if type(processor_state) is not bool:
        raise InputError("processor_state must be a strict boolean")
    if type(commanded_reach) is not bool or type(groot_contract) is not bool or (units is not None and type(units) is not bool):
        raise InputError("commanded_reach, units and groot_contract must be strict booleans (units may also be null: "
                         "automatic with a robot command or config)")
    if (openpi_config is None) != (openpi_source_root is None):
        raise InputError("openpi_config names a builtin TrainConfig read from openpi_source_root, a pinned openpi "
                         "checkout; supply both")
    for value, label in ((openpi_config, "openpi_config"), (groot_embodiment, "groot_embodiment")):
        if value is not None and (type(value) is not str or not value.strip() or len(value) > 128):
            raise InputError(f"{label} must be a nonempty name of at most 128 characters")
    if groot_version is not None and (type(groot_version) is not str or groot_version not in ("n1.5", "n1.7")):
        raise InputError("--groot-version (MCP groot_version) must be n1.5 or n1.7, or omitted")
    if groot_config_name is not None:
        if type(groot_config_name) is not str or not groot_config_name.strip() or len(groot_config_name) > 256:
            raise InputError("groot_config_name must be a nonempty name of at most 256 characters")
        if groot_version == "n1.7" and groot_modality_config is None:
            raise InputError("--groot-config-name with --groot-version n1.7 requires --groot-config "
                             "(MCP groot_modality_config)")
    if type(fetch) is not bool or type(video_headers) is not bool:
        raise InputError("fetch and video_headers must be strict booleans")
    if type(max_rows) is not int or not 1 <= max_rows <= 200000:
        raise InputError("max_rows must be a strict integer in [1, 200000]")
    if type(max_files) is not int or not 1 <= max_files <= 1024:
        raise InputError("max_files must be a strict integer in [1, 1024]")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 4 * 1024 ** 3:
        raise InputError("max_bytes must be a strict integer between 1 byte and 4 GiB")
    _input(source, "Dataset source", fetch)
    if not source.startswith('hf://'):
        source_path = Path(source).expanduser().resolve()
        if source_path.is_file() and not (source_path.name == 'info.json' and source_path.parent.name == 'meta'):
            raise InputError(f"Dataset source {source!r} names a file. Pass the dataset directory containing "
                             "meta/info.json, or that meta/info.json file. Use check-sim-data for an HDF5 recording.")
    # Authorize the modality config before any other intake, even for a Hub source where it is not opened.
    groot_modality_config = _groot_config(groot_modality_config, read_roots, base_directory)
    groot_requested = any(value is not None for value in (groot_modality_config, groot_config_name, groot_version))
    if groot_requested and not groot_contract:
        raise InputError("--no-groot-contract (MCP groot_contract=false) skips the GR00T dataset contract; it cannot "
                         "be combined with --groot-config, --groot-config-name or --groot-version "
                         "(MCP groot_modality_config, groot_config_name, groot_version)")
    openpi_source_root = _openpi_root(openpi_source_root, read_roots, base_directory)
    from .consumer_preflight import prepare_consumer_comparison

    consumer_plan = prepare_consumer_comparison(source, checkpoint,
        reference_dataset=consumer_reference_dataset, reference_checkpoint=consumer_reference_checkpoint,
        samples=consumer_samples, require_equal=consumer_require_equal, read_roots=read_roots)
    if consumer_plan and (robot_command is not None or robot_config is not None):
        raise InputError("Consumer sample comparison (--consumer-reference-dataset/--consumer-reference-checkpoint) "
                         "and --robot-command/--robot-config readiness require separate preflight calls.")
    conversion_binding = None
    if conversion_reference is not None or expected_conversion_criterion is not None:
        from .conversion_fidelity import resolve_conversion_reference

        # Restore data selectors under current authority before other intake.
        restored = resolve_conversion_reference(conversion_reference, expected_conversion_criterion,
            original_recording=original_recording, converter_source=converter_source,
            converter_config=converter_config, converter_directory=converter_directory,
            read_roots=read_roots)
        selectors = restored['selectors']
        original_recording = selectors['original_recording']
        converter_source = selectors['converter_source']
        converter_config = selectors['converter_config']
        converter_directory = selectors['converter_directory']
        expected_conversion_criterion = restored['expected_criterion']
        conversion_binding = restored['reference_binding']
    conversion_requested = any(value is not None for value in
        (original_recording, converter_source, converter_config, converter_directory,
         conversion_reference, expected_conversion_criterion))
    conversion_plan = None
    if conversion_requested:
        if any(type(value) is not str or not value.strip() for value in
               (original_recording, converter_source, converter_config)):
            raise InputError("Conversion comparison requires original_recording, converter_source and converter_config")
        if source.startswith('hf://'):
            raise InputError("Conversion comparison currently requires an explicit local converted dataset")
        from .conversion_fidelity import prepare_conversion_inputs

        # Authorize YAML-derived paths before any optional checkpoint network intake.
        conversion_plan = prepare_conversion_inputs(source, original_recording=original_recording,
            converter_source=converter_source, converter_config=converter_config,
            converter_directory=converter_directory, read_roots=read_roots,
            max_rows=max_rows, max_files=max_files,
            expected_criterion=expected_conversion_criterion, reference_binding=conversion_binding)
    model_fetch = fetch if readiness_fetch is None else readiness_fetch
    robot_plan = None
    if robot_requested:
        from .readiness_workflow import AmbiguousReadinessReference, prepare_readiness
        try:
            robot_plan = prepare_readiness(robot_command=robot_command, robot_config=robot_config,
                checkpoint=checkpoint, dataset=source, read_roots=read_roots, base_directory=base_directory)
            checkpoint = robot_plan['checkpoint']
        except AmbiguousReadinessReference:
            # The same orchestrator returns a structured UNKNOWN without probing an unknown cwd.
            pass
    readiness_bytes = min(8 * 1024 ** 2, max(1, max_bytes // 2)) if robot_requested else 0
    dataset_bytes = max_bytes - readiness_bytes if robot_requested and source.startswith('hf://') else max_bytes
    readiness_available = readiness_bytes >= 2
    dataset_bytes = max(1, dataset_bytes)
    readiness_budget = None
    if robot_requested:
        from .readiness_intake import ReadinessIntakeBudget
        readiness_budget = ReadinessIntakeBudget(max_bytes=readiness_bytes, max_files=min(max_files, 16))
    if groot_embodiment is not None and checkpoint is None:
        raise InputError("groot_embodiment selects a tag in a GR00T checkpoint's processor_config.json; supply the "
                         "checkpoint")
    if checkpoint is not None:
        if robot_requested and checkpoint.startswith('hf://') and not model_fetch:
            # A policy named by the robot command does not prevent independent local data/unit checks.
            # The checkpoint component below records missing public authority without making a request.
            hub._repo(checkpoint)
        else:
            _input(checkpoint, "Checkpoint", model_fetch if robot_requested else fetch, must_exist=False)
    if deployment is not None:
        if not isinstance(deployment, str) or not deployment.strip() or "://" in deployment:
            raise InputError("Deployment contract must be an explicit local JSON path")
    hub._revision(revision)
    hub._revision(checkpoint_revision)
    components, provenance, substitutions = {}, [], []
    dataset_root = None
    not_dataset = None if source.startswith("hf://") else _not_a_dataset(source)
    with ExitStack() as stack:
        if consumer_plan:
            from .consumer_preflight import check_consumer_comparison

            components["consumer_comparison"] = _run("consumer_comparison", lambda: check_consumer_comparison(
                consumer_plan, read_roots=read_roots))
        if conversion_requested:
            from .conversion_fidelity import check_conversion_fidelity

            components["conversion_fidelity"] = _run("conversion_fidelity", lambda: check_conversion_fidelity(
                source, original_recording=original_recording, converter_source=converter_source,
                converter_config=converter_config, converter_directory=converter_directory,
                read_roots=read_roots, max_rows=max_rows, max_files=max_files,
                expected_criterion=expected_conversion_criterion, reference_binding=conversion_binding,
                _prepared=conversion_plan))
        view = None
        if source.startswith("hf://"):
            try:
                if robot_requested and not readiness_available:
                    raise InputError("The byte budget cannot reserve public readiness intake; no public dataset request was started")
                view = stack.enter_context(hub.resolve_dataset_content(
                    source, revision=revision, fetch=fetch, max_bytes=dataset_bytes, video_headers=video_headers,
                    max_rows=max_rows, max_files=max_files))
                dataset_root = view.root
                provenance.append(_provenance(view))
                substitutions.append((str(view.root.resolve()), source + "@" + view.provenance["resolved_revision"]))
            except Exception as exc:
                components["dataset_intake"] = _unknown("dataset_intake", exc)
        else:
            from .artifacts import dataset_root as normalize_dataset_root
            from .fs_guard import refuse_special_files

            dataset_root = normalize_dataset_root(source)
            if (dataset_root / 'meta/info.json').is_file():
                refuse_special_files(dataset_root, 'dataset')
        if dataset_root is not None:
            components["dataset_statistics"] = _run("dataset_statistics", lambda: _dataset_statistics(dataset_root))
            components["numeric_rows"] = _run("numeric_rows", lambda: scan_dataset(
                str(dataset_root), max_rows=max_rows, max_files=min(max_files, 1000)))
            components["numeric_rows"].setdefault("coverage", {}).update(
                max_rows=max_rows, max_files=min(max_files, 1000))
            window = _run("episode_windows", lambda: audit_episode_windows(
                str(dataset_root), max_rows=max_rows, max_files=max_files,
                video_context=getattr(view, "video_context", None), video_headers=video_headers))
            components["episode_windows"] = window
            components["video_index"] = _run("video_index", lambda: _video_component(window))
        units_not_run = None
        if source.startswith("hf://"):
            if groot_requested:
                components["groot_contract"] = _unknown("groot_contract", "The GR00T dataset contract requires a "
                    "local dataset in this release; download the dataset and rerun with its local directory.")
            if commanded_reach:
                components["commanded_reach"] = _unknown("commanded_reach", "Recorded-position residual triage "
                    "requires a local dataset in this release; download the dataset and rerun with its local directory.")
            if units:
                components["units"] = _unknown("units", "The units manifest requires a local dataset in this release; "
                    "download the dataset and rerun with its local directory.")
            elif robot_requested and units is None:
                units_not_run = ("The units manifest reads a local dataset only in this release; download the dataset "
                                 "and rerun with its local directory to compare its units with the robot.")
        else:
            if groot_requested or (groot_contract and _groot_layout(dataset_root)):
                components["groot_contract"] = _run("groot_contract", lambda: _groot_contract(
                    dataset_root, groot_modality_config, groot_config_name, read_roots, groot_version))
            if commanded_reach:
                recorded = {a.get("path"): a.get("sha256") for c in components.values() for a in c.get("artifacts", [])}
                components["commanded_reach"] = _run("commanded_reach", lambda: _commanded_reach(
                    dataset_root, max_rows, max_files, read_roots, recorded))
            if units or (robot_requested and units is None):
                recorded = {a.get("path"): a.get("sha256") for c in components.values() for a in c.get("artifacts", [])}
                components["units"] = _run("units", lambda: _units(dataset_root, robot_command, robot_config, read_roots,
                    base_directory, max_rows, max_files, recorded, lerobot_version=lerobot_version,
                    require_consumer_version=require_consumer_version))
        if require_consumer_version and robot_requested:
            if source.startswith('hf://') or dataset_root is None:
                camera = _unknown('camera_keys', 'Camera-key comparison requires a local dataset in this release.')
                camera.update(requested=True, compared=False)
            else:
                from .readiness_workflow import _local
                from .robot_contract import check_declared_camera_keys

                camera_config = (_local(robot_config, read_roots, base_directory, explicit='robot config')
                                 if robot_config is not None else None)
                camera = check_declared_camera_keys(dataset_root, robot_command=robot_command,
                    robot_config=camera_config, lerobot_version=lerobot_version, read_roots=read_roots,
                    require_version=True)
            if camera.get('requested'):
                components['camera_keys'] = camera
        for name in TITLES.keys() & components.keys():
            components[name]["title"] = TITLES[name]
        if "groot_contract" in components:
            part = components["groot_contract"]
            part["contract_statement"] = (
                GROOT_INCOMPLETE
                if part.get("kind") != "groot_dataset_contract"
                else (GROOT_WITH_CONFIG if part["coverage"]["modality_config_supplied"] else GROOT_WITHOUT_CONFIG)
                + " " + GROOT_NOT_READ)
        if "commanded_reach" in components:
            part = components["commanded_reach"]
            part["residual_statement"] = RESIDUAL_STATEMENTS.get(part["status"], RESIDUAL_STATEMENTS["unknown"])
            part["residual_limits"] = RESIDUAL_LIMITS
        if "units" in components:
            part = components["units"]
            part["units_limits"] = UNITS_LIMITS
            robot_side = (part.get("manifest") or {}).get("deployment") if part.get("kind") == "units_manifest" else None
            part["compared_with"] = (
                f"{robot_side.get('robot_type') or 'an unrecognized robot type'} from the robot "
                f"{robot_side.get('source', {}).get('kind', 'selector')}" if robot_side
                else "dataset only (no robot command or config)" if part.get("kind") == "units_manifest" else None)
        cp_view, cp_root, cp_resolved = None, checkpoint, False
        if checkpoint is not None and not (robot_plan and robot_plan["policy_conflict"]):
            cp_view, cp_root = None, checkpoint
            try:
                if checkpoint.startswith("hf://"):
                    if robot_requested:
                        if not readiness_available:
                            raise InputError("The byte budget cannot reserve public readiness metadata intake; no model request was started")
                        from .readiness_intake import resolve_readiness_snapshot
                        cp_view = stack.enter_context(resolve_readiness_snapshot(checkpoint, repo_type="model",
                            revision=checkpoint_revision, fetch=model_fetch, budget=readiness_budget,
                            processor_state=processor_state))
                    else:
                        cp_view = stack.enter_context(hub.resolve_hub_snapshot(
                            checkpoint, repo_type="model", revision=checkpoint_revision, fetch=fetch))
                    cp_root = cp_view.root
                    provenance.append(_provenance(cp_view))
                    substitutions.append((str(cp_view.root.resolve()), checkpoint + "@" + cp_view.provenance["resolved_revision"]))
                cp_resolved = True
                name = "deployment_contract" if deployment else "checkpoint_metadata"
                components[name] = _run(name, lambda: _checkpoint(cp_root, dataset_root, deployment, cp_view))
            except Exception as exc:
                name = "deployment_contract" if deployment else "checkpoint_metadata"
                components[name] = _unknown(name, exc)
            if consumer_plan:
                part = components[name]
                part["assessment_scope"] = "lerobot_policy_interface"
                for finding in part.get("findings", []):
                    if (finding.get("code") == f"preflight.{name}_unassessed"
                            and finding.get("message", "").startswith("No config.json in ")):
                        finding["message"] = (
                            "The separate LeRobot policy-interface audit does not assess this native GR00T "
                            "saved-processor snapshot. See consumer_comparison for supported selected numeric "
                            "processing. Full checkpoint/model and deployment compatibility remain UNKNOWN.")
        elif deployment is not None:
            try:
                contract = load_deployment(deployment)
                components["deployment_contract"] = _unknown("deployment_contract",
                    "The supplied contract was read, but a checkpoint is required to compare deployment interfaces.",
                    artifacts=[asdict(a) for a in contract.artifacts])
            except Exception as exc:
                components["deployment_contract"] = _unknown("deployment_contract", exc)
        consumers, labels, refused = {}, [], None
        if checkpoint is not None and not (robot_plan and robot_plan["policy_conflict"]):
            try:  # an unreadable checkpoint is reported by the checkpoint component
                found = _convention_consumer(cp_root if cp_resolved else checkpoint)
            except (InputError, OSError, ValueError) as exc:
                found = (None, f"Cannot inspect saved consumer configuration: {exc}. Supply a readable "
                         "processor_config.json or the intended checkpoint and rerun.")
            if found and found[0] is None:
                refused = found[1]
            elif found:
                consumers.update(found[0])
                labels.append(found[1])
        if openpi_config is not None:
            consumers["openpi_config"] = {"config_name": openpi_config, "source_root": openpi_source_root}
            labels.append(f"openpi config '{openpi_config}'")
            if source.startswith('hf://') or dataset_root is None:
                components['openpi_context'] = _unknown('openpi_context',
                    'The OpenPI raw-input contract requires a local dataset; download it and rerun with its directory.')
            else:
                from .openpi_context import check_openpi_context

                components['openpi_context'] = _run('openpi_context', lambda: check_openpi_context(
                    dataset_root, config_name=openpi_config, source_root=openpi_source_root))
            components['openpi_context']['title'] = TITLES['openpi_context']
        if refused:
            components["action_conventions"] = _unknown("action_conventions", refused)
        elif groot_embodiment is not None and "groot_config" not in consumers:
            raise InputError("--groot-embodiment (MCP groot_embodiment) requires a GR00T checkpoint with "
                             "processor_config.json modality_configs. Remove the selector for a LeRobot/OpenPI consumer.")
        elif labels:
            if groot_embodiment is not None:
                consumers["groot_embodiment"] = groot_embodiment
            if source.startswith("hf://") or ("://" in str(checkpoint) and consumers.keys() & {"checkpoint", "groot_config"}):
                components["action_conventions"] = _unknown("action_conventions", "The action-convention fingerprint "
                    "reads a local dataset and checkpoint in this release; download them and rerun with their local "
                    "directories.")
            else:
                recorded = {a.get("path"): a.get("sha256") for c in components.values() for a in c.get("artifacts", [])}
                components["action_conventions"] = _run("action_conventions", lambda: _action_conventions(
                    dataset_root, consumers, read_roots, max_rows, max_files, recorded))
        if "action_conventions" in components:
            part = components["action_conventions"]
            part.update(title=TITLES["action_conventions"], convention_limits=CONVENTION_LIMITS,
                        verdicts=_convention_verdicts(part, labels))
        if robot_requested:
            from .readiness_workflow import check_robot_readiness
            # Reuse this exact dataset snapshot and checkpoint intake; no second public fetch.
            if checkpoint is not None and not cp_resolved and not (robot_plan and robot_plan['policy_conflict']):
                components['robot_readiness'] = _unknown('robot_readiness',
                    'Checkpoint intake did not complete; readiness cannot make a second unbudgeted request.')
            else:
                components['robot_readiness'] = _run('robot_readiness', lambda: check_robot_readiness(
                    robot_command=robot_command, robot_config=robot_config, checkpoint=checkpoint, dataset=source,
                    policy_revision=checkpoint_revision, dataset_revision=revision, fetch=model_fetch,
                    processor_state=processor_state, read_roots=read_roots, base_directory=base_directory,
                    _prepared=robot_plan, _resolved_dataset=str(dataset_root) if dataset_root is not None else None,
                    _checkpoint_snapshot=(str(cp_root), cp_view) if cp_resolved else None,
                    _budget=readiness_budget))
                provenance.extend(components['robot_readiness'].get('provenance', []))
        left_out = {}  # automatic components that could not fit even as a short stub: status, counts, reason

        def build():
            # Independent readers must not silently disagree about bytes at the same input path.
            artifacts, conflicts = {}, []
            for component in components.values():
                for artifact in component.get("artifacts", []):
                    key = artifact.get("path")
                    if key in artifacts and artifacts[key].get("sha256") != artifact.get("sha256"):
                        conflicts.append(key)
                    artifacts[key] = artifact
            if conflicts:
                components["input_stability"] = _unknown("input_stability",
                    "Component readers observed different content hashes at: " + ", ".join(sorted(set(conflicts))[:16]))
            findings = []
            for name, component in components.items():
                component["input_sources"] = ([source, original_recording, converter_source, converter_config]
                                              if name == "conversion_fidelity"
                                              else [source, checkpoint, consumer_plan["reference_dataset"],
                                                    consumer_plan["reference_checkpoint"]] if name == "consumer_comparison"
                                              else [checkpoint, source, deployment] if name == "deployment_contract" and checkpoint
                                              else [deployment] if name == "deployment_contract"
                                              else [checkpoint, source] if name in ("checkpoint_metadata", "robot_readiness")
                                              else [source, groot_modality_config] if name == "groot_contract" and groot_requested
                                              else [source, robot_config] if name == "units" and robot_config is not None
                                              else [x for x in (source, checkpoint, openpi_source_root) if x is not None]
                                              if name in ("action_conventions", "openpi_context")
                                              else [source])
                component["provenance_refs"] = [i for i, item in enumerate(provenance)
                                                if item.get("source") in component["input_sources"]]
                for item in component.get("findings", []):
                    finding = copy.deepcopy(item)
                    finding.setdefault("status", "error" if name == "numeric_rows" else "unknown")
                    finding.setdefault("message", str(finding.get("detail", finding.get("reason", finding.get("code", "Finding")))))
                    finding["component"] = name
                    if isinstance(finding.get("evidence"), dict):
                        finding["evidence"] = [finding["evidence"]]
                    # Video findings are already carried by the window report; do not duplicate them at the top.
                    if name != "video_index":
                        findings.append(finding)
            # A component left out at the size bound still decides the verdict through its recorded status.
            status = _status([c["status"] for c in components.values()] + [v["status"] for v in left_out.values()])
            summaries = {"error": "Supported checks found errors; inspect the cited evidence before using this input.",
                         "unknown": "Some requested checks could not be completed; inspect component coverage.",
                         "warning": "Completed checks found warnings that need review for the intended consumer.",
                         "pass": "The reported checks passed within their stated coverage."}
            summary = summaries[status]
            if status == 'unknown':
                passed = [k for k, v in components.items() if v['status'] == 'pass']
                unresolved = [k for k, v in components.items() if v['status'] == 'unknown']
                summary += (' Passed checks: ' + (', '.join(passed) or 'none') +
                            '. Unresolved checks: ' + ', '.join(unresolved) +
                            '. UNKNOWN does not invalidate a separately passed check or certify the remaining checks.')
            actions = _next_actions(components)
            if not_dataset:  # lead with what the folder is not; budget or extra remedies do not apply
                actions = [NOT_A_DATASET_ACTION] + [a for a in actions if not a.startswith(UNKNOWN_ACTION_PREFIX)]
            report = {"schema_version": 1, "kind": "dataset_preflight", "status": status,
                      "summary": not_dataset or summary, "source": source, "components": components,
                      "findings": findings, "next_actions": actions, "scope": SCOPE,
                      "deployment_status": components.get("deployment_contract", {}).get("status", "not_assessed"),
                      "artifacts": list(artifacts.values()), "provenance": provenance,
                      "coverage": {"components": {k: v["status"] for k, v in components.items()},
                                   "video_headers_requested": video_headers, "max_rows": max_rows,
                                   "max_files": max_files, "max_download_bytes": max_bytes},
                      "findings_truncated": any(c.get("findings_truncated", False) for c in components.values())}
            if lerobot_version is not None or require_consumer_version:
                from .consumer_version import consumer_version_context

                report['consumer_version'] = ((components.get('units') or {}).get('consumer_version')
                    or consumer_version_context(lerobot_version, require_version=require_consumer_version))
            if require_consumer_version and robot_requested:
                camera = components.get('camera_keys') or {}
                report['coverage']['camera_keys_requested'] = bool(camera.get('requested'))
                report['interface_scope'] = {'camera_keys': camera.get('status') if camera.get('compared') else 'not_assessed',
                    **{name: 'not_assessed' for name in ('image size', 'frame rate/fps', 'joint order')}}
            if robot_requested:
                report['coverage']['readiness_intake'] = readiness_budget.summary()
                report['coverage']['readiness_public_intake_enabled'] = model_fetch
                report['coverage']['dataset_content_byte_allocation'] = dataset_bytes
                report['coverage']['readiness_response_body_byte_allocation'] = readiness_bytes
                report['coverage']['budget_scope'] = ('Dataset content allocation plus readiness response-body allocation; '
                    'dataset API metadata/HTTP overhead are not measured wire traffic. Dataset files and readiness logical requests have distinct reported bounds.')
            if conversion_requested:
                report['coverage']['conversion_fidelity_requested'] = True
                report['scope'] += (' Original-to-export numeric association is assessed only under the explicitly selected '
                    'converter recipe. Native consumption, camera alignment and historical execution require separate evidence.')
            if consumer_plan:
                report["coverage"]["consumer_comparison_requested"] = True
                report["scope"] += (" Consumer comparison reports selected supported inner numeric values only; "
                                    "a changed value is an observation unless equality was explicitly required.")
            if "groot_contract" in components:
                report["coverage"]["groot_contract_requested"] = groot_requested
                report["scope"] += (" The GR00T dataset contract reads dataset metadata and an optional modality config "
                                    "(parsed, never executed) under the named version profiles; it reads no Parquet contents or video frames.")
            if commanded_reach:
                report["coverage"]["commanded_reach_requested"] = True
                report["scope"] += (" Opt-in recorded-position residual triage compares recorded action with recorded "
                                    "observation.state on still frames. " + RESIDUAL_LIMITS)
            if "units" in components:
                report["coverage"]["units_requested"] = "explicit" if units else "robot command or config"
                report["scope"] += " " + UNITS_LIMITS
            if units_not_run:
                report["coverage"]["units_not_run"] = units_not_run
            if units is False and robot_requested:
                report["coverage"]["units_skipped"] = "units=false (--no-units)"
            if left_out:
                report["coverage"]["components_left_out_at_size_bound"] = dict(left_out)
            if not_dataset:
                report["coverage"]["not_a_lerobot_dataset"] = True
            if robot_plan and robot_plan.get("resolved_paths"):
                report["resolved_paths"] = list(robot_plan["resolved_paths"])
            if not groot_contract:
                report["coverage"]["groot_contract_skipped"] = "groot_contract=false (--no-groot-contract)"
            if "action_conventions" in components:
                report["coverage"]["action_conventions_consumers"] = labels
                report["scope"] += " " + CONVENTION_LIMITS
            if 'openpi_context' in components:
                report['coverage']['openpi_raw_input_context_requested'] = True
                report['scope'] += (' OpenPI raw-input keys are checked against the selected fingerprint-verified '
                                    'repack declaration only; its lockfile pin is not installed-loader compatibility.')
            report["counts"] = {s: sum(f["status"] == s for f in findings) for s in PRIORITY}
            report["counts_scope"] = "available_findings"
            report["total_findings"] = sum(c.get("total_findings", c.get("findings_total",
                                             c.get("counts", {}).get("findings", len(c.get("findings", [])))))
                                             for k, c in components.items() if k != "video_index")
            report = hub._portable(report, substitutions)
            return report

        def size(value):
            return len(json.dumps(value, allow_nan=False).encode())

        report = build()
        # Automatic components never make a preflight refuse, and never hide a verdict while it fits: the largest is
        # first reduced (status, counts, per-code tally, one example per code and status), then each in turn; only if
        # that still does not fit is one replaced by a short UNKNOWN, and finally left out with the reason in coverage.
        automatic = [name for name, auto in (("groot_contract", not groot_requested),
                                             ("units", units is None and robot_requested),
                                             ("action_conventions", openpi_config is None)) if auto and name in components]
        by_size = sorted(automatic, key=lambda n: -size(components[n]))
        calm = [n for n in by_size if components[n]["status"] != "error"]
        severe = [n for n in by_size if components[n]["status"] == "error"]
        # Reduce every automatic component (largest first) before any is cut further; cut error components last.
        plan = ([("reduce", n) for n in by_size] + [("omit", n) for n in calm] + [("leave_out", n) for n in calm]
                + [("omit", n) for n in severe] + [("leave_out", n) for n in severe])
        for stage, name in plan:
            if size(report) <= REPORT_LIMIT:
                break
            part = components[name]
            if stage == "leave_out":
                components.pop(name)
                left_out[name] = {
                    "status": part["status"], "counts": part.get("counts"),
                    "reason": (f"{TITLES[name]} ({part['status']}) was left out: even as a short stub this report would "
                               "exceed its 768 KiB bound. " + (SMALLER_INPUT if name == "groot_contract" else
                               "Rerun with a smaller --max-files (MCP max_files)") + f", or skip it with {OPT_OUT[name]}.")}
            else:
                components[name] = (_reduced if stage == "reduce" else _omitted)(name, part)
            components.pop("input_stability", None)
            report = build()
        if size(report) > REPORT_LIMIT:
            # At a narrow boundary a new automatic component's omission record can be larger than its payload.
            # Compact only the generic scope prose; never remove a verdict, finding, count or component condition.
            full_scope = report['scope']
            report['scope'] = ('Bounded artifact checks only. Read each component\'s coverage and conditions; '
                               'no video decoding, model/training, robot, calibration or physical-success validation. '
                               'The longer generic scope text is omitted at the report size bound.')
            report['coverage']['scope_text_omitted'] = {'sha256': hashlib.sha256(full_scope.encode()).hexdigest(),
                                                      'characters': len(full_scope)}
        if size(report) > REPORT_LIMIT:
            raise InputError(REPORT_REFUSED)
        return report
