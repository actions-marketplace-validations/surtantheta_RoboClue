"""Bounded streaming Parquet checks; videos and semantic correctness are outside scope."""

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .artifacts import InputError, contained, load_dataset, sha256
from .diagnostics import ev
from .native_parquet_guard import UnsupportedParquetMetadata, require_unencrypted_parquet

_SCOPED_STATISTICS_RECHECK = object()
_SCOPE = ("Declared native numeric scalar/vector features, frame order and timestamp cadence. Primitive scalar storage is supported for native shape [1]; list vectors retain exact shape checks. No state-name aliases, video decode or semantic label validation; undeclared columns and higher-dimensional features are not checked.")


def scan_dataset(dataset: str, max_rows: int = 200000, max_files: int = 100, *, compare_stats: bool = True,
                 _intake_token=None) -> dict:
    if type(max_rows) is not int or not 1 <= max_rows <= 1_000_000:
        raise InputError("max_rows must be in [1, 1000000]")
    if type(max_files) is not int or not 1 <= max_files <= 1000:
        raise InputError("max_files must be in [1, 1000]")
    if type(compare_stats) is not bool:
        raise InputError('compare_stats must be a boolean')
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise InputError("Install policy-deployment-doctor[dataset] for Parquet scanning") from exc
    ds = load_dataset(dataset)
    from .numeric_statistics import NumericStatistics

    statistics = NumericStatistics(ds.config, ds.features, enabled=compare_stats)
    if ds.config.get('storage_format', 'parquet') != 'parquet':
        # Native loaders read the declared backend; leftover .parquet files are
        # not this dataset's rows, so their values are never diagnosed as current.
        reason = ('meta/info.json declares a non-Parquet storage_format; only the absent/default or '
                  'explicitly declared Parquet backend is assessed.')
        return {"schema_version": 1, "status": "unknown", "summary": "Unassessed: " + reason,
                "coverage": {"rows_scanned": 0, "files_scanned": 0, "files_discovered": 0,
                             "complete_parquet_scan": False, "videos_checked": False,
                             "observed_row_statistics_compared": False,
                             "numeric_vector_features": [], "scalar_stored_features": []},
                "findings": [{'code': 'dataset.storage_format_unassessed', 'status': 'unknown',
                              'path': str(ds.root / 'meta/info.json'), 'row': None,
                              'message': reason, 'detail': reason}],
                "total_findings": 1, "findings_truncated": False,
                "statistics_comparison": statistics.finish(complete=False, declared_rows=ds.config.get('total_frames')),
                "artifacts": [asdict(a) for a in ds.artifacts], "scope": _SCOPE}
    from .fs_guard import refuse_special_file, refuse_special_files

    if _intake_token is not _SCOPED_STATISTICS_RECHECK:
        refuse_special_files(ds.root, 'dataset')
    index_columns = ('timestamp', 'episode_index', 'frame_index')
    vector_features = []
    for key, feature in ds.features.items():
        if key in index_columns or len(feature['shape']) != 1 or feature.get('type') == 'VISUAL':
            continue
        dtype = feature.get('dtype')
        if dtype is None and feature.get('type') in ('STATE', 'ACTION'):
            vector_features.append(key)
        elif isinstance(dtype, str):
            try:
                if np.dtype(dtype).kind in 'biuf':
                    vector_features.append(key)
            except TypeError:
                # Non-numeric native features (e.g. video/string) are outside this scan.
                continue
    paths = sorted((ds.root / "data").rglob("*.parquet"))
    if not paths:
        raise InputError("No data/**/*.parquet files found; the dataset has no rows to scan")
    findings, artifacts = [], []
    count, previous, errors, cap = 0, {}, 0, False
    cap_reason = None
    scalar_stored_features = set()

    def add(code, path, row, detail):
        nonlocal errors
        errors += 1
        if len(findings) < 200:
            finding = {"code": code, "path": str(path), "row": row, "detail": detail}
            pointer = {"dataset.cadence": "/fps", "dataset.total_frames": "/total_frames"}.get(code)
            if code == "dataset.vector_invalid" and isinstance(detail, dict) and "feature" in detail:
                pointer = "/features/" + detail["feature"].replace('~', '~0').replace('/', '~1') + "/shape"
            if pointer:
                value = ds.config.get(pointer[1:]) if not pointer.startswith("/features/") else detail.get("expected_shape")
                finding["evidence"] = ev(ds, pointer, value)
            findings.append(finding)

    for file_index, path in enumerate(paths):
        if file_index >= max_files or count >= max_rows:
            cap = True
            cap_reason = (f"max_files={max_files} of {len(paths)} data files" if file_index >= max_files
                          else f"max_rows={max_rows}")
            break
        path = contained(ds.root, str(path.relative_to(ds.root)))
        refuse_special_file(path, 'dataset Parquet')
        if path.stat().st_size > 512 * 1024 * 1024:
            raise InputError(f"Parquet file exceeds 512 MiB scan limit: {path}")
        artifacts.append({"path": str(path), "sha256": sha256(path), "role": "dataset_parquet"})
        try:
            require_unencrypted_parquet(path)
            parquet = pq.ParquetFile(path)
            desired = tuple(dict.fromkeys((*vector_features, *index_columns, *statistics.columns)))
            columns = [k for k in desired if k in parquet.schema_arrow.names]
            scalar_columns = {key for key in vector_features if key in columns and
                ds.features[key]['shape'] == [1] and
                (pa.types.is_integer(parquet.schema_arrow.field(key).type) or
                 pa.types.is_floating(parquet.schema_arrow.field(key).type) or
                 pa.types.is_boolean(parquet.schema_arrow.field(key).type))}
            scalar_stored_features.update(scalar_columns)
            for key in desired:
                if key not in columns:
                    add("dataset.column_missing", path, None, key)
            row = 0
            for batch in parquet.iter_batches(batch_size=min(1024, max_rows), columns=columns):
                for record in batch.to_pylist():
                    if count >= max_rows:
                        cap = True
                        cap_reason = f"max_rows={max_rows}"
                        break
                    for key in vector_features:
                        if key in record:
                            try:
                                arr = np.asarray(record[key], dtype=float)
                                expected = ds.features.get(key, {}).get("shape")
                                shape_matches = expected is not None and (arr.shape == tuple(expected) or
                                    (key in scalar_columns and arr.ndim == 0))
                                if not shape_matches or not np.isfinite(arr).all():
                                    add("dataset.vector_invalid", path, row, {"feature": key, "expected_shape": expected})
                            except (ValueError, TypeError):
                                add("dataset.vector_invalid", path, row, {"feature": key})
                    ep, frame, stamp = (record.get(k) for k in ("episode_index", "frame_index", "timestamp"))
                    if type(ep) is not int or type(frame) is not int or not isinstance(stamp, (int, float)) or not np.isfinite(stamp):
                        add("dataset.index_or_time_invalid", path, row, "Expected finite timestamps and integer indices")
                    else:
                        if ep in previous:
                            pf, pt = previous[ep]
                            if frame != pf + 1:
                                add("dataset.frame_sequence", path, row, {"previous": pf, "current": frame, "episode": ep})
                            if stamp <= pt:
                                add("dataset.time_not_increasing", path, row, {"previous": pt, "current": stamp, "episode": ep})
                            elif abs((stamp - pt) - 1 / ds.config["fps"]) > max(1e-4, 0.05 / ds.config["fps"]):
                                add("dataset.cadence", path, row, {"delta_seconds": stamp - pt, "fps": ds.config["fps"]})
                        elif frame != 0:
                            add("dataset.episode_start", path, row, {"episode": ep, "first_frame": frame})
                        previous[ep] = (frame, stamp)
                    row += 1
                    count += 1
                    statistics.update(record)
                if cap:
                    break
        except UnsupportedParquetMetadata as exc:
            cap = True
            cap_reason = 'unassessed Parquet metadata: ' + str(exc)
            break
        except (OSError, ValueError) as exc:
            raise InputError(f"Cannot scan Parquet {path}: {exc}") from exc
    if not cap and ds.config.get("total_frames") is not None and count != ds.config["total_frames"]:
        add("dataset.total_frames", Path(ds.artifacts[0].path), None,
            {"observed": count, "declared": ds.config["total_frames"]})
    if count == 0 and not cap:
        add("dataset.empty", Path(ds.artifacts[0].path), None,
            "The inspected dataset contains no rows. Supply recorded numeric data before using this scan as a training-input check.")
    comparison = statistics.finish(complete=not cap, declared_rows=ds.config.get('total_frames'))
    for mismatch in comparison['mismatches']:
        before = len(findings)
        add('dataset.statistics_rows_mismatch', ds.root / 'meta/stats.json', None, mismatch['message'])
        if len(findings) > before:
            findings[-1].update(status='error', feature=mismatch['feature'], metric=mismatch['metric'],
                indices=mismatch['indices'], indices_total=mismatch['indices_total'],
                saved=mismatch['saved'], observed=mismatch['observed'])
    errors += comparison['mismatch_count'] - len(comparison['mismatches'])
    statistics_unassessed = bool(compare_stats and comparison['status'] == 'unknown' and (
        comparison.get('unassessed_metrics') or
        comparison.get('compared_features') and comparison.get('skipped')))
    summary = None
    if cap:
        declared = ds.config.get("total_frames")
        summary = (f"Partial scan: stopped at {cap_reason} after {count} rows"
                   + (f" of {declared} declared frames" if isinstance(declared, int) else "")
                   + (". Supply a supported unencrypted snapshot before claiming a complete scan."
                      if cap_reason.startswith('unassessed Parquet metadata:') else
                      ". Raise --max-rows (up to 1000000) or --max-files (up to 1000) for a complete scan."))
    if statistics_unassessed:
        reason = ('Native float32 summary rounding can explain an unresolved difference. Recompute numeric '
                  'statistics with higher precision to resolve it.' if comparison.get('precision_unassessed_count')
                  else comparison.get('reason') or 'Some supplied numeric statistics could not be compared.')
        detail = (f'Some supplied saved numeric statistics remain unassessed after reading {count} rows. '
                  + reason)
        if len(findings) < 200:
            findings.append({'code': 'dataset.statistics_values_unassessed', 'status': 'unknown',
                             'path': str(ds.root / 'meta/stats.json'), 'row': None,
                             'message': detail, 'detail': detail,
                             'compared_metrics': comparison['compared_metrics']})
        summary = (summary + ' ' if summary else '') + detail
    total_findings = errors + int(statistics_unassessed)
    return {"schema_version": 1, "status": "error" if errors else "unknown" if cap or statistics_unassessed else "pass",
            **({"summary": summary} if summary else {}),
            "coverage": {"rows_scanned": count, "files_scanned": len(artifacts), "files_discovered": len(paths),
                         "complete_parquet_scan": not cap, "videos_checked": False,
                         "observed_row_statistics_compared": bool(comparison['compared_features']),
                         "numeric_vector_features": vector_features,
                         "scalar_stored_features": sorted(scalar_stored_features)},
            "findings": findings, "total_findings": total_findings, "findings_truncated": total_findings > len(findings),
            "statistics_comparison": comparison,
            "artifacts": [asdict(a) for a in ds.artifacts] + artifacts,
            "scope": _SCOPE}
