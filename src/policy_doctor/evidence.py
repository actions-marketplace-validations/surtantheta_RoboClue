"""Verify a finding's native structured evidence against its recorded digest."""

import re
from pathlib import Path

from .artifacts import MAX_JSON_BYTES, InputError, read_json, sha256
from .trajectory import _read_document


def inspect_evidence(path: str, pointer: str = "", expected_sha256: str | None = None) -> dict:
    source = Path(path).expanduser().resolve()
    is_yaml = source.suffix.lower() in ('.yaml', '.yml')
    if source.suffix.lower() not in ('.json', '.yaml', '.yml'):
        raise InputError("Evidence inspection accepts JSON or native YAML artifacts only")
    if expected_sha256 is not None and (not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)):
        raise InputError("expected_sha256 must be a lowercase SHA-256 digest")
    if not isinstance(pointer, str) or len(pointer) > 4096 or (pointer and not pointer.startswith("/")) or re.search(r"~(?![01])", pointer):
        raise InputError("Expected an RFC 6901 JSON pointer with valid ~0/~1 escapes")
    try:
        if not source.is_file() or source.stat().st_size > MAX_JSON_BYTES:
            raise InputError("Expected a bounded structured evidence file")
        before = sha256(source)
    except OSError as exc:
        raise InputError(f"Cannot read evidence artifact: {exc}") from exc
    base = {"schema_version": 1, "path": str(source), "pointer": pointer, "sha256": before,
            "expected_sha256": expected_sha256, "scope": "Artifact integrity and selected JSON value, not authenticity or causal proof."}
    if is_yaml:
        base.update(source_format='YAML', scope='Artifact integrity and selected original YAML value, not authenticity or causal proof. RFC6901 addresses the original mapping, including wrappers.')
    if expected_sha256 is not None and expected_sha256 != before:
        return dict(base, status="error", code="evidence.changed", message="Artifact differs from the finding's recorded digest; rerun the audit.")
    if is_yaml:
        value, artifact = _read_document(source)
        if artifact['sha256'] != before:
            raise InputError("Evidence artifact changed during inspection; retry on a stable copy")
    else:
        value = read_json(source)
    if sha256(source) != before:
        raise InputError("Evidence artifact changed during inspection; retry on a stable copy")
    if pointer:
        for token in pointer[1:].split("/"):
            key = token.replace("~1", "/").replace("~0", "~")
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif isinstance(value, list) and re.fullmatch(r"0|[1-9][0-9]*", key) and int(key) < len(value):
                value = value[int(key)]
            else:
                return dict(base, status="unknown", code="evidence.pointer_missing", message="The selected JSON pointer is absent.")
    return dict(base, status="pass", value=value)
