"""Portable, scope-bound dataset health cards over the existing bounded preflight.

A card is a reproducible evidence envelope, not a dataset certification or a signed
attestation. Hub access is explicit and uses the existing public, pinned intake.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from html import escape
from pathlib import Path

from . import __version__
from .artifacts import InputError
from .preflight import preflight

MAX_SOURCES = 16
MAX_CARD_BYTES = 2 * 1024 * 1024
CONFIG_KEYS = frozenset({
    "checkpoint", "checkpoint_revision", "deployment", "video_headers", "max_bytes", "max_rows",
    "max_files", "groot_modality_config", "groot_config_name", "groot_version", "groot_contract",
    "units", "commanded_reach", "openpi_config", "openpi_source_root", "groot_embodiment",
})
DEFAULT_CONFIG = {key: inspect.signature(preflight).parameters[key].default for key in sorted(CONFIG_KEYS)}
# A generic card does not select a trainer's dataset-format requirements merely
# because an optional metadata file is present. Preflight keeps its own defaults.
DEFAULT_CONFIG["groot_contract"] = False
_GROOT_SELECTORS = ("groot_modality_config", "groot_config_name", "groot_version")
STATUSES = ("error", "unknown", "warning", "pass")
LIMITS = (
    "This badge describes only the reported checks, read artifacts, selected consumer label and configuration. "
    "The consumer label is supplied by the caller; it does not establish native consumer compatibility. "
    "No whole-dataset quality, video decoding, training, physical safety or task-success guarantee is made. "
    "Content hashes detect accidental changes; they are not signatures or independent proof of public origin."
)
_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _json(value):
    try:
        data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise InputError("Health-card inputs must be finite JSON values") from exc
    if len(data.encode()) > MAX_CARD_BYTES:
        raise InputError("Health card exceeds its 2 MiB JSON bound")
    return data


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _request(source, consumer, config, revision, fetch):
    from .hub import _repo, _revision

    if type(fetch) is not bool:
        raise InputError("Health-card fetch must be a strict boolean")
    if not isinstance(source, str) or not source.strip() or len(source) > 4096:
        raise InputError("Health-card source must be a nonempty explicit local path or hf://ORG/REPO")
    if "://" in source:
        _repo(source)
        _revision(revision)
        if not fetch:
            raise InputError("Public-HF health cards require explicit fetch=True")
    elif revision != "main":
        raise InputError("Local health cards use an observed-artifact digest; revision selects Hub commits only")
    if not isinstance(consumer, str) or not consumer.strip() or len(consumer) > 128:
        raise InputError("Health-card consumer must be a nonempty label of at most 128 characters")
    if config is None:
        config = {}
    if not isinstance(config, dict) or config.keys() - CONFIG_KEYS:
        raise InputError("Health-card config contains unsupported preflight options")
    effective = {**DEFAULT_CONFIG, **config}
    if "groot_contract" not in config and any(config.get(key) is not None for key in _GROOT_SELECTORS):
        effective["groot_contract"] = True
    # Round-trip into independent JSON values before passing to the existing checker.
    config = json.loads(_json(effective))
    return {"source": source, "requested_revision": revision if source.startswith("hf://") else None,
            "engine": {"package": "policy-deployment-doctor", "version": __version__},
            "consumer": consumer, "consumer_binding": "caller label; only explicitly requested checks apply",
            "config": config, "network": "public_https_authorized" if fetch else "offline"}


def _status(report):
    """Never allow missing/unsupported coverage to become a pass or hide known errors."""
    components = report.get("components", {})
    if not isinstance(components, dict):
        return "unknown"
    values = [report.get("status", "unknown")]
    values += [part.get("status", "unknown") if isinstance(part, dict) else "unknown"
               for part in components.values()]
    values += [finding.get("status", "unknown") for finding in report.get("findings", [])]
    if any(value not in STATUSES for value in values):
        values.append("unknown")
    required = {"numeric_rows", "dataset_statistics", "episode_windows", "video_index"}
    if not required.issubset(components):
        values.append("unknown")
    for part in components.values():
        coverage = part.get("coverage", {}) if isinstance(part, dict) else {}
        if not coverage or any(coverage.get(key) is False for key in ("complete", "complete_parquet_scan")):
            values.append("unknown")
        if coverage.get("status") in ("unsupported", "partial", "unknown", "timeout"):
            values.append("unknown")
    return next((status for status in STATUSES if status in values), "unknown")


def _revision_identity(request, report):
    if request["source"].startswith("hf://"):
        found = {p.get("resolved_revision") for p in report.get("provenance", [])
                 if p.get("source") == request["source"] and _SHA.fullmatch(str(p.get("resolved_revision", "")))}
        if len(found) == 1:
            resolved = next(iter(found))
            requested = request["requested_revision"]
            if not _SHA.fullmatch(requested) or requested == resolved:
                return {"kind": "hub_commit", "value": resolved}
        return {"kind": "unresolved", "value": None}
    artifacts = report.get("artifacts", [])
    if not artifacts or any(not isinstance(item.get("sha256"), str)
                            or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in artifacts):
        return {"kind": "unresolved", "value": None}
    return {"kind": "observed_artifacts_sha256", "value": _digest(artifacts),
            "scope": "Only the file manifest read by the reported checks; not a whole-directory tree hash"}


def _payload(request, report):
    identity = _revision_identity(request, report)
    status = _status(report)
    if identity["kind"] == "unresolved" and status != "error":
        status = "unknown"
    return {"schema_version": 1, "kind": "dataset_health_card", "status": status,
            "badge_label": "bounded checks", "request": request, "source_revision": identity,
            "scope": report.get("scope", "No completed checks"), "limitations": LIMITS,
            "report": report}


def health_card(source: str, *, consumer: str = "unspecified", config: dict | None = None,
                fetch: bool = False, revision: str = "main") -> dict:
    """Inspect one explicit dataset. ``config`` contains allowlisted preflight keyword arguments.

    Offline is the default. Hub errors/timeouts remain unknown preflight components;
    invalid arguments raise InputError. No files are written and no dataset is modified.
    """
    request = _request(source, consumer, config, revision, fetch)
    report = preflight(source, fetch=fetch, revision=revision, **request["config"])
    card = _payload(request, copy.deepcopy(report))
    card["card_id"] = "sha256:" + _digest(card)
    return card


def verify_health_card(card: dict) -> bool:
    """Check envelope consistency, including its derived verdict, before rendering or writing.

    This does not reread the source or authenticate who produced the evidence.
    """
    try:
        if not isinstance(card, dict) or card.get("kind") != "dataset_health_card":
            return False
        if not isinstance(card["request"]["config"], dict) or set(card["request"]["config"]) != CONFIG_KEYS:
            return False
        payload = {key: value for key, value in card.items() if key != "card_id"}
        expected = _payload(card["request"], card["report"])
        return payload == expected and card.get("card_id") == "sha256:" + _digest(expected)
    except (KeyError, TypeError, ValueError, AttributeError, InputError):
        return False


def render_badge(card: dict) -> str:
    """Render a self-contained SVG whose label and tooltip retain the bounded scope."""
    if not verify_health_card(card):
        raise InputError("Health-card binding is invalid; regenerate the card before rendering")
    status = card["status"]
    colors = {"pass": "#276b37", "warning": "#925f00", "unknown": "#59636e", "error": "#b3261e"}
    title = escape(f"Bounded checks: {status}; {card['request']['source']}; "
                   f"consumer {card['request']['consumer']}; revision {card['source_revision']['value']}; "
                   f"{card['card_id']}. {LIMITS}", quote=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="228" height="24" role="img" '
            f'aria-label="bounded checks: {status}"><title>{title}</title>'
            '<rect width="142" height="24" fill="#353d46"/>'
            f'<rect x="142" width="86" height="24" fill="{colors[status]}"/>'
            '<g fill="#fff" font-family="Verdana,sans-serif" font-size="11" text-anchor="middle">'
            '<text x="71" y="16">bounded checks</text>'
            f'<text x="185" y="16">{status}</text></g></svg>\n')


def write_health_card(card: dict, output_dir: str | Path) -> dict:
    """Create a new local CI artifact directory. Existing paths are never overwritten."""
    svg = render_badge(card)
    body = _json(card) + "\n"
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=False)
    outputs = {"report": destination / "health-card.json", "badge": destination / "badge.svg",
               "summary": destination / "summary.txt"}
    outputs["report"].write_text(body, encoding="utf-8")
    outputs["badge"].write_text(svg, encoding="utf-8")
    outputs["summary"].write_text(
        f"RoboClue bounded checks: {card['status']}\nSource: {card['request']['source']}\n"
        f"Consumer: {card['request']['consumer']}\nRevision: {card['source_revision']['value']}\n"
        f"Card: {card['card_id']}\n{LIMITS}\n", encoding="utf-8")
    return {key: str(value) for key, value in outputs.items()}


def scan_health_cards(sources: list, *, consumer: str = "unspecified", config: dict | None = None,
                      fetch: bool = False, output_dir: str | Path | None = None) -> dict:
    """Scan at most 16 explicitly listed sources sequentially; never enumerate the Hub.

    Entries are source strings or dicts with source/revision/consumer/config. Invalid
    request structures fail validation before any scan. Per-source checker/intake errors remain
    unknown cards, allowing other explicit sources to run. Batches are local artifacts.
    """
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise InputError(f"Health-card batch requires 1..{MAX_SOURCES} explicit sources")
    requests = []
    for entry in sources:
        if isinstance(entry, str):
            entry = {"source": entry}
        if not isinstance(entry, dict) or entry.keys() - {"source", "revision", "consumer", "config"}:
            raise InputError("Health-card batch entries require source and optional revision/consumer/config")
        requests.append(_request(entry.get("source"), entry.get("consumer", consumer),
                                 entry.get("config", config), entry.get("revision", "main"), fetch))
    destination = Path(output_dir).expanduser() if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    cards, outputs = [], []
    for index, request in enumerate(requests):
        try:
            card = health_card(request["source"], consumer=request["consumer"], config=request["config"],
                               revision=request["requested_revision"] or "main", fetch=fetch)
        except (InputError, OSError, TimeoutError) as exc:
            report = {"status": "unknown", "components": {}, "findings": [], "artifacts": [],
                      "provenance": [], "scope": "No completed checks",
                      "intake_error": {"type": type(exc).__name__, "message": str(exc)[:1024]}}
            card = _payload(request, report)
            card["card_id"] = "sha256:" + _digest(card)
        cards.append(card)
        if destination is not None:
            outputs.append(write_health_card(card, destination / f"{index:03d}-{card['card_id'][7:19]}"))
    result = {"schema_version": 1, "kind": "dataset_health_card_batch", "cards": cards,
              "counts": {status: sum(card["status"] == status for card in cards) for status in STATUSES},
              "scope": "Only these explicitly supplied sources and their reported checks; not a Hub census",
              "outputs": outputs}
    if destination is not None:
        # Keep the index compact; individual reports retain all evidence and coverage.
        index = {**result, "cards": [{key: card[key] for key in
                                      ("card_id", "status", "request", "source_revision")} for card in cards]}
        (destination / "index.json").write_text(_json(index) + "\n", encoding="utf-8")
    return result
