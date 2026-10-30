"""Print MCP client configuration for this installation; never edits client files."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from .artifacts import InputError


def server_command(*, canonical: bool = False) -> list[str]:
    """The MCP server launcher belonging to the interpreter running this CLI."""
    if type(canonical) is not bool:
        raise InputError('Canonical launcher selection must be a strict boolean')
    if canonical:
        return [sys.executable, '-m', 'roboclue', 'advanced', 'serve']
    launcher = Path(sys.executable).parent / ("policy-doctor-mcp.exe" if sys.platform == "win32" else "policy-doctor-mcp")
    return [str(launcher)] if launcher.is_file() else [sys.executable, "-m", "policy_doctor", "serve"]


def client_configuration(roots: list[str], *, allow_hub: bool = False, allow_hub_datasets: bool = False,
                         allow_hub_processor_state: bool = False, allow_openpi_statistics: bool = False, name: str = "roboclue",
                         clients: tuple[str, ...] = ("claude-code", "codex", "json"), canonical: bool = False) -> str:
    if any(type(flag) is not bool for flag in (allow_hub, allow_hub_datasets, allow_hub_processor_state, allow_openpi_statistics, canonical)):
        raise InputError("MCP permission flags must be strict booleans")
    resolved = []
    for root in roots:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise InputError(f"Read root must be an existing directory: {path}")
        resolved.append(str(path))
    if not name or not name.replace("-", "").replace("_", "").isalnum():
        raise InputError("Server name must contain only letters, digits, '-' or '_'")
    command = server_command(canonical=True) if canonical else server_command()
    args = [part for root in resolved for part in ("--root", root)] + (["--allow-hub"] if allow_hub else [])
    if allow_hub_datasets:
        args.append("--allow-hub-datasets")
    if allow_hub_processor_state:
        args.append("--allow-hub-processor-state")
    if allow_openpi_statistics:
        args.append("--allow-openpi-statistics")
    full = command + args
    sections = []
    if "claude-code" in clients:
        sections.append("# Claude Code\n" + shlex.join(["claude", "mcp", "add", name, "--"] + full))
    if "codex" in clients:
        sections.append("# Codex\n" + shlex.join(["codex", "mcp", "add", name, "--"] + full))
    if "json" in clients:
        config = {"mcpServers": {name: {"command": full[0], "args": full[1:]}}}
        sections.append("# Cursor, Claude Desktop and other clients (mcpServers JSON)\n" + json.dumps(config, indent=2))
    notes = ["# Notes",
             "Nothing was written; paste one of the above into your client. Read roots: " + ", ".join(resolved) + ".",
             "Public Hub metadata requests are " + ("enabled for readiness (--allow-hub-processor-state); the legacy metadata tool still requires --allow-hub." if allow_hub_processor_state and not allow_hub else "enabled (--allow-hub)." if allow_hub else "disabled; add --allow-hub to enable them."),
             "Public dataset-row preflight is " + ("enabled (--allow-hub-datasets)." if allow_hub_datasets else "disabled; add --allow-hub-datasets to enable it."),
             "Selected readiness processor statistics are " + ("enabled (--allow-hub-processor-state); no weights." if allow_hub_processor_state else "disabled; separate --allow-hub-processor-state permission is required."),
             "Selected OpenPI statistics intake is " + ("enabled (--allow-openpi-statistics); no weights or source downloads." if allow_openpi_statistics else "disabled; Hub permissions do not grant GCS access."),
             'First prompt to try: "Investigate <run folder> with response_format=\'concise\' and tell me what is unknown."']
    return "\n\n".join(sections + ["\n".join(notes)]) + "\n"
