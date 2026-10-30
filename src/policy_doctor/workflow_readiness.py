"""Choose an evidence workflow without inventing intent or running assessments."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import stat
import sys
from pathlib import Path

from .artifacts import InputError
from .conversion_fidelity import _read_small, _reference_json

GOALS = ("dataset", "training", "runtime", "upgrade", "conversion", "deploy", "recovery", "sim")
FRAMEWORKS = ("unspecified", "groot", "lerobot", "openpi", "isaaclab", "other")
PATH_KINDS = {
    "dataset": "directory", "contract": "file", "capture": "file", "recording": "file",
    "manifest": "file", "original_recording": "file", "converter_source": "directory",
    "converter_config": "file",
    "robot_config": "file", "openpi_source_root": "directory",
    'checkpoint': 'directory', 'pipeline_contract': 'file', 'runtime_source_root': 'directory',
}
GUIDES = {"groot": "docs/GR00T_TRAINING_GUARD.md", "lerobot": "docs/LEROBOT_TRAINING_GUARD.md",
          "openpi": "docs/OPENPI_TRAINING_GUARD.md"}


def _path(value, name, kind, authorize):
    if type(value) is not str or not value or len(value) > 4096 or "\x00" in value:
        raise InputError(f"{name} must be a nonempty local path")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise InputError(f"{name} must be a valid UTF-8 path") from error
    if name == "dataset" and value.startswith("hf://"):
        if not re.fullmatch(r"hf://[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", value):
            raise InputError("dataset must use hf://OWNER/REPOSITORY without a nested path")
        return value
    if "://" in value:
        raise InputError(f"{name} requires a local path")
    # Establish authority before inspecting metadata, but preserve the caller's
    # spelling: the shared guard resolves aliases and would hide symlinks.
    if authorize is not None:
        authorize(value)
    path = Path(value).expanduser().absolute()
    try:
        for component in (*reversed(path.parents), path):
            mode = component.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise InputError(f"{name} contains a symlink; select a materialized input")
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise InputError(f"{name} was not found: {path}. Check the path and rerun.") from error
    except OSError as error:
        raise InputError(f"Cannot inspect {name}: {error}") from error
    expected = stat.S_ISDIR(mode) if kind == "directory" else stat.S_ISREG(mode)
    if not expected:
        raise InputError(f"{name} must be a regular {kind}; special files are refused")
    return str(path)


def _action(description, argv, *, tool=None, arguments=None):
    value = {"description": description, "local_cli_argv": argv, "executed": False}
    if tool is not None:
        value["mcp_request"] = {"name": tool, "arguments": arguments}
    return value


def choose_workflow(goal, *, framework="unspecified", dataset=None, contract=None,
                    criterion_sha256=None, capture=None, recording=None, manifest=None,
                    original_recording=None, converter_source=None, converter_config=None,
                    authorize=None, allow_hub_datasets=False, robot_command=None, robot_config=None,
                    openpi_config=None, openpi_source_root=None, checkpoint=None,
                    pipeline_contract=None, runtime_source_root=None):
    """Return prerequisites and concrete next calls, never an assessment verdict.

    Only explicitly supplied paths are inspected. A top-level upgrade manifest
    may be read to distinguish a watcher from native comparison evidence; its
    nested paths are not opened. Actual checkers retain their own authorization,
    pin, schema, source, runtime and coverage gates.
    """
    if type(goal) is not str or goal not in GOALS:
        raise InputError("goal must be one of: " + ", ".join(GOALS))
    if type(framework) is not str or framework not in FRAMEWORKS:
        raise InputError("framework must be one of: " + ", ".join(FRAMEWORKS))
    if criterion_sha256 is not None and (type(criterion_sha256) is not str or
            re.fullmatch(r"[0-9a-f]{64}", criterion_sha256) is None):
        raise InputError("criterion_sha256 must be the independently retained 64-character lowercase SHA256")
    raw = {"dataset": dataset, "contract": contract, "capture": capture, "recording": recording,
           "manifest": manifest, "original_recording": original_recording,
           "converter_source": converter_source, "converter_config": converter_config,
           "robot_config": robot_config, "openpi_source_root": openpi_source_root,
           'checkpoint': checkpoint, 'pipeline_contract': pipeline_contract, 'runtime_source_root': runtime_source_root}
    for name, value in (("robot_command", robot_command), ("openpi_config", openpi_config)):
        if value is not None and (type(value) is not str or not value.strip() or len(value) > 8192):
            raise InputError(name + " must be nonempty bounded text; it is never executed")
    checked = {name: _path(value, name, PATH_KINDS[name], authorize)
               for name, value in raw.items() if value is not None}
    result = {
        "kind": "workflow_readiness", "assessment_status": "not_run", "goal": goal,
        "framework": framework, "readiness": "needs_inputs", "missing_evidence": [],
        "available_evidence": {name: ("unresolved_public_reference" if value.startswith("hf://")
                                       else PATH_KINDS[name] + "_present") for name, value in checked.items()},
        "can_do_now": [], "next_steps": [],
        "limits": "This is a readiness plan, not a data, target, upgrade or physical verdict. "
                  "File presence does not validate content, retained intent, native support or producer history. "
                  "Recommended calls are returned as data and are never executed.",
    }
    data = checked.get("dataset")
    if checkpoint is not None or (runtime_source_root is not None and pipeline_contract is None):
        if goal not in ('training', 'runtime') or any(value is not None for value in
                (dataset, contract, capture, recording, manifest, pipeline_contract, robot_command, robot_config,
                 openpi_config, openpi_source_root, original_recording, converter_source, converter_config, criterion_sha256)):
            raise InputError('Checkpoint image contracts use goal=training or runtime, checkpoint and runtime_source_root; other evidence needs a separate workflow.')
        missing = [name for name in ('checkpoint', 'runtime_source_root') if name not in checked]
        result['missing_evidence'] = missing
        result['readiness'] = 'needs_inputs' if missing else 'inputs_present'
        result['next_steps'] = ['Checkpoint checks retain generic interface findings and add supported FastWAM image analysis. Fresh training uses config.json normalization_mapping and replaces saved statistics with dataset statistics; unknown effective settings stay unassessed.']
        if 'checkpoint' in checked:
            arguments = {'source': checked['checkpoint']}
            argv = ['roboclue', 'check', checked['checkpoint']]
            if goal == 'training':
                arguments['runtime_mode'] = 'training'
                argv += ['--runtime-mode', 'training']
            if 'runtime_source_root' in checked:
                arguments['runtime_source_root'] = checked['runtime_source_root']
                argv += ['--runtime-source-root', checked['runtime_source_root']]
            result['can_do_now'].append(_action('Check checkpoint metadata and supported image processing for the selected use; this does not execute a model.',
                argv, tool='preflight_check', arguments=arguments))
        return result
    if checked.get('manifest'):
        from .register import is_register_input

        if is_register_input(checked['manifest']):
            result['readiness'] = 'inputs_present'
            result['can_do_now'] = [_action(
                'Check curated version/provenance predicates; nested evidence is authorized by the checker and unresolved inputs remain UNKNOWN.',
                ['roboclue', 'check', checked['manifest'], '--json'], tool='preflight_check',
                arguments={'source': checked['manifest']})]
            result['next_steps'] = ['Only the manifest role was inspected. No register verdict or source identity has been established.']
            return result
    if pipeline_contract is not None or runtime_source_root is not None:
        if goal not in ('training', 'runtime') or any(value is not None for value in
                (dataset, contract, criterion_sha256, capture, recording, manifest, original_recording,
                 converter_source, converter_config, robot_command, robot_config, openpi_config, openpi_source_root)):
            raise InputError('Pipeline contracts use goal=training/runtime and pipeline_contract/runtime_source_root; other evidence needs a separate workflow.')
        result['missing_evidence'] = [name for name in ('pipeline_contract', 'runtime_source_root') if name not in checked]
        result['readiness'] = 'needs_inputs' if result['missing_evidence'] else 'inputs_present'
        if 'pipeline_contract' in checked:
            arguments = {'source': checked['pipeline_contract']}
            argv = ['roboclue', 'check', checked['pipeline_contract']]
            if 'runtime_source_root' in checked:
                arguments['runtime_source_root'] = checked['runtime_source_root']
                argv += ['--runtime-source-root', checked['runtime_source_root']]
            result['can_do_now'] = [_action('Check the actual resolved initialization or camera-input declaration against audited source; source identity is not a live-process attestation.',
                argv, tool='preflight_check', arguments=arguments)]
        result['next_steps'] = ['Capture the existing config/batch with capture_openpi_initialization or capture_lerobot_camera_inputs; see docs/PIPELINE_CONTRACTS.md. The checker verifies the kind, schema and supported source; this chooser runs no assessment.']
        return result
    if data:
        public = data.startswith("hf://")
        argv = ["roboclue", "health-card", data]
        arguments = {"source": data}
        if public:
            argv += ["--fetch"]
            arguments["fetch"] = True
        result["can_do_now"].append(_action("Inspect supported dataset metadata/rows and contradictions; this does not infer a correct training recipe.",
            argv + ["--json"], tool="dataset_health_card", arguments=arguments))
        if public:
            result["public_access"] = {"fetch_performed": False,
                "operator_permission_enabled": bool(allow_hub_datasets),
                "requirement": "dataset_health_card requires operator --allow-hub-datasets and request fetch=true. "
                               "preflight_check requires operator --allow-hub-datasets; its CLI counterpart requires --fetch."}
    if goal == "deploy":
        result['can_do_now'] = []
        if (robot_command is not None or robot_config is not None) and openpi_config is not None:
            raise InputError('Choose one consumer question per workflow: robot units or OpenPI conventions.')
        selectors = {}
        if robot_command is not None:
            selectors['robot_command'] = robot_command
        if robot_config is not None:
            selectors['robot_config'] = checked['robot_config']
        if openpi_config is not None:
            selectors.update(openpi_config=openpi_config, openpi_source_root=checked.get('openpi_source_root'))
        missing = ([] if data else ['dataset'])
        if not selectors:
            missing.append('actual robot_command/robot_config or OpenPI config and pinned source')
        if openpi_config is not None and not openpi_source_root:
            missing.append('openpi_source_root')
        result['missing_evidence'] = missing
        result['readiness'] = 'needs_inputs' if missing else 'inputs_present'
        if not missing:
            arguments = dict(source=data, **selectors)
            argv = ['roboclue', 'check', data]
            for name, value in selectors.items():
                argv += ['--' + name.replace('_', '-'), str(value)]
            result['can_do_now'].append(_action('Compare the declared consumer: units or delta-transform choice only, not whole-pipeline compatibility.',
                argv, tool='preflight_check', arguments=arguments))
        result['next_steps'] = ['Dataset-only unit consistency does not compare a robot. Supply its actual command/config; no command is executed.',
                                'OpenPI convention consistency does not establish repack keys or loader dependency/format version compatibility.']
    elif goal == 'recovery':
        result['can_do_now'] = []
        result['missing_evidence'] = [] if data else ['dataset']
        result['readiness'] = 'inputs_present' if data else 'needs_inputs'
        if data:
            result['can_do_now'].append(_action('Inspect interrupted recording integrity and a separate-copy recovery plan; original-directory resume is not approved by a plan.',
                ['roboclue', 'check', data, '--question', 'recovery'], tool='check_recording_salvage', arguments={'dataset': data}))
    elif goal == 'sim':
        result['can_do_now'] = []
        source = checked.get('original_recording') or checked.get('recording')
        result['missing_evidence'] = [] if source else ['original_recording']
        result['readiness'] = 'inputs_present' if source else 'needs_inputs'
        if source:
            result['can_do_now'].append(_action('Inspect simulator recording conventions; transformations need a declared source/export pair.',
                ['roboclue', 'check', source], tool='check_sim_provenance', arguments={'source': source}))
    elif goal == "dataset":
        if data:
            result["readiness"] = "inputs_present"
        else:
            result["missing_evidence"] = ["dataset"]
            result["next_steps"].append("Supply --dataset with a local materialized dataset directory or hf://OWNER/REPOSITORY.")
    elif goal == "training":
        missing = [name for name in ("contract", "criterion_sha256", "capture")
                   if (criterion_sha256 if name == "criterion_sha256" else checked.get(name)) is None]
        result["missing_evidence"] = missing
        result["readiness"] = "inputs_present" if not missing else ("needs_reference" if contract is None else "needs_inputs")
        result["retrospective_target_correctness"] = "unknown"
        if not missing:
            arguments = {"contract": checked["contract"], "criterion_sha256": criterion_sha256, "capture": checked["capture"]}
            result["can_do_now"].insert(0, _action("Check saved target/mask values against the independently retained criterion; live training is a separate Python integration.",
                ["roboclue", "check-training-targets", "--contract", checked["contract"],
                 "--criterion-sha256", criterion_sha256, "--capture", checked["capture"], "--json"],
                tool="check_training_targets", arguments=arguments))
        else:
            result["next_steps"].append("A suspect batch cannot establish its own correct targets. Retain a reference from an independently reviewed intended recipe before a change; keep its pin through correction and replay.")
            result["next_steps"].append("Learn the saved-capture loop with roboclue demo-native --output NEW_DIR. It uses genuine saved native targets, not live model execution or your missing historical intent.")
            if framework in GUIDES:
                result["capture_guide"] = GUIDES[framework]
                helper = {"groot": "capture_gr00t_reference", "lerobot": "capture_lerobot_reference", "openpi": "capture_openpi_reference"}[framework]
                result["next_steps"].append(f"For a future supported {framework} run, use {helper} in {GUIDES[framework]}, then retain its criterion_sha256 independently. The guide lists qualified settings.")
            else:
                result["next_steps"].append("Select --framework groot, lerobot or openpi for the qualified capture guide; other native training profiles remain unassessed.")
    elif goal == "runtime":
        path = checked.get("recording")
        result["readiness"] = "inputs_present" if path else "needs_inputs"
        result["physical_cause"] = "unknown"
        if path:
            result["can_do_now"].append(_action("Inspect recorded software observations; physical feedback requires explicit association, units and clock evidence.",
                ["roboclue", "flight-triage", path, "--json"], tool="triage_flight_recording", arguments={"recording": path}))
        else:
            result["missing_evidence"] = ["recording"]
            result["next_steps"].append("Attach RuntimeRecorder to the existing synchronous application and retain observations with verified field/unit declarations: docs/RUNTIME_RECORDER.md. Robot/force execution and causal contact diagnosis remain unassessed.")
    elif goal == "upgrade":
        path = checked.get("manifest")
        if path:
            document = _reference_json(_read_small(path, limit=4 * 1024 * 1024)[0])
            if not isinstance(document, dict):
                raise InputError("manifest must be a JSON object")
            if document.get("kind") == "owned_upgrade_watch":
                assets = document.get("assets")
                if type(assets) is not list or not assets or any(type(asset) is not dict for asset in assets):
                    raise InputError("An owned upgrade watch requires a nonempty assets list")
                missing = sum(not isinstance(asset.get("native_comparison"), dict) for asset in assets)
                result["readiness"] = "needs_reference" if missing else "inputs_present"
                result["missing_evidence"] = ["retained native comparisons for " + str(missing) + " asset(s)"] if missing else []
                result["can_do_now"].append(_action("Observe public upstream metadata and inventory owned files. Version differences alone cannot establish native impact.",
                    ["roboclue", "upgrade-watch", "scan-once", "--config", path, "--state-dir", "NEW_DIR", "--json"]))
                result["next_steps"].append("Native comparison declarations and nested source/capture/criterion pins have not been validated by this plan. Use docs/UPGRADE_IMPACT.md before interpreting affected/unaffected.")
            elif document.get("kind") == "native_upgrade_impact":
                result["readiness"] = "inputs_present"
                result["can_do_now"].append(_action("Assess the supplied retained comparison; the actual checker authorizes nested inputs and validates hashes/source coverage.",
                    ["roboclue", "upgrade-impact", path, "--json"], tool="assess_upgrade_impact", arguments={"manifest": path}))
            else:
                raise InputError("manifest kind must be owned_upgrade_watch or native_upgrade_impact; see docs/UPGRADE_WATCH.md")
        else:
            result["missing_evidence"] = ["manifest", "retained native criterion and before/after captures"]
            result["next_steps"].append("A version or config diff is orientation only. Keep reviewed named native fields and a criterion before upgrading; bind source/config/capture identities using docs/UPGRADE_IMPACT.md. Missing history remains UNKNOWN.")
    else:
        missing = [name for name in ("dataset", "original_recording", "converter_source", "converter_config") if name not in checked]
        result["missing_evidence"] = missing
        result["readiness"] = "needs_inputs" if missing else "inputs_present"
        result["next_steps"].append("Source/export fidelity covers the documented Arena G1 recipe only; arbitrary converters, images, physical synchronization and historical execution are not inferred. See docs/CONVERSION_REFERENCE.md.")
        if not missing:
            result["can_do_now"].append(_action("Run the supported original/export comparison; the checker still validates recipe identities and actual coverage.",
                ["roboclue", "preflight", data, *(["--fetch"] if data.startswith("hf://") else []),
                 "--original-recording", checked["original_recording"],
                 "--converter-source", checked["converter_source"], "--converter-config", checked["converter_config"], "--json"],
                tool="preflight_check", arguments={"source": data, "original_recording": checked["original_recording"],
                    "converter_source": checked["converter_source"], "converter_config": checked["converter_config"]}))
    if not result["can_do_now"] and goal not in ('deploy', 'recovery', 'sim'):
        result["can_do_now"].append(_action("Replay the published saved-native target example without training or external data downloads.",
            ["roboclue", "demo-native", "--output", "NEW_DIR"]))
    if len(json.dumps(result, ensure_ascii=True).encode()) > 32 * 1024:
        raise InputError("Readiness plan exceeds the bounded interface; supply fewer or shorter input paths")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Choose checks for evidence you already have; this does not assess or change it.")
    parser.add_argument("--goal", choices=GOALS, required=True)
    parser.add_argument("--framework", choices=FRAMEWORKS, default="unspecified")
    for name in PATH_KINDS:
        parser.add_argument("--" + name.replace("_", "-"))
    parser.add_argument("--criterion-sha256")
    parser.add_argument("--robot-command")
    parser.add_argument("--openpi-config")
    parser.add_argument("--json", action="store_true")
    args = vars(parser.parse_args(argv))
    as_json = args.pop("json")
    try:
        plan = choose_workflow(**args)
        if as_json:
            print(json.dumps(plan, sort_keys=True))
        else:
            print(f"Readiness: {plan['readiness']}; assessment: NOT RUN")
            if plan["missing_evidence"]:
                print("Missing: " + ", ".join(plan["missing_evidence"]))
            for action in plan["can_do_now"]:
                print(action["description"])
                print("  " + shlex.join(action["local_cli_argv"]))
            for step in plan["next_steps"]:
                print(step)
            print(plan["limits"])
        return 0
    except (InputError, OSError) as error:
        print(json.dumps({"error": {"code": "invalid_input", "message": str(error)}}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
