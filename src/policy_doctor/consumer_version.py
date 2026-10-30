"""Caller-declared LeRobot consumer profiles, without importing the framework.

A version label selects recorded source evidence. It is never an attestation of
the robot's running environment, and unqualified labels cannot inherit defaults.
"""
from __future__ import annotations

import copy
import hashlib
import stat
from pathlib import Path

from ._consumer_profiles import SOURCE_PROFILES
from .artifacts import InputError
from .fs_guard import open_regular

SOURCE_REVISION = "146d8474932c8b566717eb2738ba459337ab71c4"
PROFILE_ID = "lerobot-0.6.2-so-146d847"
LEGACY_PROFILE_ID = "lerobot-146d847-conditional"
ROBOT_FAMILIES = {'so100_follower': 'so', 'so101_follower': 'so', 'koch_follower': 'koch',
                  'omx_follower': 'omx', 'lekiwi': 'lekiwi', 'bi_so_follower': 'bimanual-so'}
ROBOT_TYPES = tuple(ROBOT_FAMILIES)
CRITICAL_FILES = {
    "robots/so_follower/config_so_follower.py": "65eecfc4eed9239abf65015e6b89cf0a1cb7d3bc34e5971eb83a034ac166d4f4",
    "robots/so_follower/so_follower.py": "b31d7d2a45ce871466d11b4dfde893e594e1ab9d540f2c8efe06d0f591384caf",
    "motors/motors_bus.py": "aff8f6303845f897841392c59d1c411390336c05675c4f10139a4de13c2b0d6b",
    "scripts/lerobot_record.py": "3f9469cbf219e33ae49ed18141e3bd5ffb44fe8127e1cc93819e3871a0d2ede8",
    "scripts/lerobot_rollout.py": "c6f0ea75716afdefffb886fbab8080339ede058586094f9ca05de0626bf5e6ac",
}
MAX_SOURCE_BYTES = 2 * 1024**2
_CONSUMER_CODES = frozenset({"units.joint_mode_match", "units.joint_mode_mismatch",
    "units.deployment_mode_unknown", "units.joint_mode_undecided", "units.pre_pr777_calibration"})


def _critical_files(version, robot_type):
    """The fixed source slice required for this family; never search user trees."""
    family = ROBOT_FAMILIES.get(robot_type)
    if not family or version not in SOURCE_PROFILES:
        return {}
    if version == '0.6.2' and family == 'so':
        return dict(CRITICAL_FILES)
    stem = {'so': 'so_follower', 'koch': 'koch_follower', 'omx': 'omx_follower',
            'lekiwi': 'lekiwi', 'bimanual-so': 'bi_so_follower'}[family]
    names = ['motors/motors_bus.py', 'scripts/lerobot_record.py', 'scripts/lerobot_rollout.py',
             f'robots/{stem}/config_{stem}.py', f'robots/{stem}/{stem}.py']
    if family == 'bimanual-so':
        names += ['robots/so_follower/config_so_follower.py', 'robots/so_follower/so_follower.py']
    source = SOURCE_PROFILES[version]['files']
    return {name: source[name] for name in names}


def _profile_id(version, robot_type):
    family = ROBOT_FAMILIES[robot_type]
    return PROFILE_ID if version == '0.6.2' and family == 'so' else f'lerobot-{version}-{family}'


def _camera_default(version, robot_type):
    family = ROBOT_FAMILIES[robot_type]
    stem = {'so': 'so_follower', 'koch': 'koch_follower', 'omx': 'omx_follower',
            'lekiwi': 'lekiwi', 'bimanual-so': 'bi_so_follower'}[family]
    path = f'robots/{stem}/config_{stem}.py'
    source = SOURCE_PROFILES[version]
    return {'keys': ['front', 'wrist'] if family == 'lekiwi' else [],
            'path': 'src/lerobot/' + path, 'sha256': source['files'][path],
            'source_revision': source['source_revision'], 'artifact': copy.deepcopy(source['artifact']),
            'fact': 'Native camera configuration defaults: ' + ('front and wrist' if family == 'lekiwi' else 'empty mapping')}


def supported_consumer_profiles():
    """Public coverage matrix of recorded source slices, not runtime attestation."""
    return [{'lerobot_version': version, 'robot_types': list(ROBOT_TYPES),
             'source_revision': source['source_revision'], 'artifact': copy.deepcopy(source['artifact']),
             'scope': 'Native configuration defaults and body-joint/gripper normalization only; mobile-base velocities, devices, policy execution and installed fleet bytes are unassessed.'}
            for version, source in SOURCE_PROFILES.items()]


def _label(value):
    if value is None:
        return None
    if type(value) is not str or not value or len(value) > 64 or not all(c.isprintable() for c in value):
        raise InputError("lerobot_version must be a nonempty caller-supplied string of at most 64 printable characters.")
    return value


def _safe_path(value):
    text = str(value)
    if not text or not all(c.isprintable() for c in text):
        raise InputError("The trusted consumer source path must be printable.")
    path = Path(text).expanduser().absolute()
    if ".." in path.parts:
        raise InputError("Consumer source paths cannot use parent traversal.")
    for part in (*reversed(path.parents), path):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError as error:
            raise InputError("Consumer source path not found: " + str(part)) from error
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise InputError("Consumer source reads refuse symlinks and special files: " + str(part))
    return path


def verify_consumer_source(source_root, *, robot_type='so101_follower'):
    """Hash the family's fixed files only; never import supplied source code."""
    root = _safe_path(source_root)
    if not root.is_dir():
        raise InputError("Consumer source_root must be a directory.")
    package = root / "src/lerobot" if (root / "src/lerobot").exists() else root
    package = _safe_path(package)
    profiles = {version: _critical_files(version, robot_type) for version in SOURCE_PROFILES}
    names = sorted({name for expected in profiles.values() for name in expected})
    observed, missing, total = {}, [], 0
    for name in names:
        path = package / name
        if not path.exists() and not path.is_symlink():
            missing.append(name)
            continue
        path = _safe_path(path)
        with open_regular(path, "consumer critical source") as stream:
            raw = stream.read(MAX_SOURCE_BYTES - total + 1)
        total += len(raw)
        if total > MAX_SOURCE_BYTES:
            raise InputError("Consumer critical source exceeds the 2 MiB total byte limit.")
        observed[name] = hashlib.sha256(raw).hexdigest()
    version = next((version for version, expected in profiles.items()
                    if expected and all(observed.get(name) == pin for name, pin in expected.items())), None)
    matched = version is not None
    source = SOURCE_PROFILES[version] if matched else {}
    selected = {name: observed[name] for name in profiles[version]} if matched else observed
    return {"kind": "caller_supplied_local_critical_files", "matches_tested_profile": matched,
            "tested_profile": _profile_id(version, robot_type) if matched else None,
            "profile_version": version, "source_revision": source.get('source_revision'),
            "artifact": copy.deepcopy(source.get('artifact')),
            "critical_files": selected, "missing": missing if not matched else [], "bytes_read": total,
            "scope": "Only these local critical bytes were matched; no framework import, runtime/environment attestation or whole-tree verification."}


def _role(program, established):
    if not established:
        return "not_established"
    return {"record": "collection_only", "lerobot-record": "collection_only",
            "rollout": "policy_rollout", "lerobot-rollout": "policy_rollout"}.get(program, "not_established")


def consumer_version_context(lerobot_version=None, *, deployment=None, source_root=None, require_version=False):
    """Describe the selected critical-source profile, never the actual fleet."""
    declared = _label(lerobot_version)
    if type(require_version) is not bool:
        raise InputError('require_version must be a boolean.')
    if deployment is not None and type(deployment) is not dict:
        raise InputError("deployment must be the parsed units manifest declaration.")
    deployment = deployment or {}
    robot = deployment.get("robot_type")
    source = verify_consumer_source(source_root, robot_type=robot) if source_root is not None else None
    legacy = declared is None and source is None and not require_version
    exact_source = bool(source and source["matches_tested_profile"])
    label_selects = declared in SOURCE_PROFILES and source is None
    selected = (legacy or exact_source or label_selects) and robot in ROBOT_TYPES
    profile_version = '0.6.2' if legacy else source.get('profile_version') if exact_source else declared if label_selects else None
    if require_version and declared is None and source is None:
        profile = None
        reason = 'Specify the LeRobot version used by the supplied robot command or configuration; no deployment comparison was approved.'
    elif legacy:
        profile = LEGACY_PROFILE_ID
        reason = "No fleet version was declared. The existing 146d847 conditional assumptions remain; the consumer environment is unverified."
    elif selected:
        profile = _profile_id(profile_version, robot)
        reason = (f"The caller-supplied LeRobot {declared} label selects a tested units/default profile; a label alone does not attest the robot environment's source bytes."
                  if label_selects else f"The supplied local source files match the tested LeRobot {profile_version} units/default profile; the robot environment has not been verified.")
    else:
        profile = None
        reason = ("This LeRobot version/source has no tested units profile; recorded dataset units are retained, but robot defaults and unit comparison remain UNKNOWN."
                  if robot in ROBOT_TYPES else "The declared version/source profile is not qualified for this robot type; recorded dataset inference is retained and consumer comparison remains UNKNOWN.")
    basis = source or {"kind": "conditional_recorded_source" if legacy else "caller_label_selects_recorded_profile" if label_selects else "unqualified_caller_label",
                       "source_revision": SOURCE_PROFILES[profile_version]['source_revision'] if selected else None}
    if selected:
        basis = dict(basis, critical_files=_critical_files(profile_version, robot),
                     artifact=copy.deepcopy(SOURCE_PROFILES[profile_version]['artifact']),
                     qualification="Native configuration defaults and original normalization functions; not a framework-wide runtime validation")
    role = _role(deployment.get("program"), selected)
    conditional_role = _role(deployment.get('program'), robot in ROBOT_TYPES)
    allowed = selected and bool(deployment)
    return {"declared_version": declared, "tested_profile": profile, "profile_version": profile_version,
            "source_basis": basis,
            "status": "conditional_pinned_profile" if legacy else "qualified_profile" if selected else "unknown",
            "comparison_allowed": allowed, "default_inference_allowed": allowed,
            "fleet_verified": False, "robot_types": list(ROBOT_TYPES) if not legacy else None,
            "command_role": role,
            "conditional_command_role": conditional_role if role == 'not_established' else None,
            "camera_default": _camera_default(profile_version, robot) if selected else None,
            "command_role_note": ("In the tested source profile lerobot-record collects teleoperation data only; use lerobot-rollout for policy deployment."
                                  if role == "collection_only" else "A parsed policy-rollout role does not establish successful command execution or model/robot compatibility."
                                  if role == "policy_rollout" else "The command names lerobot-record. In the tested 0.6.1 and 0.6.2 sources this is collection-only; declare the actual version before using it as a deployment comparison."
                                  if conditional_role == 'collection_only' else "The command role was not established for this consumer profile."),
            "reason": reason,
            "scope": "Caller declaration or explicitly supplied local critical bytes only. No actual fleet attestation, framework import, native command execution, policy inference or physical success."}


def apply_consumer_version(units_report, lerobot_version=None, *, source_root=None, require_version=False):
    """Return a copy with safe consumer scope; recorded feature inference stays."""
    if type(units_report) is not dict or units_report.get("kind") != "units_manifest":
        raise InputError("apply_consumer_version needs the original units_manifest report.")
    report = copy.deepcopy(units_report)
    deployment = report.get("manifest", {}).get("deployment")
    context = consumer_version_context(lerobot_version, deployment=deployment, source_root=source_root,
                                       require_version=require_version)
    report["consumer_version"] = context
    if lerobot_version is None and source_root is None and not require_version:
        return report  # Preserve every legacy verdict and assumption when omitted.
    if context["comparison_allowed"]:
        version = context['profile_version']
        for arm in (deployment or {}).get("arms", {}).values():
            if type(arm.get("use_degrees")) is not bool:
                continue  # A known source does not resolve a user's missing setting.
            old_basis = arm.get("basis")
            is_default = str(arm.get("origin", "")).startswith("the default of ")
            if is_default:
                arm["origin"] = f"default of the selected LeRobot {version} source profile; robot environment not verified"
            arm["basis"] = ("use_degrees=" + str(arm["use_degrees"]).lower() + " from " +
                            (f"the recorded {version} default" if is_default else "the explicit caller setting") +
                            " selects body " + str(arm.get("mode", "unknown")) +
                            f" and gripper RANGE_0_100 under tested LeRobot {version} source; actual robot-environment source bytes remain unverified")
            if old_basis:
                for finding in report.get("findings", []):
                    if finding.get("code") in _CONSUMER_CODES and type(finding.get("message")) is str:
                        finding["message"] = finding["message"].replace(old_basis, arm["basis"])
        return report
    conditional = [f for f in report.get("findings", []) if f.get("code") in _CONSUMER_CODES]
    report["conditional_comparison"] = {"findings": conditional, "deployment": copy.deepcopy(deployment),
        "scope": "Previous conditional 146d847 comparison retained as provenance only; not an active result for the declared consumer version."}
    report["findings"] = [f for f in report.get("findings", []) if f.get("code") not in _CONSUMER_CODES]
    for arm in (deployment or {}).get("arms", {}).values():
        if str(arm.get("origin", "")).startswith("the default of "):
            arm["use_degrees"] = None  # An unqualified label cannot inherit 146d847.
        arm["mode"] = None
        arm["origin"] = "consumer version profile unqualified"
        arm["basis"] = context["reason"]
    report["findings"].append({"code": "units.consumer_version_unqualified", "status": "unknown",
        "message": context["reason"], "remedy": "Verify the actual consumer's qualified source/settings; do not change the dataset or relabel the version merely to clear this check.",
        "evidence": [{"path": None, "sha256": None, "pointer": "/lerobot_version", "value": context["declared_version"]}]})
    report["counts"] = {state: sum(f.get("status") == state for f in report["findings"])
                        for state in ("error", "warning", "unknown", "pass")}
    report["status"] = "error" if report["counts"]["error"] else "unknown"
    report.setdefault("coverage", {})["consumer_comparison"] = {"status": "unknown", "reason": context["reason"]}
    return report
