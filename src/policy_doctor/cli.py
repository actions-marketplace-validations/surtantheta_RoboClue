"""CLI for local artifact diagnostics."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from . import __version__
from .artifacts import InputError, contained, load_checkpoint, load_dataset, read_json
from .bundle import export_bundle, reproduce_bundle
from .compare import compare_configs
from .dataset import scan_dataset
from .diagnostics import audit
from .evidence import inspect_evidence
from .repair import check_repair
from .template import deployment_template
from .trace import trace_actions


def _native_sample(value: str) -> dict[str, int]:
    parts = value.split(':')
    if (len(parts) != 2 or any(not p or not p.isascii() or not p.isdecimal() for p in parts)
            or any(len(p) > 10 or int(p) > 10**9 for p in parts)):
        raise argparse.ArgumentTypeError('Use EPISODE:FRAME with nonnegative integer IDs at most 1000000000')
    return {'episode_index': int(parts[0]), 'frame_index': int(parts[1])}


def _native_row(value: str) -> dict[str, int]:
    parsed = _native_sample(value)
    return {'episode_index': parsed['episode_index'], 'physical_row': parsed['frame_index']}


def parser(*, prog: str = "policy-doctor") -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog=prog, description="Evidence-backed robot artifact and trajectory diagnostics")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    demo = commands.add_parser('demo', help='Create a checkable native trajectory incident in a new folder; core install suffices')
    demo.add_argument('destination', help='New directory; existing paths are refused')
    demo.add_argument('--kind', choices=('trajectory', 'policy'), default='trajectory',
                      help='Explicit synthetic example; trajectory remains the default')
    metadata_check = commands.add_parser('check', help='Check local or explicitly fetched public Hub JSON metadata; no weights')
    metadata_check.add_argument('source', help='Local checkpoint directory or hf://ORG/REPO')
    metadata_check.add_argument('--dataset', help='Local dataset directory or hf://ORG/REPO')
    metadata_check.add_argument('--fetch', action='store_true', help='Explicitly allow bounded public Hub JSON metadata requests')
    metadata_check.add_argument('--revision', default='main')
    metadata_check.add_argument('--dataset-revision', default='main')
    metadata_check.add_argument('--scope', choices=('deployment', 'metadata'), default='deployment',
                                help='metadata assesses saved declarations only; deployment retains the missing-contract unknown')
    metadata_check.add_argument('--case', help='Retain this local check in a new portable directory; check exit status is preserved')
    metadata_check.add_argument('--candidate-role', choices=('checkpoint', 'dataset'), help='With --case, select the role replaced by check-case; default dataset when supplied')

    metadata_check.add_argument('--fail-on', choices=('error', 'warning', 'unknown'),
                                help='CI gate: exit 1 if any finding has this status (warning also counts errors; '
                                     'unknown counts every non-pass), otherwise 0. Default keeps the 0/1/3 codes')
    preflight = commands.add_parser('preflight', help='Check dataset rows, episodes and video indexes together; optionally compare a policy and deployment')
    preflight.add_argument('source', help='Local native dataset directory or hf://ORG/REPO')
    preflight.add_argument('--policy', '--checkpoint', dest='checkpoint', help='Optional local checkpoint or hf://ORG/REPO')
    preflight.add_argument('--deployment', help='Optional local deployment contract; comparison requires a policy')
    preflight.add_argument('--require', choices=('deployment-units',),
        help='Require an actual completed dataset-to-robot units comparison; missing/unknown required evidence cannot clear a gate')
    preflight.add_argument('--fetch', action='store_true', help='Explicitly allow pinned public metadata, Parquet and bounded video-header reads')
    preflight.add_argument('--revision', default='main')
    preflight.add_argument('--policy-revision', '--checkpoint-revision', dest='checkpoint_revision', default='main')
    preflight.add_argument('--video-headers', action=argparse.BooleanOptionalAction, default=True,
                           help='Check MP4 header timing (default); opting out leaves video coverage unknown')
    preflight.add_argument('--max-download-mb', type=int, default=300,
        help='Dataset content cap in MiB; robot readiness reserves up to8MiB for its separate metadata response-body allocation; HTTP overhead is not measured')
    preflight.add_argument('--max-rows', type=int, default=200000)
    preflight.add_argument('--max-files', type=int, default=1024,
        help='Dataset file cap; robot metadata separately reports up to16 logical requests')
    preflight.add_argument('--fail-on', choices=('error', 'warning', 'unknown'))
    preflight.add_argument('--original-recording', help='Compare this local HDF5 recording with the converted numeric rows; reads up to 12 GiB')
    preflight.add_argument('--converter-source', help='Pinned supported Arena converter checkout; compared as source data without executing it')
    preflight.add_argument('--converter-config', help='Actual converter YAML; required with --original-recording and --converter-source')
    preflight.add_argument('--converter-directory', help='Explicit intended native converter cwd for relative YAML paths; historical execution is not inferred')
    preflight.add_argument('--conversion-reference', help='Reuse conversion inputs and criterion from a full saved preflight JSON report; current read authority still applies')
    preflight.add_argument('--expected-conversion-criterion', help='Additionally require this exact lowercase SHA256 conversion criterion; drift cannot certify the candidate')
    preflight.add_argument('--consumer-reference-checkpoint', help='Compare saved GR00T numeric processing with this local reference checkpoint folder')
    preflight.add_argument('--consumer-reference-dataset', help='Reference local dataset; omitted means evaluate both processors on the current dataset')
    preflight.add_argument('--consumer-sample', action='append', metavar='EPISODE:ANCHOR',
        help='Stored episode ID and physical anchor row; repeat up to 9 times. Omit for bounded automatic complete windows')
    preflight.add_argument('--consumer-require-equal', action='store_true',
        help='Require selected supported numeric values to equal the reference; other changes remain observations')
    preflight.add_argument('--groot-config', metavar='PATH',
        help='GR00T training modality config .py, parsed without executing it. The GR00T dataset contract runs '
             'without it when meta/modality.json exists, leaving config checks unknown')
    preflight.add_argument('--groot-config-name', metavar='NAME',
        help='Config in --groot-config, or a known built-in N1.5 data config name')
    preflight.add_argument('--groot-version', choices=('n1.5', 'n1.7'),
        help='Check this GR00T version. Omit to detect from the config, or check both versions without a config')
    preflight.add_argument('--no-groot-contract', dest='groot_contract', action='store_false',
        help='Skip the GR00T dataset contract that otherwise runs automatically when meta/modality.json exists')
    preflight.add_argument('--commanded-reach', action='store_true',
        help='Opt in to recorded-position residual triage: recorded action against observation.state on still '
             'frames of a local dataset. Not a calibration, actuator or robot-health check')
    preflight_units = preflight.add_mutually_exclusive_group()
    preflight_units.add_argument('--units', action='store_true', default=None,
        help='Also infer joint units, frames and timebase from the recorded values of a local dataset; runs '
             'automatically with --robot-command or --robot-config. The policy is not read')
    preflight_units.add_argument('--no-units', dest='units', action='store_false', default=None,
        help='Skip the units manifest that otherwise runs automatically with --robot-command or --robot-config')
    preflight.add_argument('--openpi-config', metavar='NAME',
        help='openpi builtin TrainConfig whose DeltaActions mask is compared with the recorded action convention; '
             'needs --openpi-source-root')
    preflight.add_argument('--openpi-source-root', metavar='PATH',
        help='Pinned openpi checkout (215abfb) the config is read from as text; nothing is imported or run')
    preflight.add_argument('--groot-embodiment', metavar='TAG',
        help="Embodiment tag to compare in the checkpoint's GR00T processor_config.json when it declares several")
    preflight_robot = preflight.add_mutually_exclusive_group()
    preflight_robot.add_argument('--robot-command', help='Also assess the declared robot command against the selected policy')
    preflight_robot.add_argument('--robot-config', help='Also assess the declared robot config against the selected policy')
    preflight.add_argument('--lerobot-version', help='Actual LeRobot version used by the declared robot command/config; qualified source profiles only')
    readiness = commands.add_parser('readiness', help='Compare the robot command/config with its selected policy before running; no robot or model execution')
    selectors = readiness.add_mutually_exclusive_group(required=True)
    selectors.add_argument('--robot-command', help='Quoted native robot command; parsed without execution')
    selectors.add_argument('--robot-config', help='Local native robot command/config JSON')
    readiness.add_argument('--policy', '--checkpoint', dest='checkpoint', help='Policy when absent from the command/config; conflicts are errors')
    readiness.add_argument('--dataset', help='Optional declared comparison dataset; provenance does not authenticate training origin')
    readiness.add_argument('--policy-revision', default='main')
    readiness.add_argument('--dataset-revision')
    readiness.add_argument('--command-directory', help='Explicit absolute intended command cwd; OpenPI source_root must match')
    readiness.add_argument('--source-root', help='OpenPI checkout/profile matching the intended command directory; defaults to that directory')
    readiness.add_argument('--reference-command', help='Actual declared OpenPI reference command; not historical working proof')
    readiness.add_argument('--statistics-generation', help='Exact current GCS statistics-object generation, separate from model revision')
    readiness.add_argument('--reference-statistics-generation', help='Exact reference GCS statistics-object generation')
    readiness.add_argument('--restore-reference-intent', action='store_true', help='Opt into a conditional restoration of the supplied reference selection')
    readiness.add_argument('--fetch', action='store_true', help='Allow bounded pinned public metadata and selected processor statistic tensors; no weights')
    readiness.add_argument('--max-download-mb', type=int, default=8,
        help='Shared response-body/failed-reservation cap in MiB,1-64; excludes HTTP overhead')
    readiness.add_argument('--max-files', type=int, default=16, help='Shared metadata logical-request cap,1-64')
    readiness.add_argument('--fail-on', choices=('error', 'warning', 'unknown'))
    setup = commands.add_parser('setup', help='Print MCP client configuration for this installation; writes nothing')
    setup.add_argument('--root', action='append', help='Read root for artifacts (repeatable; default: current directory)')
    setup.add_argument('--allow-hub-datasets', action='store_true', help='Enable public dataset-row preflight in the printed configuration')
    setup.add_argument('--allow-hub-processor-state', action='store_true', help='Enable selected readiness statistic tensors in the printed configuration; no weights')
    setup.add_argument('--allow-openpi-statistics', action='store_true', help='Enable selected official OpenPI GCS statistics; Hub permissions remain separate')
    setup.add_argument('--allow-hub', action='store_true', help='Also expose public Hub metadata checks')
    setup.add_argument('--client', choices=('claude-code', 'codex', 'json'), action='append',
                       help='Limit output to these clients (repeatable; default: all)')
    setup.add_argument('--name', default='roboclue', help='Server name registered in the client')
    inspect = commands.add_parser("inspect", help="Inspect a native checkpoint or dataset")
    inspect.add_argument("path")
    inspect.add_argument("--kind", choices=("checkpoint", "dataset"), default="checkpoint")
    check = commands.add_parser("audit", help="Compare checkpoint, dataset and deployment metadata")
    check.add_argument("checkpoint")
    check.add_argument("--dataset")
    check.add_argument("--deployment")
    investigate = commands.add_parser("investigate", help="Discover local run artifacts and triage without writing a contract first")
    investigate.add_argument("directory")
    investigate.add_argument("--reference", help="Explicit known-good run folder; compare supported native evidence by documented alignment")
    investigate.add_argument("--requirements", help="Shared native requirement JSON, including native-processor-source; also accepted by matching create-case")
    investigate.add_argument("--named-contract", help="Existing named-action JSON criteria; equality is enforced only when explicitly requested")
    investigate.add_argument("--max-files", type=int, default=256)
    investigate.add_argument("--max-depth", type=int, default=4)
    rollouts = commands.add_parser("compare-rollouts", help="Find observed numeric recording divergence; preserves native dataset keys")
    rollouts.add_argument("reference")
    rollouts.add_argument("candidate")
    rollouts.add_argument("--alignment-key")
    rollouts.add_argument("--reference-episode")
    rollouts.add_argument("--candidate-episode")
    rollouts.add_argument("--streams", nargs='+', help="NPZ only: exact native numeric keys to decode; report all excluded arrays without decoding their values")
    rollouts.add_argument("--atol", type=float, default=1e-6)
    rollouts.add_argument("--rtol", type=float, default=1e-5)
    rollouts.add_argument("--max-rows", type=int, default=100000)
    rollouts.add_argument('--case', help='Retain this comparison and frozen reference in a new portable directory')
    capture = commands.add_parser("audit-capture", help="Verify and audit an effective runtime snapshot")
    capture.add_argument("capture")
    capture.add_argument("--dataset")
    capture.add_argument('--named-contract', help='JSON object of explicit declared order, software driver namespace and optional canonical/driver-return equality requirement')
    capture.add_argument('--command-contract', help='JSON object of explicit effective-processor, declared action-order and software-consumer criteria')
    capture.add_argument('--max-query-gap', type=int, help='ACT: explicit maximum selection calls per observed query; no universal policy-quality claim')
    capture.add_argument('--expected-names', nargs='+', help='Resolved-layout capture: expected ordered BODY names for the raw term')
    capture.add_argument('--expected-frame')
    capture.add_argument('--expected-quantity')
    capture.add_argument('--expected-prim-path', help='Expected captured sensor namespace; does not certify model equivalence')
    capture_diff = commands.add_parser("compare-captures", help="Compare same-family saved runtime captures and observed software-command stages")
    capture_diff.add_argument("reference")
    capture_diff.add_argument("candidate")
    trajectory = commands.add_parser('sample-trajectory', help='Sample native ROS JointTrajectory references offline; no ROS or robot required')
    trajectory.add_argument('trajectory')
    trajectory.add_argument('--times', nargs='+', type=float, required=True, help='Seconds from trajectory start')
    trajectory.add_argument('--method', choices=('splines', 'none'), default='splines')
    trajectory.add_argument('--case', help='Save this single-message check in a new portable case directory')
    trajectory.add_argument('--position-limits', help='JSON file of explicitly supplied joint:[lower,upper] reference bounds')
    trajectory_diff = commands.add_parser('compare-trajectories', help='Explain sampled reference differences with native joint-name alignment')
    trajectory_diff.add_argument('reference')
    trajectory_diff.add_argument('candidate')
    trajectory_diff.add_argument('--times', nargs='+', type=float, required=True)
    trajectory_diff.add_argument('--reference-method', choices=('splines', 'none'), default='splines')
    trajectory_diff.add_argument('--candidate-method', choices=('splines', 'none'), default='splines')
    case_export = commands.add_parser('export-trajectory-case', help='Package native trajectories and an exact declarative regression check')
    case_export.add_argument('reference')
    case_export.add_argument('candidate')
    case_export.add_argument('destination')
    case_export.add_argument('--control', help='Optional native comparison control; no physical repair is inferred')
    case_export.add_argument('--times', nargs='+', type=float, required=True)
    case_export.add_argument('--reference-method', choices=('splines', 'none'), default='splines')
    case_export.add_argument('--candidate-method', choices=('splines', 'none'), default='splines')
    case_export.add_argument('--control-method', choices=('splines', 'none'), default='splines')
    case_export.add_argument('--position-limits', help='JSON file of explicitly supplied joint:[lower,upper] numeric bounds')
    retained = commands.add_parser('retain-check', help='Keep native metadata or a recording comparison as a portable check for later inputs')
    retained_checks = retained.add_subparsers(dest='check', required=True)
    retained_metadata = retained_checks.add_parser('policy-metadata', help='Retain original saved checkpoint/dataset declarations and their scoped finding')
    retained_metadata.add_argument('source', help='Local supported checkpoint; no model weights are loaded')
    retained_metadata.add_argument('destination', help='New portable check directory')
    retained_metadata.add_argument('--dataset', help='Optional local native dataset metadata')
    retained_metadata.add_argument('--candidate-role', choices=('checkpoint', 'dataset'), help='Role replaced by check-case; default dataset when supplied, otherwise checkpoint')
    retained_metadata.add_argument('--scope', choices=('metadata', 'deployment'), default='metadata')
    retained_recording = retained_checks.add_parser('recording-equivalence', help='Retain a native numeric comparison and its fixed reference; differences do not establish a physical fault')
    retained_recording.add_argument('source', help='Original candidate recording file')
    retained_recording.add_argument('destination', help='New portable check directory')
    retained_recording.add_argument('--reference', required=True, help='Retained native reference recording')
    retained_recording.add_argument('--alignment-key')
    retained_recording.add_argument('--reference-episode')
    retained_recording.add_argument('--candidate-episode')
    retained_recording.add_argument('--streams', nargs='+', help='Exact native NPZ numeric keys; include the alignment key when selecting streams')
    retained_recording.add_argument('--atol', type=float, default=1e-6)
    retained_recording.add_argument('--rtol', type=float, default=1e-5)
    retained_recording.add_argument('--max-rows', type=int, default=100000)
    video_recovery = commands.add_parser('recover-video-reference', help='Prove a native video association against a pinned original clip and retain an offline case')
    video_recovery.add_argument('repo_id', help='Public dataset ORG/REPO or hf://ORG/REPO')
    video_recovery.add_argument('destination', help='New offline recovery evidence directory')
    video_recovery.add_argument('--revision', required=True, help='Exact v3 dataset commit SHA')
    video_recovery.add_argument('--original-revision', required=True, help='Exact original v2.1 dataset commit SHA')
    video_recovery.add_argument('--episode-index', type=int, required=True)
    video_recovery.add_argument('--video-key', required=True, help='Exact native camera feature key')
    bulk_recovery = commands.add_parser('recover-video-references', help='Detect supported video overruns and retain one derived native metadata proposal')
    bulk_recovery.add_argument('repo_id', help='Public dataset ORG/REPO or hf://ORG/REPO')
    bulk_recovery.add_argument('destination', help='New offline batch evidence directory; original dataset is unchanged')
    bulk_recovery.add_argument('--revision', required=True, help='Exact current v3 dataset commit SHA')
    bulk_recovery.add_argument('--original-revision', required=True, help='Exact original v2.1 source commit SHA; caller must identify the source')
    bulk_recovery.add_argument('--fetch', action='store_true', help='Authorize bounded public metadata and compressed media intake')
    bulk_recovery.add_argument('--max-associations', type=int, default=32, help='Episode/camera associations attempted, 1-256; a subset stays UNKNOWN')
    bulk_recovery.add_argument('--max-response-body-mb', type=int, default=512, help='Aggregate response-body upper bound in MiB, 1-8192; includes conservative intake reservation')
    bulk_recovery.add_argument('--max-intake-mb', type=int, default=16, help='Native metadata intake bound in MiB, 1-300; separate from aggregate bound')
    bulk_recovery.add_argument('--max-rows', type=int, default=100000)
    bulk_recovery.add_argument('--max-files', type=int, default=128)
    bulk_replay = commands.add_parser('reproduce-video-references', help='Replay a retained batch offline; reproduction is separate from proposal completeness')
    bulk_replay.add_argument('directory')
    bulk_replay.add_argument('--patch', action='store_true', help='Also list every replayed before->after metadata cell (the complete patch from proposals.json)')
    case_reproduce = commands.add_parser('reproduce-case', help='Verify a retained check, portable incident or legacy trajectory case without writes')
    case_reproduce.add_argument('directory')
    case_reproduce.add_argument('--patch', action='store_true', help='Video-reference batches: also list every replayed before->after metadata cell')
    case_check = commands.add_parser('check-case', help='Apply a verified retained check or incident regression to a later native artifact')
    case_check.add_argument('directory', help='Retained check or shared incident case')
    case_check.add_argument('candidate', help='Later native input replacing the case-defined role; same retained parameters')
    incident = commands.add_parser('create-case', help='Save and verify a native incident and its correction in a new portable case')
    incident_checks = incident.add_subparsers(dest='check', required=True)
    window_case = incident_checks.add_parser('episode-windows', help='Retain an exact native episode metadata correction and its verified windows')
    window_case.add_argument('source', help='Original native dataset directory or meta/info.json')
    window_case.add_argument('destination', help='New case directory outside all source datasets')
    correction = window_case.add_mutually_exclusive_group(required=True)
    correction.add_argument('--apply-metadata-corrections', action='store_true', help='Apply freshly verified metadata-only proposals to a separate case copy')
    correction.add_argument('--corrected', help='Separately supplied corrected native dataset')
    window_case.add_argument('--episode-index', type=int)
    window_case.add_argument('--anchor-index', type=int, help='Absolute dataset index')
    window_case.add_argument('--delta-indices', nargs='+', type=int)
    window_case.add_argument('--max-rows', type=int, default=100000)
    window_case.add_argument('--max-files', type=int, default=64)
    trajectory_case = incident_checks.add_parser('trajectory-sampling', help='Retain an original/corrected reference check under the same explicit bounds')
    trajectory_case.add_argument('source', help='Original native JSON/YAML trajectory')
    trajectory_case.add_argument('destination', help='New case directory')
    trajectory_case.add_argument('--corrected', required=True, help='Supplied corrected native JSON/YAML trajectory')
    trajectory_case.add_argument('--times', nargs='+', type=float, required=True)
    trajectory_case.add_argument('--method', choices=('splines', 'none'), default='splines')
    trajectory_case.add_argument('--position-limits', required=True, help='JSON object of explicit joint:[lower,upper] reference bounds')
    layout_case = incident_checks.add_parser('resolved-layout', help='Retain a raw-term layout incompatibility and a supplied corrected snapshot')
    layout_case.add_argument('source', help='Original resolved-layout snapshot directory')
    layout_case.add_argument('destination', help='New incident-case directory')
    layout_case.add_argument('--corrected', required=True, help='Reviewed corrected resolved-layout snapshot')
    layout_case.add_argument('--expected-names', nargs='+', help='Expected ordered BODY names; native rule derives their six components')
    layout_case.add_argument('--expected-frame')
    layout_case.add_argument('--expected-quantity')
    layout_case.add_argument('--expected-prim-path')
    cadence_case = incident_checks.add_parser('act-query-cadence', help='Retain an observed query-cadence violation and a separately executed correction')
    cadence_case.add_argument('source', help='Original ACT selection capture directory')
    cadence_case.add_argument('destination', help='New incident-case directory')
    cadence_case.add_argument('--corrected', required=True, help='Separately captured corrected native execution')
    cadence_case.add_argument('--max-query-gap', type=int, required=True, help='Explicit maximum selections per observed in-select query')
    command_case = incident_checks.add_parser('act-command-path', help='Retain a selection-to-software-consumer contract violation and a supplied corrected capture')
    command_case.add_argument('source', help='Original joined command-path capture directory')
    command_case.add_argument('destination', help='New incident-case directory')
    command_case.add_argument('--corrected', required=True, help='Supplied corrected full-chain capture; its effective criteria are retained if no contract is given')
    command_case.add_argument('--command-contract', help='JSON object of explicit retained processor/order/software-namespace criteria')
    named_case = incident_checks.add_parser('native-named-action', help='Retain an explicit native named-action boundary contract and supplied correction')
    named_case.add_argument('source', help='Original native named-action capture directory')
    named_case.add_argument('destination', help='New portable incident-case directory')
    named_case.add_argument('--corrected', required=True, help='Supplied corrected complete named boundary capture')
    named_case.add_argument('--named-contract', help='JSON contract; canonical/driver equality is enforced only when explicitly true')
    context_case = incident_checks.add_parser('native-run-context', help='Retain selected recorded evaluation settings and a bounded numeric excerpt; omits images explicitly')
    context_case.add_argument('source', help='Native RoboLab task folder')
    context_case.add_argument('destination', help='New portable comparison-case directory')
    context_case.add_argument('--corrected', required=True, help='Supplied accepted reference folder; not a demonstrated behavioral repair')
    context_case.add_argument('--requirements', required=True, help='Same native-run-context kind+parameters file as investigate')
    cohort_case = incident_checks.add_parser('evaluation-cohort', help='Retain observed episode identity and selected context checks over native JSON')
    cohort_case.add_argument('source', help='Original native evaluation JSON folder')
    cohort_case.add_argument('destination', help='New portable comparison-case directory')
    cohort_case.add_argument('--corrected', required=True, help='Explicitly nominated accepted-context reference; not a behavioral repair')
    cohort_case.add_argument('--requirements', required=True, help='Same evaluation-cohort kind+parameters file as investigate')
    source_case = incident_checks.add_parser('native-processor-source', help='Retain a caller restore-source contradiction and its same-input selected-source correction')
    source_case.add_argument('source', help='Recorded native processor-source capture directory')
    source_case.add_argument('destination', help='New portable incident-case directory')
    source_case.add_argument('--corrected', required=True, help='Reviewed source-corrected capture on the same original input')
    source_case.add_argument('--requirements', help='Optional native-processor-source requirement; selected source is frozen from corrected evidence')
    observation_case = incident_checks.add_parser('native-observation-source', help='Retain a state/camera input contract violation and its same-input native correction')
    observation_case.add_argument('source')
    observation_case.add_argument('destination')
    observation_case.add_argument('--corrected', required=True)
    observation_case.add_argument('--requirements', help='Optional native-observation-source contract; model identity is frozen from corrected evidence')
    for child in (layout_case, named_case):
        child.add_argument('--requirements', help='Reuse the same kind+parameters file supplied to investigate; conflicts with separate criterion flags')
    for child in (window_case, trajectory_case, layout_case, cadence_case, command_case, named_case, context_case, cohort_case, source_case, observation_case):
        child.add_argument('--controls', help='JSON object with controls: [{name,source,parameters,expected_status}]; source paths are relative to this file')
    dispatch = commands.add_parser('audit-dispatch', help='Check native recorded action roles against supplemental boundary snapshots')
    dispatch.add_argument('evidence')
    dispatch.add_argument('--recording', help='Native LeRobot Parquet; no format conversion')
    dispatch.add_argument('--episode', type=int)
    dispatch.add_argument('--action-names', nargs='+', help='Explicit native action vector component order')
    dispatch.add_argument('--atol', type=float, default=1e-6)
    dispatch.add_argument('--rtol', type=float, default=1e-5)
    trace = commands.add_parser("trace", help="Decode saved model outputs without running a policy")
    trace.add_argument("checkpoint")
    trace.add_argument("sample")
    trace.add_argument("--deployment")
    compare = commands.add_parser("compare", help="Compare two JSON configurations")
    compare.add_argument("before")
    compare.add_argument("after")
    template = commands.add_parser("template", help="Generate an incomplete deployment contract for you to fill")
    template.add_argument("checkpoint")
    template.add_argument("--dataset", help="Training dataset: take state/action shapes and names, fps and robot_type from it")
    scan = commands.add_parser("scan", help="Stream numeric checks over local dataset Parquet files")
    scan.add_argument("dataset")
    scan.add_argument("--max-rows", type=int, default=200000)
    scan.add_argument("--max-files", type=int, default=100)
    windows = commands.add_parser('audit-windows', help='Check native LeRobot v3 episode spans and optional temporal action-window membership')
    windows.add_argument('dataset')
    windows.add_argument('--episode-index', type=int)
    windows.add_argument('--anchor-index', type=int, help='Absolute dataset index, not the episode-relative frame_index')
    windows.add_argument('--delta-indices', nargs='+', type=int, help='Explicit integer dataset-row offsets')
    windows.add_argument('--max-rows', type=int, default=200000)
    windows.add_argument('--max-files', type=int, default=1024)
    for child in (scan, windows):
        child.add_argument('--fetch', action='store_true',
                           help='Allow a bounded public Hub download for hf://ORG/REPO: metadata and Parquet rows, never videos')
        child.add_argument('--revision', default='main', help='Hub branch, tag or commit for hf:// datasets')
        child.add_argument('--max-download-mb', type=int, default=300, help='Refuse hf:// datasets larger than this')
    windows.add_argument('--video-headers', action='store_true',
                         help='With --fetch: also read only the MP4 header boxes of referenced video files (bounded '
                              'HTTPS range reads) to check episode intervals against real frame counts')
    bundle = commands.add_parser("bundle", help="Export metadata/stats and expected diagnostic findings")
    bundle.add_argument("checkpoint")
    bundle.add_argument("destination")
    bundle.add_argument("--dataset")
    bundle.add_argument("--deployment")
    reproduce = commands.add_parser("reproduce", help="Verify artifact hashes and reproduce bundle diagnostics")
    reproduce.add_argument("bundle")
    evidence = commands.add_parser("evidence", help="Verify a finding's JSON pointer and artifact hash")
    evidence.add_argument("path")
    evidence.add_argument("--pointer", default="")
    evidence.add_argument("--sha256")
    repair = commands.add_parser("check-repair", help="Re-audit a proposed deployment contract with unchanged checkpoint/dataset artifacts")
    repair.add_argument("checkpoint")
    repair.add_argument("before")
    repair.add_argument("candidate")
    repair.add_argument("--dataset")
    native_probe = commands.add_parser('probe-native', help='Probe an explicitly configured native CPU processing path; optional numerical capsule')
    native_probe.add_argument('provider', choices=('lerobot', 'groot_state', 'groot_adapter'))
    native_probe.add_argument('model')
    native_probe.add_argument('dataset')
    native_probe.add_argument('--native-runtime', required=True, help='Operator-configured trusted native runtime JSON')
    native_selection = native_probe.add_mutually_exclusive_group(required=True)
    native_selection.add_argument('--sample', type=_native_sample, action='append',
                                  help='Explicit native EPISODE:FRAME; repeat for at most two unique samples')
    native_selection.add_argument('--row', type=_native_row, action='append',
                                  help='GR1 only: explicit EPISODE:PHYSICAL_ROW in an original single-episode file; no inferred frame column')
    native_probe.add_argument('--requirements', help='groot_adapter: explicit native-observation-source contract JSON; frozen model identities apply through investigate/check-case')
    native_probe.add_argument('--case', help='Retain numerical inputs/exports in a new capsule directory')
    native_probe.add_argument('--mode', choices=('observation', 'processor_source'), default='observation',
                              help='processor_source: one original SmolVLA observation, explicit source intent and retained source evidence')
    native_probe.add_argument('--processor-source', choices=('selected_checkpoint', 'serialized_pretrained_path', 'fresh_training'))
    native_probe.add_argument('--intent', choices=('restore_selected_checkpoint', 'fresh_training', 'explicit_base_source'),
                              help='Explicit caller intent; omission cannot establish a restoration fault')
    native_probe.add_argument('--approved-sources', help='JSON {sources:[{repo_id,revision,directory}]} explicitly authorizing alternate original snapshots')
    native_reproduce = commands.add_parser('reproduce-native-probe', help='Replay retained native numerical inputs with the same trusted runtime')
    native_reproduce.add_argument('directory')
    native_reproduce.add_argument('--native-runtime', required=True, help='Operator-configured trusted native runtime JSON')
    serve = commands.add_parser("serve", help="Run MCP diagnostics; optionally enable new incident cases under --case-root")
    serve.add_argument("--root", action="append", default=None, help="Allowed read root; repeatable, defaults to cwd")
    serve.add_argument('--case-root', help='Opt in to creating new incident cases strictly below this existing directory; inputs remain read-only')
    serve.add_argument('--native-runtime', help='Opt in to native CPU processing tools using this trusted operator runtime JSON')
    serve.add_argument('--allow-hub', action='store_true', help='Expose an additional tool allowing bounded public Hub JSON metadata fetches')
    serve.add_argument('--allow-hub-datasets', action='store_true', help='Enable public dataset-row preflight; separate from JSON-only Hub checks')
    serve.add_argument('--allow-hub-processor-state', action='store_true', help='Enable bounded readiness intake of selected normalizer/unnormalizer statistics; separate from JSON-only Hub checks')
    serve.add_argument('--allow-openpi-statistics', action='store_true', help='Enable bounded selected official OpenPI GCS statistics; separate from every Hub permission')
    serve.add_argument('--allow-hub-media', action='store_true', help='With --case-root, expose bounded pinned public video recovery; separate from metadata fetch')
    for command in (demo, metadata_check, preflight, readiness, inspect, check, investigate, rollouts, capture, capture_diff, trajectory, trajectory_diff, retained_metadata, retained_recording, video_recovery, bulk_recovery, bulk_replay, case_export, case_reproduce, case_check, window_case, trajectory_case, layout_case, cadence_case, command_case, named_case, context_case, cohort_case, source_case, observation_case, dispatch, trace, compare, template, scan, windows, bundle, reproduce, evidence, repair, native_probe, native_reproduce):
        formats = command.add_mutually_exclusive_group()
        formats.add_argument("--json", action="store_true", help="Print machine-readable JSON")
        if command in (demo, preflight, readiness, scan, investigate, rollouts, capture, capture_diff, trajectory, trajectory_diff, windows, window_case, trajectory_case, layout_case, cadence_case, command_case, named_case, context_case, cohort_case, source_case, observation_case, case_check, case_reproduce, native_probe, native_reproduce):
            formats.add_argument("--markdown", action="store_true", help="Create a portable evidence report")
        if command in (retained_metadata, retained_recording, video_recovery, window_case, trajectory_case, layout_case, cadence_case, command_case, named_case, context_case, cohort_case, source_case, observation_case):
            command.set_defaults(output=None)
        else:
            command.add_argument("--output", type=Path, help="Write this report to a new file")
    from .sim_salvage_frontdoor import add_cli_commands

    add_cli_commands(commands)  # check-sim-data, check-salvage: their own output and exit handling
    return root


def _requirement_file(path):
    from .investigation import _requirements

    return _requirements(read_json(path))


def _named_contract(path):
    contract = read_json(path)
    if set(contract) - {'expected_action_names', 'expected_driver_namespace', 'require_canonical_matches_driver_return', 'require_recorded_matches_driver_return'}:
        raise InputError('Named contract contains unsupported keys')
    return contract


def _command_contract(path):
    contract = read_json(path)
    allowed = {'expected_processor', 'expected_action_names', 'expected_consumer_namespace'}
    if set(contract) - allowed:
        raise InputError('Command contract contains unsupported keys')
    return contract


def _interleaved(findings: list, rank: dict) -> list:
    """Most severe first; within one severity, each component's first finding before any component's second, so a
    component with a few deployment-blocking errors is never crowded out by one with many."""
    seen: dict = {}
    keyed = []
    for index, finding in enumerate(findings):
        group = (finding.get('component'), finding.get('status'))
        seen[group] = seen.get(group, -1) + 1
        keyed.append((rank.get(finding.get('status'), len(rank)), seen[group], index))
    return [findings[index] for _, _, index in sorted(keyed)]


def _resolved_lines(result: dict) -> list[str]:
    """An explicitly named path reached through a symlink was resolved once; say so in its own words."""
    return [f"resolved {item.get('given')} → {item.get('real')}" for item in result.get('resolved_paths') or []
            if isinstance(item, dict)]


def human(result: dict, *, brand: str = "Policy Doctor", command_prefix: tuple[str, ...] = ("policy-doctor",)) -> str:
    if result.get('kind') == 'openpi_readiness':
        lines = [f"RoboClue serving readiness: {result['status'].upper()}", str(result.get('summary', ''))]
        lines.extend(_resolved_lines(result))
        for role in ('current', 'reference'):
            part = result.get(role, {})
            outcome = part.get('normalization', {})
            lines.append(f"  {role}: {outcome.get('status', 'unknown')} — {outcome.get('message', '')}")
        for change in result.get('consequential_changes', []):
            lines.append(f"  Reference change {change['field']}: {change['current']} -> {change['restored_reference']}")
        correction = result.get('correction')
        if correction:
            lines.extend(['Conditional supported correction (restore declared reference intent):', correction['command'],
                '  Targeted replay: ' + correction['targeted_requirement_status'] + '; complete serving remains ' + correction['overall_after_correction']])
        for action in result.get('next_actions', []):
            lines.append('Next: ' + action)
        for name, reason in result.get('unknowns', {}).items():
            lines.append('  Unknown ' + name.replace('_', ' ') + ': ' + str(reason))
        lines.append(str(result.get('scope', '')))
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'robot_readiness':
        lines = [f"RoboClue robot readiness: {result['status'].upper()}", str(result.get('summary', ''))]
        lines.extend(_resolved_lines(result))
        for name, axis in result.get('axes', {}).items():
            lines.append(f"  {name.replace('_', ' ')}: {axis.get('status', 'unknown')}")
            lines.extend('    ' + str(reason) for reason in axis.get('reasons', []))
        for finding in result.get('findings', []):
            if finding.get('status') != 'pass':
                lines.append(f"[{finding.get('status', 'unknown').upper()}] {finding.get('code')}: {finding.get('message', '')}")
                if finding.get('remedy'):
                    lines.append('  Next: ' + str(finding['remedy']))
        lines.append(str(result.get('scope', '')))
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'dataset_video_reference_batch':
        from .bulk_workflow import concise_bulk_report
        from .recovery_handoff import handoff_lines

        text = concise_bulk_report(result).splitlines()
        cut = next((i for i, line in enumerate(text) if line.startswith('Next: ')), len(text))
        return '\n'.join(text[:cut] + handoff_lines(result) + text[cut:]) + '\n'
    if result.get('kind') == 'dataset_preflight':
        lines = [f"RoboClue preflight: {result['status'].upper()}", str(result.get('summary', result.get('scope', '')))]
        for name, requirement in result.get('coverage', {}).get('required_assessments', {}).items():
            lines.append('Required ' + name + ': ' + str(requirement.get('status', 'unknown')).upper() +
                         ' — ' + str(requirement.get('scope', 'Only this required assessment.')))
        lines.extend(_resolved_lines(result))
        if result.get('source'):
            lines.append('Source: ' + str(result['source']))
        for name, component in result.get('components', {}).items():
            lines.append(f"  {component.get('title', name.replace('_', ' '))}: {component.get('status', 'unknown')}")
            if name == 'groot_contract' and component.get('headline'):
                lines.append('    ' + str(component['headline']))
                for version, verdict in component.get('versions', {}).items():
                    lines.append(f"    {version}: {verdict.get('status', 'unknown')} — {verdict.get('statement', '')}")
                if component.get('remedy'):
                    lines.append('    Next: ' + str(component['remedy']))
            if name == 'groot_contract' and component.get('consumer'):
                coverage = component.get('coverage', {})
                lines.append('    consumer: ' + str(component['consumer']) + '; modality config: '
                             + (str(component.get('modality_config_name') or 'supplied')
                                if coverage.get('modality_config_supplied') else 'not supplied'))
            if name == 'groot_contract' and component.get('contract_statement'):
                lines.append('    ' + str(component['contract_statement']))
                if component.get('findings_truncated'):
                    counts = component.get('counts') or {}
                    lines.append(f"    findings: {len(component.get('findings', []))} of {component.get('total_findings')} "
                                 "kept in this report, one example of every code and status first, then most severe; "
                                 "component counts cover all: "
                                 + ', '.join(f"{counts.get(s, 0)} {s}" for s in ('error', 'warning', 'unknown', 'pass')))
            if component.get('finding_codes') and (component.get('findings_truncated') or component.get('reduced')):
                lines.append('    by code: ' + '; '.join(
                    f"{code} " + ', '.join(f"{n} {s}" for s, n in by_status.items())
                    for code, by_status in list(component['finding_codes'].items())[:24]
                    if isinstance(by_status, dict)))
            if component.get('reduced') and name != 'groot_contract':
                lines.append('    ' + str(component.get('reduced_statement', '')))
            if name == 'commanded_reach':
                lines.append('    ' + str(component.get('residual_statement', '')))
                coverage = component.get('coverage') or {}
                if 'rows' in coverage:
                    lines.append(f"    read rows: {coverage['rows']} in {coverage.get('episodes')} episodes "
                                 f"({coverage.get('data_files_read')} data file(s))")
                lines.append('    ' + str(component.get('residual_limits', '')))
            if name == 'units' and component.get('compared_with'):
                coverage = component.get('coverage') or {}
                lines.append('    compared with: ' + str(component['compared_with']))
                if coverage.get('values_read') is False and coverage.get('reason'):
                    lines.append('    values not read: ' + str(coverage['reason']))
                summarized = [(key, entry['coordinates_summary']) for key, entry in
                              ((component.get('manifest') or {}).get('features') or {}).items()
                              if isinstance(entry, dict) and isinstance(entry.get('coordinates_summary'), dict)]
                if summarized:
                    lines.append('    summarized: ' + ', '.join(f"{key} ({summary['not_listed']} of {summary['total']} "
                                                                f"non-motor coordinates counted, not listed)"
                                                                for key, summary in summarized[:6]))
                lines.append('    ' + str(component.get('units_limits', '')))
            if name == 'action_conventions' and component.get('verdicts'):
                verdicts = component['verdicts']
                lines.append(f"    delta transform consistency: {verdicts['consumer']}; dataset fingerprint: {verdicts['dataset_fingerprint']}")
                if verdicts.get('consumers'):
                    lines.append('    declared transform recipes: ' + ', '.join(verdicts['consumers']))
                if len(verdicts.get('per_consumer', {})) > 1:
                    for consumer, verdict in verdicts['per_consumer'].items():
                        lines.append(f'      {consumer} delta transform consistency: {verdict}')
                lines.append('    This compares declared delta transforms; keys, framework versions and native training compatibility are not established.')
                lines.append('    ' + str(component.get('convention_limits', '')))
            if name == 'consumer_comparison' and component.get('kind') == 'consumer_sample_comparison':
                from .consumer_preflight import concise_consumer_comparison

                compact = concise_consumer_comparison(component)
                lines.append('    selected numeric values: ' + compact['numeric_projection_value_effect']
                             + '; equality required: ' + str(compact['require_equal']))
                lines.append('    saved artifact identity changed: ' + str(compact['artifact_identity_changed'])
                             + '; active interpretation changed: ' + str(compact['active_interpretation_changed']))
                lines.append('    window coverage: ' + json.dumps(compact['coverage'], sort_keys=True))
                first = compact['first_observed_divergence']
                if first:
                    lines.append('    first computed divergence: ' + first['stage'] + ' ' + first['field']
                                 + '; change: ' + str(first.get('change')))
                witness = compact['first_value_divergence']
                if witness:
                    lines.append('    first value/source witness: ' + json.dumps(witness, sort_keys=True))
                if compact['first_final_value_difference']:
                    lines.append('    first final value difference: '
                                 + json.dumps(compact['first_final_value_difference'], sort_keys=True))
                    lines.append('    final value source: ' + json.dumps(compact['first_final_value_source'], sort_keys=True))
                lines.append('    full native processor and model target: UNKNOWN')
            if name == 'conversion_fidelity':
                lines.append('    numeric association: ' + str(component.get('numeric_association_status', 'unknown')))
                binding = component.get('criterion_reuse')
                if isinstance(binding, dict) and binding.get('requested'):
                    lines.append('    saved criterion: ' + str(binding.get('status', 'unknown'))
                                 + '; matches: ' + str(binding.get('matches')))
                    lines.append('      expected: ' + str(binding.get('expected_sha256')))
                    lines.append('      current: ' + str(binding.get('current_sha256')))
                    lines.append('      candidate evaluated under saved criterion: '
                                 + str(binding.get('candidate_evaluated_under_expected_criterion', False)))
                    if binding.get('status') != 'pass':
                        if binding.get('drift_fields'):
                            lines.append('      changed criterion field: ' + str(binding['drift_fields'][0]))
                        lines.append('      Next: restore intended original/recipe/verifier, or explicitly save a new baseline; keep the old report.')
                if 'joint_numeric_association_status' in component:
                    lines.append('    joint association: ' + str(component['joint_numeric_association_status']))
                if 'auxiliary_numeric_association_status' in component:
                    lines.append('    auxiliary commands: ' + str(component['auxiliary_numeric_association_status']))
                for field, checked in component.get('auxiliary_fields', {}).items():
                    lines.append(f"      {field}: {checked.get('status', 'unknown')} "
                                 f"({checked.get('compared_rows', 0)}/{checked.get('expected_rows', '?')} rows)")
                coverage = component.get('coverage', {})
                compared = coverage.get('compared_numeric_rows', coverage.get('compared_selected_rows'))
                if compared is not None:
                    lines.append(f"    compared source/export rows: {compared}")
                if 'findings_total' in component:
                    lines.append(f"    conversion findings: {component['findings_total']} "
                                 f"({len(component.get('findings', []))} retained in this report)")
                lines.append('    native consumption: ' + str(component.get('native_consumption', {}).get('status', 'unknown')))
            if name == 'robot_readiness':
                for axis, outcome in component.get('axes', {}).items():
                    lines.append(f"    {axis.replace('_', ' ')}: {outcome.get('status', 'unknown')}")
        if result.get('coverage', {}).get('units_not_run'):
            lines.append('  units: not run. ' + str(result['coverage']['units_not_run']))
        if result.get('coverage', {}).get('units_skipped'):
            lines.append('  units: skipped (' + str(result['coverage']['units_skipped']) + ')')
        for left_out in (result.get('coverage', {}).get('components_left_out_at_size_bound') or {}).values():
            lines.append('  ' + str(left_out.get('reason') if isinstance(left_out, dict) else left_out))
        attention = _interleaved([f for f in result.get('findings', []) if f.get('status') != 'pass'],
                                 {'error': 0, 'unknown': 1, 'warning': 2})
        for finding in attention[:12]:
            if finding.get('status') != 'pass':
                lines.append(f"[{finding.get('status', 'unknown').upper()}] {finding.get('code')}: "
                             + (f"[{finding['groot_version']}] " if finding.get('groot_version') else '')
                             + str(finding.get('message', finding.get('reason', finding.get('detail', ''))))
                             + (f" (applies to {finding['applies_to']})" if finding.get('applies_to') else ''))
                if finding.get('remedy'):
                    lines.append('  Next step: ' + str(finding['remedy']))
        lines.extend('Next: ' + str(action) for action in result.get('next_actions', [])[:6])
        lines.append(str(result.get('scope', '')))
        return '\n'.join(lines) + '\n'
    if result.get('kind', '').startswith('video_reference_recovery'):
        from .video_recovery_report import concise_video_recovery

        return concise_video_recovery(result)
    if result.get('kind') in ('retained_check_export', 'retained_check_run', 'retained_check_reproduction', 'retained_candidate_check'):
        from .retained_report import concise_retained_check

        return concise_retained_check(result)
    if result.get('kind') == 'checkpoint_check':
        from .check_report import concise_check

        return concise_check(result)
    if result.get('kind') == 'generated_policy_demo':
        return ('Created synthetic policy-order example: ' + str(result['destination']) + '\n'
                'Saved processor order and dataset order deliberately disagree. No robot or public incident.\n'
                'Next: ' + result['next_command'] + '\n'
                'Corrected metadata control: ' + str(result['corrected_dataset']) + '\n')
    if result.get('kind') in ('native_processor_source_observation', 'native_processor_source_reproduction',
                              'native_observation_source_observation', 'native_observation_source_reproduction'):
        from .reports import _investigation_details

        observation = result['kind'].startswith('native_observation_source')
        audit_kind = 'native_observation_source_audit' if observation else 'native_processor_source_audit'
        lines = [('Native input preservation check: ' if observation else 'Native checkpoint source check: ') + result.get('status', 'unknown'),
                 'Native execution: ' + result.get('execution_status', 'not evaluated') + '.']
        for side in ('candidate', 'corrected'):
            if type(result.get(side)) is dict:
                lines.append(side.capitalize() + ' source evidence:')
                lines.extend(_investigation_details({'audits': [dict(result[side], kind=audit_kind)]}))
        for key in ('reproduction_matches', 'verifier_matches'):
            if key in result:
                lines.append(key.replace('_', ' ') + ': ' + str(result[key]) + '.')
        for key in ('candidate_directory', 'corrected_directory'):
            if result.get(key):
                lines.append(key.replace('_', ' ') + ': ' + result[key])
        if result.get('candidate_directory') and result.get('corrected_directory'):
            lines.append('Investigate: ' + shlex.join([*command_prefix, 'investigate',
                result['candidate_directory'], '--reference', result['corrected_directory']]))
        if result.get('retention_note'):
            lines.append(result['retention_note'])
        lines.append(result.get('scope', 'Native observations do not establish policy competence.'))
        return '\n'.join(lines) + '\n'
    if result.get('kind') in ('native_processing_probe', 'native_processing_reproduction'):
        probe = result.get('probe', {})
        lines = [f"Native processing observation: {result.get('status', 'unknown')}",
                 f"Provider: {probe.get('provider', 'unknown')}; execution: {probe.get('execution_status', 'not evaluated')}.",
                 result.get('scope', probe.get('scope', ''))]
        if 'reproduction_matches' in result:
            lines.append(f"Retained numerical replay matches: {result['reproduction_matches']}.")
        if result.get('verifier_matches') is False:
            lines.append('Configured runtime/source differs from the retained verifier; no candidate evaluated.')
        if result.get('input_selection'):
            selection = result['input_selection']
            lines.append(f"Input selection: {len(selection.get('files', []))} original dependency files, copied without changes.")
            for omission in selection.get('omissions', []):
                lines.append('Input scope: ' + str(omission))
        for sample in probe.get('samples', []):
            identity = sample.get('identity', {})
            row_selection = sample.get('selection_basis') == 'single_episode_physical_row'
            coordinate = ('physical row ' + str(identity.get('physical_row', 'unknown')) if row_selection
                          else 'frame ' + str(identity.get('frame_index', 'unknown')))
            lines.append(f"Sample episode {identity.get('episode_index', 'unknown')}, {coordinate}.")
            if sample.get('selection_basis'):
                lines.append('  Selection basis: ' + sample['selection_basis'] + '.')
            for route in sample.get('routes', []):
                line = f"  {route.get('name', 'native route')}: {route.get('status', 'unknown')}"
                if route.get('dimensions'):
                    line += '; dimensions ' + json.dumps(route['dimensions'], sort_keys=True)
                if route.get('error'):
                    line += '; ' + json.dumps(route['error'], sort_keys=True)
                lines.append(line)
        for difference in result.get('observed_differences', []):
            frame = difference.get('frame_index')
            coordinate = ('frame ' + str(frame) if frame is not None else
                          'physical row ' + str(difference.get('physical_row', 'unknown')))
            lines.append(f"Observed route difference, episode {difference.get('episode_index', 'unknown')}, {coordinate}: "
                         f"{difference.get('reference_route')} -> {difference.get('candidate_route')}.")
            for changed in difference.get('changed_outputs', []):
                def compact(value):
                    if not isinstance(value, dict):
                        return json.dumps(value, allow_nan=False)
                    parts = []
                    if 'shape' in value:
                        parts.append('shape ' + json.dumps(value['shape']))
                    if 'dtype' in value:
                        parts.append(str(value['dtype']))
                    if 'min' in value or 'max' in value:
                        parts.append('range ' + json.dumps([value.get('min'), value.get('max')], allow_nan=False))
                    if isinstance(value.get('preview'), list) and len(value['preview']) <= 4:
                        parts.append('values ' + json.dumps(value['preview'], allow_nan=False))
                    return ', '.join(parts) if parts else json.dumps(value, sort_keys=True, allow_nan=False)
                lines.append('  ' + str(changed.get('key', 'output')) + ': ' + compact(changed.get('reference'))
                             + ' -> ' + compact(changed.get('candidate')) + '.')
        if probe.get('error'):
            lines.append('Native observation failed: ' + json.dumps(probe['error'], sort_keys=True))
        if result.get('directory'):
            lines.append('Numerical capsule: ' + str(result['directory']))
        for unknown in probe.get('unverified', []):
            lines.append('Unverified: ' + str(unknown))
        lines.append('Use --json for source hashes, native parameters, input/output summaries and retained exports.')
        return '\n'.join(lines) + '\n'
    if (result.get('applied_requirements') or {}).get('kind') == 'evaluation-cohort' and 'audits' in result:
        from .reports import _cohort_investigation_lines

        return '\n'.join(_cohort_investigation_lines(result)) + '\n'
    if result.get('kind') == 'run_boundary_comparison':
        from .reports import _investigation_details

        return '\n'.join(_investigation_details({'comparison': result}) + [result.get('scope', '')]) + '\n'
    if result.get('kind') == 'named_action_audit':
        lines = [f"Native named-action boundary: {result['status']}",
                 f"Binding: {result['binding_status']}; numeric replay: {result['numeric_replay_status']}; explicit contract: {result['acceptance_contract']['status']}.",
                 'Actual named order: ' + json.dumps(result['action_names']),
                 'Declared engine order: ' + json.dumps(result['engine_action_names']),
                 'Software driver namespace: ' + json.dumps(result['driver_namespace']),
                 f"Observed driver changes: {result['driver_change_events']}; canonical dictionary changes: {result['canonical_mutation_events']}."]
        for violation in result['acceptance_contract']['violations'][:4]:
            lines.append('Contract violation: ' + json.dumps(violation))
        recording = result.get('recording', {})
        lines.append(f"Recorded rows: {result.get('recorded_action_status', 'unknown')}; buffer binding {recording.get('buffer_binding_status', 'unknown')}; persistence {recording.get('persistence_status', 'unknown')}.")
        if recording.get('first_difference') is not None:
            lines.append('First recorded/driver difference: ' + json.dumps(recording['first_difference']))
        lines.extend(['Legal limiting is not automatically a fault. Earlier policy selection/processing and physical execution remain unknown.',
                      result['scope'], 'Use --json for native snapshots and original-byte evidence.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') in ('command_path_audit', 'command_path_comparison'):
        lines = [f"Software command path: {result['status']}"]
        if result['kind'] == 'command_path_audit':
            criterion = result['acceptance_contract']
            lines.extend([f"Binding: {result['binding_status']}; numeric replay: {result['numeric_replay_status']}; explicit contract: {criterion['status']}.",
                          'Declared tensor order: ' + json.dumps(result['action_names']),
                          'Software consumer namespace: ' + json.dumps(result['consumer_namespace']),
                          'Observed effective processor: ' + json.dumps(result.get('observed_processor'))])
            for tick in result.get('ticks', [])[:8]:
                lines.append(f"Tick {tick['tick_id']}: {tick['outcome']}; binding {tick['binding_status']}; replay {tick['numeric_replay_status']}.")
            for violation in criterion.get('violations', [])[:4]:
                lines.append('Contract violation: ' + json.dumps(violation))
        else:
            lines.extend([f"Comparison supported: {result.get('comparison_supported')}; equal over compared ticks: {result.get('equal_over_compared_ticks')}.",
                          'First differing stage: ' + json.dumps(result.get('first_difference')),
                          'Context differences: ' + json.dumps(result.get('context_differences', [])),
                          'A lawful recipe or consumer-return difference is not automatically a fault.'])
        lines.extend(['Declared names describe tensor order, not an executed named-joint mapping.',
                      'Physical execution, recorded actions and historical query lineage remain unknown.',
                      result.get('scope', ''), 'Use --json or --markdown for native locators and source hashes.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'query_cadence_audit':
        criterion = result['cadence_criterion']
        return (f"{brand} observed query cadence: {result['status']}\n"
                f"Maximum selection calls per query: {result['consumer_requirement']['max_query_gap']}.\n"
                f"Selections: {criterion['selection_count']}; queries: {criterion['query_count']}.\n"
                f"Criterion: {criterion['status']}; {criterion['reason']}\n"
                f"First violation: {json.dumps(criterion['first_violation'])}\n"
                f"{result['scope']}\n")
    if result.get('kind') in ('action_selection_audit', 'action_selection_comparison'):
        lines = [f"{brand} action selection: {result['status']}"]
        if result['kind'] == 'action_selection_audit':
            lines.append(f"Selection calls: {result['selection_count']}; supported source: {result['source_supported']}.")
            for event in result['events'][:12]:
                lines.append(f"Event {event['event_id']} ({event['type']}): replay {event['numeric_replay']['status']}; origin {event['origin']['status']}.")
                if event.get('local_binding'):
                    binding = event['local_binding']
                    lines.append(f"  Local binding: {binding['status']}; {binding.get('basis', binding.get('reason', ''))}.")
                arithmetic = event.get('arithmetic_contributors', {})
                if arithmetic.get('status') == 'pass':
                    lines.append('  Arithmetic contributions, conditional on no external state mutation:')
                for item in arithmetic.get('contributors', [])[:4]:
                    lines.append(f"  Query {item['query_id']}, offset {item['offset']}, weight {item['weight']:.6g}.")
        else:
            lines.append('First selected difference: ' + json.dumps(result['first_selected_difference']))
            lines.append(result['alignment_basis'])
        lines.extend([result['scope'], 'Use --json or --markdown for source hashes, saved-array locators and full query contributions.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') in ('resolved_layout_audit', 'resolved_layout_comparison'):
        lines = [f"{brand} raw layout: {result['status']}"]
        if result['kind'] == 'resolved_layout_audit':
            lines.append('Resolved BODY order: ' + json.dumps(result['layout'].get('body_names')))
            lines.append('Expected BODY order: ' + json.dumps(result['acceptance_contract'].get('expected_names')))
            lines.append('Saved-buffer/raw-term replay: ' + str(result['raw_binding'].get('status')))
            lines.append('Processed-term stage: ' + str(result['processed_binding'].get('status')))
            for finding in result.get('findings', [])[:4]:
                lines.append(f"{finding.get('status')}: {finding.get('code')}; {finding.get('message', '')}")
        else:
            lines.extend([f"Same ordered layout: {result.get('ordered_layout_equal')}; same raw values: {result.get('raw_values_equal')}; same named values: {result.get('named_values_equal')}.",
                          'Different raw order can preserve named values. Matching numbers alone do not establish the same policy layout.'])
        lines.extend([result.get('scope', ''),
                      'Physical sampling, live origin and policy concatenation remain unverified. Use --json or --markdown for native evidence.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'incident_candidate_check':
        lines = [f"{brand} retained regression: {result['status']}",
                 f"Case verified: {result['case_verified']}; candidate evaluated: {result['candidate_evaluated']}."]
        if result.get('candidate'):
            candidate = result['candidate']
            lines.append(f"Later artifact: {candidate['status']}; expected {result['expected_status']}; same retained parameters.")
            if candidate.get('kind') == 'named_action_audit':
                lines.append('Named boundary contract: ' + candidate['acceptance_contract']['status'])
                lines.append('Actual named order: ' + json.dumps(candidate['action_names']))
                lines.append('Legal driver limiting is only incompatible when the retained contract explicitly requires equality.')
            elif candidate.get('kind') == 'command_path_audit':
                lines.append('Software command contract: ' + candidate['acceptance_contract']['status'])
                lines.append('Declared tensor order: ' + json.dumps(candidate['action_names']))
                for violation in candidate['acceptance_contract'].get('violations', [])[:2]:
                    lines.append('Contract violation: ' + json.dumps(violation))
                lines.append('Physical execution and recorded actions remain unknown.')
            elif candidate.get('kind') == 'query_cadence_audit':
                criterion = candidate['cadence_criterion']
                lines.append(f"Observed query cadence: {criterion['status']}; max_query_gap {candidate['consumer_requirement']['max_query_gap']}.")
                lines.append('First violation: ' + json.dumps(criterion['first_violation']))
            elif candidate.get('kind') == 'resolved_layout_audit':
                lines.append('Resolved BODY order: ' + json.dumps(candidate['layout'].get('body_names')))
                lines.append('Expected BODY order: ' + json.dumps(candidate['acceptance_contract'].get('expected_names')))
                lines.append('Saved raw binding: ' + candidate['raw_binding']['status'])
                for finding in candidate.get('findings', [])[:2]:
                    lines.append(f"{finding['code']}: {finding.get('message', '')}")
            elif candidate.get('findings'):
                first = candidate['findings'][0]
                if isinstance(first, dict) and 'code' in first:
                    lines.append(f"First finding: {first['code']}; observed {json.dumps(first.get('observed'))}; expected {json.dumps(first.get('expected'))}.")
                else:
                    lines.append('First finding: ' + json.dumps(first, ensure_ascii=False))
            if candidate.get('sampled_limit_violation_count'):
                lines.append(f"Supplied-bound sample violations: {candidate['sampled_limit_violation_count']}.")
        lines.extend([result.get('scope', ''), 'Use --json for retained parameters and native evidence.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') in ('exported_incident_case', 'incident_case_reproduction'):
        lines = [f"{brand} incident case: {result['status']}"]
        if result.get('retained_context'):
            lines.append('Additional retained corrected-source context: ' + str(result['retained_context']))
        if result.get('directory'):
            lines.extend(['Created: ' + json.dumps(result['directory']),
                          'Rerun: ' + shlex.join([*command_prefix, 'reproduce-case', result['directory']])])
        if 'reproduction_matches' in result:
            lines.append(f"Recorded findings reproduced: {result['reproduction_matches']}; verifier matches: {result['verifier_matches']}; lifecycle verified: {result.get('lifecycle_verified')}.")
        checks = result.get('checks', {})
        for name in ('before', 'after'):
            check = checks.get(name, {})
            lines.append(f"{'Original finding' if name == 'before' else 'Corrected check'}: {check.get('status', 'unknown')}.")
        for control in checks.get('controls', []):
            lines.append(f"Control {json.dumps(control['name'])}: {control['check']['status']}; expected {control['expected_status']}.")
        change = result.get('change_verification', {})
        if change.get('method') == 'exact_native_metadata_copy':
            lines.append(f"Metadata cells changed: {change['logical_cells_changed']}; schema preserved: {change['schema_preserved']}; non-target files unchanged: {change['non_target_files_unchanged']}.")
            for cell in change['cells'][:3]:
                lines.append(f"  {json.dumps(cell['path'])}, row {cell['row']}, {cell['column']}: {cell['observed']} -> {cell['expected']}.")
        elif change.get('method') == 'supplied_native_named_action':
            lines.append('Supplied named-action snapshots retain the same explicit order, software namespace and optional equality requirement; new values are permitted.')
        elif 'command_path' in change.get('method', ''):
            lines.append('Separate supplied command-path captures; same effective-processor, declared tensor-order and software-namespace criteria. New action values and lawful consumer returns are permitted.')
        elif change.get('method') == 'supplied_act_query_cadence_capture':
            lines.append('Separate supplied ACT executions; same explicit selection-call query requirement and numeric tolerances.')
        elif change.get('method') == 'supplied_resolved_layout_capture':
            lines.append('Supplied corrected raw-term layout; same evidenced sensor namespace, selected bodies and retained criteria.')
        elif change.get('method') == 'supplied_recorded_context_reference':
            lines.append('Selected recorded settings are checked against the supplied reference; numeric evidence is excerpted, media omissions are explicit. This is a comparison selection, not a behavioral repair.')
        elif change:
            lines.append('Supplied corrected native trajectory; matching joint set and unchanged check parameters verified.')
        lines.extend([result.get('scope', ''), 'Use --json for input identities, native findings and verifier details.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'episode_window_audit':
        lines = [f"{brand} episode windows: {result['status']}", result['scope']]
        for finding in result['findings'][:6]:
            if finding.get('message'):
                # Episode consistency findings carry their own explanation and LeRobot consequence.
                lines.append(f"{finding['status']}: {finding['code']}; {finding['message']}")
            elif finding.get('reason'):
                lines.append(f"{finding['status']}: {finding['code']}; {finding['reason']}")
            else:
                lines.append(f"{finding['status']}: {finding['code']}; episode {finding.get('episode_index')}, {finding.get('column', 'metadata')}; declared {json.dumps(finding.get('observed'))}; actual {json.dumps(finding.get('expected'))}.")
        if len(result['findings']) > 6:
            lines.append(f"{len(result['findings']) - 6} more finding(s); use --json for all of them.")
        consistency = result.get('coverage', {}).get('episode_consistency', '')
        if consistency.startswith('not checked'):
            lines.append('Episode count/span consistency ' + consistency + '.')
        if result.get('metadata_patches'):
            lines.append('Metadata-only candidate changes; source files were not changed:')
            for patch in result['metadata_patches'][:4]:
                lines.append(f"  {json.dumps(patch['path'])}, row {patch['row']}, {patch['column']}: {patch['observed']} -> {patch['expected']}.")
        query = result.get('query')
        if query:
            lines.append(f"Episode {query['episode_index']}, absolute anchor {query['anchor_index']}, offsets {query['delta_indices']}.")
            for key in ('declared', 'corrected_what_if'):
                window = query.get(key)
                if window:
                    lines.append(f"{key}: status {str(window.get('status', 'unknown')).upper()}; indices {window.get('indices')}; action_is_pad {window.get('action_is_pad')}; episode IDs {window.get('selected_episode_indices')}.")
                    if window.get('reason'):
                        lines.append('  Reason: ' + str(window['reason']))
                    lines.append(f"  Unpadded cross-episode witnesses: {len(window.get('cross_episode_witnesses', []))}.")
                    if window.get('values') is not None:
                        values = window['values']
                        preview = [row[:8] if isinstance(row, list) else row for row in values[:3]]
                        lines.append(f"  Action preview (first 3 rows, up to 8 coordinates): {json.dumps(preview)}; value status {window.get('value_status')}.")
        if result.get('query'):
            lines.append('The corrected window is a numerical what-if. Audit a saved corrected copy to verify its metadata.')
        lines.extend([
                      'Video alignment, policy outcomes and robot execution remain outside this check.',
                      'Use --json or --markdown for native row locators, exact hashes, coverage and bounded previews.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'action_boundary_audit':
        lines = [f"{brand} action boundary: {result['status']}",
                 f"Events checked: {result['events_checked']}; recording contradictions: {result['counts']['error']}; unknown: {result['counts']['unknown']}",
                 f"Driver input/return changes: {result['driver_change_events']} (observations)"]
        first = result['first_recording_contradiction']
        if first:
            lines.append(f"First contradiction: {json.dumps(first['event_id'])}, frame {first['frame_index']}, declared role {first['declared_recorded_role']}")
            for d in first['recording_difference_preview']:
                lines.append(f"  {json.dumps(d['channel'])}: expected {d['expected']}; recorded {d['observed']}")
        lines.extend([result['coverage'], result['scope'], 'Use --json for event previews, locators and file hashes.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') in ('exported_trajectory_case', 'trajectory_case_reproduction'):
        lines = [f"{brand} case: {result['status']}"]
        if 'directory' in result:
            lines.append(f"Created: {result['directory']}")
            lines.append('Rerun: ' + shlex.join([*command_prefix, 'reproduce-case', result['directory']]))
        else:
            lines.append(f"Recorded findings reproduced: {result['reproduction_matches']}; verifier matches: {result['verifier_matches']}")
        for name, check in result['checks'].items():
            status = check['status'] if isinstance(check, dict) else check
            if name == 'sample' and isinstance(check, dict) and 'sample_count' in check:
                lines.append(f"Sampled reference: {status}; complete: {check['sampling_complete']}; requested samples: {check['sample_count']}.")
                lines.append(f"Supplied-bound violations: {check['sampled_limit_violation_count']}; envelope review hints: {check['excursion_count']}.")
                lines.append('Execution and physical cause: unknown. This case records sampling, not a comparison control.')
                continue
            lines.append(f"{name.capitalize()} comparison: {status}")
            if isinstance(check, dict) and check['first_position_difference']:
                first = check['first_position_difference']
                lines.append(f"  First sampled position difference: {first['time_from_start_ns']} ns; see --json for joint values.")
            if isinstance(check, dict):
                for role, sampling in check['sampling'].items():
                    if sampling['sampled_limit_violations']:
                        lines.append(f"  {role.capitalize()}: {sampling['sampled_limit_violation_count']} supplied-bound violations; first 8 at most in --json, full recorded evidence in expected.json.")
                if not check['comparison_complete']:
                    lines.append('  Incomplete comparison: equality remains unknown.')
        lines.append(result['scope'])
        if result.get('version_notice'):
            lines.append(result['version_notice'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'generated_trajectory_demo':
        return (f"Demo created: {result['destination']}\n"
                "Same waypoint positions and times. At 0.75 seconds:\n"
                f"  Linear reference: 0.75; candidate: {result['candidate_position_at_750ms'][0]}.\n"
                f"Fixture bounds: {result['candidate_bounds_status']}; corrected: {result['corrected_status']}; "
                f"valid joint reorder: {result['valid_reordering']}; missing initial state: {result['missing_initial_state']}.\n"
                f"Read: {result['destination']}/README.md\n"
                "Generated arithmetic example. Execution and physical cause remain unknown.\n")
    if result.get('kind') == 'joint_trajectory_comparison':
        def preview(values):
            return json.dumps(values[:8]) + (f" (first 8 of {len(values)})" if len(values) > 8 else '')

        lines = [f"{brand}: {result['status']}",
                 f"Same waypoint positions and times: {result['same_waypoint_positions_and_times']}",
                 f"Joint alignment: {result['joint_alignment']}",
                 'Aligned joint order: ' + preview(result['reference']['joint_names']),
                 f"Comparison complete: {result['comparison_complete']}; sampled reference equality: {result['sampled_references_equal']}"]
        for field, difference in result['different_sample_fields'].items():
            left, right = difference['reference_values'], difference['candidate_values']
            changed = [index for index in range(max(len(left), len(right)))
                       if index >= len(left) or index >= len(right) or left[index] != right[index]]
            lines.append(f"First {field} difference at {difference['time_from_start_ns']} ns:")
            for index in changed[:8]:
                name = result['reference']['joint_names'][index]
                before = json.dumps(left[index]) if index < len(left) else 'absent'
                after = json.dumps(right[index]) if index < len(right) else 'absent'
                lines.append(f"  {json.dumps(name)}[{index}]: reference {before}; candidate {after}")
            if len(changed) > 8:
                lines.append(f"  {len(changed) - 8} more changed coordinates; see --json for all values.")
        if not result['comparison_complete']:
            lines.append('Incomplete comparison: equal sampled references are not established.')
        for label in ('reference', 'candidate'):
            report = result[label]
            computed = sum(row['status'] == 'sampled' for row in report['samples'])
            lines.append(f"{label.capitalize()} sampling/bounds: {report['status']}; "
                         f"samples {computed}/{len(report['samples'])}; "
                         f"declared bounds: {'supplied' if report['position_limits'] else 'none supplied'}; "
                         f"waypoint-envelope hints: {report['excursion_count']}")
        lines.append('Envelope hints are review items, not physical limits; bounds cover requested samples only.')
        for artifact in result['artifacts']:
            lines.append(f"Evidence: {json.dumps(artifact['path'])}; SHA-256 {artifact['sha256']}")
        lines.extend(['Execution status: unknown. Physical cause: unknown.',
                      'Use --json for full sampled arrays or --markdown for the evidence report.'])
        return '\n'.join(lines) + '\n'
    if result.get('kind') == 'joint_trajectory_samples':
        sampled = sum(row['status'] == 'sampled' for row in result['samples'])
        lines = [f"{brand} reference samples: {result['status']}",
                 f"Sampled {sampled}/{len(result['samples'])} times across {len(result['joint_names'])} named joints; selected model: {result['interpolation_method']}.",
                 (f"Supplied-bound violations: {len(result['sampled_limit_violations'])}." if result['position_limits'] else 'Reference bounds: none supplied; bounds check not performed.'),
                 f"Adjacent-waypoint-envelope review hints: {result['excursion_count']}."]
        for hint in result['waypoint_envelope_excursions'][:3]:
            lines.append(f"  {json.dumps(hint['joint'])} at {hint['time_from_start_ns']} ns: {hint['value']}; waypoint envelope {hint['waypoint_envelope']}.")
        for row in result['samples']:
            if row['status'] == 'unknown':
                lines.append(f"Unresolved at {row['time_from_start_ns']} ns: {row['reason']}")
                break
        lines.extend(['Envelope hints can be intentional; physical limits are not inferred.',
                      f"Pinned reference: {result['upstream_reference']['url']}",
                      'Execution and physical cause: unknown. Current controller upsampling/replacement behavior is unmodeled.',
                      'Use --json for all sampled values and hashes or --markdown for an evidence report.'])
        return '\n'.join(lines) + '\n'
    lines = [f"{brand}: {result.get('status', result.get('kind', 'report'))}"]
    if "counts" in result:
        lines.append(" | ".join(f"{key}: {value}" for key, value in result["counts"].items()))
    if result.get("scope"):
        lines.append(result["scope"])
    if result.get("summary"):
        lines.append(result["summary"])
    if result.get('retained_context'):
        lines.append('Additional retained corrected-source context: ' + str(result['retained_context']))
    if "first_observed_difference" in result:
        first = result["first_observed_difference"]
        lines.append("First observed difference: " + (f"{first['stream']} row {first['row']}, coordinate {first['coordinate']}" if first else "none in compared rows"))
        selected = any(a.get('stream_selection') for a in result['artifacts'])
        coverage = 'Selected-stream comparison complete' if selected else 'Comparison complete'
        lines.append(f"{coverage}: {result['comparison_complete']}; causal conclusion: {result['causal_conclusion']}")
        for artifact in result['artifacts']:
            selection = artifact.get('stream_selection')
            if selection:
                lines.append(f"Selected streams in {json.dumps(artifact['path'])}: {json.dumps(selection['selected'])}; excluded: {json.dumps(selection['excluded'])}.")
                lines.append(selection['scope'])
    if "candidate_findings" in result:
        lines.append(f"Before: {result['before_status']} → candidate: {result['candidate_status']}")
        for key in ("resolved_error_counts", "introduced_error_counts", "remaining_error_counts", "candidate_unknown_counts"):
            lines.append(f"{key}: {json.dumps(result[key], sort_keys=True)}")
    for finding in result.get("findings", result.get("candidate_findings", [])):
        if finding.get("status") == "pass":
            continue
        lines.append(f"\n[{finding.get('status', 'error').upper()}] {finding['code']}: {finding.get('message', finding.get('detail', ''))}")
        evidence_items = finding.get("evidence", [])
        if isinstance(evidence_items, dict):
            evidence_items = [evidence_items]
        for evidence in evidence_items[:2]:
            lines.append(f"  {evidence['path']}#{evidence.get('pointer', 'row=' + str(evidence.get('row')) + ', column=' + str(evidence.get('column')))}")
        if finding.get("remedy"):
            lines.append("  Next: " + finding["remedy"])
    if 'inventory' in result:
        from .reports import _investigation_details

        lines.extend(_investigation_details(result))
        coverage = result.get('coverage') or {}
        for key, label in (('tabular_files_unrecognized', 'Not a recognized rollout recording'),
                           ('tabular_files_not_inspected', 'Not inspected')):
            listed = coverage.get(key) or []
            for item in listed[:3]:
                lines.append(f"{label}: {json.dumps(item['path'])}: {item['reason']}")
            more = coverage.get(key + '_count', len(listed)) - min(3, len(listed))
            if more > 0:
                lines.append(f"  ... and {more} more (listed in coverage; use --json).")
        for item in result['inventory'].get('trajectories', [])[:3]:
            if 'point_count' in item:
                lines.append(f"Trajectory: {json.dumps(item['path'])}; {item['point_count']} points, {len(item['joint_names'])} joints; derivatives {json.dumps(item['derivative_points'])}.")
        for problem in result.get('intake_problems', [])[:3]:
            lines.append(f"Intake: {json.dumps(problem['path'])}: {problem['reason']}")
        for action in result['next_actions'][:8]:
            args = action['arguments']
            if action['tool'] == 'sample_ros_trajectory' and args is not None:
                command = [*command_prefix, 'sample-trajectory', args['trajectory'], '--times',
                           *map(str, args['sample_times']), '--method', args['interpolation_method']]
                lines.append('Suggested check: ' + shlex.join(command))
            elif action['tool'] == 'reproduce_incident_case' and args is not None:
                lines.append('Suggested check: ' + shlex.join([*command_prefix, 'reproduce-case', args['directory']]))
            elif action['tool'] == 'audit_dataset_windows' and args is not None:
                lines.append('Suggested check: ' + shlex.join([*command_prefix, 'audit-windows', args['dataset']]))
            else:
                lines.append(f"Next: {action['tool']}: {action['purpose']}")
        lines.append('Suggestions do not establish pairing, active controller configuration or execution. Use --json or --markdown for full evidence and callable tool arguments.')
    elif ("findings" not in result and "candidate_findings" not in result) or "commands" in result or "coverage" in result:
        lines.append(json.dumps(result, indent=2, allow_nan=False))
    return "\n".join(lines) + "\n"


def _dataset_source(args, run, *, with_video_context=False):
    """Run a dataset check on a local path, or on an explicitly fetched, hash-verified public Hub copy."""
    if not str(args.dataset).startswith('hf://'):
        return run(args.dataset)
    from .hub import portable_dataset_report, resolve_dataset_content

    if type(args.max_download_mb) is not int or args.max_download_mb < 1:
        raise InputError('--max-download-mb must be a positive integer')
    with resolve_dataset_content(args.dataset, revision=args.revision, fetch=args.fetch,
                                 max_bytes=args.max_download_mb * 1024 * 1024,
                                 video_headers=bool(getattr(args, 'video_headers', False))) as view:
        try:
            value = (run(str(view.root), video_context=getattr(view, 'video_context', None))
                     if with_video_context else run(str(view.root)))
            return portable_dataset_report(value, view)
        except (InputError, OSError) as exc:
            # Never print the private temporary path; name the pinned Hub identity instead.
            identity = f"{args.dataset}@{view.provenance['resolved_revision']}"
            message = str(exc).replace(str(view.root.resolve()), identity).replace(str(view.root), identity)
            raise InputError(message) from exc


def main(argv: list[str] | None = None, *, prog: str = "policy-doctor", brand: str = "Policy Doctor",
         command_prefix: tuple[str, ...] = ("policy-doctor",)) -> int:
    args = parser(prog=prog).parse_args(argv)
    from .sim_salvage_frontdoor import COMMANDS, run_cli

    if args.command in COMMANDS:
        return run_cli(args)
    required_units = None
    try:
        match args.command:
            case 'demo':
                from .demo import create_controller_demo, create_policy_demo

                result = (create_policy_demo if args.kind == 'policy' else create_controller_demo)(args.destination)
            case 'setup':
                from .setup_client import client_configuration

                canonical_options = {'canonical': True} if brand == 'RoboClue' else {}
                sys.stdout.write(client_configuration(args.root or ['.'], allow_hub=args.allow_hub, allow_hub_datasets=args.allow_hub_datasets, allow_hub_processor_state=args.allow_hub_processor_state, allow_openpi_statistics=args.allow_openpi_statistics, name=args.name,
                                                      clients=tuple(args.client or ('claude-code', 'codex', 'json')), **canonical_options))
                return 0
            case 'readiness':
                from .openpi_workflow import check_readiness

                if not 1 <= args.max_download_mb <= 64:
                    raise InputError('--max-download-mb must be between 1 and 64')
                result = check_readiness(robot_command=args.robot_command, robot_config=args.robot_config,
                    checkpoint=args.checkpoint, dataset=args.dataset, policy_revision=args.policy_revision,
                    dataset_revision=args.dataset_revision, fetch=args.fetch, processor_state=args.fetch,
                    max_bytes=args.max_download_mb * 1024 ** 2, max_files=args.max_files,
                    base_directory=str(Path.cwd()), command_directory=args.command_directory,
                    source_root=args.source_root, reference_command=args.reference_command,
                    statistics_generation=args.statistics_generation, reference_statistics_generation=args.reference_statistics_generation,
                    restore_reference_intent=args.restore_reference_intent)
            case 'preflight':
                from .preflight import preflight as run_preflight

                canonical = brand == 'RoboClue'
                selectors = bool(args.robot_command or args.robot_config)
                if args.lerobot_version and not selectors:
                    raise InputError('--lerobot-version needs the actual --robot-command or --robot-config.')
                if args.require and args.units is False:
                    raise InputError('--require deployment-units conflicts with --no-units; the required comparison cannot be skipped.')
                if args.require or (canonical and selectors):
                    from .robot_contract import validate_robot_declaration

                    validate_robot_declaration(robot_command=args.robot_command, robot_config=args.robot_config)
                if args.require or (canonical and selectors and args.units is not False):
                    if not selectors:
                        raise InputError('--require deployment-units needs the actual --robot-command or --robot-config; dataset-only units are not a deployment comparison.')
                    if args.source.startswith('hf://'):
                        raise InputError('Required deployment-units currently needs the materialized local dataset; a Hub label is not a completed units comparison.')
                    from .product_cli import _document
                    from .product_questions import dataset_root

                    _document(dataset_root(args.source) / 'meta/info.json')
                    required_units = {'requirement': 'deployment-units',
                                      'requested_by': 'explicit --require' if args.require else 'canonical robot selector',
                                      'scope': 'Recorded dataset units against the declared robot settings only; other requested preflight components retain their own status.'}
                if not 1 <= args.max_download_mb <= 4096:
                    raise InputError('--max-download-mb must be between 1 and 4096')
                consumer_samples = None
                if args.consumer_sample is not None:
                    consumer_samples = []
                    for value in args.consumer_sample:
                        parts = value.split(':')
                        if len(parts) != 2 or any(not part.isascii() or not part.isdecimal() for part in parts):
                            raise InputError('--consumer-sample requires EPISODE:ANCHOR as nonnegative integers')
                        numbers = [part.lstrip('0') or '0' for part in parts]
                        if len(numbers[0]) > 6 or len(numbers[1]) > 4:
                            raise InputError('--consumer-sample requires bounded stored episode IDs and anchor rows')
                        consumer_samples.append(dict(zip(('episode_index', 'anchor_index'), map(int, numbers), strict=True)))
                result = run_preflight(args.source, checkpoint=args.checkpoint, deployment=args.deployment,
                    fetch=args.fetch, revision=args.revision, checkpoint_revision=args.checkpoint_revision,
                    video_headers=args.video_headers, max_bytes=args.max_download_mb * 1024 * 1024,
                    max_rows=args.max_rows, max_files=args.max_files,
                    robot_command=args.robot_command, robot_config=args.robot_config, processor_state=args.fetch,
                    base_directory=str(Path.cwd()), original_recording=args.original_recording,
                    converter_source=args.converter_source, converter_config=args.converter_config,
                    converter_directory=args.converter_directory, conversion_reference=args.conversion_reference,
                    expected_conversion_criterion=args.expected_conversion_criterion,
                    consumer_reference_dataset=args.consumer_reference_dataset,
                    consumer_reference_checkpoint=args.consumer_reference_checkpoint,
                    consumer_samples=consumer_samples, consumer_require_equal=args.consumer_require_equal,
                    groot_modality_config=args.groot_config, groot_config_name=args.groot_config_name,
                    groot_version=args.groot_version,
                    lerobot_version=args.lerobot_version,
                    require_consumer_version=bool(required_units is not None),
                    commanded_reach=args.commanded_reach, units=args.units, groot_contract=args.groot_contract,
                    openpi_config=args.openpi_config, openpi_source_root=args.openpi_source_root,
                    groot_embodiment=args.groot_embodiment)
                if required_units is not None:
                    part = result.get('components', {}).get('units')
                    state = part.get('status') if isinstance(part, dict) else None
                    required_units['status'] = state if state in ('pass', 'warning', 'error') else 'unknown'
                    deployment = ((part.get('manifest') or {}).get('deployment')
                                  if isinstance(part, dict) else None)
                    if isinstance(deployment, dict) and deployment.get('unknown'):
                        raise InputError('The declared robot settings could not be interpreted: ' + str(deployment['unknown']) + '. No required units gate passed.')
                    if required_units['status'] == 'unknown':
                        required_units['reason'] = 'The required units comparison did not complete; inspect components.units coverage/findings.'
                    result.setdefault('coverage', {}).setdefault('required_assessments', {})['deployment-units'] = required_units
            case 'check':
                from .hub import check_checkpoint

                if args.case:
                    if args.output:
                        raise InputError('Choose --case or --output; the case already retains the full original report')
                    if any(isinstance(value, str) and value.startswith('hf://') for value in (args.source, args.dataset)):
                        raise InputError('Retained metadata checks currently require local original artifacts; Hub intake is not retained')
                    from .retained import retain_check

                    saved = retain_check('policy_metadata', args.source, args.case, dataset=args.dataset,
                        candidate_role=args.candidate_role, parameters={'scope': args.scope})
                    result = dict(saved, kind='retained_check_run', status=saved['original_status'], retained_status=saved['status'])
                else:
                    if args.candidate_role is not None:
                        raise InputError('--candidate-role requires --case')
                    result = check_checkpoint(args.source, args.dataset, fetch=args.fetch,
                        revision=args.revision, dataset_revision=args.dataset_revision, scope=args.scope)
            case "inspect":
                result = (load_checkpoint(args.path) if args.kind == "checkpoint" else load_dataset(args.path)).summary()
            case "audit":
                result = audit(args.checkpoint, args.dataset, args.deployment)
            case "investigate":
                from .investigation import investigate_run

                result = investigate_run(args.directory, reference=args.reference,
                    named_contract=_named_contract(args.named_contract) if args.named_contract else None,
                    requirements=_requirement_file(args.requirements) if args.requirements else None,
                    max_files=args.max_files, max_depth=args.max_depth)
            case "compare-rollouts":
                from .rollouts import compare_rollouts

                if args.case:
                    if args.output:
                        raise InputError('Choose --case or --output; the case already retains the full original report')
                    from .retained import retain_check

                    parameters = {key: getattr(args, key) for key in
                        ('alignment_key', 'reference_episode', 'candidate_episode', 'streams', 'atol', 'rtol', 'max_rows')}
                    saved = retain_check('recording_equivalence', args.candidate, args.case,
                        reference=args.reference, parameters=parameters)
                    result = dict(saved, kind='retained_check_run', status=saved['original_status'], retained_status=saved['status'])
                else:
                    result = compare_rollouts(args.reference, args.candidate, alignment_key=args.alignment_key,
                        reference_episode=args.reference_episode, candidate_episode=args.candidate_episode,
                        atol=args.atol, rtol=args.rtol, max_rows=args.max_rows, streams=args.streams)
            case "audit-capture":
                from .capture import audit_capture

                result = audit_capture(args.capture, args.dataset, expected_names=args.expected_names,
                    expected_frame=args.expected_frame, expected_quantity=args.expected_quantity,
                    expected_identity={'prim_path': args.expected_prim_path} if args.expected_prim_path else None,
                    max_query_gap=args.max_query_gap,
                    command_contract=_command_contract(args.command_contract) if args.command_contract else None,
                    named_contract=_named_contract(args.named_contract) if args.named_contract else None)
            case "compare-captures":
                from .runtime_diff import compare_runtime_captures

                result = compare_runtime_captures(args.reference, args.candidate)
            case 'sample-trajectory':
                from .trajectory import sample_joint_trajectory

                if args.case and args.output:
                    raise InputError('Choose --case or --output; a case already contains the full recorded report')
                limits = read_json(args.position_limits) if args.position_limits else None
                if args.case:
                    from .cases import export_sample_case

                    exported = export_sample_case(args.trajectory, args.case, args.times,
                                                  interpolation_method=args.method, position_limits=limits)
                    result = dict(exported['sampling'], case_directory=exported['directory'], case_verifier=exported['verifier'])
                else:
                    result = sample_joint_trajectory(args.trajectory, args.times, interpolation_method=args.method,
                                                     position_limits=limits)
            case 'compare-trajectories':
                from .trajectory import compare_joint_trajectories

                result = compare_joint_trajectories(args.reference, args.candidate, args.times,
                    reference_method=args.reference_method, candidate_method=args.candidate_method)
            case 'export-trajectory-case':
                from .cases import export_trajectory_case

                result = export_trajectory_case(args.reference, args.candidate, args.destination, args.times,
                    control=args.control, reference_method=args.reference_method, candidate_method=args.candidate_method,
                    control_method=args.control_method, position_limits=read_json(args.position_limits) if args.position_limits else None)
            case 'retain-check':
                from .retained import retain_check

                if args.check == 'policy-metadata':
                    result = retain_check('policy_metadata', args.source, args.destination,
                        dataset=args.dataset, candidate_role=args.candidate_role, parameters={'scope': args.scope})
                else:
                    parameters = {key: getattr(args, key) for key in
                        ('alignment_key', 'reference_episode', 'candidate_episode', 'streams', 'atol', 'rtol', 'max_rows')}
                    result = retain_check('recording_equivalence', args.source, args.destination,
                        reference=args.reference, parameters=parameters)
            case 'recover-video-reference':
                from .video_recovery import recover_video_reference

                repo_id = args.repo_id.removeprefix('hf://')
                result = recover_video_reference(repo_id, args.revision, args.original_revision,
                    args.episode_index, args.video_key, args.destination)
            case 'recover-video-references':
                from .bulk_workflow import check_report_output, recover_video_references

                check_report_output(args.output, args.destination)
                result = recover_video_references(args.repo_id, args.revision, args.original_revision,
                    args.destination, fetch=args.fetch, max_associations=args.max_associations,
                    max_response_body_mb=args.max_response_body_mb, max_intake_mb=args.max_intake_mb,
                    max_rows=args.max_rows, max_files=args.max_files)
                from .recovery_handoff import batch_handoff

                result = batch_handoff(result)
            case 'reproduce-video-references':
                from .bulk_workflow import check_report_output, reproduce_video_references

                check_report_output(args.output, args.directory)
                from .recovery_handoff import batch_handoff

                result = batch_handoff(reproduce_video_references(args.directory), include_patch=args.patch)
            case 'reproduce-case':
                if not args.directory:
                    raise InputError('Require a nonempty case directory')
                directory = Path(args.directory).expanduser().resolve()
                from .bulk_workflow import is_bulk_case, reproduce_video_references

                if is_bulk_case(directory):
                    from .bulk_workflow import check_report_output

                    check_report_output(args.output, args.directory)
                    from .recovery_handoff import batch_handoff

                    result = batch_handoff(reproduce_video_references(args.directory), include_patch=args.patch)
                    manifest = {'kind': 'batch_already_reproduced'}
                else:
                    manifest = read_json(contained(directory, 'manifest.json'))
                    if args.patch and manifest.get('kind') != 'robot_video_reference_recovery':
                        raise InputError('--patch applies to video-reference recovery batches and cases')
                if manifest.get('kind') == 'robot_video_reference_recovery':
                    from .recovery_handoff import case_handoff
                    from .video_recovery import reproduce_video_recovery

                    result = case_handoff(reproduce_video_recovery(args.directory), args.directory)
                elif manifest.get('kind') == 'robotics_retained_check':
                    from .retained import reproduce_retained_check

                    result = reproduce_retained_check(args.directory)
                elif manifest.get('kind') == 'robot_incident_case':
                    from .incidents import reproduce_incident_case

                    result = reproduce_incident_case(args.directory)
                elif manifest.get('kind') != 'batch_already_reproduced':
                    from .cases import reproduce_trajectory_case

                    result = reproduce_trajectory_case(args.directory)
            case 'create-case':
                from .incidents import export_incident_case

                control_file = read_json(args.controls) if args.controls else {'controls': None}
                if set(control_file) != {'controls'}:
                    raise InputError('Control file must contain only a controls list')
                controls = control_file['controls']
                if isinstance(controls, list) and args.controls:
                    base = Path(args.controls).expanduser().resolve().parent
                    controls = [dict(control, source=str((base / Path(control['source']).expanduser()).resolve()))
                                if isinstance(control, dict) and isinstance(control.get('source'), str) else control
                                for control in controls]
                if getattr(args, 'requirements', None):
                    requirements = _requirement_file(args.requirements)
                    if requirements['kind'] != args.check:
                        raise InputError('Requirements kind must match the selected create-case check')
                    if any(getattr(args, key, None) is not None for key in ('named_contract', 'expected_names', 'expected_frame', 'expected_quantity', 'expected_prim_path')):
                        raise InputError('Choose requirements or separate criterion flags; do not provide them together')
                    from .capture import audit_capture
                    from .investigation import _requirement_arguments, _retained_parameters

                    if requirements['kind'] == 'native-processor-source':
                        from .native_processor_source import (
                            audit_native_processor_source,
                            bind_native_processor_source_parameters,
                        )

                        parameters = bind_native_processor_source_parameters(args.corrected, requirements['parameters'])
                        healthy = audit_native_processor_source(args.corrected, **parameters)
                    elif requirements['kind'] == 'native-observation-source':
                        from .native_observation_source import (
                            audit_native_observation_source,
                            bind_native_observation_source_parameters,
                        )

                        parameters = bind_native_observation_source_parameters(args.corrected, requirements['parameters'])
                        healthy = audit_native_observation_source(args.corrected, **parameters)
                    elif requirements['kind'] == 'evaluation-cohort':
                        from .evaluation_cohort import audit_evaluation_cohort

                        healthy = audit_evaluation_cohort(args.corrected, reference=args.corrected, **requirements['parameters'])
                    elif requirements['kind'] == 'native-run-context':
                        from .native_context import audit_native_run_context

                        healthy = audit_native_run_context(args.corrected, reference=args.corrected, **requirements['parameters'])
                    else:
                        healthy = audit_capture(args.corrected, **_requirement_arguments(requirements))
                    if healthy['status'] != 'pass' or (requirements['kind'] not in ('native-run-context', 'evaluation-cohort', 'native-processor-source', 'native-observation-source') and healthy.get('acceptance_contract', {}).get('status') != 'pass'):
                        raise InputError('Corrected capture must pass the same supplied requirement before retaining a case')
                    if requirements['kind'] == 'native-processor-source':
                        retained_context = {'expected_source': parameters['expected_source']}
                    elif requirements['kind'] == 'native-observation-source':
                        retained_context = {'model_identity': parameters['model_identity']}
                    else:
                        parameters, retained_context = _retained_parameters(requirements, healthy)
                elif args.check in ('native-processor-source', 'native-observation-source'):
                    parameters = {}
                elif args.check == 'episode-windows':
                    parameters = {'episode_index': args.episode_index, 'anchor_index': args.anchor_index,
                                  'delta_indices': args.delta_indices, 'max_rows': args.max_rows, 'max_files': args.max_files}
                elif args.check == 'native-named-action':
                    parameters = _named_contract(args.named_contract) if args.named_contract else {}
                elif args.check == 'act-command-path':
                    parameters = _command_contract(args.command_contract) if args.command_contract else {}
                elif args.check == 'act-query-cadence':
                    parameters = {'max_query_gap': args.max_query_gap}
                elif args.check == 'resolved-layout':
                    parameters = {'expected_names': args.expected_names}
                    for key in ('expected_frame', 'expected_quantity'):
                        value = getattr(args, key)
                        if value is not None:
                            parameters[key] = value
                    if args.expected_prim_path:
                        parameters['expected_identity'] = {'prim_path': args.expected_prim_path}
                else:
                    parameters = {'sample_times': args.times, 'interpolation_method': args.method,
                                  'position_limits': read_json(args.position_limits)}
                result = export_incident_case(args.check.replace('-', '_'), args.source, args.destination, parameters,
                    corrected=args.corrected, apply_metadata_corrections=getattr(args, 'apply_metadata_corrections', False),
                    controls=controls)
                if getattr(args, 'requirements', None):
                    result['retained_context'] = retained_context
                    result['applied_requirements'] = requirements
            case 'check-case':
                if not args.directory or not args.candidate:
                    raise InputError('Require nonempty case and candidate paths')
                directory = Path(args.directory).expanduser().resolve()
                manifest = read_json(contained(directory, 'manifest.json'))
                if manifest.get('kind') == 'robot_video_reference_recovery':
                    from .video_recovery import check_video_recovery_candidate

                    result = check_video_recovery_candidate(args.directory, args.candidate)
                elif manifest.get('kind') == 'robotics_retained_check':
                    from .retained import check_retained_candidate

                    result = check_retained_candidate(args.directory, args.candidate)
                else:
                    from .incidents import check_incident_candidate

                    result = check_incident_candidate(args.directory, args.candidate)
            case 'audit-dispatch':
                from .dispatch import audit_dispatch

                result = audit_dispatch(args.evidence, args.recording, episode=args.episode, action_names=args.action_names,
                                        atol=args.atol, rtol=args.rtol)
            case "trace":
                result = trace_actions(args.checkpoint, args.sample, args.deployment)
            case "compare":
                result = compare_configs(args.before, args.after)
            case "template":
                result = deployment_template(args.checkpoint, args.dataset)
                args.json = True
            case "scan":
                result = _dataset_source(args, lambda path: scan_dataset(path, args.max_rows, args.max_files))
            case 'audit-windows':
                from .windows import audit_episode_windows

                result = _dataset_source(args, lambda path, video_context=None: audit_episode_windows(path, video_context=video_context, episode_index=args.episode_index,
                    anchor_index=args.anchor_index, delta_indices=args.delta_indices,
                    max_rows=args.max_rows, max_files=args.max_files), with_video_context=True)
            case "bundle":
                result = export_bundle(args.checkpoint, args.destination, args.dataset, args.deployment)
            case "reproduce":
                result = reproduce_bundle(args.bundle)
            case "evidence":
                result = inspect_evidence(args.path, args.pointer, args.sha256)
            case "check-repair":
                result = check_repair(args.checkpoint, args.before, args.candidate, args.dataset)
            case 'probe-native':
                from .native_processing import load_native_runtimes, probe_native_processing

                selected = args.sample or args.row
                if args.row and args.provider != 'groot_state':
                    raise InputError('--row is supported only for original GR1 state samples')
                if len(selected) > 2 or any(sample in selected[:index] for index, sample in enumerate(selected)):
                    raise InputError('Choose one or two unique EPISODE:FRAME samples or EPISODE:PHYSICAL_ROW coordinates')
                if args.case and args.output:
                    raise InputError('Choose --case or --output; a native capsule already retains the numerical evidence')
                source_options = {}
                if args.mode != 'observation' or any(value is not None for value in (args.processor_source, args.intent, args.approved_sources)):
                    approved = None
                    if args.approved_sources is not None:
                        document = read_json(args.approved_sources)
                        if type(document) is not dict or set(document) != {'sources'} or type(document['sources']) is not list:
                            raise InputError('Approved source file requires only a sources list')
                        base = Path(args.approved_sources).expanduser().resolve().parent
                        approved = [dict(item, directory=str((base / Path(item['directory']).expanduser()).absolute()))
                            if type(item) is dict and type(item.get('directory')) is str else item
                            for item in document['sources']]
                    source_options = {'mode': args.mode, 'processor_source': args.processor_source,
                                      'intent': args.intent, 'approved_sources': approved}
                if args.requirements:
                    requirement = _requirement_file(args.requirements)
                    if args.provider != 'groot_adapter' or requirement['kind'] != 'native-observation-source':
                        raise InputError('Native probe requirements require groot_adapter and native-observation-source')
                    if 'model_identity' in requirement['parameters']:
                        raise InputError('Probe accepts contract only; apply frozen model identity through investigate/check-case')
                    source_options['contract'] = requirement['parameters']['contract']
                result = probe_native_processing(args.provider, args.model, args.dataset,
                    runtimes=load_native_runtimes(args.native_runtime), samples=selected, destination=args.case,
                    **source_options)
            case 'reproduce-native-probe':
                from .native_processing import load_native_runtimes, reproduce_native_processing

                result = reproduce_native_processing(args.directory,
                    runtimes=load_native_runtimes(args.native_runtime, require_initial_cache=False))
            case "serve":
                from .server import serve

                options = {'allow_hub': True} if args.allow_hub else {}
                if brand == 'RoboClue':
                    options['server_name'] = 'RoboClue'
                if args.allow_hub_datasets:
                    options['allow_hub_datasets'] = True
                if args.allow_hub_processor_state:
                    options['allow_hub_processor_state'] = True
                if args.allow_openpi_statistics:
                    options['allow_openpi_statistics'] = True
                if args.allow_hub_media:
                    options['allow_hub_media'] = True
                if args.native_runtime:
                    serve(args.root, case_root=args.case_root, native_runtime=args.native_runtime, **options)
                else:
                    serve(args.root, case_root=args.case_root, **options)
                return 0
            case _:
                raise InputError("Unknown command")
        if getattr(args, 'markdown', False):
            from .reports import markdown_report

            if result.get('kind') in ('native_processing_probe', 'native_processing_reproduction',
                                      'native_processor_source_observation', 'native_processor_source_reproduction',
                                      'native_observation_source_observation', 'native_observation_source_reproduction', 'dataset_preflight',
                                      'dataset_video_reference_batch', 'robot_readiness', 'openpi_readiness') or args.command == 'scan':
                from .reports import _text

                title = ('RoboClue derived video-reference proposal' if result.get('kind') == 'dataset_video_reference_batch' else
                         'RoboClue robot readiness' if result.get('kind') in ('robot_readiness', 'openpi_readiness') else
                         'RoboClue dataset preflight' if result.get('kind') == 'dataset_preflight' else
                         'RoboClue dataset values' if args.command == 'scan' else 'Native processing observation')
                rendered = '# ' + title + '\n\n' + '\n\n'.join(_text(line) for line in human(result, brand=brand, command_prefix=command_prefix).splitlines()) + '\n'
            else:
                rendered = markdown_report(result)
        else:
            rendered = json.dumps(result, indent=2, allow_nan=False) + "\n" if args.json else human(result, brand=brand, command_prefix=command_prefix)
        if result.get('case_directory') and not args.json and not getattr(args, 'markdown', False):
            rendered += '\nCase created: ' + json.dumps(result['case_directory']) + '\nRerun: ' + shlex.join([*command_prefix, 'reproduce-case', result['case_directory']]) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        else:
            sys.stdout.write(rendered)
        if required_units is not None:
            # Required coverage cannot be waived by an error-only severity
            # filter. Known contradictions elsewhere in this full report still
            # fail rather than being hidden by a passed units component.
            if result.get('status') == 'error' or required_units['status'] == 'error':
                return 1
            if required_units['status'] not in ('pass', 'warning'):
                return 3
            if args.require and not args.fail_on:
                return 1 if brand == 'RoboClue' and required_units['status'] == 'warning' else 0
        if getattr(args, 'fail_on', None):
            failing = {'error': {'error'}, 'warning': {'error', 'warning'},
                       'unknown': {'error', 'warning', 'unknown'}}[args.fail_on]
            # A component status counts too: a capped or reduced component can hold errors no listed finding shows.
            statuses = {f.get('status') for f in result.get('findings', []) if isinstance(f, dict)}
            statuses |= {c.get('status') for c in (result.get('components') or {}).values() if isinstance(c, dict)}
            statuses |= {c.get('status') for c in
                         ((result.get('coverage') or {}).get('components_left_out_at_size_bound') or {}).values()
                         if isinstance(c, dict)}
            if result.get('kind') == 'dataset_preflight':  # its verdict also covers components left out at the bound
                statuses.add(result.get('status'))
            return 1 if statuses & failing else 0
        if brand == 'RoboClue' and result.get('status') == 'warning':
            return 1
        return {"error": 1, "different": 1, "unknown": 3}.get(result.get("status"), 0)
    except (InputError, OSError, ImportError) as exc:
        message = {"error": {"code": "invalid_input", "message": str(exc)}}
        sys.stderr.write(json.dumps(message) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
