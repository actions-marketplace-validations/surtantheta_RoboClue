"""Stable discovery metadata for supported checks, not an executable plugin registry.

A rule page groups related diagnostic codes. Looking a code up never changes its
severity, evidence or supported-version scope, and never executes artifact code.
"""
from __future__ import annotations

import copy
import re
from urllib.parse import urlsplit

from .artifacts import InputError

SCHEMA_VERSION = 1
SCOPE = ('Rule IDs document supported checks; catalog membership does not certify any dataset, framework version, '
         'training pipeline or physical behavior. Read the selected report and each rule\'s tested profiles.')
LEROBOT = '146d8474932c8b566717eb2738ba459337ab71c4'
ISAACLAB = 'e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4'
GROOT15 = '4af2b622892f7dcb5aae5a3fb70bcb02dc217b96'
GROOT17 = '944fe70c4fd54e3fcb80a7d7258a328f9c061176'
OPENPI = '215abfb217dbac7d5f1273282331b9b1866c0479'


def _source(repository, revision, path, title):
    return {'title': title, 'url': f'https://github.com/{repository}/blob/{revision}/{path}'}


_RULES = (
    {
        'id': 'RC001', 'title': 'Recorded motor units versus the declared robot',
        'summary': 'The same joint labels can hold degrees or calibrated -100..100 values.',
        'diagnostic_codes': ['units.joint_mode_mismatch', 'units.joint_mode_match', 'units.joint_mode_undecided',
                             'units.deployment_mode_unknown', 'units.state_action_mode_disagree'],
        'frameworks': [{'name': 'LeRobot', 'revision': LEROBOT, 'profile': 'Supported motor grids and robot defaults; not all releases'}],
        'upstream': [_source('huggingface/lerobot', LEROBOT, 'src/lerobot/robots/so_follower/config_so_follower.py', 'SO follower use_degrees default'),
                     _source('huggingface/lerobot', LEROBOT, 'src/lerobot/motors/motors_bus.py', 'Motor normalization and inverse')],
        'trigger': 'Recorded supported motor values decide a mode, and an explicit robot command/config requests a different mode.',
        'scope': 'Recorded-number inference compared with declared robot settings; dataset-only inference is not deployment approval.',
        'not_checked': ['Policy processors', 'Physical calibration or achieved behavior', 'Defaults in untested framework revisions'],
        'remedy': 'Review the actual robot settings and intended policy units; recheck the matching command. Do not rescale public data merely to clear this check.',
        'cli': "roboclue check DATASET --robot-command 'YOUR_DEPLOY_COMMAND' --require deployment-units",
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'robot_command']},
        'docs': 'docs/rules/RC001.md', 'guide': 'docs/UNITS_MANIFEST.md',
        'tests': ['tests/test_units_manifest.py', 'tests/test_preflight_units.py', 'tests/test_product_question_routes.py'],
    },
    {
        'id': 'RC002', 'title': 'Isaac Lab quaternion declarations survive conversion',
        'summary': 'A dropped format_version can make a reader interpret XYZW as legacy WXYZ.',
        'diagnostic_codes': ['isaaclab.format_version', 'isaaclab.declared_order_vs_content',
                             'isaaclab.mixed_order_across_demos', 'isaaclab.quaternions_outside_root_pose',
                             'isaaclab.conversion_quaternion_order'],
        'frameworks': [{'name': 'Isaac Lab', 'revision': ISAACLAB, 'profile': 'v3.0.0 HDF5 conventions; source/export declaration and bounded comparisons'}],
        'upstream': [_source('isaac-sim/IsaacLab', ISAACLAB, 'source/isaaclab/isaaclab/utils/datasets/hdf5_dataset_file_handler.py', 'HDF5 format version and quaternion conversion'),
                     _source('isaac-sim/IsaacLab', 'c7d50a61689c9e971b37f90a70fd9d4bc3ea9062', 'scripts/tools/merge_hdf5_datasets.py', 'Merge attribute handling')],
        'trigger': 'A declaration/content contradiction or a supported source/export quaternion-order mismatch.',
        'scope': 'Bounded HDF5 declarations and supported quaternion comparisons; numeric values do not generally identify the intended order.',
        'not_checked': ['Physical mounting without declared assumptions', 'Mixed-source intent without originals', 'Simulator execution or task success'],
        'remedy': 'Compare retained source/export files, retain mounting/version conditions, correct only a separate owned candidate, and rerun the same comparison.',
        'cli': 'roboclue check SOURCE.hdf5 --converted EXPORT.hdf5',
        'mcp': {'tool': 'check_sim_provenance', 'arguments': ['source', 'converted']},
        'docs': 'docs/rules/RC002.md', 'guide': 'docs/SIM_DATA_PROVENANCE.md',
        'tests': ['tests/test_sim_guards.py', 'tests/test_product_question_routes.py'],
    },
    {
        'id': 'RC003', 'title': 'GR00T dataset metadata matches the selected consumer',
        'summary': 'N1.5 and N1.7 load modality, configuration and statistics declarations differently.',
        'diagnostic_codes': ['groot.version', 'groot.format', 'groot.modality', 'groot.modality_schema',
                             'groot.metadata', 'groot.config', 'groot.config_key', 'groot.state_config',
                             'groot.action_config', 'groot.video_mapping', 'groot.statistics', 'groot.state_slice',
                             'groot.action_slice',
                             'groot.state_width', 'groot.action_width'],
        'frameworks': [{'name': 'Isaac-GR00T', 'revision': GROOT15, 'profile': 'N1.5 static metadata/config contract'},
                       {'name': 'Isaac-GR00T', 'revision': GROOT17, 'profile': 'N1.7 static metadata/config contract'}],
        'upstream': [_source('NVIDIA/Isaac-GR00T', GROOT15, 'gr00t/data/dataset.py', 'N1.5 dataset loader'),
                     _source('NVIDIA/Isaac-GR00T', GROOT17, 'gr00t/data/dataset/lerobot_episode_loader.py', 'N1.7 episode loader')],
        'trigger': 'Supported metadata/config declarations contradict the selected pinned loader profile.',
        'scope': 'Version-specific static contract only; each finding retains its version and conditions. Matching metadata does not certify fine-tuning.',
        'not_checked': ['Decoded cameras and all Parquet values', 'Arbitrary Python config behavior', 'Whole-model or native training compatibility'],
        'remedy': 'First select a training config matching the dataset. Preserve public metadata; make source-backed corrections only in a separate owned candidate.',
        'cli': 'roboclue check DATASET --groot-version n1.5 --groot-config-name CONFIG_NAME',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'groot_version', 'groot_config_name']},
        'docs': 'docs/rules/RC003.md', 'guide': 'docs/GROOT_DATASET_CONTRACT.md',
        'tests': ['tests/test_groot_contract.py', 'tests/test_groot_contract_n15.py', 'tests/test_product_question_routes.py'],
    },
    {
        'id': 'RC004', 'title': 'Delta-target transformation matches declared action intent',
        'summary': 'Subtracting state is meaningful only for the supported declared quantity and convention.',
        'diagnostic_codes': ['convention.relative_on_absolute', 'convention.relative_on_incremental',
                             'convention.relative_undecided', 'convention.relative_unmodeled', 'convention.relative_width'],
        'frameworks': [{'name': 'openpi', 'revision': OPENPI, 'profile': 'Pinned builtin config and DeltaActions source'},
                       {'name': 'LeRobot', 'revision': LEROBOT, 'profile': 'Supported saved relative-action processors'},
                       {'name': 'Isaac-GR00T', 'revision': GROOT17, 'profile': 'Supported saved RELATIVE declarations'}],
        'upstream': [_source('Physical-Intelligence/openpi', OPENPI, 'src/openpi/transforms.py', 'DeltaActions and AbsoluteActions'),
                     _source('huggingface/lerobot', LEROBOT, 'src/lerobot/processor/relative_action_processor.py', 'Relative-action processor')],
        'trigger': 'A modeled transform contradicts source-backed declared intent; inference alone can instead warn or stay unknown.',
        'scope': 'Delta-transform consistency, not complete consumer compatibility. Recorded-number fingerprints can remain undecided.',
        'not_checked': ['Dataset repack keys', 'LeRobot format/dependency version compatibility', 'Physical frames, native training or task success'],
        'remedy': 'Review the actual declared transform and dataset intent, correct a separate owned configuration, and recheck the same criterion; do not change documented LIBERO behavior on inference alone.',
        'cli': 'roboclue check DATASET --openpi-config pi0_libero --openpi-source-root OPENPI_SOURCE',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'openpi_config', 'openpi_source_root']},
        'docs': 'docs/rules/RC004.md', 'guide': 'docs/ACTION_CONVENTIONS.md',
        'tests': ['tests/test_action_conventions.py', 'tests/test_preflight_action_conventions.py', 'tests/test_product_question_routes.py'],
    },
    {
        'id': 'RC005', 'title': 'Interrupted recordings must not be resumed as-is',
        'summary': 'Unflushed metadata, orphan rows or next-shard files can make resume replace or misindex data.',
        'diagnostic_codes': ['salvage.resume_hazard', 'salvage.file_unreadable', 'salvage.episode_metadata_missing',
                             'salvage.metadata_plan', 'salvage.not_metadata_only', 'salvage.coverage_unknown',
                             'salvage.metadata_rows_repairable', 'salvage.metadata_row_conflict', 'salvage.video_conflict',
                             'salvage.row_identity', 'salvage.episode_integrity'],
        'frameworks': [{'name': 'LeRobot', 'revision': LEROBOT, 'profile': 'Native v3 standard recording metadata and paths'}],
        'upstream': [_source('huggingface/lerobot', LEROBOT, 'src/lerobot/datasets/dataset_writer.py', 'Data, video and episode write order'),
                     _source('huggingface/lerobot', LEROBOT, 'src/lerobot/datasets/dataset_metadata.py', 'Metadata flush and resume spans')],
        'trigger': 'Supported byte/metadata contradictions or incomplete recording evidence make current-directory resume unsafe or unassessed.',
        'scope': 'Read-only bounded recovery ledger. A predicted separate-copy plan never approves the original directory.',
        'not_checked': ['Footerless Parquet page recovery', 'Video decoding', 'Reconstructed recording intent or physical validity'],
        'remedy': 'Preserve originals, assemble an explicitly accounted separate candidate, complete any required row rewrite, and rerun recovery and the intended native reader.',
        'cli': 'roboclue check DATASET --question recovery',
        'mcp': {'tool': 'check_recording_salvage', 'arguments': ['dataset']},
        'docs': 'docs/rules/RC005.md', 'guide': 'docs/RECORDING_SALVAGE.md',
        'tests': ['tests/test_salvage.py', 'tests/test_salvage_product_truth.py', 'tests/test_product_question_routes.py',
                  'tests/test_critic_real_cases.py', 'tests/test_compound_metadata_recovery.py'],
    },
    {
        'id': 'RC006', 'title': 'Saved statistics must describe a consistent finite cohort',
        'summary': 'A mean outside min/max, or exact zero std with a meaningful range, is internally inconsistent.',
        'diagnostic_codes': ['dataset.statistics_inconsistent', 'dataset.statistics_unassessed',
                             'dataset.statistics_rows_mismatch', 'dataset.statistics_count_mismatch',
                             'dataset.statistics_values_unassessed'],
        'frameworks': [{'name': 'LeRobot metadata', 'revision': LEROBOT, 'profile': 'Supported saved vector summaries; arithmetic invariant is not loader-version certification'}],
        'upstream': [_source('huggingface/lerobot', LEROBOT, 'src/lerobot/datasets/compute_stats.py', 'Dataset statistics context')],
        'trigger': 'Finite, width-consistent saved summaries violate a same-cohort arithmetic constraint beyond the stated rounding tolerance.',
        'scope': 'Saved-summary self-consistency and, separately, bounded supported numeric-row comparison. Partial cohorts, unsupported metrics and ambiguous native float32 rounding remain unknown.',
        'not_checked': ['Decoded image summaries in the default check', 'Whether all summaries use the same historical cohort', 'Native loader compatibility'],
        'remedy': 'Confirm the intended feature/cohort and recompute source-backed summaries into a separate owned candidate. Preserve the original statistics.',
        'cli': 'roboclue check DATASET --question dataset',
        'mcp': {'tool': 'dataset_health_card', 'arguments': ['source']},
        'docs': 'docs/rules/RC006.md', 'guide': 'docs/PREFLIGHT.md',
        'tests': ['tests/test_brand_and_preview_feedback.py', 'tests/test_rule_catalog.py'],
    },
    {
        'id': 'RC007', 'title': 'Declared camera keys correspond to dataset identities',
        'summary': 'A robot can declare different camera names before any policy or checkpoint exists.',
        'diagnostic_codes': ['camera.keys_mismatch', 'camera.keys_consistent', 'camera.partial_keys',
                             'camera.extra_keys', 'camera.keys_not_requested', 'camera.declaration_unassessed',
                             'camera.dataset_unassessed', 'camera.coverage_unknown', 'camera.rename_unassessed',
                             'camera.default_empty', 'camera.default_unqualified'],
        'frameworks': [{'name': 'LeRobot', 'revision': LEROBOT, 'profile': 'Explicit supported robot/camera declarations and metadata feature-key sets only'}],
        'upstream': [_source('huggingface/lerobot', LEROBOT, 'src/lerobot/utils/feature_utils.py', 'Robot camera names become observation feature keys'),
                     _source('huggingface/lerobot', LEROBOT, 'src/lerobot/robots/bi_so_follower/bi_so_follower.py', 'Per-arm camera-name prefixes')],
        'trigger': 'An explicit nonempty camera-key set has disjoint dataset identities without an applicable declared rename_map; partial or additional views instead require review.',
        'scope': 'Declared camera-key correspondence only. Dataset metadata does not establish every camera is required by a future policy.',
        'not_checked': ['Image size', 'Frame rate', 'Joint order', 'Physical camera identity/calibration', 'Policy view selection or compatibility'],
        'remedy': 'Review intended views and explicit raw-robot-to-dataset rename_map; recheck an owned command/config. Do not rename public data merely to clear the check.',
        'cli': "roboclue check DATASET --robot-command 'YOUR_CONSUMER_COMMAND_WITH_CAMERA_MAPPING'",
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'robot_command']},
        'docs': 'docs/rules/RC007.md', 'guide': 'docs/ROBOT_DECLARATIONS.md',
        'tests': ['tests/test_robot_contract.py', 'tests/test_rule_catalog.py'],
    },
    {
        'id': 'RC008', 'title': 'OpenPI raw input keys match the selected adapter',
        'summary': 'A consistent delta transform does not ensure the selected repack adapter can find dataset keys.',
        'diagnostic_codes': ['openpi.raw_key_missing', 'openpi.raw_keys_consistent',
                             'openpi.rlds_backend', 'openpi.input_context_unresolved'],
        'frameworks': [{'name': 'openpi', 'revision': OPENPI, 'profile': 'Fingerprint-verified literal selected repack declaration and lockfile facts'}],
        'upstream': [_source('Physical-Intelligence/openpi', OPENPI, 'src/openpi/training/config.py', 'Selected training configuration and raw-key mappings'),
                     _source('Physical-Intelligence/openpi', OPENPI, 'src/openpi/transforms.py', 'Native RepackTransform key lookup')],
        'trigger': 'A statically required source key is absent from the dataset feature declarations. Dynamic mappings and other backends remain unknown.',
        'scope': 'Declared raw-key availability plus separate dataset-format and verified consumer lockfile facts; no installed-loader compatibility claim.',
        'not_checked': ['Image contents and dtype/shape compatibility', 'Installed LeRobot reader support', 'Model execution or native training'],
        'remedy': 'Choose or supply a matching owned repack adapter and recheck. Do not rename public dataset features merely to clear a finding.',
        'cli': 'roboclue check DATASET --openpi-config CONFIG --openpi-source-root OPENPI_SOURCE',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'openpi_config', 'openpi_source_root']},
        'docs': 'docs/rules/RC008.md', 'guide': 'docs/ACTION_CONVENTIONS.md',
        'tests': ['tests/test_openpi_context.py', 'tests/test_r4_frontdoor.py'],
    },
    {
        'id': 'RC009', 'title': 'FastWAM saved image scaling matches the supplied model source',
        'summary': 'An extra image normalization can compose with the model scaling to 4*x-3 instead of 2*x-1.',
        'diagnostic_codes': ['fastwam.image_map'],
        'frameworks': [{'name': 'LeRobot FastWAM', 'revision': '77710dee471d5cfb070fad1adc30c8f94dc14757',
                        'profile': 'Exact audited source files; saved float32 RGB image affine map only'}],
        'upstream': [{'title': 'Reported FastWAM double normalization', 'url': 'https://github.com/huggingface/lerobot/issues/5116'},
                     _source('huggingface/lerobot', '77710dee471d5cfb070fad1adc30c8f94dc14757',
                             'src/lerobot/policies/fastwam/wan/modular.py', 'Native model image scaling and VAE input contract')],
        'trigger': 'A saved supported image preprocessor composed with the exact supplied source contradicts the model image map.',
        'scope': 'Static file/source composition for RGB inputs in [0,1]; source is caller-supplied, not attested as the active trainer. No native execution.',
        'not_checked': ['Model weights and task success', 'State and action normalization',
                        'Unknown source revisions and other video backbones', 'Live overrides and active runtime identity'],
        'remedy': 'Test VISUAL=IDENTITY in a separate owned processor copy, preserve state/action statistics, then repeat against the same actual runtime source.',
        'cli': 'roboclue check CHECKPOINT --runtime-source-root ACTUAL_LEROBOT_SOURCE',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'runtime_source_root']},
        'docs': 'docs/rules/RC009.md', 'guide': 'docs/rules/RC009.md',
        'tests': ['tests/test_fastwam_contract.py'],
    },
    {
        'id': 'RC013', 'title': 'Declared instructions versus the linked task catalog',
        'summary': 'Lexical catalog membership is separate from native token equivalence and physical capability.',
        'diagnostic_codes': ['language.out_of_catalog', 'language.catalog_unresolved',
                             'language.lexical_equivalence_unknown', 'language.catalog_provenance_unknown'],
        'frameworks': [{'name': 'LeRobot', 'revision': '72e1d9e5e76ee719bacfc78bc34d817bee5fb521',
                        'profile': 'Saved train_config dataset link; bounded v2 JSONL/v3 Parquet catalog; lexical matching only'}],
        'upstream': [_source('huggingface/lerobot', '72e1d9e5e76ee719bacfc78bc34d817bee5fb521',
                             'src/lerobot/rollout/planner.py#L283-L302', 'Linked task vocabulary is an inference, not a contract'),
                     {'title': 'Reported training/evaluation instruction mismatch',
                      'url': 'https://github.com/huggingface/lerobot/issues/4548'},
                     {'title': 'CloseFridge versus natural-language evaluation',
                      'url': 'https://github.com/huggingface/lerobot/issues/4045'}],
        'trigger': 'An explicit instruction has no exact or documented lexical-fold match in the bounded linked catalog.',
        'scope': 'WARNING-only lexical observation. Null/mutable revision, local association and episode subsets do not authenticate historical training coverage. Lexical folding does not establish native processor equivalence.',
        'not_checked': ['Native tokenizer/policy execution', 'Physical capability or success', 'Complete historical training mixture'],
        'remedy': 'Preserve the intended task. Use an exact catalog string only when its meaning matches; never auto-rewrite an instruction to the nearest lexical suggestion.',
        'cli': 'roboclue check CHECKPOINT --instruction "DECLARED INSTRUCTION" --language-catalog LOCAL_DATASET',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'instruction', 'language_catalog']},
        'docs': 'docs/rules/RC013.md', 'guide': 'docs/LANGUAGE_VOCABULARY.md',
        'tests': ['tests/test_language_vocab.py'],
    },
    {
        'id': 'RC010', 'title': 'OpenPI pretrained loading is actually selected',
        'summary': 'The PyTorch trainer ignores the JAX weight_loader when deciding whether to load pretrained weights.',
        'diagnostic_codes': ['openpi.pretrained_load_selection'],
        'frameworks': [{'name': 'OpenPI PyTorch', 'revision': '3e0325a5e6032a9ab6beef7395fe47ca724980a5',
                        'profile': 'Exact train_pytorch.py load-selection branch; resolved config and explicit intent'}],
        'upstream': [{'title': 'Reported ignored weight loader', 'url': 'https://github.com/Physical-Intelligence/openpi/issues/804'},
                     _source('Physical-Intelligence/openpi', '3e0325a5e6032a9ab6beef7395fe47ca724980a5', 'scripts/train_pytorch.py', 'PyTorch load predicate at line442')],
        'trigger': 'Pretrained intent, PyTorch backend, fresh training and a null resolved pytorch_weight_path under audited source.',
        'scope': 'File/source contract for branch selection only; a selected path does not establish a successful weight load.',
        'not_checked': ['Checkpoint tensor compatibility', 'Actual big-model loading or training', 'Resume restoration and JAX initialization', 'Unprofiled source'],
        'remedy': 'Set TrainConfig.pytorch_weight_path to the converted PyTorch checkpoint directory, recapture and repeat the same gate.',
        'cli': 'roboclue check actual-initialization.json --fail-on error',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'runtime_source_root']},
        'docs': 'docs/rules/RC010.md', 'guide': 'docs/PIPELINE_CONTRACTS.md',
        'tests': ['tests/test_pipeline_contracts.py'],
    },
    {
        'id': 'RC011', 'title': 'Missing cameras do not silently shift present cameras into earlier slots',
        'summary': 'An earlier missing camera changes compacted input-slot identity in supported LeRobot preprocessors.',
        'diagnostic_codes': ['lerobot.camera_slot_order'],
        'frameworks': [{'name': 'LeRobot', 'revision': '77710dee471d5cfb070fad1adc30c8f94dc14757',
                        'profile': 'Exact pi0/pi05/SmolVLA source; also official PyPI0.6.1 file profiles'}],
        'upstream': [{'title': 'Reported silent camera-slot shift', 'url': 'https://github.com/huggingface/lerobot/issues/5118'},
                     _source('huggingface/lerobot', '77710dee471d5cfb070fad1adc30c8f94dc14757', 'src/lerobot/policies/pi05/modeling_pi05.py', 'Present-first image packing')],
        'trigger': 'A missing earlier declared camera moves a supplied later camera into a different native slot.',
        'scope': 'Runtime key guard plus exported-key/source contract. No camera pixels, calibration or model forward are validated.',
        'not_checked': ['Physical camera identity behind a key', 'Policy adequacy with missing cameras', 'Arbitrary native processors', 'Whole-model inference or robot success'],
        'remedy': 'Restore the genuine missing camera under its trained key, recapture at the native boundary and recheck; never invent an unmasked replacement.',
        'cli': 'roboclue check actual-cameras.json --fail-on error',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'runtime_source_root']},
        'docs': 'docs/rules/RC011.md', 'guide': 'docs/PIPELINE_CONTRACTS.md',
        'tests': ['tests/test_pipeline_contracts.py'],
    },
    {
        'id': 'RC012', 'title': 'Bounded version and provenance register intake',
        'summary': 'Evaluate seven scoped version/provenance and publisher-recipe entries against explicit local evidence.',
        'diagnostic_codes': ['register.input_unknown', 'register.catalog_unknown',
                             'register.MUJOCO-310-ROBOSUITE-140-CRASH', 'register.MUJOCO-333-LIBERO-RENDER',
                             'register.MUJOCO-GE340-LIBERO-SPATIAL-INIT-OPENVLA', 'register.ROBOTWIN-SAPIEN-COMMIT',
                             'register.RECIPE-LIBERO-GROOT-N17', 'register.RECIPE-LIBERO-OPENVLA-OFT',
                             'register.RECIPE-LIBERO-XVLA'],
        'profile_kind': 'engine', 'frameworks': [],
        'upstream': [{'title': 'Python distribution metadata does not attest installed source',
                      'url': 'https://docs.python.org/3/library/importlib.metadata.html'}],
        'trigger': 'A scoped entry matches evidenced declarations, or missing/ambiguous evidence prevents assessment.',
        'scope': 'Seven bounded entries, each with its own applicability and evidence. No whole-framework compatibility or native execution approval.',
        'not_checked': ['Installed source identity from a version alone', 'Uncurated incidents', 'Native execution and task performance'],
        'remedy': 'Supply the named missing evidence and independently validated register entries; never infer a repair from UNKNOWN.',
        'cli': 'roboclue check provenance-input.json --launch-command "ACTUAL COMMAND"',
        'mcp': {'tool': 'preflight_check', 'arguments': ['source', 'launch_command']},
        'docs': 'docs/rules/RC012.md', 'guide': 'docs/PROVENANCE_REGISTER.md',
        'tests': ['tests/test_provenance_register.py', 'tests/test_register_recipes.py'],
    },
)


def rule_catalog() -> dict:
    """Return independent JSON-compatible catalog metadata; no checks are executed."""
    return {'schema_version': SCHEMA_VERSION, 'kind': 'rule_catalog', 'count': len(_RULES),
            'rules': copy.deepcopy(sorted(_RULES, key=lambda row: row['id'])), 'scope': SCOPE}


def get_rule(rule_id: str) -> dict:
    """Look up a stable rule ID without treating unknown IDs as an empty success."""
    if not isinstance(rule_id, str) or not re.fullmatch(r'RC[0-9]{3}', rule_id):
        raise InputError('Rule ID must have the form RC001; list rules with roboclue advanced rules.')
    for rule in _RULES:
        if rule['id'] == rule_id:
            return copy.deepcopy(rule)
    raise InputError(f'Unknown rule {rule_id}; list rules with roboclue advanced rules.')


def rules_for_code(code: str) -> list[dict]:
    """Find exact documented code mappings; unmapped codes carry no catalog verdict."""
    if not isinstance(code, str):
        raise InputError('Diagnostic code must be a string.')
    return [copy.deepcopy(rule) for rule in _RULES if code in rule['diagnostic_codes']]


def validate_catalog(catalog: dict | None = None) -> list[str]:
    """Return structural contribution errors without importing detector/plugin code.

    Repository tests separately verify links and exact diagnostic-code existence.
    This helper does not validate scientific claims or upstream version support.
    """
    data = rule_catalog() if catalog is None else catalog
    if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data.get('schema_version') != SCHEMA_VERSION or not isinstance(data.get('rules'), list):
        return ['Catalog needs schema_version=1 and a rules list.']
    errors, seen_ids, seen_codes = [], set(), set()
    if type(data.get('count')) is not int or data.get('count') != len(data['rules']):
        errors.append('Catalog count does not match its rules list.')
    for index, rule in enumerate(data['rules']):
        label = f'rules[{index}]'
        if not isinstance(rule, dict):
            errors.append(label + ' must be an object.')
            continue
        rule_id = rule.get('id')
        if not isinstance(rule_id, str) or not re.fullmatch(r'RC[0-9]{3}', rule_id):
            errors.append(label + ' needs a stable RC001-style ID.')
        elif rule_id in seen_ids:
            errors.append(label + ' duplicates rule ID ' + rule_id + '.')
        else:
            seen_ids.add(rule_id)
        for field in ('title', 'summary', 'trigger', 'scope', 'remedy', 'cli', 'docs', 'guide'):
            if not isinstance(rule.get(field), str) or not rule[field].strip():
                errors.append(label + ' needs nonempty ' + field + '.')
        for field in ('diagnostic_codes', 'not_checked', 'tests'):
            values = rule.get(field)
            if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v.strip() for v in values):
                errors.append(label + ' needs nonempty string list ' + field + '.')
        for code in rule.get('diagnostic_codes', []) if isinstance(rule.get('diagnostic_codes'), list) else []:
            if not isinstance(code, str):
                continue
            if code in seen_codes:
                errors.append(label + ' duplicates diagnostic mapping ' + code + '.')
            seen_codes.add(code)
        profiles = rule.get('frameworks')
        if rule.get('profile_kind') == 'engine' and profiles == []:
            pass  # Infrastructure discovery does not fabricate tested native profiles.
        elif not isinstance(profiles, list) or not profiles:
            errors.append(label + ' needs tested framework profiles.')
        else:
            for profile in profiles:
                if not isinstance(profile, dict) or any(not isinstance(profile.get(k), str) or not profile[k].strip() for k in ('name', 'revision', 'profile')):
                    errors.append(label + ' has an incomplete framework profile.')
        sources = rule.get('upstream')
        if not isinstance(sources, list) or not sources:
            errors.append(label + ' needs primary upstream evidence.')
        else:
            for source in sources:
                if not isinstance(source, dict) or not isinstance(source.get('title'), str) or not isinstance(source.get('url'), str):
                    errors.append(label + ' has incomplete upstream evidence.')
                    continue
                try:
                    url = urlsplit(source['url'])
                except ValueError:
                    errors.append(label + ' has an invalid upstream URL.')
                    continue
                if url.scheme != 'https' or not url.netloc:
                    errors.append(label + ' upstream URLs must use https.')
        route = rule.get('mcp')
        if not isinstance(route, dict) or not isinstance(route.get('tool'), str) or not isinstance(route.get('arguments'), list):
            errors.append(label + ' needs an MCP discovery route.')
    return errors


def render_catalog() -> str:
    """Render short discovery text without running a diagnostic."""
    lines = ['RoboClue rules', '']
    lines.extend(f"{rule['id']}  {rule['title']}" for rule in _RULES)
    lines.extend(['', 'Explain a rule: roboclue advanced rules RC001', SCOPE])
    return '\n'.join(lines)


def render_rule(rule: dict) -> str:
    """Render metadata returned by get_rule; no verdict is inferred from it."""
    lines = [f"{rule['id']}: {rule['title']}", rule['summary'], '', 'Scope: ' + rule['scope'],
             'Trigger: ' + rule['trigger'], 'Not checked: ' + '; '.join(rule['not_checked']),
             'Remedy: ' + rule['remedy'], 'Run: ' + rule['cli'],
             'MCP: ' + rule['mcp']['tool'], 'Guide: ' + rule['docs'], '', 'Tested profiles:']
    lines.extend(f"- {profile['name']} {profile['revision']}: {profile['profile']}" for profile in rule['frameworks'])
    lines.extend(['', 'Upstream evidence:'])
    lines.extend(f"- {source['title']}: {source['url']}" for source in rule['upstream'])
    return '\n'.join(lines)
