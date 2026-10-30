"""RoboClue's public Python entry point.

Existing ``policy_doctor`` imports remain compatible. Adapters are imported only
when requested, so core installation does not import robotics frameworks.
"""
from importlib import import_module

from policy_doctor import __version__

_EXPORTS = {
    "Gr00tTrainingGuard": "groot_training_guard",
    "capture_gr00t_reference": "groot_training_guard",
    "LeRobotTrainingGuard": "lerobot_training_guard",
    "capture_lerobot_reference": "lerobot_training_guard",
    "OpenpiTrainingGuard": "openpi_training_guard",
    "capture_openpi_reference": "openpi_training_guard",
    "RuntimeRecorder": "runtime_recorder",
    "SignalSpec": "runtime_recorder",
    "SendReceipt": "runtime_recorder",
    "health_card": "health_cards",
    "triage_flight_recording": "flight_recorder",
    "FlightRecorder": "flight_recorder",
    "OpenpiSourceDataset": "openpi_training_guard",
    "fingerprint_training_dataset": "training_dataset_identity",
    "scan_health_cards": "health_cards",
    "write_health_card": "health_cards",
    "scan_once": "health_monitor",
    "flight_view": "first_use_views",
    "check_sim_provenance": "sim_guards",
    "remedy_list": "sim_guards",
    "BoundaryContract": "upgrade_impact",
    "save_boundary_capture": "upgrade_impact",
    "assess_upgrade_impact": "upgrade_impact",
    "write_upgrade_alerts": "upgrade_impact",
    "register_mcp_tools": "big_bets_frontdoor",
    "rule_catalog": "rule_catalog",
    "get_rule": "rule_catalog",
    "rules_for_code": "rule_catalog",
    "collect_framework_versions": "framework_versions",
    "supported_framework_profiles": "framework_versions",
    "validate_robot_declaration": "robot_contract",
    "check_declared_camera_keys": "robot_contract",
    "write_recovery_copy": "recovery_copy",
    "consumer_version_context": "consumer_version",
    "PipelineContractError": "pipeline_contracts",
    "guard_openpi_initialization": "pipeline_contracts",
    "capture_openpi_initialization": "pipeline_contracts",
    "check_pipeline_contract": "pipeline_contracts",
    "capture_lerobot_camera_inputs": "pipeline_contracts",
    "guard_lerobot_camera_inputs": "pipeline_contracts",
    "install_lerobot_camera_guard": "pipeline_contracts",
}
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module 'roboclue' has no attribute {name!r}")
    value = getattr(import_module("policy_doctor." + _EXPORTS[name]), name)
    globals()[name] = value
    return value
