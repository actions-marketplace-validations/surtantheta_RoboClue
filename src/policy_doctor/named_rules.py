"""Pinned native named-action boundary identities; no native imports."""

from .command_rules import (
    MAX_BYTES,
    MAX_DIM,
    MAX_MANIFEST,
    MAX_TICKS,
    REPOSITORY,
    REVISION,
)
from .command_rules import (
    SOURCE_HASHES as COMMAND_HASHES,
)
from .command_rules import (
    SOURCE_MODULES as COMMAND_MODULES,
)

SOURCE_MODULES = {
    **COMMAND_MODULES,
    "core.py": "lerobot.rollout.strategies.core",
    "robot_wrapper.py": "lerobot.rollout.robot_wrapper",
}
SOURCE_HASHES = {
    **COMMAND_HASHES,
    "core.py": "c9ae5fb4c93d22835c8898f43b19cfb56559583fa4ea9e326befd123e733d8ab",
    "robot_wrapper.py": "717dece22bebdcd4a2dd06c4298c7358035a4cb27165597b6bfda26ba3fe69e0",
}

RECORDING_SOURCE_MODULES = {
    "episodic.py": "lerobot.rollout.strategies.episodic",
    "feature_utils.py": "lerobot.utils.feature_utils",
    "lerobot_dataset.py": "lerobot.datasets.lerobot_dataset",
    "dataset_writer.py": "lerobot.datasets.dataset_writer",
}
RECORDING_SOURCE_HASHES = {
    "episodic.py": "4a6f63c6e2e3feab58c666bf5fe87e041ac0f110ff078f14714cb2f24a0d7fe1",
    "feature_utils.py": "312ed4eb29fa50f5a3e186e7579ccbf901cc60d2d757f623898b04789c7f2adc",
    "lerobot_dataset.py": "9837856f0b7611adb2ae71dc87922b2403b5f96c370a1a76925147036a2d5cf2",
    "dataset_writer.py": "9151b19d3bd30b343b6917563b3c728d50cd629c9dbd4caf87c1becdc5824f8d",
}

__all__ = ["MAX_BYTES", "MAX_DIM", "MAX_MANIFEST", "MAX_TICKS", "REPOSITORY",
           "REVISION", "SOURCE_HASHES", "SOURCE_MODULES"]
