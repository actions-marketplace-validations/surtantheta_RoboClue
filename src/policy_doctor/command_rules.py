"""Pinned native command-path rules; plain data only, no policy imports."""

REPOSITORY = "https://github.com/huggingface/lerobot"
REVISION = "146d8474932c8b566717eb2738ba459337ab71c4"

SOURCE_MODULES = {
    "sync.py": "lerobot.inference.sync",
    "dispatch.py": "lerobot.inference.dispatch",
    "base.py": "lerobot.inference.base",
    "pipeline.py": "lerobot.processor.pipeline",
    "normalize_processor.py": "lerobot.processor.normalize_processor",
    "device_processor.py": "lerobot.processor.device_processor",
    "converters.py": "lerobot.processor.converters",
    "action_interpolator.py": "lerobot.utils.action_interpolator",
    "modeling_act.py": "lerobot.policies.act.modeling_act",
    "configuration_act.py": "lerobot.policies.act.configuration_act",
}

SOURCE_HASHES = {
    "sync.py": "8a00f1c3acce08b2896c59ff718136a7ccebf10bf2f4adf9678740d4ff223dd5",
    "dispatch.py": "1aa946b0096afefa68d1ad0a3e8a820ee5bb24c0ac31aaaa361261df8caa7c37",
    "base.py": "e254402956574938f270349e0a5de6909f8e5763313c617600698cf2b9b9f71f",
    "pipeline.py": "43fa3f47ebe134e69bff2834c80edec00ec2c4396dc71a4caeef4a7921277a89",
    "normalize_processor.py": "0b4beb2b9743d47256be53e67330d05c8552f07258a06eeeade1c9bbf217558c",
    "device_processor.py": "2e98b89c039005e40baf31f7c1c82a5d946df39c2328ef999ac63c1dd5e8a287",
    "converters.py": "51099b6ed8330f1b2fd88cf8afd3a09a4466e6a262d55e3151dbc38185ca75f4",
    "action_interpolator.py": "3879afec081f84cf0a9a5530db417e4ff4490b12685a1387075afa7406fdccc3",
    "modeling_act.py": "83d954f79ccd3eaa6774dbea92524939c8ac08359bb355ba7cd37b8601e1e3af",
    "configuration_act.py": "35a60c0dd3c811b137b5eb035bdbd40d40e2c24ba0cda61756052e6e1c829282",
    "LICENSE.txt": "9e108af26a7ae92a6d2cb0273856a4a0a84da8819c2cffee8b2657cc4f905718",
}

MAX_TICKS = 64
MAX_VALUES = 1_000_000
MAX_BYTES = 32 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024
MAX_DIM = 256
MAX_MULTIPLIER = 16
