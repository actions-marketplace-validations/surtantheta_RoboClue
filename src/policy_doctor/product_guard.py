"""Guided prospective startup-reference setup for supported native adapters.

The default creates a local integration packet. Importing application code and
capturing a reference requires an explicit factory and reviewed-intent consent.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import shlex
import stat
import sys
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path

from .artifacts import InputError
from .training_guard import TargetContract
from .training_guard_capture import _write_new

FRAMEWORKS = ("groot", "lerobot", "openpi")
FACTORY_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", re.ASCII)


def _directory(value):
    if not isinstance(value, (str, Path)) or not str(value) or len(str(value)) > 4096:
        raise InputError("--output must name a new local directory")
    if not all(character.isprintable() for character in str(value)):
        raise InputError("--output must contain no terminal control characters")
    path = Path(value).expanduser().absolute()
    if ".." in path.parts or "://" in str(value) or "\x00" in str(value):
        raise InputError("--output requires a local directory without parent traversal")
    try:
        str(path).encode("utf-8")
        for parent in reversed(path.parents):
            mode = parent.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise InputError("--output parents must be materialized directories, without symlinks")
        if path.exists() or path.is_symlink():
            raise InputError("--output already exists; choose a new directory. Existing references are never replaced")
    except UnicodeError as error:
        raise InputError("--output requires a valid UTF-8 path") from error
    except FileNotFoundError as error:
        raise InputError("--output parent was not found; create the parent first") from error
    except OSError as error:
        raise InputError(f"Cannot inspect --output: {error}") from error
    return path


def _factory(value):
    module_name, attribute = value.split(":")
    try:
        result = importlib.import_module(module_name)
        for name in attribute.split("."):
            result = getattr(result, name)
    except (ImportError, AttributeError) as error:
        raise InputError(f"--factory {value!r} could not be loaded: {error}") from error
    if not callable(result):
        raise InputError("--factory must name a callable taking no arguments")
    return result()


@contextmanager
def _trusted_factory_imports():
    """Search the explicitly trusted working project only during native setup.

    Console scripts and isolated Python omit cwd from sys.path. Native factory
    construction, lazy batch iteration and resource cleanup may all import
    project modules. Restore the caller's exact path object and entries even if
    user setup changes them or fails. Imported modules remain loaded normally.
    """
    original_path = sys.path
    original_entries = original_path[:]
    try:
        sys.path.insert(0, str(Path.cwd()))
        yield
    finally:
        original_path[:] = original_entries
        sys.path = original_path


def _capture(framework, native, *, output, intent):
    if framework == "groot":
        from .groot_training_guard import capture_gr00t_reference

        return capture_gr00t_reference(native, output=output, intent=intent)
    if framework == "openpi":
        from .openpi_training_guard import capture_openpi_reference

        return capture_openpi_reference(native, output=output, intent=intent)
    from .lerobot_training_guard import capture_lerobot_reference

    allowed = {"batches", "dataset_id", "padding", "input_fields", "declarations"}
    if not isinstance(native, Mapping) or not {"batches", "dataset_id"} <= native.keys() or set(native) - allowed:
        raise InputError("LeRobot factory must return batches and dataset_id, plus optional padding, input_fields and declarations")
    contract = capture_lerobot_reference(intent=intent, **native)
    contract.save(output)
    return contract


def _factory_template(framework):
    result = {
        "groot": "Return your configured native Gr00tTrainer, before trainer.train().",
        "openpi": "Return your indexed native DataLoaderImpl; wrap the final transformed dataset with OpenpiSourceDataset before constructing the loader.",
        "lerobot": "Return {'batches': intended_preprocessed_batches, 'dataset_id': fingerprint['dataset_id']}; preserve original index and action_is_pad. Optional keys: padding, input_fields, declarations.",
    }[framework]
    return ('"""Edit this factory in your existing training project/environment.\n'
            'Construct the independently reviewed intended recipe before a change.\n'
            'Do not train, call a model or take optimizer steps here.\n'
            f'{result}\n"""\n\n'
            'def build_reference_context():\n'
            '    raise NotImplementedError("Replace this with your own reviewed native setup; no reference has been captured")\n')


def _attachment_template(framework):
    start = ('"""Attach only after capturing and retaining the pin independently.\n'
             'Keep the original criterion and pin through fault, correction and replay.\n'
             'Use the same supported native settings; later batches are unassessed.\n"""\n'
             'from pathlib import Path\n\n'
             'CONTRACT = Path(__file__).parent / "startup-reference.json"\n\n')
    if framework == "groot":
        return start + ('from roboclue import Gr00tTrainingGuard\n\n'
                        'def attach(trainer, independently_retained_pin):\n'
                        '    """Use: with attach(trainer, retained_pin): trainer.train()."""\n'
                        '    return Gr00tTrainingGuard(trainer, contract=CONTRACT,\n'
                        '                              criterion_sha256=independently_retained_pin)\n')
    if framework == "openpi":
        return start + ('from roboclue import OpenpiTrainingGuard\n\n'
                        'def attach(loader, model, independently_retained_pin):\n'
                        '    """Use returned context as guarded_loader; preserve native cast/device steps."""\n'
                        '    return OpenpiTrainingGuard(loader, model=model, contract=CONTRACT,\n'
                        '                                criterion_sha256=independently_retained_pin)\n')
    return start + ('from roboclue import LeRobotTrainingGuard\n\n'
                    'def attach(policy, dataset_id, independently_retained_pin):\n'
                    '    """Use returned context around the existing policy/update call."""\n'
                    '    return LeRobotTrainingGuard(policy, dataset_id=dataset_id, contract=CONTRACT,\n'
                    '                                criterion_sha256=independently_retained_pin)\n')


def initialize_guard(*, framework, output, factory=None, intent=None, reviewed_intended_recipe=False):
    """Prepare integration files and optionally invoke the explicit native factory.

    Factory code is application code, not sandboxed. The caller must ensure it
    only constructs the reviewed native setup, without training or model calls.
    A context-manager return value is entered and closed around capture.
    """
    if framework not in FRAMEWORKS:
        raise InputError("--framework must be groot, lerobot or openpi")
    if type(reviewed_intended_recipe) is not bool:
        raise InputError("reviewed_intended_recipe must be Boolean")
    if factory is None:
        if intent is not None or reviewed_intended_recipe:
            raise InputError("--intent and --reviewed-intended-recipe require --factory")
    else:
        if type(factory) is not str or len(factory) > 512 or not FACTORY_PATTERN.fullmatch(factory):
            raise InputError("--factory must use module:callable, from your existing trusted training project")
        if not reviewed_intended_recipe or type(intent) is not str or not intent.strip() or len(intent) > 2048:
            raise InputError("Reference capture requires --reviewed-intended-recipe and a nonempty --intent (up to 2048 characters); a suspect batch cannot establish its own correctness")
        if any(ord(character) < 32 for character in intent):
            raise InputError("--intent must be a single line of reviewed recipe intent")
    directory = _directory(output)
    directory.mkdir(mode=0o700)
    _write_new(directory / "reference_factory.py", _factory_template(framework).encode())
    _write_new(directory / "attach_guard.py", _attachment_template(framework).encode())
    next_command = shlex.join(["roboclue", "guard", "init", "--framework", framework,
        "--factory", "your_training_module:build_reference_context", "--intent",
        "Describe the independently reviewed intended recipe", "--reviewed-intended-recipe",
        "--output", str(directory.with_name(directory.name + "-reference"))])
    readme = (f"# Your {framework} startup guard\n\n"
              "1. Construct the independently reviewed intended recipe in a separate process BEFORE changing it.\n"
              "2. Implement a trusted zero-argument factory in your existing training project. See reference_factory.py.\n"
              "3. Run the capture command below in that environment. It consumes startup batches and sampler state (one batch for GR00T/OpenPI; your declared cohort for LeRobot).\n"
              "4. Retain the printed pin independently (for example in your experiment record), then use attach_guard.py.\n"
              "5. Correct candidate problems and verify against that ORIGINAL criterion and pin. Never re-capture to clear a fault.\n\n"
              f"```sh\n{next_command}\n```\n\n"
              "A factory may return a context manager to release application-owned workers/loaders after capture. "
              "It must not call training, the model or an optimizer. RoboClue cannot sandbox arbitrary application code. "
              "Explicit capture temporarily searches your working project for its factory and helper imports, including cleanup; "
              "the Python search path is restored afterwards. Default setup never imports application code.\n\n"
              "GR00T requires the supported N1.7 source, one process, zero loader workers, remove_unused_columns=False, "
              "eval_on_start=False and allow_padding=False. OpenPI requires its supported indexed synchronous PyTorch "
              "loader, one process and zero workers. LeRobot requires original global row indices and a deterministic "
              "preprocessed startup cohort; declare any padding/model-input objective explicitly.\n\n"
              "Reference capture checks no candidate and does not prove physical or model correctness. Later batches remain "
              "unassessed. If the run already failed without retained intent, historical target correctness remains unknown.\n")
    _write_new(directory / "README.md", readme.encode())
    result = {"kind": "guard_setup", "framework": framework, "assessment_status": "not_run",
              "reference_status": "not_captured", "output": str(directory),
              "next_command": next_command, "guide": f"docs/{'GR00T' if framework == 'groot' else framework.upper()}_TRAINING_GUARD.md"}
    if factory is not None:
        with _trusted_factory_imports(), ExitStack() as stack:
            native = _factory(factory)
            if callable(getattr(native, "__enter__", None)) and callable(getattr(native, "__exit__", None)):
                native = stack.enter_context(native)
            contract = _capture(framework, native, output=directory / "startup-reference.json",
                                intent={"objective": intent.strip()})
        if not isinstance(contract, TargetContract):
            raise InputError("The native capture helper did not return a target contract")
        TargetContract.load(directory / "startup-reference.json", expected_sha256=contract.criterion_sha256)
        _write_new(directory / "criterion-sha256.txt", (contract.criterion_sha256 + "\n").encode())
        result.update(reference_status="captured", criterion_sha256=contract.criterion_sha256,
                      next_command="Retain criterion_sha256 independently, then attach with attach_guard.py in this output directory")
    _write_new(directory / "setup.json", (json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return result


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['resume']:
        from .resume_guard import main as resume_main

        return resume_main(arguments[1:])
    parser = argparse.ArgumentParser(prog="roboclue guard", description="Prepare intended training inputs or check a recording before resume.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create integration files; capture only with explicit reviewed factory")
    init.add_argument("--framework", choices=FRAMEWORKS, required=True)
    init.add_argument("--output", required=True, help="New directory; existing references are never replaced")
    init.add_argument("--factory", help="Explicit local application execution: trusted module:callable constructing the native setup")
    init.add_argument("--intent", help="Single-line independently reviewed intended recipe, not a suspect candidate")
    init.add_argument("--reviewed-intended-recipe", action="store_true", help="Confirm the factory constructs the intended recipe before a change")
    init.add_argument("--json", action="store_true")
    sub.add_parser('resume', add_help=False, help='Check the declared recording root before lerobot-record --resume (see resume --help)')
    args = parser.parse_args(arguments)
    try:
        result = initialize_guard(framework=args.framework, output=args.output, factory=args.factory,
                                  intent=args.intent, reviewed_intended_recipe=args.reviewed_intended_recipe)
        code = 0
    except Exception as error:
        result = {"kind": "guard_setup", "assessment_status": "not_run", "reference_status": "not_completed",
                  "status": "input_error", "message": str(error), "error_type": type(error).__name__}
        code = 2
    if args.json:
        print(json.dumps(result, sort_keys=True))
    elif code:
        print("Could not prepare the guard: " + result["message"])
        print("Check the declared setup and choose a new output directory; no candidate was verified.")
    elif result["reference_status"] == "captured":
        print("Startup reference captured; no training or candidate verification was performed by RoboClue.")
        print("Retain this pin independently BEFORE changing the pipeline: " + result["criterion_sha256"])
        print("Next: use attach_guard.py in " + result["output"] + " with that original pin.")
    else:
        print("Guard setup ready; no reference has been captured and no candidate checked.")
        print("Copy the reference_factory.py template from " + result["output"] + " into your existing training module.")
        print("Next: " + result["next_command"])
    return code
