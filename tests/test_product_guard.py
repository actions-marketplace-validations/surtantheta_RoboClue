"""Prospective setup controls; native framework qualification is separate."""
import ast
import importlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from policy_doctor import product_guard
from policy_doctor.artifacts import InputError
from policy_doctor.lerobot_training_guard import lerobot_target_view
from policy_doctor.training_guard import TargetContract


def _context():
    return {"batches": [{"action": np.array([[[0.25, 0.5]]], dtype=np.float32),
                         "index": np.array([7], dtype=np.int64),
                         "action_is_pad": np.zeros((1, 1), dtype=bool)}],
            "dataset_id": "sha256:" + "1" * 64}


def _consent(output, **kwargs):
    return dict(framework="lerobot", output=output, factory="my_training:make_reference",
                intent="Preserve independently reviewed minmax-relative targets", reviewed_intended_recipe=True, **kwargs)


@pytest.mark.parametrize("framework", product_guard.FRAMEWORKS)
def test_init_default_never_imports_or_captures(tmp_path, monkeypatch, framework):
    monkeypatch.setattr(product_guard, "_factory", lambda *_: pytest.fail("Default may not import application code"))
    output = tmp_path / "new-packet"
    result = product_guard.initialize_guard(framework=framework, output=output)
    assert result["reference_status"] == "not_captured" and result["assessment_status"] == "not_run"
    assert set(p.name for p in output.iterdir()) == {"README.md", "reference_factory.py", "attach_guard.py", "setup.json"}
    assert not (output / "startup-reference.json").exists()
    assert "--reviewed-intended-recipe" in result["next_command"]
    for name in ("reference_factory.py", "attach_guard.py"):
        ast.parse((output / name).read_text())
    assert "from roboclue import " in (output / "attach_guard.py").read_text()
    assert "policy_doctor" not in (output / "attach_guard.py").read_text()
    assert json.loads((output / "setup.json").read_text()) == result


@pytest.mark.parametrize("control", ["\n", "\x1b", "\r", "\t", "\x7f"])
def test_output_control_characters_refused_before_creation(tmp_path, control):
    output = tmp_path / ("packet" + control)
    with pytest.raises(InputError, match="control"):
        product_guard.initialize_guard(framework="openpi", output=output)
    assert not output.exists()


def test_explicit_lerobot_capture_is_real_retained_contract(tmp_path, monkeypatch):
    context = _context()
    monkeypatch.setattr(product_guard, "_factory", lambda _: context)
    output = tmp_path / "reference"
    result = product_guard.initialize_guard(**_consent(output))
    assert result["reference_status"] == "captured" and result["assessment_status"] == "not_run"
    pin = result["criterion_sha256"]
    assert (output / "criterion-sha256.txt").read_text() == pin + "\n"
    reference = output / "startup-reference.json"
    original = reference.read_bytes()
    criterion = TargetContract.load(reference, expected_sha256=pin)
    view, keys = lerobot_target_view(context["batches"][0], dataset_id=context["dataset_id"])
    assert criterion.check(view, keys)["status"] == "pass"
    view["inputs"]["action"][0, 0, 1] += 0.25
    assert criterion.check(view, keys)["status"] == "error"
    view["inputs"]["action"][0, 0, 1] -= 0.25
    assert criterion.check(view, keys)["status"] == "pass"
    assert reference.read_bytes() == original
    with pytest.raises(InputError, match="already exists"):
        product_guard.initialize_guard(**_consent(output))
    assert reference.read_bytes() == original


@pytest.mark.parametrize("mutation", [
    {"reviewed_intended_recipe": False}, {"intent": None}, {"intent": " "},
    {"intent": "x" * 2049}, {"intent": "bad\nintent"}, {"factory": "shell command"},
    {"factory": "path/to/file:factory"}, {"factory": "m:$(command)"}, {"factory": "x" * 513},
])
def test_incomplete_or_malformed_consent_does_not_import_or_write(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(product_guard, "_factory", lambda *_: pytest.fail("Consent gates must precede import"))
    options = _consent(tmp_path / "output")
    options.update(mutation)
    with pytest.raises(InputError):
        product_guard.initialize_guard(**options)
    assert not options["output"].exists()


@pytest.mark.parametrize("options", [{"intent": "future recipe"}, {"reviewed_intended_recipe": True}])
def test_capture_flags_require_factory(tmp_path, options):
    with pytest.raises(InputError, match="require --factory"):
        product_guard.initialize_guard(framework="groot", output=tmp_path / "new", **options)


@pytest.mark.parametrize("kind", ["directory", "regular", "fifo", "symlink", "dangling_symlink", "parent_symlink", "parent_fifo", "missing_parent"])
def test_output_paths_refuse_without_factory_execution(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(product_guard, "_factory", lambda *_: pytest.fail("Path refusal must precede import"))
    output = tmp_path / "output"
    if kind == "directory":
        output.mkdir()
    elif kind == "regular":
        output.write_text("preserve")
    elif kind == "fifo":
        os.mkfifo(output)
    elif kind in ("symlink", "dangling_symlink"):
        output.symlink_to(tmp_path if kind == "symlink" else tmp_path / "missing")
    elif kind == "parent_symlink":
        output.symlink_to(tmp_path, target_is_directory=True)
        output /= "new"
    elif kind == "parent_fifo":
        os.mkfifo(output)
        output /= "new"
    elif kind == "missing_parent":
        output /= "new"
    with pytest.raises(InputError):
        product_guard.initialize_guard(**_consent(output))


def test_context_lifetime_closed_and_failure_not_reported_as_reference(tmp_path, monkeypatch, capsys):
    state = []

    @contextmanager
    def native():
        state.append("enter")
        try:
            yield {"batches": []}  # Missing source identity; not valid capture.
        finally:
            state.append("closed")

    monkeypatch.setattr(product_guard, "_factory", lambda _: native())
    output = tmp_path / "failed"
    assert product_guard.main(["init", "--framework", "lerobot", "--output", str(output),
                               "--factory", "my_training:make_reference", "--intent", "reviewed", "--reviewed-intended-recipe", "--json"]) == 2
    assert state == ["enter", "closed"]
    result = json.loads(capsys.readouterr().out)
    assert result["reference_status"] == "not_completed" and result["assessment_status"] == "not_run"
    assert not (output / "criterion-sha256.txt").exists()
    assert not (output / "setup.json").exists()


@pytest.mark.parametrize("framework", ["groot", "openpi"])
def test_supported_native_helpers_receive_exact_native_object_and_intent(tmp_path, monkeypatch, framework):
    # This routing control does not replace the separately qualified native proofs.
    context = _context()
    from policy_doctor.lerobot_training_guard import capture_lerobot_reference

    expected = capture_lerobot_reference(intent={"objective": "reviewed"}, **context)
    native = object()
    module = importlib.import_module("policy_doctor." + ("groot_training_guard" if framework == "groot" else "openpi_training_guard"))
    helper_name = "capture_" + ("gr00t" if framework == "groot" else "openpi") + "_reference"

    def helper(value, *, output, intent):
        assert value is native and intent == {"objective": "reviewed"}
        expected.save(output)
        return expected

    monkeypatch.setattr(module, helper_name, helper)
    monkeypatch.setattr(product_guard, "_factory", lambda _: native)
    result = product_guard.initialize_guard(framework=framework, output=tmp_path / framework,
        factory="my_training:reference", intent="reviewed", reviewed_intended_recipe=True)
    assert result["reference_status"] == "captured" and result["criterion_sha256"] == expected.criterion_sha256


def test_factory_only_explicit_callable_and_attribute_lookup(monkeypatch):
    monkeypatch.setattr(product_guard.importlib, "import_module", lambda _: SimpleNamespace(nested=SimpleNamespace(build=lambda: 7)))
    assert product_guard._factory("my_project:nested.build") == 7
    with pytest.raises(InputError, match="could not be loaded"):
        product_guard._factory("my_project:missing")
    with pytest.raises(InputError, match="callable"):
        product_guard._factory("my_project:nested")


def test_cli_plain_setup_is_not_an_assessment(tmp_path, capsys):
    output = tmp_path / "new"
    assert product_guard.main(["init", "--framework", "groot", "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "no reference has been captured" in text and "no candidate checked" in text
    assert text.count("\n") == 3


def test_actual_cli_factory_import_and_capture(tmp_path, monkeypatch, capsys):
    module = tmp_path / "native_setup.py"
    module.write_text("import numpy as np\ndef intended():\n    return {'batches': [{'action': np.ones((1,1,1), dtype=np.float32), 'index': np.array([0]), 'action_is_pad': np.zeros((1,1), dtype=bool)}], 'dataset_id': 'sha256:' + '2' * 64}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    output = tmp_path / "reference"
    assert product_guard.main(["init", "--framework", "lerobot", "--output", str(output),
        "--factory", "native_setup:intended", "--intent", "Reviewed deterministic startup targets", "--reviewed-intended-recipe"]) == 0
    text = capsys.readouterr().out
    assert "Retain this pin independently" in text
    assert (output / "startup-reference.json").is_file()


def test_isolated_console_finds_explicit_project_factory_and_deferred_imports(tmp_path):
    """Only the product source is bootstrapped; cwd must be enabled by the product.

    The fresh installed smoke separately executes the actual installed console.
    """
    (tmp_path / "project_values.py").write_text(
        "import numpy as np\n"
        "def values():\n"
        "    return {'action': np.ones((1,1,1), dtype=np.float32), 'index': np.array([0]), 'action_is_pad': np.zeros((1,1), dtype=bool)}\n")
    (tmp_path / "project_cleanup.py").write_text("def done():\n    return True\n")
    (tmp_path / "local_training_factory.py").write_text(
        "from contextlib import contextmanager\n"
        "@contextmanager\n"
        "def intended():\n"
        "    import project_values\n"
        "    def batches():\n"
        "        from project_values import values\n"
        "        yield values()\n"
        "    try:\n"
        "        yield {'batches': batches(), 'dataset_id': 'sha256:' + '2' * 64}\n"
        "    finally:\n"
        "        from project_cleanup import done\n"
        "        assert done()\n")
    source = str(Path(product_guard.__file__).resolve().parents[1])
    launcher = "import sys; sys.path.insert(0, " + repr(source) + "); from roboclue.__main__ import main"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", launcher,
        "guard", "init", "--framework", "lerobot", "--output", str(tmp_path / "reference"),
        "--factory", "local_training_factory:intended", "--intent", "Reviewed deterministic targets",
        "--reviewed-intended-recipe", "--json"], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["reference_status"] == "captured"
    assert (tmp_path / "reference" / "startup-reference.json").is_file()


@pytest.mark.parametrize("failure", [None, "factory", "context_enter", "capture", "context_exit"])
def test_trusted_project_import_scope_restored_on_success_and_errors(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    original_path, original_entries = sys.path, sys.path[:]

    @contextmanager
    def native():
        assert sys.path[0] == str(tmp_path)
        if failure == "context_enter":
            raise RuntimeError("context enter failure")
        try:
            yield _context() if failure != "capture" else {"batches": []}
        finally:
            assert sys.path[0] == str(tmp_path)
            if failure == "context_exit":
                raise RuntimeError("context exit failure")

    def factory(_):
        assert sys.path[0] == str(tmp_path)
        if failure == "factory":
            raise RuntimeError("factory failure")
        return native()

    monkeypatch.setattr(product_guard, "_factory", factory)
    if failure:
        with pytest.raises((RuntimeError, InputError)):
            product_guard.initialize_guard(**_consent(tmp_path / "output"))
    else:
        assert product_guard.initialize_guard(**_consent(tmp_path / "output"))["reference_status"] == "captured"
    assert sys.path is original_path and sys.path == original_entries


def test_trusted_import_scope_restores_path_even_if_application_reassigns_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_path, original_entries = sys.path, sys.path[:]
    with pytest.raises(RuntimeError, match="application failure"):
        with product_guard._trusted_factory_imports():
            sys.path = ["application replacement"]
            raise RuntimeError("application failure")
    assert sys.path is original_path and sys.path == original_entries
