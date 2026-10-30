import json

import pytest

from policy_doctor.artifacts import InputError, sha256
from policy_doctor.evidence import inspect_evidence


@pytest.mark.parametrize("pointer,value", [("/a~1b/~0key/1", 2), ("/a~1b/~0key/0", None),
                                          ("/a~1b/~0key", [None, 2])])
def test_json_pointer_escapes_arrays_and_null(tmp_path, pointer, value):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"a/b": {"~key": [None, 2]}}))
    result = inspect_evidence(str(path), pointer, sha256(path))
    assert result["status"] == "pass" and result["value"] == value


def test_stale_evidence_never_returns_a_current_value_as_verified(case):
    original = sha256(case["deployment"])
    case["deployment"].write_text(case["deployment"].read_text() + " ")
    result = inspect_evidence(str(case["deployment"]), "/units", original)
    assert result["status"] == "error" and result["code"] == "evidence.changed"
    assert "value" not in result


@pytest.mark.parametrize("pointer", ["/absent", "/items/01", "/items/-", "/items/20"])
def test_missing_and_noncanonical_array_indices_are_unknown(tmp_path, pointer):
    path = tmp_path / "source.json"
    path.write_text('{"items":[1,2]}')
    assert inspect_evidence(str(path), pointer)["status"] == "unknown"


@pytest.mark.parametrize("pointer", ["not/a/pointer", "/bad~2escape", "/bad~"])
def test_invalid_json_pointer_is_rejected(case, pointer):
    with pytest.raises(InputError, match="pointer"):
        inspect_evidence(str(case["deployment"]), pointer)


def test_digest_and_artifact_type_are_validated(case):
    with pytest.raises(InputError, match="digest"):
        inspect_evidence(str(case["deployment"]), "", "incorrect")
    with pytest.raises(InputError, match="JSON"):
        inspect_evidence(str(case["checkpoint"] / "pre.safetensors"))
