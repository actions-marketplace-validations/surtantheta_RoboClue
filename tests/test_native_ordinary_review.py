"""Independent small witnesses for ordinary-folder native dependency boundaries."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from policy_doctor import native_intake as intake
from policy_doctor.artifacts import InputError
from policy_doctor.native_processing import _durable_payload


def _receipt(root, relative):
    path = root / relative
    return {
        "path": relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _payload(root):
    """Declared generated provider-owned manifest; no native API or data used."""
    root.mkdir()
    files = {
        "result.json": "{}",
        "samples/sample_0.json": "{}",
        "samples/sample_0.npz": "generated numeric-array placeholder",
        "traces/sample_0_saved.json": "[]",
        "traces/sample_0_current_factory.json": "[]",
        "exports/saved/policy_preprocessor.json": json.dumps(
            {"steps": [{"registry_name": "device_processor", "config": {"device": "cpu"}}]}
        ),
        "exports/current_factory/policy_preprocessor.json": json.dumps(
            {"steps": [{"registry_name": "device_processor", "config": {"device": "cpu"}}]}
        ),
    }
    for name, value in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    groups = []
    routes = []
    for name in ("saved", "current_factory"):
        groups.append(
            {
                "route": name,
                "path": "exports/" + name,
                "config_filename": "policy_preprocessor.json",
                "files": [_receipt(root / ("exports/" + name), "policy_preprocessor.json")],
            }
        )
        routes.append(
            {
                "name": name,
                "status": "complete",
                "input": {"state": [1]},
                "output": {"state": [1]},
                "steps": [{"class": "generated.NativeStep"}],
                "trace": _receipt(root, "traces/sample_0_" + name + ".json"),
            }
        )
    samples = [
        {
            "identity": {"episode_index": 0, "frame_index": 0, "index": 0},
            "arrays": _receipt(root, "samples/sample_0.npz"),
            "metadata": _receipt(root, "samples/sample_0.json"),
        }
    ]
    manifest = {
        "schema": "policy_doctor.native_lerobot.v1",
        "provider": "lerobot",
        "samples": samples,
        "exports": groups,
        "source_files": [],
        "expected_samples": [],
    }
    (root / "worker_manifest.json").write_text(json.dumps(manifest))
    return {
        "provider": "lerobot",
        "execution_status": "complete",
        "samples": [{"identity": samples[0]["identity"], "routes": routes}],
        "exports": groups,
    }


def test_declared_native_payload_positive_control(tmp_path):
    stage = tmp_path / "stage"
    report = _payload(stage)
    _durable_payload(stage, report)


@pytest.mark.parametrize(
    "relative", ["exports/_home_author_cache.lock", "samples/undeclared.npy", "metadata/private-path.txt"]
)
def test_unreferenced_file_under_allowed_payload_prefix_refused(tmp_path, relative):
    stage = tmp_path / "stage"
    report = _payload(stage)
    path = stage / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Generated unexpected private runtime content")
    with pytest.raises(InputError):
        _durable_payload(stage, report)


def _selection(tmp_path):
    roots = {name: tmp_path / ("original-" + name) for name in ("model", "dataset")}
    staged = {name: tmp_path / ("view-" + name) for name in roots}
    for path in [*roots.values(), *staged.values()]:
        path.mkdir()
    for name in intake.MODEL_SEEDS["groot_state"]:
        (roots["model"] / name).write_text("{}")
    (roots["dataset"] / "meta").mkdir()
    for name in ("info.json", "modality.json"):
        (roots["dataset"] / "meta" / name).write_text("{}")
    (roots["dataset"] / "data").mkdir()
    (roots["dataset"] / "data/selected.parquet").write_bytes(b"Generated exact-byte copy witness")
    return (
        {
            "provider": "groot_state",
            "roots": {k: str(v) for k, v in roots.items()},
            "staged_roots": {k: str(v) for k, v in staged.items()},
            "files": [],
            "total_bytes": 0,
            "seed_membership": intake._membership("groot_state", roots),
            "tokenizer_membership": {},
        },
        roots,
        staged,
    )


def test_source_copy_exact_bytes_and_later_mutation_refusal(tmp_path):
    selection, roots, staged = _selection(tmp_path)
    intake._copy_selected(selection, "dataset", "data/selected.parquet", "native_data")
    source, target = roots["dataset"] / "data/selected.parquet", staged["dataset"] / "data/selected.parquet"
    assert source.read_bytes() == target.read_bytes() and source.stat().st_ino != target.stat().st_ino
    intake.verify_native_inputs(selection)
    source.write_bytes(b"Changed generated selected source")
    with pytest.raises(InputError):
        intake.verify_native_inputs(selection)


def test_selected_source_ancestor_symlink_refused_before_copy(tmp_path):
    selection, roots, staged = _selection(tmp_path)
    (roots["dataset"] / "data").rename(roots["dataset"] / "actual-data")
    (roots["dataset"] / "data").symlink_to(roots["dataset"] / "actual-data", target_is_directory=True)
    with pytest.raises((OSError, InputError)):
        intake._copy_selected(selection, "dataset", "data/selected.parquet", "native_data")
    assert not (staged["dataset"] / "data/selected.parquet").exists()


def test_selected_source_swapped_during_copy_refused(tmp_path, monkeypatch):
    selection, roots, _ = _selection(tmp_path)
    source = roots["dataset"] / "data/selected.parquet"
    original_read = intake.os.read
    changed = False

    def read(fd, count):
        nonlocal changed
        data = original_read(fd, count)
        if data and not changed:
            changed = True
            source.rename(source.with_suffix(".old"))
            source.write_bytes(b"Generated replacement inode")
        return data

    monkeypatch.setattr(intake.os, "read", read)
    with pytest.raises(InputError):
        intake._copy_selected(selection, "dataset", "data/selected.parquet", "native_data")


def test_unreferenced_empty_payload_directory_refused(tmp_path):
    stage = tmp_path / "stage"
    report = _payload(stage)
    (stage / "exports/private-empty-cache").mkdir()
    with pytest.raises(InputError):
        _durable_payload(stage, report)


def test_oversized_footer_refused_before_arrow_constructor(tmp_path, monkeypatch):
    import pyarrow.parquet as pq

    from policy_doctor import native_parquet_guard as guard

    path = tmp_path / "bad-footer.parquet"
    path.write_bytes(b"PAR1" + (guard.MAX_FOOTER + 1).to_bytes(4, "little") + b"PAR1")

    def deny(*args, **kwargs):
        pytest.fail("Arrow constructor reached despite oversized footer")

    monkeypatch.setattr(pq, "ParquetFile", deny)
    with pytest.raises(ValueError, match="footer"):
        guard.guard_file(path)


def test_string_dictionary_expansion_bound_before_value_read(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from policy_doctor import native_parquet_guard as guard

    path = tmp_path / "bounded-metadata-only.parquet"
    # Real footer bytes satisfy the plaintext Compact-Thrift scan that runs
    # before Arrow; Arrow then sees only the declared metadata below.
    pq.write_table(pa.table({"native_string": ["x"]}), path)
    column = SimpleNamespace(num_values=2000, total_uncompressed_size=32768)
    group = SimpleNamespace(num_columns=1, column=lambda index: column)
    metadata = SimpleNamespace(num_rows=2000, num_columns=1, num_row_groups=1, row_group=lambda index: group)

    # This is declared metadata only: never allocates the hypothetical 64 MiB strings.
    class MetadataOnly:
        schema_arrow = pa.schema([pa.field("native_string", pa.string())])

        def __init__(self, *args, **kwargs):
            self.metadata = metadata

        def read(self, *args, **kwargs):
            pytest.fail("String values read before allocation guard")

        def iter_batches(self, *args, **kwargs):
            pytest.fail("String batches decoded before allocation guard")

    monkeypatch.setattr(pq, "ParquetFile", MetadataOnly)
    with pytest.raises(ValueError, match="decoded allocation"):
        guard.guard_file(path)


def _plan_fixture():
    info = {
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"observation.camera": {"dtype": "video"}},
    }
    data = "data/chunk-000/file-007.parquet"
    video = "videos/observation.camera/chunk-000/file-003.mp4"
    plan = {
        "provider": "lerobot",
        "operation": "plan",
        "execution_status": "complete",
        "metadata_files": ["meta/episodes/chunk-000/file-000.parquet"],
        "files": [{"path": data, "role": "native_data"}, {"path": video, "role": "native_video"}],
        "samples": [
            {
                "episode_index": 7,
                "frame_index": 0,
                "episode_length": 2,
                "data_path": data,
                "video_paths": {"observation.camera": video},
            }
        ],
    }
    return plan, [{"episode_index": 7, "frame_index": 0}], info, plan["metadata_files"][:]


def test_native_plan_exact_closure_positive_control():
    plan, samples, info, metadata = _plan_fixture()
    assert len(intake._planned_files(plan, samples, info, metadata)) == 2


@pytest.mark.parametrize(
    "change", ["bool_identity", "missing_camera", "extra_file", "missing_metadata", "wrong_template"]
)
def test_native_plan_exact_closure_refusals(change):
    plan, samples, info, metadata = _plan_fixture()
    if change == "bool_identity":
        samples[0]["episode_index"] = 1
        plan["samples"][0]["episode_index"] = True
    elif change == "missing_camera":
        plan["samples"][0]["video_paths"] = {}
    elif change == "extra_file":
        plan["files"].append({"path": "data/chunk-000/file-999.parquet", "role": "native_data"})
    elif change == "missing_metadata":
        plan["metadata_files"] = []
    elif change == "wrong_template":
        plan["samples"][0]["data_path"] = "outside/file-007.parquet"
    with pytest.raises(InputError):
        intake._planned_files(plan, samples, info, metadata)
