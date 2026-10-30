"""Independent no-allocation witnesses for native row intake boundaries."""

import json
from types import SimpleNamespace

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.native_processing import _validate_inputs, _validate_report
from policy_doctor.native_rows_worker import run


def _parquet_witness(monkeypatch, *, state_type, state_num_values):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    names = ['episode_index', 'frame_index', 'index', 'timestamp', 'observation.state']
    columns = [SimpleNamespace(path_in_schema=name + ('.list.element' if name == 'observation.state' else ''),
                               total_uncompressed_size=64,
                               num_values=state_num_values if name == 'observation.state' else 1)
               for name in names]
    group = SimpleNamespace(num_columns=5, column=lambda index: columns[index])
    metadata = SimpleNamespace(num_rows=1, num_columns=5, num_row_groups=1,
                               row_group=lambda _: group)
    schema = pa.schema([(name, state_type if name == 'observation.state' else
                         pa.float64() if name == 'timestamp' else pa.int64()) for name in names])

    class MetadataOnlyParquet:
        schema_arrow = schema

        def __init__(self, _path):
            self.metadata = metadata

        def read(self, **_kwargs):
            raise AssertionError('Arrow decoding reached before selected-value/type preflight')

    monkeypatch.setattr(pq, 'ParquetFile', MetadataOnlyParquet)
    return {'path': '/generated/never-opened.parquet', 'relative_path': 'data/episode.parquet',
            'columns': ['observation.state'], 'samples': [{'episode_index': 0, 'frame_index': 0}]}


def test_selected_parquet_value_count_precedes_decoding(monkeypatch):
    pa = pytest.importorskip('pyarrow')
    # Valid list metadata can represent huge repeated vectors in tiny encoded pages.
    # No large array or file is allocated; read is an explicit failure sentinel.
    request = _parquet_witness(monkeypatch, state_type=pa.list_(pa.float64()),
                               state_num_values=8 * 1024**2)
    with pytest.raises(ValueError):
        run(request)


def test_selected_binary_media_column_refused_before_read(monkeypatch):
    pa = pytest.importorskip('pyarrow')
    request = _parquet_witness(monkeypatch, state_type=pa.binary(), state_num_values=1)
    with pytest.raises(ValueError):
        run(request)


def test_groot_v3_template_declines_instead_of_format_keyerror(tmp_path):
    dataset = tmp_path / 'dataset'
    meta = dataset / 'meta'
    meta.mkdir(parents=True)
    model = tmp_path / 'model'
    model.mkdir()
    (meta / 'modality.json').write_text(json.dumps({'state': {'arm': {'start': 0, 'end': 1}}}))
    (meta / 'info.json').write_text(json.dumps({'chunks_size': 1000,
        'features': {'observation.state': {'dtype': 'float32', 'shape': [1]}},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'}))
    with pytest.raises(InputError):
        _validate_inputs('groot_state', model, dataset)


@pytest.mark.parametrize('sample', [
    {},
    {'identity': {'episode_index': 99, 'frame_index': 0, 'index': 0}, 'routes': [{'status': 'complete'}]},
    {'identity': {'episode_index': 0, 'frame_index': 0}, 'routes': []},
])
def test_complete_native_report_needs_requested_identity_and_routes(sample):
    report = {'provider': 'lerobot', 'execution_status': 'complete', 'samples': [sample],
              'native_versions': {}, 'source_files': [], 'exports': [], 'artifact_receipts': [],
              'unverified': ['Generated invalid worker-output witness']}
    with pytest.raises(InputError):
        _validate_report(report, 'lerobot', [{'episode_index': 0, 'frame_index': 0}])


@pytest.mark.parametrize('overrides', [
    {'status': 'error'},
    {'input': None, 'output': None, 'steps': None},
])
def test_complete_native_report_cannot_hide_failed_or_missing_trace(overrides):
    routes = [{'name': name, 'status': 'complete', 'input': {'state': [1]},
               'output': {'state': [1]}, 'steps': [{'class': 'generated.NativeStep'}]}
              for name in ('saved', 'current_factory')]
    routes[0].update(overrides)
    report = {'provider': 'lerobot', 'execution_status': 'complete', 'samples': [
        {'identity': {'episode_index': 0, 'frame_index': 0, 'index': 0}, 'routes': routes}],
        'native_versions': {}, 'source_files': [], 'exports': [], 'artifact_receipts': [],
        'unverified': ['Generated invalid worker-output witness']}
    with pytest.raises(InputError):
        _validate_report(report, 'lerobot', [{'episode_index': 0, 'frame_index': 0}])


def _real_rows(tmp_path, *, episodes=(0, 0), indexes=(40, 90), frames=None):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    payload = {'episode_index': list(episodes), 'index': list(indexes),
               'timestamp': [0.1, 0.2], 'observation.state': [[1.0], [2.0]]}
    if frames is not None:
        payload['frame_index'] = list(frames)
    path = tmp_path / 'original-schema.parquet'
    pq.write_table(pa.table(payload), path)
    return {'path': str(path), 'relative_path': path.name, 'columns': ['observation.state']}


def test_explicit_physical_row_preserves_native_index_and_missing_frame(tmp_path):
    request = _real_rows(tmp_path)
    request['samples'] = [{'episode_index': 0, 'physical_row': 1}]
    selected = run(request)['raw_samples'][0]
    assert selected['identity'] == {'episode_index': 0, 'index': 90, 'timestamp': 0.2,
                                    'physical_row': 1, 'frame_index': None}
    assert selected['selection_basis'] == 'single_episode_physical_row'
    assert selected['columns']['observation.state'] == [2.0]


@pytest.mark.parametrize('variant', ['no_optin', 'mixed_episode', 'wrong_episode',
                                     'duplicate_index', 'out_of_range', 'native_frame_available'])
def test_frame_less_row_selection_refuses_unsupported_identity(tmp_path, variant):
    request = _real_rows(tmp_path,
        episodes=(0, 1) if variant == 'mixed_episode' else (0, 0),
        indexes=(40, 40) if variant == 'duplicate_index' else (40, 90),
        frames=(0, 1) if variant == 'native_frame_available' else None)
    sample = {'episode_index': 1 if variant == 'wrong_episode' else 0,
              'physical_row': 2 if variant == 'out_of_range' else 1}
    if variant == 'no_optin':
        sample = {'episode_index': 0, 'frame_index': 1}
    request['samples'] = [sample]
    with pytest.raises(ValueError):
        run(request)
