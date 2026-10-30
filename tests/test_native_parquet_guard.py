"""Actual generated Parquet controls; no native robotics runtime is invoked."""

import builtins
import json
import subprocess
import sys

import pytest

from policy_doctor import native_parquet_guard as guard

pa = pytest.importorskip('pyarrow')
pq = pytest.importorskip('pyarrow.parquet')


def write(path, values=None, **kwargs):
    table = pa.table(values or {'episode_index': [0, 1], 'length': [4, 7],
                                'tasks': [['pick red'], ['move blue']]})
    pq.write_table(table, path, **kwargs)
    return path


def test_normal_numeric_and_task_strings_are_guarded_without_value_read(tmp_path, monkeypatch):
    path = write(tmp_path / 'tasks.parquet')
    original = pq.ParquetFile
    options = []

    class MetadataOnly:
        def __init__(self, *args, **kwargs):
            options.append(kwargs)
            self.parquet = original(*args, **kwargs)

        def __getattr__(self, key):
            if key in {'read', 'read_row_group', 'iter_batches'}:
                raise AssertionError('Preflight must not read column values')
            return getattr(self.parquet, key)

    monkeypatch.setattr(pq, 'ParquetFile', MetadataOnly)
    result = guard.run({'operation': 'guard_parquet', 'paths': [str(path)]})
    assert result['execution_status'] == 'complete'
    assert result['files'][0]['rows'] == 2
    assert 0 < result['decoded_bound_bytes'] < guard.MAX_DECODED
    assert options == [{'thrift_string_size_limit': guard.MAX_FOOTER, 'thrift_container_size_limit': 100_000}]


def test_repeated_dictionary_strings_use_expansion_bound_not_stats_max(tmp_path):
    # A small dictionary page can encode many repeated long strings.
    path = write(tmp_path / 'dictionary.parquet', {'task': ['x' * 16_384] * 2500}, use_dictionary=True)
    assert path.stat().st_size < 100_000
    with pytest.raises(ValueError, match='decoded allocation'):
        guard.guard_file(path)


def test_numeric_rle_values_are_not_bounded_by_small_encoded_pages(tmp_path):
    vectors = pa.array([[0] * 16] * 40_000, type=pa.list_(pa.int64(), 16))
    path = write(tmp_path / 'rle.parquet', {'state': vectors}, use_dictionary=True)
    assert path.stat().st_size < 100_000
    with pytest.raises(ValueError, match='decoded allocation'):
        guard.guard_file(path)


def test_aggregate_limit_is_not_reset_per_file(tmp_path):
    paths = [write(tmp_path / f'{i}.parquet', {'state': pa.array([[0] * 4] * 35_000,
                                                              type=pa.list_(pa.int64(), 4))}) for i in range(2)]
    assert all(guard.guard_file(path)['decoded_bound_bytes'] < guard.MAX_DECODED for path in paths)
    with pytest.raises(ValueError, match='Aggregate'):
        guard.run({'operation': 'guard_parquet', 'paths': [str(path) for path in paths]})


@pytest.mark.parametrize('trailer', [(guard.MAX_FOOTER + 1).to_bytes(4, 'little') + b'PAR1',
                                    (1).to_bytes(4, 'little') + b'PARE',
                                    (0).to_bytes(4, 'little') + b'PAR1'])
def test_bad_footer_is_refused_before_arrow_import(tmp_path, monkeypatch, trailer):
    path = tmp_path / 'bad.parquet'
    path.write_bytes(b'PAR1' + trailer)
    original = builtins.__import__

    def no_arrow(name, *args, **kwargs):
        if name.startswith('pyarrow'):
            raise AssertionError('Invalid physical footer reached Arrow')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', no_arrow)
    with pytest.raises(ValueError, match='footer'):
        guard.guard_file(path)


def test_sparse_oversized_file_is_refused_without_content_read(tmp_path):
    path = tmp_path / 'large.parquet'
    with path.open('wb') as stream:
        stream.truncate(guard.MAX_FILE + 1)
    with pytest.raises(ValueError, match='byte bounds'):
        guard.guard_file(path)


@pytest.mark.parametrize('dtype,values', [(pa.binary(), [b'a']), (pa.timestamp('ns'), [0]),
                                         (pa.map_(pa.string(), pa.int64()), [[('a', 1)]])])
def test_unsupported_schema_is_refused(tmp_path, dtype, values):
    path = write(tmp_path / 'unsupported.parquet', {'bad': pa.array(values, type=dtype)})
    with pytest.raises(ValueError, match='Unsupported Parquet schema'):
        guard.guard_file(path)


def test_deep_nested_lists_are_refused(tmp_path):
    dtype, value = pa.int64(), 1
    for _ in range(7):
        dtype, value = pa.list_(dtype), [value]
    path = write(tmp_path / 'deep.parquet', {'bad': pa.array([value], type=dtype)})
    with pytest.raises(ValueError, match='nesting'):
        guard.guard_file(path)


def test_fixed_width_vector_limit(tmp_path):
    path = write(tmp_path / 'wide.parquet', {'bad': pa.array([[1] * 4097], type=pa.list_(pa.int64(), 4097))})
    with pytest.raises(ValueError, match='vector width'):
        guard.guard_file(path)


def test_row_group_and_row_limits(tmp_path):
    path = write(tmp_path / 'groups.parquet', {'x': list(range(129))}, row_group_size=1)
    with pytest.raises(ValueError, match='dimensions'):
        guard.guard_file(path)
    path = write(tmp_path / 'rows.parquet', {'x': [1] * 100_001})
    with pytest.raises(ValueError, match='dimensions'):
        guard.guard_file(path)


@pytest.mark.parametrize('ancestor', [False, True])
def test_symlink_leaf_or_ancestor_is_refused(tmp_path, ancestor):
    real = tmp_path / 'real'
    real.mkdir()
    path = write(real / 'x.parquet')
    link = tmp_path / 'link'
    link.symlink_to(real if ancestor else path, target_is_directory=ancestor)
    with pytest.raises(ValueError, match='symlinks'):
        guard.guard_file(link / 'x.parquet' if ancestor else link)


def test_change_during_footer_parse_refuses(tmp_path, monkeypatch):
    path = write(tmp_path / 'change.parquet')
    original = pq.ParquetFile

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        with path.open('ab') as stream:
            stream.write(b'changed')
        return result

    monkeypatch.setattr(pq, 'ParquetFile', mutate)
    with pytest.raises(ValueError, match='changed'):
        guard.guard_file(path)


@pytest.mark.parametrize('payload', [{}, {'operation': 'arbitrary', 'paths': []},
                                    {'operation': 'guard_parquet', 'paths': [True]},
                                    {'operation': 'guard_parquet', 'paths': ['x', 'x']},
                                    {'operation': 'guard_parquet', 'paths': ['x'], 'module': 'artifact'}])
def test_fixed_request_schema(payload):
    with pytest.raises(ValueError):
        guard.run(payload)


def test_isolated_script_protocol(tmp_path):
    path = write(tmp_path / 'x.parquet')
    output = tmp_path / 'output'
    result = subprocess.run([sys.executable, '-I', '-B', guard.__file__],
                            input=json.dumps({'operation': 'guard_parquet', 'paths': [str(path)],
                                              'output': str(output)}), text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'execution_status': 'complete'}
    assert json.loads((output / 'result.json').read_text())['files'][0]['rows'] == 2
