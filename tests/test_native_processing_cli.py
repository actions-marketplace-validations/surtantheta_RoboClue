"""CLI boundaries for opt-in native probes and replay; no native imports needed."""

import copy
import json

import pytest

from policy_doctor import cli, native_processing
from policy_doctor.artifacts import InputError


@pytest.fixture
def native_calls(monkeypatch):
    calls = []
    runtimes = {'groot_state': {'python': '/trusted/python', 'source_root': '/trusted/source'}}
    report = {'kind': 'native_processing_probe', 'status': 'observed', 'scope': 'State component only; physical behavior unknown.',
              'probe': {'provider': 'groot_state', 'execution_status': 'complete',
                        'samples': [{'identity': {'episode_index': 0, 'frame_index': 1},
                                     'routes': [{'name': 'native_state', 'status': 'complete',
                                                 'dimensions': {'native_declared_base': 29, 'observed_output': 58}}]}],
                        'unverified': ['Training meaning remains unknown.']}, 'retained': False}
    def load(path, *, require_initial_cache=True):
        assert type(require_initial_cache) is bool
        calls.append(('load', path))
        return runtimes
    def probe(provider, model, dataset, **kwargs):
        calls.append(('probe', provider, model, dataset, kwargs))
        return copy.deepcopy(report)
    def replay(directory, **kwargs):
        calls.append(('replay', directory, kwargs))
        return dict(copy.deepcopy(report), kind='native_processing_reproduction', status='pass', reproduction_matches=True)
    monkeypatch.setattr(native_processing, 'load_native_runtimes', load)
    monkeypatch.setattr(native_processing, 'probe_native_processing', probe)
    monkeypatch.setattr(native_processing, 'reproduce_native_processing', replay)
    return calls, runtimes


def test_probe_forwards_explicit_two_samples_runtime_and_capsule(native_calls, capsys):
    calls, runtimes = native_calls
    assert cli.main(['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json',
                     '--sample', '0:0', '--sample', '0:1', '--case', 'new-capsule', '--json']) == 0
    assert calls == [('load', 'trusted.json'), ('probe', 'groot_state', 'model', 'dataset', {
        'runtimes': runtimes, 'samples': [{'episode_index': 0, 'frame_index': 0}, {'episode_index': 0, 'frame_index': 1}],
        'destination': 'new-capsule'})]
    result = json.loads(capsys.readouterr().out)
    assert result['probe']['execution_status'] == 'complete'


def test_reproduce_forwards_configured_runtime(native_calls, capsys):
    calls, runtimes = native_calls
    assert cli.main(['reproduce-native-probe', 'capsule', '--native-runtime', 'trusted.json', '--json']) == 0
    assert calls == [('load', 'trusted.json'), ('replay', 'capsule', {'runtimes': runtimes})]
    assert json.loads(capsys.readouterr().out)['reproduction_matches'] is True


@pytest.mark.parametrize('command', [
    ['probe-native', 'groot_state', 'model', 'dataset', '--sample', '0:0'],
    ['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json'],
    ['reproduce-native-probe', 'capsule'],
])
def test_native_execution_requires_runtime_and_explicit_probe_sample(command, native_calls):
    with pytest.raises(SystemExit) as error:
        cli.main(command)
    assert error.value.code == 2
    assert native_calls[0] == []


@pytest.mark.parametrize('sample', ['0', '0:1:2', '-1:0', '0:-1', 'a:b', '0:1000000001', '1.0:2', '٠:١'])
def test_malformed_or_out_of_bounds_sample_refused_before_api(sample, native_calls):
    with pytest.raises(SystemExit) as error:
        cli.main(['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json', '--sample', sample])
    assert error.value.code == 2
    assert native_calls[0] == []


@pytest.mark.parametrize('samples', [['0:0', '0:0'], ['0:0', '0:1', '0:2']])
def test_sample_uniqueness_and_count_checked_before_runtime_loading(samples, native_calls, capsys):
    command = ['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json']
    for sample in samples:
        command += ['--sample', sample]
    assert cli.main(command) == 2
    assert native_calls[0] == []
    assert 'unique' in json.loads(capsys.readouterr().err)['error']['message']


def test_case_and_output_ambiguity_refused(native_calls, capsys):
    assert cli.main(['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json',
                     '--sample', '0:0', '--case', 'capsule', '--output', 'report.json']) == 2
    assert native_calls[0] == []
    assert 'Choose --case or --output' in capsys.readouterr().err


@pytest.mark.parametrize('output_format', [None, '--markdown'])
def test_human_and_markdown_explain_scope_identity_and_actual_route(native_calls, capsys, output_format):
    args = ['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json', '--sample', '0:1']
    if output_format:
        args.append(output_format)
    assert cli.main(args) == 0
    text = capsys.readouterr().out
    assert 'State component only' in text
    assert 'episode 0, frame 1' in text
    assert 'native_state: complete' in text.replace('\\_', '_')
    assert 'Training meaning remains unknown' in text
    if output_format:
        assert text.startswith('# Native processing observation')


def test_json_output_file_uses_existing_new_file_convention(native_calls, tmp_path, capsys):
    destination = tmp_path / 'report.json'
    args = ['reproduce-native-probe', 'capsule', '--native-runtime', 'trusted.json', '--json', '--output', str(destination)]
    assert cli.main(args) == 0
    assert json.loads(destination.read_text())['kind'] == 'native_processing_reproduction'
    assert not capsys.readouterr().out
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().err)['error']['code'] == 'invalid_input'


def test_native_input_error_retains_existing_cli_error_contract(native_calls, monkeypatch, capsys):
    def refused(path, **kwargs):
        raise InputError('Configured native provider unavailable')
    monkeypatch.setattr(native_processing, 'load_native_runtimes', refused)
    assert cli.main(['reproduce-native-probe', 'capsule', '--native-runtime', 'trusted.json']) == 2
    assert json.loads(capsys.readouterr().err) == {'error': {'code': 'invalid_input', 'message': 'Configured native provider unavailable'}}


def test_native_unknown_and_failed_replay_exit_codes(native_calls, monkeypatch, capsys):
    monkeypatch.setattr(native_processing, 'reproduce_native_processing', lambda *a, **kw: {
        'kind': 'native_processing_reproduction', 'status': 'unknown', 'scope': 'Only software behavior',
        'verifier_matches': False, 'candidate_evaluated': False})
    assert cli.main(['reproduce-native-probe', 'capsule', '--native-runtime', 'trusted.json']) == 3
    assert 'no candidate evaluated' in capsys.readouterr().out
    monkeypatch.setattr(native_processing, 'reproduce_native_processing', lambda *a, **kw: {
        'kind': 'native_processing_reproduction', 'status': 'error', 'scope': 'Only software behavior',
        'reproduction_matches': False})
    assert cli.main(['reproduce-native-probe', 'capsule', '--native-runtime', 'trusted.json', '--json']) == 1
    assert json.loads(capsys.readouterr().out)['reproduction_matches'] is False


def test_serve_preserves_old_call_signature_and_passes_runtime_only_when_supplied(monkeypatch):
    from policy_doctor import server

    calls = []
    monkeypatch.setattr(server, 'serve', lambda *args, **kwargs: calls.append((args, kwargs)))
    assert cli.main(['serve', '--root', 'read-root', '--case-root', 'case-root']) == 0
    assert calls == [((['read-root'],), {'case_root': 'case-root'})]
    assert cli.main(['serve', '--root', 'read-root', '--native-runtime', 'trusted.json']) == 0
    assert calls[1] == ((['read-root'],), {'case_root': None, 'native_runtime': 'trusted.json'})


def test_explicit_physical_row_is_forwarded_without_frame_alias(native_calls, capsys):
    calls, runtimes = native_calls
    assert cli.main(['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json',
                     '--row', '0:0', '--row', '0:1', '--json']) == 0
    assert calls[-1][-1]['samples'] == [{'episode_index': 0, 'physical_row': 0}, {'episode_index': 0, 'physical_row': 1}]
    assert calls[-1][-1]['runtimes'] == runtimes
    assert json.loads(capsys.readouterr().out)['status'] == 'observed'


def test_physical_row_and_native_frame_flags_cannot_mix(native_calls):
    with pytest.raises(SystemExit) as error:
        cli.main(['probe-native', 'groot_state', 'model', 'dataset', '--native-runtime', 'trusted.json',
                  '--row', '0:0', '--sample', '0:1'])
    assert error.value.code == 2 and native_calls[0] == []


def test_lerobot_does_not_accept_gr1_row_coordinate(native_calls, capsys):
    assert cli.main(['probe-native', 'lerobot', 'model', 'dataset', '--native-runtime', 'trusted.json', '--row', '0:0']) == 2
    assert not native_calls[0]
    assert 'original GR1' in capsys.readouterr().err


def test_human_displays_row_provenance_and_actual_route_changes():
    text = cli.human({'kind': 'native_processing_probe', 'status': 'observed', 'scope': 'Numerical software only',
        'probe': {'provider': 'groot_state', 'execution_status': 'complete', 'samples': [
            {'identity': {'episode_index': 0, 'frame_index': None, 'physical_row': 1},
             'selection_basis': 'single_episode_physical_row', 'routes': [{'name': 'native_state', 'status': 'complete'}]}]},
        'observed_differences': [{'episode_index': 0, 'frame_index': None, 'physical_row': 1,
            'reference_route': 'saved', 'candidate_route': 'current_factory', 'changed_outputs': [
                {'key': 'tokens', 'reference': {'shape': [1, 64], 'dtype': 'int64'},
                 'candidate': {'shape': [1, 200], 'dtype': 'int64'}},
                {'key': 'domain_id', 'reference': {'shape': [1], 'dtype': 'int64', 'preview': [11]},
                 'candidate': {'shape': [1], 'dtype': 'int64', 'preview': [4]}},
                {'key': 'pixels', 'reference': {'shape': [3, 8, 8], 'dtype': 'float32', 'min': 0, 'max': 1},
                 'candidate': {'shape': [3, 8, 8], 'dtype': 'float32', 'min': -1, 'max': 1}}]}]})
    assert 'episode 0, physical row 1' in text and 'frame None' not in text
    assert 'single_episode_physical_row' in text
    assert 'shape [1, 64]' in text and 'shape [1, 200]' in text
    assert 'values [11]' in text and 'values [4]' in text
    assert 'range [0, 1]' in text and 'range [-1, 1]' in text
