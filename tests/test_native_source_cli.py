import json

from policy_doctor import cli


def test_native_source_probe_cli_passes_explicit_authority(tmp_path, monkeypatch, capsys):
    from policy_doctor import native_processing

    config = tmp_path / 'runtime.json'
    config.write_text('{}')
    sources = tmp_path / 'sources.json'
    sources.write_text(json.dumps({'sources': [{
        'repo_id': 'example/base', 'revision': 'a' * 40, 'directory': 'base'}]}))
    observed = []
    monkeypatch.setattr(native_processing, 'load_native_runtimes', lambda _: {'configured': True})

    def probe(*args, **kwargs):
        observed.append((args, kwargs))
        return {'kind': 'native_processor_source_probe', 'status': 'error'}

    monkeypatch.setattr(native_processing, 'probe_native_processing', probe)
    assert cli.main(['probe-native', 'lerobot', str(tmp_path / 'selected'), str(tmp_path / 'data'),
        '--sample', '1678:0', '--native-runtime', str(config), '--mode', 'processor_source',
        '--processor-source', 'serialized_pretrained_path', '--intent', 'restore_selected_checkpoint',
        '--approved-sources', str(sources), '--json']) == 1
    reply = json.loads(capsys.readouterr().out)
    assert reply['kind'] == 'native_processor_source_probe'
    assert observed[0][1]['mode'] == 'processor_source'
    assert observed[0][1]['processor_source'] == 'serialized_pretrained_path'
    assert observed[0][1]['intent'] == 'restore_selected_checkpoint'
    assert observed[0][1]['approved_sources'][0]['directory'] == str(tmp_path / 'base')


def test_native_source_case_cli_uses_existing_case_engine(tmp_path, monkeypatch, capsys):
    from policy_doctor import incidents

    observed = []

    def create(*args, **kwargs):
        observed.append((args, kwargs))
        return {'status': 'pass', 'scope': 'Evidence-only reproduction.'}

    monkeypatch.setattr(incidents, 'export_incident_case', create)
    assert cli.main(['create-case', 'native-processor-source', 'before', 'case',
                     '--corrected', 'after', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'pass'
    assert observed[0][0] == ('native_processor_source', 'before', 'case', {})
    assert observed[0][1]['corrected'] == 'after'
