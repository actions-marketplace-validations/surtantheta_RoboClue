"""Declared checkpoint intent selects its route without increasing read authority."""
import copy
import json

import pytest
from test_language_vocab import fixture

from policy_doctor import hub, language_vocab, product_cli


@pytest.fixture
def remote_control(tmp_path, monkeypatch):
    checkpoint, catalog = fixture(tmp_path)
    instructions = tmp_path / 'instructions.json'
    instructions.write_text(json.dumps(['CloseFridge']))
    calls = []
    original = language_vocab.check_language_vocabulary
    base = {'kind': 'checkpoint_check', 'status': 'error', 'counts': {'error': 1},
            'findings': [{'code': 'independent.error', 'status': 'error', 'message': 'Independent checkpoint error'}]}

    def audit(source, **kwargs):
        calls.append(('audit', source, kwargs))
        return copy.deepcopy(base)

    def vocabulary(source, **kwargs):
        calls.append(('language', source, kwargs))
        return original(checkpoint, instruction=kwargs.get('instruction'), launch_command=kwargs.get('launch_command'),
                        instruction_file=kwargs.get('instruction_file'), catalog_directory=str(catalog))

    monkeypatch.setattr(hub, 'check_checkpoint', audit)
    monkeypatch.setattr(language_vocab, 'check_language_vocabulary', vocabulary)
    monkeypatch.setattr(hub, '_request', lambda *a, **k: pytest.fail('Routing controls must not make real network requests'))
    return {'calls': calls, 'catalog': catalog, 'instructions': instructions}


def selectors(control, name):
    return {'instruction': ['--instruction', 'CloseFridge'], 'task': ['--task', 'CloseFridge'],
            'command': ['--launch-command', 'python evaluate.py --task CloseFridge'],
            'file': ['--instruction-file', str(control['instructions'])],
            'catalog': ['--language-catalog', str(control['catalog'])]}[name]


@pytest.mark.parametrize('selector', ['instruction', 'task', 'command', 'file', 'catalog'])
@pytest.mark.parametrize('explicit_kind', [False, True])
def test_hub_checkpoint_selectors_reach_same_audit_and_preserve_independent_errors(remote_control, capsys, selector, explicit_kind):
    args = ['check', 'hf://publisher/checkpoint', '--fetch', '--revision', 'a' * 40,
            *selectors(remote_control, selector), '--json', '--fail-on', 'error']
    if explicit_kind:
        args.extend(['--kind', 'checkpoint'])
    assert product_cli.main(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'error' and report['findings'][0]['code'] == 'independent.error'
    assert report['counts']['error'] == 1 and report['language_vocabulary']['status'] in ('unknown', 'warning')
    assert [row[0] for row in remote_control['calls']] == ['audit', 'language']
    assert all(row[1] == 'hf://publisher/checkpoint' for row in remote_control['calls'])
    assert remote_control['calls'][0][2] == {'fetch': True, 'revision': 'a' * 40}


@pytest.mark.parametrize('selector', ['instruction', 'task', 'command', 'file', 'catalog'])
def test_checkpoint_intent_does_not_enable_implicit_fetch(remote_control, capsys, selector):
    assert product_cli.main(['check', 'hf://publisher/checkpoint', *selectors(remote_control, selector)]) == 2
    assert 'explicit download permission' in capsys.readouterr().err
    assert remote_control['calls'] == []


@pytest.mark.parametrize('selector', ['instruction', 'task', 'command', 'file', 'catalog'])
def test_explicit_dataset_role_is_never_overridden(remote_control, capsys, selector):
    assert product_cli.main(['check', 'hf://publisher/dataset', '--kind', 'dataset', '--fetch',
                            *selectors(remote_control, selector)]) == 2
    assert 'checkpoint' in capsys.readouterr().err
    assert remote_control['calls'] == []


@pytest.mark.parametrize('extra', [['--question', 'dataset'], ['--question', 'recovery'], ['--units'],
                                 ['--groot-version', 'n1.7'], ['--no-video-headers'],
                                 ['--max-rows', '1'], ['--runtime-mode', 'training']])
def test_mixed_or_unsupported_remote_inputs_fail_before_metadata_reads(remote_control, capsys, extra):
    assert product_cli.main(['check', 'hf://publisher/checkpoint', '--fetch', '--instruction', 'CloseFridge', *extra]) == 2
    assert capsys.readouterr().err.startswith('Input:')
    assert remote_control['calls'] == []


def test_bare_hub_source_keeps_dataset_default(remote_control, monkeypatch):
    from policy_doctor import health_cards
    called = []
    expected = {'kind': 'dataset_health_card', 'status': 'unknown'}
    monkeypatch.setattr(health_cards, 'health_card', lambda source, **kwargs: called.append((source, kwargs)) or expected)
    args = product_cli._parser().parse_args(['check', 'hf://publisher/dataset', '--fetch'])
    report, kind = product_cli._assess(args)
    assert kind == 'dataset' and report == expected
    assert called[0][0] == 'hf://publisher/dataset' and called[0][1]['fetch'] is True
    assert remote_control['calls'] == []
