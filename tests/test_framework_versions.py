"""Generated dist-info controls; not installation/native framework qualification."""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from policy_doctor import framework_versions as versions
from policy_doctor.artifacts import InputError

LEROBOT_PIN = '146d8474932c8b566717eb2738ba459337ab71c4'


def distribution(root, name='lerobot', version='0.6.2', direct=None):
    path = root / f'{name}-{version}.dist-info'
    path.mkdir()
    (path / 'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    if direct is not None:
        (path / 'direct_url.json').write_text(json.dumps(direct))
    return metadata.PathDistribution(path)


def vcs(commit=LEROBOT_PIN):
    return {'url': 'https://github.com/huggingface/lerobot.git',
            'vcs_info': {'vcs': 'git', 'commit_id': commit}}


def fake_discovery(monkeypatch, mapping):
    monkeypatch.setattr(versions.metadata, 'distributions', lambda *, name: iter(mapping.get(name, [])))


def row(result, framework='lerobot'):
    return next(r for r in result['frameworks'] if r['id'] == framework)


def test_primary_distribution_names_and_catalog_pins_are_explicit():
    profiles = versions.supported_framework_profiles()
    assert {p['id']: p['distribution_names'] for p in profiles['frameworks']} == {
        'lerobot': ['lerobot'], 'isaaclab': ['isaaclab'], 'groot': ['gr00t'], 'openpi': ['openpi']}
    for profile in profiles['frameworks']:
        assert profile['tested_commits'] and profile['repository'].count('/') == 1
        assert all(len(pin) == 40 for pin in profile['tested_commits'])
        assert profile['rule_ids']
    assert LEROBOT_PIN in profiles['frameworks'][0]['tested_commits']


def test_absent_frameworks_are_context_not_compatibility_failure(monkeypatch):
    fake_discovery(monkeypatch, {})
    result = versions.collect_framework_versions()
    assert all(item['source_identity'] == 'not_installed' for item in result['frameworks'])
    assert result['runtime_revalidated'] is False
    assert 'child recorder/trainer environment may differ' in result['environment']
    assert 'status' not in result


@pytest.mark.parametrize('commit, expected', [(LEROBOT_PIN, 'declared_tested_commit'), ('a' * 40, 'not_revalidated')])
def test_full_declared_commit_match_is_not_runtime_attestation(tmp_path, monkeypatch, commit, expected):
    dist = distribution(tmp_path, direct=vcs(commit))
    fake_discovery(monkeypatch, {'lerobot': [dist]})
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dist._path.iterdir()}
    result = versions.collect_framework_versions()
    observed = row(result)
    assert observed['version'] == '0.6.2' and observed['declared_commit'] == commit
    assert observed['source_identity'] == expected
    assert result['runtime_revalidated'] is False
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dist._path.iterdir()} == before
    compact = versions.compact_framework_context(result, 'lerobot')
    assert compact['source_identity'] == expected and compact['runtime_revalidated'] is False
    assert 'consumer environment unverified' in compact['environment']
    assert 'compatible' not in compact and 'pass' not in compact.values()


@pytest.mark.parametrize('direct', [None, {'url': 'file:///editable', 'dir_info': {'editable': True}}])
def test_version_or_editable_path_does_not_establish_qualified_source(tmp_path, monkeypatch, direct):
    dist = distribution(tmp_path, direct=direct)
    fake_discovery(monkeypatch, {'lerobot': [dist]})
    observed = row(versions.collect_framework_versions())
    assert observed['source_identity'] == 'not_revalidated'
    assert observed['version'] == '0.6.2' and observed['declared_commit'] is None


@pytest.mark.parametrize('mutation', ['bad_json', 'duplicate_key', 'short_commit', 'non_git', 'huge_url', 'directory', 'symlink', 'metadata_missing', 'repeated_version', 'wrong_name', 'nonascii_version', 'deep_json', 'missing_url', 'mixed_source_identity'])
def test_malformed_ambiguous_and_unsupported_metadata_is_unknown(tmp_path, monkeypatch, mutation):
    dist = distribution(tmp_path, direct=vcs())
    direct = dist._path / 'direct_url.json'
    if mutation == 'bad_json':
        direct.write_text('{')
    elif mutation == 'duplicate_key':
        direct.write_text('{"vcs_info":{},"vcs_info":' + json.dumps(vcs()['vcs_info']) + '}')
    elif mutation == 'short_commit':
        direct.write_text(json.dumps(vcs(LEROBOT_PIN[:7])))
    elif mutation == 'non_git':
        data = vcs()
        data['vcs_info']['vcs'] = 'other'
        direct.write_text(json.dumps(data))
    elif mutation == 'huge_url':
        direct.write_text('x' * (versions.MAX_DIRECT_URL_BYTES + 1))
    elif mutation == 'directory':
        direct.unlink()
        direct.mkdir()
    elif mutation == 'symlink':
        outside = tmp_path / 'outside.json'
        outside.write_text(json.dumps(vcs()))
        direct.unlink()
        direct.symlink_to(outside)
    elif mutation == 'metadata_missing':
        (dist._path / 'METADATA').unlink()
    elif mutation == 'repeated_version':
        with (dist._path / 'METADATA').open('a') as f:
            f.write('Version: 0.6.2\n')
    elif mutation == 'wrong_name':
        (dist._path / 'METADATA').write_text('Name: some_other_package\nVersion: 0.6.2\n')
    elif mutation == 'nonascii_version':
        (dist._path / 'METADATA').write_bytes(b'Name: lerobot\nVersion: \xff\n')
    elif mutation == 'deep_json':
        direct.write_text('[' * 3000 + '0' + ']' * 3000)
    elif mutation == 'missing_url':
        direct.write_text(json.dumps({'vcs_info': vcs()['vcs_info']}))
    elif mutation == 'mixed_source_identity':
        direct.write_text(json.dumps(dict(vcs(), dir_info={'editable': True})))
    fake_discovery(monkeypatch, {'lerobot': [dist]})
    observed = row(versions.collect_framework_versions())
    assert observed['source_identity'] == 'unknown'
    assert 'unreadable' in observed['reason']


def test_multiple_distributions_never_choose_a_convenient_version(tmp_path, monkeypatch):
    first = distribution(tmp_path, direct=vcs())
    second = distribution(tmp_path, version='999.0', direct=vcs('f' * 40))
    fake_discovery(monkeypatch, {'lerobot': [first, second]})
    observed = row(versions.collect_framework_versions())
    assert observed['source_identity'] == 'unknown' and observed['declared_commit'] is None
    assert 'Multiple' in observed['reason']


def test_framework_code_is_not_imported_for_metadata_observation(tmp_path, monkeypatch):
    package = tmp_path / 'lerobot'
    package.mkdir()
    (package / '__init__.py').write_text('raise RuntimeError("framework code must not run")')
    monkeypatch.syspath_prepend(str(tmp_path))
    dist = distribution(tmp_path, direct=vcs())
    fake_discovery(monkeypatch, {'lerobot': [dist]})
    was_loaded = sys.modules.get('lerobot')
    assert row(versions.collect_framework_versions())['source_identity'] == 'declared_tested_commit'
    assert sys.modules.get('lerobot') is was_loaded


@pytest.mark.skipif(not hasattr(os, 'mkfifo'), reason='POSIX named-pipe control')
def test_special_metadata_file_does_not_hang_a_console_process(tmp_path):
    dist = distribution(tmp_path)
    os.mkfifo(dist._path / 'direct_url.json')
    code = '''import importlib.metadata as m,json,sys
from pathlib import Path
from policy_doctor.framework_versions import _inspect_distribution
print(json.dumps(_inspect_distribution(m.PathDistribution(Path(sys.argv[1])),['lerobot'])))
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parents[1] / 'src')
    result = subprocess.run([sys.executable, '-B', '-c', code, str(dist._path)], env=env,
                            capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(result.stdout)['source_identity'] == 'unknown'


def test_unsupported_metadata_provider_and_discovery_errors_remain_unknown(monkeypatch):
    fake_discovery(monkeypatch, {'lerobot': [object()]})
    assert row(versions.collect_framework_versions())['source_identity'] == 'unknown'
    def fail(*, name):
        raise OSError('generated metadata discovery error')
    monkeypatch.setattr(versions.metadata, 'distributions', fail)
    assert all(r['source_identity'] == 'unknown' for r in versions.collect_framework_versions()['frameworks'])


def test_compact_context_refuses_missing_or_ambiguous_frameworks(monkeypatch):
    fake_discovery(monkeypatch, {})
    report = versions.collect_framework_versions()
    with pytest.raises(InputError):
        versions.compact_framework_context(report, 'not-a-framework')
    with pytest.raises(InputError):
        versions.compact_framework_context({}, 'lerobot')
    report['frameworks'].append(row(report))
    with pytest.raises(InputError):
        versions.compact_framework_context(report, 'lerobot')
    text = versions.render_framework_versions()
    assert 'not_installed' in text and 'never establishes support' in text


def test_compact_version_omission_is_explicit_and_keeps_full_observation(monkeypatch):
    fake_discovery(monkeypatch, {})
    report = versions.collect_framework_versions()
    unusual = '\U0001f916' * 200
    row(report)['version'] = unusual
    row(report)['source_identity'] = 'not_revalidated'
    compact = versions.compact_framework_context(report, 'lerobot')
    assert compact['version'] is None and compact['version_omitted'] is True
    assert compact['version_sha256'] == hashlib.sha256(unusual.encode()).hexdigest()
    assert compact['details'] == 'roboclue advanced versions --json'
    assert compact['source_identity'] == 'not_revalidated' and compact['runtime_revalidated'] is False
    assert len(json.dumps(compact, indent=2, ensure_ascii=True).encode()) < 512
    assert row(report)['version'] == unusual


def test_actual_cli_versions_separates_context_from_qualification(capsys, monkeypatch, tmp_path):
    from policy_doctor.product_cli import main

    dist = distribution(tmp_path)
    fake_discovery(monkeypatch, {'lerobot': [dist]})
    assert main(['advanced', 'versions', '--json']) == 0
    observed = json.loads(capsys.readouterr().out)
    assert row(observed)['version'] == '0.6.2'
    assert row(observed)['source_identity'] == 'not_revalidated'
    assert observed['runtime_revalidated'] is False
    assert main(['advanced', 'versions']) == 0
    text = capsys.readouterr().out
    assert 'lerobot: version=0.6.2; source identity=not_revalidated' in text
    assert 'child recorder/trainer environment may differ' in text


def test_fresh_console_uses_generated_distribution_metadata_without_framework_import(tmp_path):
    distribution(tmp_path)
    package = tmp_path / 'lerobot'
    package.mkdir()
    marker = tmp_path / 'framework-was-imported'
    (package / '__init__.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("bad")\nraise RuntimeError("framework code ran")\n')
    code = '''import importlib.metadata as m,sys
from pathlib import Path
fixture=Path(sys.argv[1]);sys.path.insert(0,str(fixture))
original=m.distributions
m.distributions=lambda *,name: original(name=name,path=[str(fixture)])
from policy_doctor.product_cli import main
result=main(['advanced','versions','--json'])
assert 'lerobot' not in sys.modules
raise SystemExit(result)
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).parents[1] / 'src')
    result = subprocess.run([sys.executable, '-B', '-c', code, str(tmp_path)], env=env, capture_output=True,
                            text=True, timeout=20, check=True, cwd=tmp_path)
    observed = json.loads(result.stdout)
    assert row(observed)['version'] == '0.6.2' and row(observed)['source_identity'] == 'not_revalidated'
    assert not marker.exists()
