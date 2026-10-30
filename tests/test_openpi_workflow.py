"""Original recorded bodies through the existing command/tool; no live HTTP."""
import asyncio
import json
import os
import shlex
import shutil
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.openpi_workflow import check_readiness, concise_openpi_summary

FIXTURES = Path(__file__).parent / 'fixtures'
BASE = 'uv run scripts/serve_policy.py policy:checkpoint --policy.dir=gs://openpi-assets/checkpoints/pi0_base '
CURRENT = BASE + '--policy.config=pi05_droid'
REFERENCE = BASE + '--policy.config=pi0_droid'
GENERATION = '1748456810418896'
META = 'https://storage.googleapis.com/storage/v1/b/openpi-assets/o/checkpoints%2Fpi0_base%2Fassets%2Fdroid%2Fnorm_stats.json'
BODY = 'https://storage.googleapis.com/openpi-assets/checkpoints/pi0_base/assets/droid/norm_stats.json?generation=' + GENERATION


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'openpi'
    shutil.copytree(FIXTURES / 'openpi_source_profile_215abfb', root)
    for file in root.rglob('*.txt'):
        file.rename(file.with_suffix(''))
    return root


@pytest.fixture
def recorded(monkeypatch):
    from policy_doctor import openpi_intake

    calls = []
    class Response:
        status = 200
        def __init__(self, raw, url):
            self.raw, self.url, self.offset = raw, url, 0
            self.headers = {'Content-Length': str(len(raw)), 'x-goog-generation': GENERATION}
        def geturl(self):
            return self.url
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, count):
            result = self.raw[self.offset:self.offset + count]
            self.offset += len(result)
            return result
        read1 = read
    class Opener:
        def open(self, request, timeout):
            url = request.full_url
            assert url in [META, META + '?generation=' + GENERATION, BODY]
            calls.append(url)
            filename = 'norm_stats.json' if url == BODY else 'object_metadata.json'
            return Response((FIXTURES / 'openpi_original_20261024' / filename).read_bytes(), url)
    monkeypatch.setattr(openpi_intake, '_opener', Opener)
    return calls


def options(source):
    return {'robot_command': CURRENT, 'reference_command': REFERENCE, 'command_directory': str(source),
            'read_roots': [str(source)], 'statistics_generation': GENERATION,
            'reference_statistics_generation': GENERATION, 'restore_reference_intent': True,
            'openpi_fetch': True, 'max_bytes': 2 * 1024 ** 2, 'max_files': 8}


def test_actual_command_shared_original_snapshot_and_exact_correction(source, recorded):
    report = check_readiness(**options(source))
    assert report['status'] == 'error' and report['current']['normalization']['status'] == 'error'
    assert report['correction']['command'] == REFERENCE
    assert report['correction']['overall_after_correction'] == 'unknown'
    assert report['snapshot_reuse']['distinct_attempt_keys'] == 1 and len(recorded) == 2
    assert report['intake_budget']['logical_requests'] == 2
    assert report['intake_budget']['successful_response_body_bytes'] == 3459
    assert report['unknowns']['checkpoint_weights_and_model_architecture']
    assert report['command_context']['source_root'] == str(source)
    concise = concise_openpi_summary(report)
    assert len(json.dumps(concise).encode()) <= 8192
    assert concise['correction']['command'] == REFERENCE and concise['status'] == 'error'
    assert concise['correction']['replay']['normalization']['status'] == 'pass'
    replay_options = options(source)
    replay_options['robot_command'] = report['correction']['command']
    replay_options['reference_command'] = None
    replay_options['reference_statistics_generation'] = None
    replay_options['restore_reference_intent'] = False
    replay = check_readiness(**replay_options)
    assert replay['status'] == 'unknown' and replay['current']['normalization']['status'] == 'pass'
    assert replay['correction'] is None and len(recorded) == 4


def test_standalone_current_error_without_invented_reference(source, recorded):
    report = check_readiness(robot_command=CURRENT, command_directory=str(source), read_roots=[str(source)],
                             statistics_generation=GENERATION, openpi_fetch=True)
    assert report['status'] == 'error' and report['reference']['normalization']['status'] == 'unknown'
    assert report['correction'] is None and len(recorded) == 2
    assert 'intended config' in report['next_actions'][0]
    assert 'actual reference_command' in report['next_actions'][1]


def test_no_explicit_restoration_no_correction(source, recorded):
    args = options(source)
    args['restore_reference_intent'] = False
    report = check_readiness(**args)
    assert report['status'] == 'error' and report['correction'] is None
    assert len(report['consequential_changes']) == 3


def test_mcp_missing_cwd_not_inferred_from_root_or_source(source, recorded):
    args = options(source)
    args['command_directory'] = None
    args['source_root'] = str(source)
    report = check_readiness(**args)
    assert report['status'] == 'unknown' and not recorded
    assert 'command_directory' in report['next_actions'][0]


def test_unrelated_source_and_all_reference_local_paths_authorized_before_http(source, tmp_path, recorded):
    other = tmp_path / 'other'
    other.mkdir()
    args = options(source)
    args['source_root'] = str(other)
    args['read_roots'].append(str(other))
    with pytest.raises(InputError, match='source_root'):
        check_readiness(**args)
    args = options(source)
    args['reference_command'] = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=/etc'
    with pytest.raises(InputError, match='outside'):
        check_readiness(**args)
    assert not recorded


@pytest.mark.parametrize('extra', [{'robot_config': '/etc/robot.json'}, {'dataset': '/etc'}, {'checkpoint': 'hf://x/y'},
    {'policy_revision': 'abc'}, {'dataset_revision': 'abc'}, {'statistics_generation': True},
    {'reference_statistics_generation': 1}, {'restore_reference_intent': 1}])
def test_unsupported_combinations_strict_before_http(source, recorded, extra):
    with pytest.raises(InputError):
        check_readiness(**dict(options(source), **extra))
    assert not recorded


def test_parent_command_directory_before_probes_and_no_public_attempt(source, recorded):
    args = options(source)
    args['command_directory'] = str(source / 'link/..')
    with pytest.raises(InputError):
        check_readiness(**args)
    assert not recorded


def test_relative_selected_local_stats_use_known_command_directory(source, recorded):
    target = source / 'checkpoint/assets/droid'
    target.mkdir(parents=True)
    shutil.copyfile(FIXTURES / 'openpi_original_20261024/norm_stats.json', target / 'norm_stats.json')
    command = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=checkpoint'
    report = check_readiness(robot_command=command, command_directory=str(source), read_roots=[str(source)])
    assert report['statistics_intake']['current']['provenance']['source'] == str(target / 'norm_stats.json')
    assert report['status'] == 'error' and not recorded
    assert report['current']['statistics_binding']['status'] == 'pass'
    assert report['current']['statistics_binding']['identity_kind'] == 'local_content_snapshot'
    assert 'generation' not in report['current']['statistics_binding']
    assert report['current']['normalization']['code'] == 'openpi.normalization.missing_quantile_bounds'
    assert report['correction'] is None
    assert report['commands']['current'] == command
    assert str(source / 'checkpoint') in report['commands']['effective_current']



def test_local_declared_comparison_no_fetch_diagnoses_without_cloud_correction(source, recorded, capsys):
    from policy_doctor.cli import main

    target = source / 'fine-tune/assets/droid'
    target.mkdir(parents=True)
    shutil.copyfile(FIXTURES / 'openpi_original_20261024/norm_stats.json', target / 'norm_stats.json')
    current = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=fine-tune'
    reference = current.replace('pi05_droid', 'pi0_droid')
    report = check_readiness(robot_command=current, reference_command=reference, command_directory=str(source),
                             read_roots=[str(source)], restore_reference_intent=True)
    assert report['status'] == 'error' and report['reference']['normalization']['status'] == 'pass'
    assert report['current']['statistics_binding']['identity_kind'] == 'local_content_snapshot'
    assert report['current']['statistics_binding']['sha256'] == 'c6f9a4794b1df4901e618db3d3b5e02126dcc7c8d95b8f9735b5bc9ed75b95c9'
    assert report['correction'] is None and 'versioned public object' in report['correction_reason']
    assert report['snapshot_reuse']['distinct_attempt_keys'] == 1
    assert report['intake_budget']['logical_requests'] == 0 and not recorded
    concise = concise_openpi_summary(report)
    assert concise['current']['statistics_binding']['identity_kind'] == 'local_content_snapshot'
    assert 'path can change' in concise['current']['statistics_binding']['origin_scope']
    assert concise['status'] == 'error' and concise['correction'] is None
    assert main(['readiness', '--robot-command', current, '--reference-command', reference,
                 '--command-directory', str(source), '--restore-reference-intent', '--json']) == 1
    actual = json.loads(capsys.readouterr().out)
    assert actual['status'] == 'error' and actual['correction'] is None
    assert actual['intake_budget']['logical_requests'] == 0 and not recorded


def test_one_failed_snapshot_not_retried_for_reference(source, monkeypatch):
    from policy_doctor import openpi_intake

    calls = []
    class Failure:
        def open(self, request, timeout):
            calls.append(request.full_url)
            raise OSError('Generated unavailable response')
    monkeypatch.setattr(openpi_intake, '_opener', Failure)
    report = check_readiness(**options(source))
    assert report['status'] == 'unknown' and report['correction'] is None
    assert len(calls) == 1 and report['snapshot_reuse']['distinct_attempt_keys'] == 1
    assert report['intake_budget']['response_body_bytes_or_failed_reservations'] == 65537


def test_impossible_budget_unknown_without_http(source, recorded):
    report = check_readiness(**dict(options(source), max_bytes=1))
    assert report['status'] == 'unknown' and not recorded


def test_cli_human_json_and_unknown_exit_preserve_scope(source, recorded, capsys):
    from policy_doctor.cli import main

    args = ['readiness', '--robot-command', CURRENT, '--reference-command', REFERENCE,
            '--command-directory', str(source), '--statistics-generation', GENERATION,
            '--reference-statistics-generation', GENERATION, '--restore-reference-intent', '--fetch']
    assert main(args) == 1
    human = capsys.readouterr().out
    assert REFERENCE in human and 'complete serving remains unknown' in human
    assert 'model action' not in human.lower() or 'unknown' in human
    assert main(args + ['--json']) == 1
    body = json.loads(capsys.readouterr().out)
    assert body['status'] == 'error' and body['correction']['overall_after_correction'] == 'unknown'
    assert main(['readiness', '--robot-command', REFERENCE, '--command-directory', str(source), '--fetch', '--fail-on', 'unknown']) == 1
    capsys.readouterr()


BOOTSTRAP = r'''import json,os,sys
from pathlib import Path
from policy_doctor import openpi_intake as intake
base=Path(os.environ['ROBOCLUE_ORIGINAL']);ledger=Path(os.environ['ROBOCLUE_CALLS'])
GEN='1748456810418896'
META='https://storage.googleapis.com/storage/v1/b/openpi-assets/o/checkpoints%2Fpi0_base%2Fassets%2Fdroid%2Fnorm_stats.json'
BODY='https://storage.googleapis.com/openpi-assets/checkpoints/pi0_base/assets/droid/norm_stats.json?generation='+GEN
class Response:
 status=200
 def __init__(self,raw,url):self.raw=raw;self.url=url;self.offset=0;self.headers={'Content-Length':str(len(raw)),'x-goog-generation':GEN}
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def geturl(self):return self.url
 def read(self,n):value=self.raw[self.offset:self.offset+n];self.offset+=len(value);return value
 read1=read
class Original:
 def open(self,request,timeout):
  url=request.full_url
  assert url in [META,META+'?generation='+GEN,BODY], 'Unrecorded transport refused'
  with ledger.open('a') as f:f.write(json.dumps({'url':url})+'\n')
  return Response((base/('norm_stats.json' if url==BODY else 'object_metadata.json')).read_bytes(),url)
intake._opener=Original
from policy_doctor.cli import main
raise SystemExit(main(sys.argv[1:]))
'''


def test_actual_sdk_existing_tool_default_and_every_old_permission_zero_http(source, tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    local_stats = source / 'local-checkpoint/assets/droid'
    local_stats.mkdir(parents=True)
    shutil.copyfile(FIXTURES / 'openpi_original_20261024/norm_stats.json', local_stats / 'norm_stats.json')
    entry = tmp_path / 'operator_only.py'
    entry.write_text(BOOTSTRAP)
    ledger = tmp_path / 'requests.jsonl'
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / 'src'),
               ROBOCLUE_ORIGINAL=str(FIXTURES / 'openpi_original_20261024'), ROBOCLUE_CALLS=str(ledger))
    def count():
        return len(ledger.read_text().splitlines()) if ledger.exists() else 0
    async def run():
        for label, flags in [('default', []), ('hub', ['--allow-hub']), ('state', ['--allow-hub-processor-state']),
                             ('rows', ['--allow-hub-datasets']), ('openpi', ['--allow-openpi-statistics'])]:
            params = StdioServerParameters(command=sys.executable, args=[str(entry), 'serve', '--root', str(source), *flags], env=env)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write, read_timeout_seconds=timedelta(seconds=120)) as client:
                    await client.initialize()
                    tools = (await client.list_tools()).tools
                    tool = next(t for t in tools if t.name == 'check_robot_readiness')
                    assert tool.inputSchema['additionalProperties'] is False
                    request = {'robot_command': CURRENT, 'reference_command': REFERENCE,
                        'command_directory': str(source), 'statistics_generation': GENERATION,
                        'reference_statistics_generation': GENERATION, 'restore_reference_intent': True}
                    before = count()
                    if label == 'default':
                        local_request = {'robot_command': CURRENT.replace('gs://openpi-assets/checkpoints/pi0_base', 'local-checkpoint'),
                            'reference_command': REFERENCE.replace('gs://openpi-assets/checkpoints/pi0_base', 'local-checkpoint'),
                            'command_directory': str(source), 'restore_reference_intent': True}
                        for mode in ['concise', 'detailed']:
                            local = await client.call_tool(tool.name, dict(local_request, response_format=mode))
                            assert not local.isError and count() == before
                            assert local.structuredContent['status'] == 'error'
                            assert local.structuredContent['current']['normalization']['code'] == 'openpi.normalization.missing_quantile_bounds'
                            assert local.structuredContent['current']['statistics_binding']['identity_kind'] == 'local_content_snapshot'
                            assert local.structuredContent['correction'] is None
                    if label != 'openpi':
                        result = await client.call_tool(tool.name, request)
                        assert result.isError and count() == before
                        continue
                    for mode in ['concise', 'detailed']:
                        result = await client.call_tool(tool.name, dict(request, response_format=mode))
                        assert not result.isError
                        report = result.structuredContent
                        assert report['status'] == 'error' and report['current']['normalization']['status'] == 'error'
                        assert report['correction']['command'] == REFERENCE
                        assert report['correction']['overall_after_correction'] == 'unknown'
                    for extra in [{'restore_reference_intent': 1}, {'statistics_generation': True}, {'execute': True},
                                  {'reference_command': 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir=/etc'},
                                  {'robot_command': "uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir='~/checkpoint'"},
                                  {'reference_command': "uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid --policy.dir '~other/checkpoint'"}]:
                        before = count()
                        result = await client.call_tool(tool.name, dict(request, **extra))
                        assert result.isError and count() == before
                    before = count()
                    request.pop('command_directory')
                    request['source_root'] = str(source)
                    result = await client.call_tool(tool.name, request)
                    assert not result.isError and result.structuredContent['status'] == 'unknown' and count() == before
    asyncio.run(run())


def test_setup_separate_gcs_authority_and_strict_flag(source):
    from policy_doctor.setup_client import client_configuration

    out = client_configuration([str(source)], allow_openpi_statistics=True, clients=('json',))
    assert '--allow-openpi-statistics' in out and '--allow-hub"' not in out
    with pytest.raises(InputError):
        client_configuration([str(source)], allow_openpi_statistics=1)


@pytest.mark.parametrize('directory', ['checkpoint/../checkpoint', '$CHECKPOINT', '`pwd`', 'checkpoint\n'])
def test_relative_canonicalization_never_erases_unsupported_input(source, recorded, directory):
    command = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=' + shlex.quote(directory)
    if '..' in directory:
        with pytest.raises(InputError):
            check_readiness(robot_command=command, command_directory=str(source), read_roots=[str(source)])
    else:
        report = check_readiness(robot_command=command, command_directory=str(source), read_roots=[str(source)])
        assert report['status'] == 'unknown' and report['correction'] is None
    assert not recorded


@pytest.mark.parametrize('value', ['~/checkpoint', '~other/checkpoint', '~'])
@pytest.mark.parametrize('joined', [True, False])
def test_literal_tilde_checkpoint_refused_before_local_read_or_intake(source, recorded, monkeypatch, value, joined):
    from policy_doctor import openpi_workflow

    def no_read(*args, **kwargs):
        raise AssertionError('Tilde checkpoint reached a local path or statistics reader')
    monkeypatch.setattr(openpi_workflow, '_local', no_read)
    monkeypatch.setattr(openpi_workflow, 'resolve_openpi_statistics', no_read)
    argument = '--policy.dir=' + shlex.quote(value) if joined else '--policy.dir ' + shlex.quote(value)
    command = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid ' + argument
    with pytest.raises(InputError, match='intended absolute checkpoint path'):
        check_readiness(robot_command=command, command_directory=str(source), read_roots=[str(source)])
    assert not recorded


def test_absolute_checkpoint_with_interior_tilde_retains_literal_selection(source, recorded):
    target = source / '~/checkpoint/assets/droid'
    target.mkdir(parents=True)
    shutil.copyfile(FIXTURES / 'openpi_original_20261024/norm_stats.json', target / 'norm_stats.json')
    command = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_droid --policy.dir=' + shlex.quote(str(source / '~/checkpoint'))
    report = check_readiness(robot_command=command, command_directory=str(source), read_roots=[str(source)])
    assert report['status'] == 'error' and report['current']['statistics_binding']['source'] == str(target / 'norm_stats.json')
    assert report['current']['statistics_binding']['identity_kind'] == 'local_content_snapshot'
    assert report['correction'] is None and not recorded


@pytest.mark.parametrize('joined', [True, False])
def test_reference_tilde_refusal_precedes_any_current_public_intake(source, recorded, monkeypatch, joined):
    from policy_doctor import openpi_workflow

    def no_read(*args, **kwargs):
        raise AssertionError('Reference tilde reached local path or intake')
    monkeypatch.setattr(openpi_workflow, '_local', no_read)
    monkeypatch.setattr(openpi_workflow, 'resolve_openpi_statistics', no_read)
    args = options(source)
    argument = "--policy.dir='~/checkpoint'" if joined else "--policy.dir '~other/checkpoint'"
    args['reference_command'] = 'uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_droid ' + argument
    with pytest.raises(InputError, match='intended absolute checkpoint path'):
        check_readiness(**args)
    assert not recorded
