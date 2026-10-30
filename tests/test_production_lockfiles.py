"""Useful exact-version intake without interpreting an install environment."""
import hashlib
import json

import pytest
from test_provenance_register import control as control
from test_provenance_register import dump

from policy_doctor.register.engine import MAX_LOCK_BYTES, assess_register

# Explicitly export the shared fixture for pytest and supported Ruff versions.
__all__ = ['control']


@pytest.mark.parametrize('prefix', ['', '\ufeff', '--index-url https://user:SECRET@private.invalid/simple\n',
                                  '--extra-index-url=https://example.org/simple\n--no-index\n',
                                  '--find-links ./wheels\n--require-hashes\n'])
def test_bom_and_inert_resolution_options_keep_declared_exact_pin(control, prefix):
    control['lock'].write_text(prefix + 'synthetic==1.2.0\n')
    report = assess_register(control['path'])
    assert report['status'] == 'error'
    assert not report['version_errors'] and 'SECRET' not in json.dumps(report)
    witness = report['entries'][0]['applicability'][0]
    assert witness['observations'][0]['source_identity'] == 'not_attested'
    assert witness['observations'][0]['sha256'] == hashlib.sha256(control['lock'].read_bytes()).hexdigest()
    dump(control['config'], {'mode': 'corrected'})
    assert assess_register(control['path'])['status'] == 'pass'
    dump(control['config'], {})
    assert assess_register(control['path'])['status'] == 'unknown'


def test_pip_compile_hash_continuations_do_not_download_or_authenticate_artifacts(control):
    continuation = ' ' + chr(92) + '\n'
    content = ('synthetic==1.2.0' + continuation + '    --hash=sha256:' + 'a' * 64
               + continuation + '    --hash sha256:' + 'b' * 64 + '  # via a declared dependency\n')
    control['lock'].write_text(content)
    report = assess_register(control['path'])
    assert report['status'] == 'error' and not report['version_errors']
    assert report['entries'][0]['applicability'][0]['observations'][0]['source_identity'] == 'not_attested'


@pytest.mark.parametrize('name', ['uv.lock', 'poetry.lock'])
def test_wheel_rich_lock_is_bounded_separately_from_rule_json(control, name):
    lock = control['path'].parent / name
    wheel = '{url="https://example.org/wheel", hash="sha256:' + 'a' * 64 + '", size=1}'
    rows = []
    for index in range(180):
        rows.append('[[package]]\nname="unrelated-' + str(index) + '"\nversion="1.0"\nwheels=[' + ','.join([wheel] * 32) + ']\n')
    rows.append('[[package]]\nname="synthetic"\nversion="1.2.0"\nsource={registry="https://pypi.org/simple"}\n')
    lock.write_text(''.join(rows))
    assert 256 * 1024 < lock.stat().st_size < MAX_LOCK_BYTES
    control['manifest']['lockfiles'] = [name]
    dump(control['path'], control['manifest'])
    report = assess_register(control['path'])
    assert report['status'] == 'error' and not report['version_errors']
    assert report['entries'][0]['applicability'][0]['observations'][0]['version'] == '1.2.0'


@pytest.mark.parametrize('line', ['synthetic==1.2; python_version >= "3.11"',
                                 'synthetic @ https://example.org/patched.whl',
                                 'synthetic>=1.2', 'synthetic==1.*',
                                 'synthetic==1.2 --hash=sha256:invalid',
                                 '-r other-requirements.txt', '-e ./local-project'])
def test_constraints_markers_direct_sources_and_includes_never_become_exact_pins(control, line):
    control['lock'].write_text(line + '\n')
    assert assess_register(control['path'])['status'] == 'unknown'


@pytest.mark.parametrize('extra', ['source={editable="."}', 'source={git="https://example.org/repo"}',
                                  'source={url="https://example.org/patched.whl"}',
                                  'resolution-markers=["python_version >= 3.11"]'])
def test_selected_toml_local_or_conditional_package_stays_unknown(control, extra):
    lock = control['path'].parent / 'uv.lock'
    lock.write_text('[[package]]\nname="synthetic"\nversion="1.2"\n' + extra + '\n')
    control['manifest']['lockfiles'] = ['uv.lock']
    dump(control['path'], control['manifest'])
    assert assess_register(control['path'])['status'] == 'unknown'


def test_unrelated_local_lock_package_does_not_erase_selected_registry_pin(control):
    lock = control['path'].parent / 'uv.lock'
    lock.write_text('[[package]]\nname="unrelated"\nversion="1.2"\nsource={editable="."}\n'
                    '[[package]]\nname="synthetic"\nversion="1.2"\n')
    control['manifest']['lockfiles'] = ['uv.lock']
    dump(control['path'], control['manifest'])
    assert assess_register(control['path'])['status'] == 'error'


@pytest.mark.parametrize('content', ['synthetic==1.2\nsynthetic==2.0\n',
                                     'synthetic==1.2\n' * 64 + 'synthetic==2.0\n',
                                     'x' * (MAX_LOCK_BYTES + 1), 'synthetic==1.2 \\'])
def test_conflicts_overflow_and_unfinished_continuations_fail_closed(control, content):
    control['lock'].write_text(content)
    report = assess_register(control['path'])
    assert report['status'] == 'unknown'


@pytest.mark.parametrize('content', ['[[package]]\nname="synthetic"\nversion="1.2"\nx=nan\n',
                                     '[[package]]\nname="synthetic"\nversion="1.2"\nx=' + '[' * 30 + '1' + ']' * 30,
                                     '[[package]]\nname="synthetic"\nversion="1.2"\nx=[' + ','.join('0' for _ in range(100001)) + ']'])
def test_toml_nonfinite_depth_and_node_exhaustion_remain_structured_unknown(control, content):
    lock = control['path'].parent / 'uv.lock'
    lock.write_text(content)
    control['manifest']['lockfiles'] = ['uv.lock']
    dump(control['path'], control['manifest'])
    report = assess_register(control['path'])
    assert report['status'] == 'unknown' and report['version_errors']
