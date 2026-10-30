"""Prospective parent source-capture tests with an explicitly fake fixed worker.

Publication, auditor and immutable replay checks are real. Generated source/load
witnesses and fake invoke results are not native execution qualification.
"""

import copy
import hashlib
import importlib
import json
from pathlib import Path

import pytest
from test_native_processor_source import make_capture

from policy_doctor.artifacts import InputError


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


@pytest.fixture
def harness(tmp_path, monkeypatch):
    api = importlib.import_module('policy_doctor.native_source_processing')
    fixture = make_capture(tmp_path / 'generated-native-fixture')
    selected = fixture / 'originals/selected'
    alternate = fixture / 'originals/alternate/snapshot'
    dataset = tmp_path / 'dataset'
    dataset.mkdir()
    (dataset / 'unit.parquet').write_bytes(b'Generated unit data, not native Parquet')
    prefix = 'models--prior--source/snapshots/' + 'a' * 40 + '/'
    raw = json.loads((fixture / 'raw/result.json').read_text())
    raw['provider'] = 'lerobot_processor_source'
    raw['operation'] = 'probe'
    raw['routes'] = {'consumer': raw['routes']['suspect'], 'selected_reference': raw['routes']['selected']}
    for number, (name, route) in enumerate(raw['routes'].items(), 1):
        route['name'] = name
        route['call_id'] = f'native-call-{number:04d}'
    raw['identity'] = {'episode_index': 4, 'frame_index': 2, 'index': 99}
    for route in raw['routes'].values():
        for receipt in route['loaded_files']:
            if receipt['root'] == 'alternate':
                receipt['root'], receipt['path'] = 'hf_cache', prefix + receipt['path'].removeprefix('snapshot/')
    bindings = []
    for item in json.loads((fixture / 'capture.json').read_text())['bindings']:
        if item['root'] == 'attempt':
            continue
        item = copy.deepcopy(item)
        if item['root'] == 'alternate':
            item['root'], item['path'] = 'hf_cache', prefix + item['path'].removeprefix('snapshot/')
        item['retained_path'] = 'originals/' + item['root'] + '/' + item['path']
        bindings.append(item)
    # Alternate config is an approved original dependency even if no config load was witnessed.
    write(alternate / 'config.json', {'type': 'generated', 'pretrained_path': None})
    alt_config = alternate / 'config.json'
    bindings.append({'root': 'hf_cache', 'path': prefix + 'config.json',
                     'retained_path': 'originals/hf_cache/' + prefix + 'config.json',
                     'sha256': hashlib.sha256(alt_config.read_bytes()).hexdigest(), 'size_bytes': alt_config.stat().st_size})
    calls, state = [], {'verifier': {'unit_fixed_verifier': 1}, 'change_output': False, 'mutate': None}
    def receipts(root, label):
        return [{'root': label, 'path': p.relative_to(root).as_posix(), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                 'size_bytes': p.stat().st_size, 'role': 'generated_unit_fixture'} for p in root.rglob('*') if p.is_file()]
    selections = []
    def prep_model(model, workspace, *, model_family):
        assert model_family == 'smolvla'
        root = Path(model)
        recorded = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
        selection = {'unit_originals': recorded}
        selections.append(selection)
        return {'model': root, 'selection': selection, 'receipts': receipts(root, 'model'), 'omissions': ['Generated unit intake, not native proof']}
    def prep(provider, model, data, samples, workspace, runtime, invoke, *, model_family):
        assert provider == 'lerobot'
        result = prep_model(model, workspace, model_family=model_family)
        result['dataset'] = Path(data)
        result['receipts'] += receipts(Path(data), 'dataset')
        result['selection']['unit_originals'].update({p: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(data).rglob('*') if p.is_file()})
        return result
    def verify(selection):
        for path, sha in selection['unit_originals'].items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise InputError('Generated unit selected source changed')
    def invoke(script, runtime, request, workspace):
        assert script.name == 'native_processor_source_worker.py'
        calls.append(copy.deepcopy(request))
        output = Path(request['output'])
        output.mkdir()
        report = copy.deepcopy(raw)
        report['operation'] = request['operation']
        report['config_lineage'] = {'serialized_training_source': 'prior/source', 'parsed_pretrained_path': 'prior/source',
                                    'intent': request['intent'], 'processor_source': request['processor_source']}
        if request['processor_source'] == 'selected_checkpoint':
            report['routes']['consumer'] = copy.deepcopy(report['routes']['selected_reference'])
            report['routes']['consumer']['name'] = 'consumer'
        if state['change_output']:
            report['routes']['consumer']['output']['robot.proprio']['sha256'] = 'e' * 64
        worker_bindings = copy.deepcopy(bindings)
        if request['processor_source'] != 'serialized_pretrained_path':
            worker_bindings = [item for item in worker_bindings if item['root'] != 'hf_cache']
        for item in worker_bindings:
            source = selected / item['path'] if item['root'] == 'selected' else alternate / item['path'].removeprefix(prefix)
            target = output / item['retained_path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        for route in report['routes'].values():
            original = fixture / 'originals/attempt' / route['trace']['path']
            target = output / route['trace']['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(original.read_bytes())
        arrays = output / 'samples/sample_0.npz'
        arrays.parent.mkdir()
        arrays.write_bytes(b'Generated unit retained input, no array decode/native claim')
        metadata = output / 'samples/sample_0.json'
        write(metadata, {'identity': report['identity'], 'task': 'generated task', 'fields': [], 'source_bindings': ['unit.parquet']})
        def record(path):
            return {'path': path.relative_to(output).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'size_bytes': path.stat().st_size}
        sample = {'identity': report['identity'], 'input': report['input'], 'routes': report['routes'],
                  'arrays': record(arrays), 'metadata': record(metadata)}
        report['samples'], report['bindings'], report['exports'] = [sample], worker_bindings, []
        if state['mutate']:
            state['mutate'](report, worker_bindings, output)
        write(output / 'result.json', report)
        manifest = {'schema': 'native_processor_source.v1', 'provider': 'lerobot_processor_source', 'operation': request['operation'],
                    'result_path': 'result.json', 'samples': [{k: sample[k] for k in ('identity', 'arrays', 'metadata')}],
                    'exports': [], 'bindings': worker_bindings}
        payload_paths = {'result.json', arrays.relative_to(output).as_posix(), metadata.relative_to(output).as_posix()}
        payload_paths |= {r['trace']['path'] for r in report['routes'].values()}
        payload_paths |= {item['retained_path'] for item in worker_bindings}
        manifest['payload_files'] = [record(output / relative) for relative in sorted(payload_paths)]
        write(output / 'worker_manifest.json', manifest)
        descriptor = {'provider': 'lerobot_processor_source', 'operation': request['operation'], 'execution_status': report['execution_status'],
                      'result': record(output / 'result.json'), 'manifest': record(output / 'worker_manifest.json'), 'payload_files': manifest['payload_files']}
        write(output / 'worker_receipt.json', descriptor)
        # Ephemeral and unlisted files must never be published by the parent.
        (output / 'runtime/private-author-cache').mkdir(parents=True)
        (output / 'runtime/private-author-cache/lock').write_text('/private/author/path')
        (output / 'unlisted.py').write_text('raise RuntimeError("must not execute or copy")')
        return descriptor
    monkeypatch.setattr(api, 'prepare_native_inputs', prep)
    monkeypatch.setattr(api, 'prepare_native_model', prep_model)
    monkeypatch.setattr(api, 'verify_native_inputs', verify)
    monkeypatch.setattr(api, '_invoke', invoke)
    monkeypatch.setattr(api, '_fingerprint', lambda *a: copy.deepcopy(state['verifier']))
    return {'api': api, 'selected': selected, 'alternate': alternate, 'dataset': dataset, 'calls': calls,
            'state': state, 'runtime': {'lerobot': {'python': '/unit/python', 'source_root': '/unit/source'}},
            'approved': [{'repo_id': 'prior/source', 'revision': 'a' * 40, 'directory': str(alternate)}],
            'samples': [{'episode_index': 4, 'frame_index': 2}], 'destination': tmp_path / 'retained-attempt'}


def probe(harness, **kwargs):
    values = {'runtimes': harness['runtime'], 'samples': harness['samples'], 'destination': harness['destination'],
              'processor_source': 'serialized_pretrained_path', 'intent': 'restore_selected_checkpoint',
              'approved_sources': harness['approved']}
    values.update(kwargs)
    return harness['api'].probe_native_processor_source(harness['selected'], harness['dataset'], **values)


def test_one_attempt_publishes_two_self_contained_derived_views(harness):
    result = probe(harness)
    assert result['retained'] and len(harness['calls']) == 1
    assert result['candidate']['status'] == 'error' and result['corrected']['status'] == 'pass'
    for key in ('candidate_directory', 'corrected_directory'):
        root = Path(result[key])
        wrapper = json.loads((root / 'capture.json').read_text())
        assert wrapper['evidence_role'] == 'derived_route_view'
        assert wrapper['result']['path'] == 'raw/result.json'
        assert (root / 'native_verifier.json').is_file()
        assert not (root / 'runtime').exists() and not (root / 'unlisted.py').exists()
    first = json.loads((Path(result['candidate_directory']) / 'capture.json').read_text())
    second = json.loads((Path(result['corrected_directory']) / 'capture.json').read_text())
    assert first['derivation']['attempt_sha256'] == second['derivation']['attempt_sha256']
    assert second['consumer_route'] == second['reference_route'] == 'selected_reference'
    assert second['derivation']['consumer_call_id'] == second['derivation']['reference_call_id']


def test_no_destination_preserves_audit_but_reports_no_retained_paths(harness):
    result = probe(harness, destination=None)
    assert result['retained'] is False and 'candidate_directory' not in result
    assert result['candidate']['status'] == 'error'


def test_no_overwrite_and_no_destination_in_source_roots(harness):
    probe(harness)
    before = sorted(p.relative_to(harness['destination']).as_posix() for p in harness['destination'].rglob('*'))
    with pytest.raises(InputError):
        probe(harness)
    with pytest.raises(InputError):
        probe(harness, destination=harness['selected'] / 'inside')
    assert len(harness['calls']) == 1
    assert before == sorted(p.relative_to(harness['destination']).as_posix() for p in harness['destination'].rglob('*'))


@pytest.mark.parametrize('bad', [[], [{'episode_index': 4, 'physical_row': 2}], [{'episode_index': 4, 'frame_index': 2}] * 2])
def test_only_one_explicit_native_frame_is_supported(harness, bad):
    with pytest.raises(InputError):
        probe(harness, samples=bad)
    assert not harness['calls']


@pytest.mark.parametrize('bad', [None, [], [{'repo_id': 'other/source', 'revision': 'b' * 40, 'directory': '/unapproved'}],
    [{'repo_id': 'prior/source', 'revision': 'main', 'directory': '/unapproved'}]])
def test_serialized_source_requires_exact_explicit_approved_revision(harness, bad):
    with pytest.raises(InputError):
        probe(harness, approved_sources=bad)
    assert not harness['calls']


def test_duplicate_approved_ids_are_refused(harness):
    with pytest.raises(InputError):
        probe(harness, approved_sources=harness['approved'] * 2)
    assert not harness['calls']


def test_unknown_intent_is_retained_without_manufactured_incident(harness):
    result = probe(harness, intent=None)
    assert result['candidate']['status'] == result['corrected']['status'] == 'unknown'
    assert result['retained']


def test_selected_source_alias_does_not_invent_second_native_call(harness):
    result = probe(harness, processor_source='selected_checkpoint', approved_sources=[])
    assert result['candidate']['status'] == result['corrected']['status'] == 'pass'
    first = json.loads((Path(result['candidate_directory']) / 'capture.json').read_text())
    assert first['consumer_route'] == first['reference_route'] == 'selected_reference'


@pytest.mark.parametrize('kind', ['unapproved_root', 'mismatched_hash', 'escaping_retained_path'])
def test_worker_binding_cannot_grant_original_source_authority(harness, kind):
    def mutate(report, bindings, output):
        if kind == 'unapproved_root':
            bindings[0]['root'] = 'client_authorized_itself'
        elif kind == 'mismatched_hash':
            bindings[0]['sha256'] = 'f' * 64
        else:
            bindings[0]['retained_path'] = '../outside'
            # Let the generated producer return its malformed receipt so the
            # parent guard is exercised, rather than failing in this fixture.
            (output.parent / 'outside').write_bytes(b'generated outside-payload witness')
    harness['state']['mutate'] = mutate
    with pytest.raises(InputError):
        probe(harness)
    assert not harness['destination'].exists()


def test_selected_source_change_during_attempt_refuses_publication(harness):
    def mutate(report, bindings, output):
        path = harness['selected'] / 'config.json'
        path.write_bytes(path.read_bytes() + b' ')
    harness['state']['mutate'] = mutate
    with pytest.raises(InputError, match='changed'):
        probe(harness)
    assert not harness['destination'].exists()


def test_replay_verifies_view_then_reuses_retained_arrays_not_video_reader(harness):
    first = probe(harness)
    result = harness['api'].reproduce_native_processor_source(first['candidate_directory'], runtimes=harness['runtime'])
    assert result['reproduction_matches'] and result['status'] == 'pass'
    request = harness['calls'][-1]
    assert request['operation'] == 'replay' and 'dataset' not in request
    assert set(request['raw_samples'][0]) == {'arrays', 'metadata'}
    assert result['candidate']['status'] == 'error' and result['corrected']['status'] == 'pass'


def test_replay_tamper_and_verifier_drift_precede_native_invocation(harness):
    first = probe(harness)
    harness['state']['verifier'] = {'unit_fixed_verifier': 2}
    result = harness['api'].reproduce_native_processor_source(first['candidate_directory'], runtimes=harness['runtime'])
    assert result['status'] == 'unknown' and result['candidate_evaluated'] is False
    assert len(harness['calls']) == 1
    raw = Path(first['candidate_directory']) / 'raw/result.json'
    raw.write_text('{}')
    with pytest.raises(InputError):
        harness['api'].reproduce_native_processor_source(first['candidate_directory'], runtimes=harness['runtime'])
    assert len(harness['calls']) == 1


def test_changed_native_numerical_result_fails_stable_replay(harness):
    first = probe(harness)
    harness['state']['change_output'] = True
    result = harness['api'].reproduce_native_processor_source(first['candidate_directory'], runtimes=harness['runtime'])
    assert result['reproduction_matches'] is False and result['status'] == 'error'


@pytest.mark.parametrize('kind', ['role_name', 'aliased_different_outputs', 'unapproved_load'])
def test_observed_route_identity_and_load_membership_are_independent_guards(harness, kind):
    def mutate(report, bindings, output):
        if kind == 'role_name':
            report['routes']['consumer']['name'] = 'invented_role'
        elif kind == 'aliased_different_outputs':
            report['routes']['consumer']['call_id'] = report['routes']['selected_reference']['call_id']
        else:
            report['routes']['consumer']['loaded_files'][0]['root'] = 'unapproved'
    harness['state']['mutate'] = mutate
    with pytest.raises(InputError):
        probe(harness)
    assert not harness['destination'].exists()


def test_fixed_notice_is_preserved_from_intake_receipts_without_native_load_claim(harness):
    notice = harness['selected'] / 'LICENSE.txt'
    notice.write_text('Generated local notice; no public redistribution conclusion.')
    result = probe(harness)
    retained = Path(result['candidate_directory']) / 'originals/selected/LICENSE.txt'
    assert retained.read_bytes() == notice.read_bytes()
    raw = json.loads((Path(result['candidate_directory']) / 'raw/result.json').read_text())
    assert not any(item['path'] == 'LICENSE.txt' for item in raw.get('original_config_loaded_files', []))


def test_replay_preserves_selected_reference_role_of_corrected_view(harness):
    first = probe(harness)
    result = harness['api'].reproduce_native_processor_source(first['corrected_directory'], runtimes=harness['runtime'])
    assert result['execution_status'] == 'complete'
    assert result['status'] == 'pass' and result['reproduction_matches'] is True
    assert result['candidate']['source_criterion']['status'] == 'pass'
    assert result['corrected']['source_criterion']['status'] == 'pass'
    assert harness['calls'][-1]['processor_source'] == 'serialized_pretrained_path'
