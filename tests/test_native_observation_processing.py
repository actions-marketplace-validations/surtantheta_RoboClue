"""Generated lifecycle controls; fake native calls are not execution evidence."""

import copy
import importlib
from pathlib import Path

import pytest

from policy_doctor.artifacts import InputError


@pytest.fixture
def harness(tmp_path, monkeypatch):
    api = importlib.import_module('policy_doctor.native_observation_processing')
    model, dataset = tmp_path / 'model', tmp_path / 'dataset'
    model.mkdir()
    dataset.mkdir()
    (model / 'processor_config.json').write_text('{}')
    (model / 'statistics.json').write_text('{}')
    (dataset / 'row.parquet').write_bytes(b'unit fixture, not Parquet')
    calls, state = [], {'verifier': 'fixed', 'mutate': None, 'changed': False}
    contract = {'kind': 'unit supported contract'}
    def validate(parameters):
        if parameters.get('contract') != contract:
            raise InputError('Generated unit unsupported contract')
        return dict(parameters)
    def prepare(model, dataset, samples, workspace):
        receipts = []
        for label, root in [('model', Path(model)), ('dataset', Path(dataset))]:
            for path in root.iterdir():
                sha, size = api._hash(path)
                receipts.append({'root': label, 'path': path.name, 'sha256': sha,
                                 'size_bytes': size, 'role': 'unit_original'})
        return {'model': Path(model), 'dataset': Path(dataset), 'receipts': receipts,
                'selection': {'model': Path(model), 'dataset': Path(dataset)}}
    def invoke(script, runtime, request, workspace):
        assert script.name == 'native_groot_adapter_worker.py'
        calls.append(copy.deepcopy(request))
        stage = Path(request['output'])
        stage.mkdir()
        (stage / 'samples').mkdir()
        (stage / 'samples/arrays.npz').write_bytes(b'unit arrays, never decoded by this fake auditor')
        metadata = {'episode_index': 4, 'frame_index': 2, 'global_index': 12,
                    'cameras': {'image': {'source_file': str(request.get('dataset', '/moved')),
                                           'pixel_sha256': 'a' * 64}}}
        api._write(stage / 'samples/metadata.json', metadata)
        raw = {'kind': 'native_adapter_boundary_capture', 'input_identity': {'episode_index': 4,
                    'frame_index': 2, 'global_index': 12}, 'contract': contract,
               'native_processing_status': 'complete', 'source_module_sha256': 'b' * 64,
               'captured': {'state': {'generated': [2 if state['changed'] else 1]}},
               'unit_role': 'reference' if request['source_root'] == '/trusted/reference' else 'candidate'}
        api._write(stage / 'result.json', raw)
        descriptor = {'provider': 'groot_adapter', 'operation': request['operation'],
                      'execution_status': 'complete', 'result': api._record(stage, 'result.json'),
                      'payload_files': [api._record(stage, path) for path in
                                       ('result.json', 'samples/arrays.npz', 'samples/metadata.json')]}
        (stage / 'unlisted.py').write_text('never retained or executed')
        if state['mutate']:
            state['mutate'](stage, descriptor)
        return descriptor
    def audit(root, **kwargs):
        raw = api._json(Path(root) / 'raw/result.json')
        status = 'pass' if raw['unit_role'] == 'reference' else 'error'
        return {'kind': 'native_observation_source_audit', 'status': status,
                'source_criterion': {'status': status, 'code': 'source_preserved' if status == 'pass'
                                     else 'source_contradiction', 'reason': 'unit independent auditor'},
                'input_digest': 'input-fixed', 'native_identity': raw['input_identity'],
                'model_identity': 'model-fixed', 'native_processing_status': 'complete',
                'native_state_component_status': 'complete', 'native_replay_status': 'unknown'}
    monkeypatch.setattr(api, 'validate_parameters', validate)
    monkeypatch.setattr(api, 'prepare_native_groot_inputs', prepare)
    monkeypatch.setattr(api, 'verify_native_groot_inputs', lambda selection: None)
    monkeypatch.setattr(api, '_verifier', lambda runtime: {'fixed': state['verifier']})
    monkeypatch.setattr(api, '_invoke', invoke)
    monkeypatch.setattr(api, 'audit_native_observation_source', audit)
    return {'api': api, 'model': model, 'dataset': dataset, 'contract': contract,
            'samples': [{'episode_index': 4, 'frame_index': 2}], 'calls': calls, 'state': state,
            'runtimes': {'groot_adapter': {'python': '/trusted/python', 'source_root': '/trusted/candidate',
                                         'groot_source_root': '/trusted/groot',
                                         'reference_source_root': '/trusted/reference'}},
            'destination': tmp_path / 'retained'}


def probe(h, **overrides):
    args = {key: h[key] for key in ('runtimes', 'samples', 'contract', 'destination')}
    args.update(overrides)
    return h['api'].probe_native_observation_source(h['model'], h['dataset'], **args)


def test_separate_native_routes_and_self_contained_views(harness):
    h = harness
    result = probe(h)
    assert [call['source_root'] for call in h['calls']] == ['/trusted/candidate', '/trusted/reference']
    assert result['candidate']['status'] == 'error' and result['corrected']['status'] == 'pass'
    for key in ('candidate_directory', 'corrected_directory'):
        root = Path(result[key])
        assert not (root / 'unlisted.py').exists()
        capture = h['api']._json(root / 'capture.json')
        assert capture['evidence_role'] == 'native_adapter_dry_run'
        assert {item['root'] for item in capture['bindings']} == {'model', 'dataset'}
        assert (root / 'samples/arrays.npz').is_file()
        h['api']._load_committed(root, h['api'].VIEW)


def test_no_reference_never_fabricates_corrected_view(harness):
    h = harness
    del h['runtimes']['groot_adapter']['reference_source_root']
    result = probe(h)
    assert len(h['calls']) == 1 and result['corrected'] is None
    assert result.get('corrected_directory') is None
    assert not (Path(result['directory']) / 'corrected').exists()


@pytest.mark.parametrize('role', ['candidate', 'corrected'])
def test_relocated_view_replay_uses_retained_role_and_no_initial_cache(harness, tmp_path, role):
    h = harness
    result = probe(h)
    old = Path(result[role + '_directory'])
    moved = tmp_path / ('moved-' + role)
    old.rename(moved)
    h['model'].rename(tmp_path / 'unavailable-model')
    h['dataset'].rename(tmp_path / 'unavailable-dataset')
    replay = h['api'].reproduce_native_observation_source(moved, runtimes=h['runtimes'])
    assert replay['status'] == 'pass' and replay['reproduction_matches']
    assert len(h['calls']) == 3
    assert h['calls'][-1]['source_root'] == '/trusted/' + ('reference' if role == 'corrected' else 'candidate')
    assert h['calls'][-1]['operation'] == 'replay'
    assert 'dataset' not in h['calls'][-1] and 'cache_root' not in h['calls'][-1]


def test_outer_attempt_dispatches_candidate_only(harness):
    h = harness
    result = probe(h)
    assert h['api'].reproduce_native_observation_source(result['directory'], runtimes=h['runtimes'])['status'] == 'pass'
    assert len(h['calls']) == 3 and h['calls'][-1]['source_root'] == '/trusted/candidate'


def test_runtime_drift_is_unknown_before_execution(harness):
    h = harness
    result = probe(h)
    h['state']['verifier'] = 'changed'
    replay = h['api'].reproduce_native_observation_source(result['candidate_directory'], runtimes=h['runtimes'])
    assert replay['status'] == 'unknown' and replay['candidate_evaluated'] is False
    assert len(h['calls']) == 2


def test_native_output_change_fails_full_projection(harness):
    h = harness
    result = probe(h)
    h['state']['changed'] = True
    replay = h['api'].reproduce_native_observation_source(result['candidate_directory'], runtimes=h['runtimes'])
    assert replay['status'] == 'error' and not replay['reproduction_matches']


@pytest.mark.parametrize('mutator', [
    lambda stage, d: d['payload_files'].append(dict(d['result'])),
    lambda stage, d: d['payload_files'].pop(),
    lambda stage, d: d.update(provider='artifact-selected-code'),
    lambda stage, d: d['result'].update(path='../escape'),
    lambda stage, d: d['result'].update(sha256='0' * 64),
    lambda stage, d: d['payload_files'].append({'path': 'unlisted.py', 'sha256': 'a' * 64, 'size_bytes': 1}),
])
def test_malformed_or_extra_worker_payload_refused_before_publication(harness, mutator):
    h = harness
    h['state']['mutate'] = mutator
    with pytest.raises(InputError):
        probe(h)
    assert not h['destination'].exists()


def test_tampered_retained_membership_refused_before_execution(harness):
    h = harness
    result = probe(h)
    root = Path(result['candidate_directory'])
    (root / 'extra.json').write_text('{}')
    with pytest.raises(InputError):
        h['api'].reproduce_native_observation_source(root, runtimes=h['runtimes'])
    assert len(h['calls']) == 2


def test_absent_contract_refused_before_native_call(harness):
    with pytest.raises(InputError):
        probe(harness, contract=None)
    assert not harness['calls']


@pytest.mark.parametrize('samples', [[], [{'episode_index': 4}], [{'episode_index': 4, 'frame_index': 2}] * 2])
def test_single_explicit_row_required(harness, samples):
    with pytest.raises(InputError):
        probe(harness, samples=samples)
    assert not harness['calls']


def test_ephemeral_probe_reports_no_retention(harness):
    result = probe(harness, destination=None)
    assert result['retained'] is False and 'directory' not in result


def test_existing_destination_refused(harness):
    harness['destination'].mkdir()
    with pytest.raises(InputError):
        probe(harness)
    assert not harness['calls']


def restamp(h, stage, descriptor, relative):
    item = h['api']._record(stage, relative)
    descriptor['payload_files'] = [item if value['path'] == relative else value
                                   for value in descriptor['payload_files']]
    if relative == 'result.json':
        descriptor['result'] = item


def test_imported_worker_answer_does_not_override_independent_auditor(harness):
    h = harness
    def mutate(stage, descriptor):
        raw = h['api']._json(stage / 'result.json')
        raw['criterion'] = {'status': 'pass', 'reason': 'forged helper conclusion'}
        h['api']._write(stage / 'replacement.json', raw)
        (stage / 'replacement.json').replace(stage / 'result.json')
        restamp(h, stage, descriptor, 'result.json')
    h['state']['mutate'] = mutate
    assert probe(h)['candidate']['status'] == 'error'


@pytest.mark.parametrize('field', ['global_index', 'frame_index', 'episode_index'])
def test_worker_sample_identity_contradiction_refused(harness, field):
    h = harness
    def mutate(stage, descriptor):
        metadata = h['api']._json(stage / 'samples/metadata.json')
        metadata[field] += 1
        h['api']._write(stage / 'replacement.json', metadata)
        (stage / 'replacement.json').replace(stage / 'samples/metadata.json')
        restamp(h, stage, descriptor, 'samples/metadata.json')
    h['state']['mutate'] = mutate
    with pytest.raises(InputError):
        probe(h)
    assert not h['destination'].exists()


def test_reference_cannot_change_original_pixel_witness(harness):
    h = harness
    def mutate(stage, descriptor):
        if len(h['calls']) == 2:
            metadata = h['api']._json(stage / 'samples/metadata.json')
            metadata['cameras']['image']['pixel_sha256'] = 'c' * 64
            h['api']._write(stage / 'replacement.json', metadata)
            (stage / 'replacement.json').replace(stage / 'samples/metadata.json')
            restamp(h, stage, descriptor, 'samples/metadata.json')
    h['state']['mutate'] = mutate
    with pytest.raises(InputError, match='same original'):
        probe(h)


@pytest.mark.parametrize('field', ['input_digest', 'model_identity', 'native_identity'])
def test_reference_audit_identity_must_match_original(harness, monkeypatch, field):
    h = harness
    audit = h['api'].audit_native_observation_source
    def altered(root, **kwargs):
        report = audit(root, **kwargs)
        if Path(root).name == 'corrected':
            report[field] = 'changed'
        return report
    monkeypatch.setattr(h['api'], 'audit_native_observation_source', altered)
    with pytest.raises(InputError, match='same original'):
        probe(h)


def test_projection_excludes_only_known_host_locator(harness):
    api = harness['api']
    raw = {'source_file': 'actual native field', 'criterion': {'status': 'forged'}}
    metadata = {'source_file': 'not an approved locator', 'cameras': {'image': {
        'source_file': '/host/video.mp4', 'pixel_sha256': 'a' * 64}}}
    projection = api._projection(raw, metadata)
    assert projection['raw'] == {'source_file': 'actual native field'}
    assert projection['sample_metadata']['source_file'] == 'not an approved locator'
    assert projection['sample_metadata']['cameras']['image'] == {'pixel_sha256': 'a' * 64}


def test_runtime_mutation_during_capture_refused(harness):
    h = harness
    h['state']['mutate'] = lambda stage, descriptor: h['state'].update(verifier='changed')
    with pytest.raises(InputError, match='verifier changed'):
        probe(h)
    assert not h['destination'].exists()


def test_destination_inside_originals_refused(harness):
    with pytest.raises(InputError, match='outside original'):
        probe(harness, destination=harness['dataset'] / 'capture')
    assert not harness['calls']


def test_unsafe_capture_root_symlink_refused(harness, tmp_path):
    result = probe(harness)
    link = tmp_path / 'linked-view'
    link.symlink_to(result['candidate_directory'], target_is_directory=True)
    with pytest.raises(InputError, match='symlink'):
        harness['api'].reproduce_native_observation_source(link, runtimes=harness['runtimes'])
    assert len(harness['calls']) == 2


def test_copy_stops_if_source_grows_after_open(harness, tmp_path, monkeypatch):
    import os

    api = harness['api']
    source, destination = tmp_path / 'source-copy', tmp_path / 'destination-copy'
    source.mkdir()
    destination.mkdir()
    path = source / 'payload.bin'
    path.write_bytes(b'a' * (2 * 1024**2))
    record = api._record(source, 'payload.bin')
    read, changed = os.read, False
    def growing(fd, count):
        nonlocal changed
        block = read(fd, count)
        if not changed:
            changed = True
            with path.open('ab') as stream:
                stream.write(b'b' * (2 * 1024**2))
        return block
    monkeypatch.setattr(os, 'read', growing)
    with pytest.raises(InputError):
        api._copy(source, 'payload.bin', destination, 'payload.bin', record)
    assert (destination / 'payload.bin').stat().st_size <= record['size_bytes'] + 1


@pytest.mark.parametrize('contradiction', [False, True])
def test_real_core_audits_retained_typed_view(tmp_path, contradiction):
    import numpy as np
    from test_native_observation_source import fixture, sha

    api = importlib.import_module('policy_doctor.native_observation_processing')
    root = tmp_path / 'generated-source'
    capture, raw = fixture(root)
    if contradiction:
        leaf = raw['captured']['state']['roll']
        leaf['values'] = raw['captured']['state']['pitch']['values'][:]
        leaf['sha256'] = sha(np.asarray(leaf['values'], dtype='<f4').tobytes())
    stage = tmp_path / 'worker-stage'
    stage.mkdir()
    api._write(stage / 'result.json', raw)
    for relative in ('samples/arrays.npz', 'samples/metadata.json'):
        api._copy(root, relative, stage, relative, api._record(root, relative))
    records = {relative: api._record(stage, relative) for relative in api.PAYLOAD}
    allowed = {(binding['root'], binding['path']): dict(binding,
               source_root=root / 'originals' / binding['root'], source_path=binding['path'])
               for binding in capture['bindings']}
    metadata = api._json(stage / 'samples/metadata.json')
    result = api._view(tmp_path / 'actual-core-view', stage, records, allowed, raw, metadata,
                       capture['contract'], {'unit_verifier': True},
                       {'role': 'candidate', 'samples': [{'episode_index': 0, 'frame_index': 0}],
                        'contract': capture['contract']})
    assert result['status'] == ('error' if contradiction else 'pass')
    assert result['native_processing_status'] == 'complete'
    assert result['native_replay_status'] == 'unknown'
