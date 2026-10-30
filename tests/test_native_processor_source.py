"""Prospective evidence-only source criteria; all fixtures here are generated."""

import copy
import hashlib
import importlib
import json
import shutil

import pytest

from policy_doctor.artifacts import InputError


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def tensor(label, width=3):
    return {'dtype': 'float32', 'shape': [width], 'sha256': digest(label.encode())}


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def make_capture(root, *, consumer='suspect', intent='restore_selected_checkpoint', label='first'):
    root.mkdir(parents=True)
    bindings = []

    def retain(source_root, path, raw):
        relative = 'originals/' + source_root + '/' + path
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        item = {'root': source_root, 'path': path, 'retained_path': relative,
                'sha256': digest(raw), 'size_bytes': len(raw)}
        bindings.append(item)
        return {key: item[key] for key in ('root', 'path', 'sha256', 'size_bytes')} | {
            'witnesses': ['Generated unit-fixture load witness']}

    selected_config = retain('selected', 'config.json', encode({'type': 'generated', 'pretrained_path': 'prior/source'}))
    receipts = {}
    for source, prefix in [('selected', ''), ('alternate', 'snapshot/')]:
        records = []
        for filename, registry, state in [('policy_preprocessor.json', 'normalizer_processor', 'norm.safetensors'),
                                          ('policy_postprocessor.json', 'unnormalizer_processor', 'decode.safetensors')]:
            records.append(retain(source, prefix + filename, encode({'steps': [
                {'registry_name': registry, 'state_file': state, 'config': {}}]})))
            records.append(retain(source, prefix + state, (source + ':' + state).encode()))
        receipts[source] = records
    raw_input = {'robot.proprio': tensor(label), 'task': 'generated task'}
    routes = {}
    for name, source in [('suspect', 'alternate'), ('selected', 'selected'), ('fresh', None)]:
        stats_key = 'robot.proprio' if source == 'selected' else 'robot.other'
        stats = {} if source is None else {stats_key: {'mean': tensor(source + 'mean'), 'std': tensor(source + 'std')}}
        step = {'registry_name': 'normalizer_processor', 'qualified_name': 'native.Normalizer',
                'parameters': {'features': {'robot.proprio': {'type': 'STATE', 'shape': [3]}},
                               'norm_map': {'STATE': 'MEAN_STD'}},
                'effective_tensor_stats': stats, 'native_state_dict': copy.deepcopy(stats),
                'stats_explicitly_provided': False}
        output = {'robot.proprio': tensor(label + ('selected' if source == 'selected' else 'other'))}
        trace = [{'position': 'before', 'step_index': 0, 'transition': {'observation': raw_input}},
                 {'position': 'after', 'step_index': 0, 'transition': {'observation': output}}]
        trace_path = 'traces/' + name + '.json'
        trace_receipt = retain('attempt', trace_path, encode(trace))
        routes[name] = {'name': name, 'status': 'complete', 'input': copy.deepcopy(raw_input),
                        'output': output, 'loaded_files': copy.deepcopy(receipts.get(source, [])),
                        'factory_arguments': {'pretrained_path': source}, 'steps': [copy.deepcopy(step)],
                        'steps_after': [copy.deepcopy(step)],
                        'trace': {'path': trace_path, 'sha256': trace_receipt['sha256']}}
    raw = {'config_lineage': {'caller_selected': 'community/example@pinned',
                              'restore_intent': 'restore_selected_checkpoint',
                              'serialized_training_source': 'prior/source', 'parsed_pretrained_path': 'prior/source'},
           'execution_status': 'complete', 'identity': {'episode_index': 4, 'frame_index': 2, 'index': 99},
           'input': raw_input, 'routes': routes, 'original_config_loaded_files': [selected_config]}
    result_path = root / 'raw/result.json'
    result_path.parent.mkdir()
    result_path.write_bytes(encode(raw))
    result_sha = digest(result_path.read_bytes())
    wrapper = {'schema_version': 1, 'kind': 'native_processor_source_capture',
               'evidence_role': 'derived_route_view', 'result': {'path': 'raw/result.json', 'sha256': result_sha},
               'consumer_route': consumer, 'reference_route': 'selected', 'restore_intent': intent,
               'selected_source': {'identity': 'community/example@pinned', 'root': 'selected', 'prefix': ''},
               'bindings': bindings, 'derivation': {'attempt_sha256': result_sha,
               'consumer_call_id': result_sha + ':' + consumer, 'reference_call_id': result_sha + ':selected',
               'description': 'Generated derived views share one attempt; not two native launches.'}}
    (root / 'capture.json').write_bytes(encode(wrapper))
    return root


def mutate_capture(root, transform):
    path = root / 'capture.json'
    wrapper = json.loads(path.read_text())
    transform(wrapper)
    path.write_bytes(encode(wrapper))


def mutate_raw(root, transform):
    path = root / 'raw/result.json'
    raw = json.loads(path.read_text())
    transform(raw)
    path.write_bytes(encode(raw))
    result_sha = digest(path.read_bytes())

    def update(wrapper):
        wrapper['result']['sha256'] = result_sha
        wrapper['derivation']['attempt_sha256'] = result_sha
        wrapper['derivation']['consumer_call_id'] = result_sha + ':' + wrapper['consumer_route']
        wrapper['derivation']['reference_call_id'] = result_sha + ':' + wrapper['reference_route']
    mutate_capture(root, update)


@pytest.fixture
def api():
    # Imported only when a test runs, so the prospective red run records each
    # missing implementation rather than stopping at collection.
    return importlib.import_module('policy_doctor.native_processor_source')


@pytest.fixture
def captures(tmp_path):
    return make_capture(tmp_path / 'before'), make_capture(tmp_path / 'after', consumer='selected')


def test_real_source_contradiction_without_native_exception(api, captures):
    before, _ = captures
    report = api.audit_native_processor_source(before)
    assert report['status'] == 'error'
    assert report['source_criterion']['code'] == 'source_contradiction'
    assert report['native_processing_status'] == 'complete'
    assert report['native_replay_status'] == 'unknown'
    assert report['captured_output_parity_status'] == 'error'
    assert report['normalizer_input_statistics']['consumer'][0]['observed_input_stat_keys'] == []
    assert report['normalizer_input_statistics']['reference'][0]['observed_input_stat_keys'] == ['robot.proprio']


def test_selected_source_and_same_input_reference_pass(api, captures):
    _, after = captures
    report = api.audit_native_processor_source(after)
    assert report['status'] == 'pass'
    assert report['source_binding_status'] == 'pass'
    assert report['captured_output_parity_status'] == 'pass'
    assert report['effective_statistics_status'] == 'pass'


@pytest.mark.parametrize('intent', [None, 'fresh_training', 'restore_explicit_source'])
def test_missing_or_legal_other_intent_does_not_become_fault(api, captures, intent):
    before, _ = captures
    mutate_capture(before, lambda value: value.update(restore_intent=intent))
    report = api.audit_native_processor_source(before)
    assert report['status'] == 'unknown'
    assert report['source_criterion']['status'] == 'unknown'
    assert report['applicability'] == ('unknown' if intent is None else 'not_applicable')


@pytest.mark.parametrize('change', ['missing_state_receipt', 'missing_witness', 'empty_witness'])
def test_incomplete_source_witness_is_unknown_not_positive_mismatch(api, captures, change):
    before, _ = captures

    def alter(raw):
        records = raw['routes']['suspect']['loaded_files']
        if change == 'missing_state_receipt':
            records[:] = [record for record in records if not record['path'].endswith('norm.safetensors')]
        elif change == 'missing_witness':
            records[0].pop('witnesses')
        else:
            records[0]['witnesses'] = ['']
    mutate_raw(before, alter)
    report = api.audit_native_processor_source(before)
    assert report['status'] == 'unknown'
    assert report['source_criterion']['code'] != 'source_contradiction'


@pytest.mark.parametrize('change', ['empty_input', 'bad_hash', 'bool_shape', 'different_input',
                                   'empty_reference_output', 'reference_error'])
def test_input_and_reference_prerequisites_precede_output_comparison(api, captures, change):
    before, _ = captures

    def alter(raw):
        reference = raw['routes']['selected']
        if change == 'empty_input':
            reference['input'] = {}
        elif change == 'bad_hash':
            reference['input']['robot.proprio']['sha256'] = ''
        elif change == 'bool_shape':
            reference['input']['robot.proprio']['shape'] = [True]
        elif change == 'different_input':
            reference['input']['robot.proprio']['sha256'] = 'f' * 64
        elif change == 'empty_reference_output':
            reference['output'] = {}
        else:
            reference['status'] = 'error'
            reference.pop('output')
    mutate_raw(before, alter)
    report = api.audit_native_processor_source(before)
    assert report['status'] == 'unknown'
    assert report['captured_output_parity_status'] == 'unknown'


@pytest.mark.parametrize('value', [True, None, 'false'])
def test_statistics_overrides_and_missing_flags_cannot_certify_restore(api, captures, value):
    _, after = captures

    def alter(raw):
        for key in ('steps', 'steps_after'):
            raw['routes']['selected'][key][0]['stats_explicitly_provided'] = value
    mutate_raw(after, alter)
    assert api.audit_native_processor_source(after)['status'] == 'unknown'


def test_same_source_numerical_exception_is_not_source_incident(api, captures):
    _, after = captures

    def alter(raw):
        route = copy.deepcopy(raw['routes']['selected'])
        route.update(name='numerical_error', status='error', error={'type': 'ValueError', 'message': 'generated'})
        route.pop('output')
        raw['routes']['numerical_error'] = route
    mutate_raw(after, alter)
    mutate_capture(after, lambda value: value.update(consumer_route='numerical_error',
        derivation=value['derivation'] | {'consumer_call_id': value['result']['sha256'] + ':numerical_error'}))
    report = api.audit_native_processor_source(after)
    assert report['status'] != 'error'
    assert report['source_criterion']['code'] != 'source_contradiction'


def test_positive_source_contradiction_can_precede_numerical_exception(api, captures):
    before, _ = captures
    mutate_raw(before, lambda raw: raw['routes']['suspect'].update(status='error', output=None,
                                                                error={'type': 'ValueError', 'message': 'generated'}))
    assert api.audit_native_processor_source(before)['source_criterion']['code'] == 'source_contradiction'


def test_frozen_bundle_cannot_be_renominated_on_later_input(api, captures, tmp_path):
    _, after = captures
    parameters = api.bind_native_processor_source_parameters(after, {})
    later = make_capture(tmp_path / 'later', consumer='selected', label='new original numeric input')
    assert api.audit_native_processor_source(later, **parameters)['status'] == 'pass'
    mutate_capture(later, lambda value: value['selected_source'].update(identity='different/checkpoint@new'))
    assert api.audit_native_processor_source(later, **parameters)['status'] == 'error'


def test_direct_parameter_validation_is_strict(api):
    assert api.validate_parameters({}) == {'expected_source': 'selected_checkpoint',
                                          'require_same_input_native_reference': True}
    for value in ({'require_same_input_native_reference': 1}, {'require_same_input_native_reference': False},
                  {'expected_source': {}}, {'callback': 'artifact'}):
        with pytest.raises(InputError):
            api.validate_parameters(value)


@pytest.mark.parametrize('mutation', ['hash', 'duplicate_binding', 'duplicate_retained_path', 'unsafe_path',
                                      'bad_attempt', 'conflicting_calls'])
def test_capture_integrity_and_relative_binding_controls(api, captures, mutation):
    _, after = captures

    def alter(value):
        if mutation == 'hash':
            value['bindings'][0]['sha256'] = 'f' * 64
        elif mutation == 'duplicate_binding':
            value['bindings'].append(copy.deepcopy(value['bindings'][0]))
        elif mutation == 'duplicate_retained_path':
            value['bindings'][1]['retained_path'] = value['bindings'][0]['retained_path']
        elif mutation == 'unsafe_path':
            value['bindings'][0]['retained_path'] = '../outside'
        elif mutation == 'bad_attempt':
            value['derivation']['attempt_sha256'] = '0' * 64
        else:
            value['derivation']['consumer_call_id'] += ':different'
    mutate_capture(after, alter)
    with pytest.raises(InputError):
        api.audit_native_processor_source(after)


def test_symlink_retained_file_refused(api, captures, tmp_path):
    _, after = captures
    target = after / 'originals/selected/config.json'
    other = tmp_path / 'other.json'
    shutil.copyfile(target, other)
    target.unlink()
    target.symlink_to(other)
    with pytest.raises(InputError):
        api.audit_native_processor_source(after)


def test_incident_retains_same_source_rule_for_later_input(api, captures, tmp_path):
    from policy_doctor.incidents import (
        check_incident_candidate,
        export_incident_case,
        reproduce_incident_case,
    )

    before, after = captures
    case = tmp_path / 'case'
    result = export_incident_case('native_processor_source', before, case, {}, corrected=after)
    assert result['status'] == 'pass'
    assert reproduce_incident_case(case)['status'] == 'pass'
    manifest = json.loads((case / 'manifest.json').read_text())
    assert type(manifest['parameters']['expected_source']) is dict
    later = make_capture(tmp_path / 'later', consumer='selected', label='new input')
    assert check_incident_candidate(case, later)['status'] == 'pass'
    mutate_capture(later, lambda value: value.update(restore_intent=None))
    assert check_incident_candidate(case, later)['status'] == 'unknown'
    mutate_capture(later, lambda value: value.update(restore_intent='restore_selected_checkpoint', consumer_route='suspect',
        derivation=value['derivation'] | {'consumer_call_id': value['result']['sha256'] + ':suspect'}))
    assert check_incident_candidate(case, later)['status'] == 'error'


def test_healthy_capture_cannot_be_primary_source_incident(api, captures, tmp_path):
    from policy_doctor.incidents import export_incident_case

    _, after = captures
    with pytest.raises(InputError):
        export_incident_case('native_processor_source', after, tmp_path / 'case', {}, corrected=after)


def test_different_input_cannot_be_called_same_input_correction(api, captures, tmp_path):
    from policy_doctor.incidents import export_incident_case

    before, _ = captures
    different = make_capture(tmp_path / 'different', consumer='selected', label='different frame')
    with pytest.raises(InputError, match='same parameters'):
        export_incident_case('native_processor_source', before, tmp_path / 'case', {}, corrected=different)


@pytest.mark.parametrize('change', ['missing_reference_stats', 'empty_both_outputs', 'invalid_dtype',
                                   'empty_tensor', 'missing_own_stats', 'missing_parameters'])
def test_no_vacuous_numerical_or_statistics_parity(api, captures, change):
    _, after = captures

    def alter(raw):
        route = raw['routes']['selected']
        if change == 'missing_reference_stats':
            route['steps_after'][0].pop('effective_tensor_stats')
        elif change == 'empty_both_outputs':
            route['output'] = {}
        elif change == 'invalid_dtype':
            route['input']['robot.proprio']['dtype'] = 'pickle'
        elif change == 'empty_tensor':
            route['output']['robot.proprio']['shape'] = [0]
        elif change == 'missing_own_stats':
            route['steps'][0].pop('native_state_dict')
        else:
            route['steps'][0].pop('parameters')
    mutate_raw(after, alter)
    assert api.audit_native_processor_source(after)['status'] == 'unknown'


@pytest.mark.parametrize('change', ['duplicate_receipt', 'foreign_route_field'])
def test_loaded_receipts_are_unambiguous_and_call_scoped(api, captures, change):
    before, _ = captures

    def alter(raw):
        files = raw['routes']['suspect']['loaded_files']
        if change == 'duplicate_receipt':
            files.append(copy.deepcopy(files[0]))
        else:
            files[0]['route'] = 'another call'
    mutate_raw(before, alter)
    with pytest.raises(InputError):
        api.audit_native_processor_source(before)


def test_native_trace_must_have_complete_step_pairs(api, captures):
    _, after = captures
    trace = after / 'originals/attempt/traces/selected.json'
    trace.write_bytes(encode([{'position': 'before', 'step_index': 0, 'transition': {'observation': {'x': 1}}}]))
    new_sha = digest(trace.read_bytes())

    def wrapper_change(value):
        for item in value['bindings']:
            if item['root'] == 'attempt' and item['path'] == 'traces/selected.json':
                item.update(sha256=new_sha, size_bytes=trace.stat().st_size)
    mutate_capture(after, wrapper_change)
    mutate_raw(after, lambda value: value['routes']['selected']['trace'].update(sha256=new_sha))
    assert api.audit_native_processor_source(after)['status'] == 'unknown'


def test_empty_or_duplicate_required_bundle_is_refused(api, captures):
    _, after = captures
    parameters = api.bind_native_processor_source_parameters(after)
    for mode in ('empty', 'duplicate'):
        changed = copy.deepcopy(parameters)
        files = changed['expected_source']['files']
        if mode == 'empty':
            files.clear()
        else:
            files.append(copy.deepcopy(files[0]))
        with pytest.raises(InputError):
            api.validate_parameters(changed)


def test_equal_values_from_different_native_frame_are_not_a_same_input_correction(api, captures, tmp_path):
    from policy_doctor.incidents import export_incident_case

    before, after = captures
    mutate_raw(after, lambda raw: raw['identity'].update(frame_index=3, index=100))
    with pytest.raises(InputError, match='same parameters'):
        export_incident_case('native_processor_source', before, tmp_path / 'case', {}, corrected=after)
