"""Independent generated controls; fixture wire reused from author tests, no native data."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from policy_doctor import native_processor_source as api
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



def test_independent_positive_generated_source_content(tmp_path):
    capture = make_capture(tmp_path / 'positive', consumer='selected')
    report = api.audit_native_processor_source(capture)
    assert report['status'] == 'pass'
    assert report['original_authenticity_status'] == 'unknown'


@pytest.mark.parametrize('field', ['effective_tensor_stats', 'native_state_dict'])
def test_empty_selected_nonidentity_statistics_cannot_pass(tmp_path, field):
    capture = make_capture(tmp_path / 'empty-stats', consumer='selected')
    def alter(raw):
        for inventory in ('steps', 'steps_after'):
            raw['routes']['selected'][inventory][0][field] = {}
    mutate_raw(capture, alter)
    assert api.audit_native_processor_source(capture)['status'] == 'unknown'


@pytest.mark.parametrize('endpoint', ['input', 'output'])
def test_trace_endpoint_contradiction_is_unknown(tmp_path, endpoint):
    capture = make_capture(tmp_path / 'endpoint', consumer='selected')
    trace_file = capture / 'originals/attempt/traces/selected.json'
    trace = json.loads(trace_file.read_text())
    phase = 0 if endpoint == 'input' else -1
    trace[phase]['transition']['observation']['robot.proprio']['sha256'] = 'f' * 64
    trace_file.write_bytes(encode(trace))
    new_sha = digest(trace_file.read_bytes())
    def update_binding(wrapper):
        for binding in wrapper['bindings']:
            if binding['root'] == 'attempt' and binding['path'] == 'traces/selected.json':
                binding.update(sha256=new_sha, size_bytes=trace_file.stat().st_size)
    mutate_capture(capture, update_binding)
    mutate_raw(capture, lambda raw: raw['routes']['selected']['trace'].update(sha256=new_sha))
    assert api.audit_native_processor_source(capture)['status'] == 'unknown'


def test_json_has_no_unbounded_second_read_after_guard(tmp_path, monkeypatch):
    path = tmp_path / 'bounded.json'
    path.write_text('{"generated": true}')
    original = Path.read_bytes
    def sentinel(self):
        if self == path:
            pytest.fail('Unbounded Path.read_bytes reached after preliminary file hash')
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', sentinel)
    assert api._json(path) == {'generated': True}


@pytest.mark.parametrize('change', ['all_inputs_missing', 'wrong_reference_input_and_output',
    'missing_reference_completion', 'reference_error', 'reference_stats_missing', 'both_outputs_missing'])
def test_original_criterion_missing_evidence_gaps_closed(tmp_path, change):
    capture = make_capture(tmp_path / change, consumer='selected')
    def alter(raw):
        route = raw['routes']['selected']
        if change == 'all_inputs_missing':
            raw.pop('input')
            for value in raw['routes'].values():
                value.pop('input')
        elif change == 'wrong_reference_input_and_output':
            # Use separate consumer to keep the current input intact.
            raw['routes']['copy_consumer'] = copy.deepcopy(route)
            raw['routes']['copy_consumer']['name'] = 'copy_consumer'
            route['input']['robot.proprio']['sha256'] = 'f' * 64
            route['output']['robot.proprio']['sha256'] = 'e' * 64
        elif change == 'missing_reference_completion':
            route.pop('status')
        elif change == 'reference_error':
            route['status'] = 'error'
        elif change == 'reference_stats_missing':
            route['steps_after'][0].pop('effective_tensor_stats')
        else:
            route.pop('output')
    mutate_raw(capture, alter)
    if change == 'wrong_reference_input_and_output':
        mutate_capture(capture, lambda w: w.update(consumer_route='copy_consumer',
            derivation=w['derivation'] | {'consumer_call_id': w['result']['sha256'] + ':copy_consumer'}))
    assert api.audit_native_processor_source(capture)['status'] == 'unknown'


@pytest.mark.parametrize('change', ['duplicate_binding', 'duplicate_retained', 'duplicate_receipt', 'unsafe_retained'])
def test_duplicate_and_unsafe_source_bindings_refused(tmp_path, change):
    capture = make_capture(tmp_path / change, consumer='selected')
    if change == 'duplicate_receipt':
        mutate_raw(capture, lambda raw: raw['routes']['selected']['loaded_files'].append(
            copy.deepcopy(raw['routes']['selected']['loaded_files'][0])))
    else:
        def alter(w):
            if change == 'duplicate_binding':
                w['bindings'].append(copy.deepcopy(w['bindings'][0]))
            elif change == 'duplicate_retained':
                w['bindings'][1]['retained_path'] = w['bindings'][0]['retained_path']
            else:
                w['bindings'][0]['retained_path'] = '../escape'
        mutate_capture(capture, alter)
    with pytest.raises(InputError):
        api.audit_native_processor_source(capture)


@pytest.mark.parametrize('mode', ['empty', 'duplicate', 'callback'])
def test_strict_frozen_criterion_bundle(tmp_path, mode):
    capture = make_capture(tmp_path / mode, consumer='selected')
    parameters = api.bind_native_processor_source_parameters(capture)
    if mode == 'empty':
        parameters['expected_source']['files'] = []
    elif mode == 'duplicate':
        parameters['expected_source']['files'].append(copy.deepcopy(parameters['expected_source']['files'][0]))
    else:
        parameters['callback'] = 'arbitrary.python:execute'
    with pytest.raises(InputError):
        api.validate_parameters(parameters)


def test_content_identical_alternate_origin_is_not_false_fault(tmp_path):
    capture = make_capture(tmp_path / 'equal-content', consumer='suspect')
    # Explicit root-approved semantics: content identity may pass even though
    # the logical origin differs. This proves no repository-origin authenticity.
    def copy_original_content(w):
        selected = {x['path']: x for x in w['bindings'] if x['root'] == 'selected'}
        for item in w['bindings']:
            if item['root'] == 'alternate':
                relative = item['path'].removeprefix('snapshot/')
                source = capture / selected[relative]['retained_path']
                target = capture / item['retained_path']
                target.write_bytes(source.read_bytes())
                item.update(sha256=digest(target.read_bytes()), size_bytes=target.stat().st_size)
    mutate_capture(capture, copy_original_content)
    wrapper = json.loads((capture / 'capture.json').read_text())
    index = {(x['root'],x['path']):x for x in wrapper['bindings']}
    def alter(raw):
        alternate = raw['routes']['suspect']
        for item in alternate['loaded_files']:
            binding = index[(item['root'],item['path'])]
            item.update(sha256=binding['sha256'], size_bytes=binding['size_bytes'])
        for field in ('input','output','steps','steps_after'):
            alternate[field] = copy.deepcopy(raw['routes']['selected'][field])
    mutate_raw(capture, alter)
    # Align alternate trace to its genuinely declared generated output.
    trace = capture / 'originals/attempt/traces/suspect.json'
    trace.write_bytes((capture / 'originals/attempt/traces/selected.json').read_bytes())
    new_sha = digest(trace.read_bytes())
    mutate_capture(capture, lambda w: [x.update(sha256=new_sha, size_bytes=trace.stat().st_size)
        for x in w['bindings'] if x['root']=='attempt' and x['path']=='traces/suspect.json'])
    mutate_raw(capture, lambda raw: raw['routes']['suspect']['trace'].update(sha256=new_sha))
    report = api.audit_native_processor_source(capture)
    assert report['status'] == 'pass'
    assert report['original_authenticity_status'] == 'unknown'


def test_frozen_source_rule_later_input_and_corrected_association(tmp_path):
    from policy_doctor.incidents import check_incident_candidate, export_incident_case
    before = make_capture(tmp_path / 'before')
    after = make_capture(tmp_path / 'after', consumer='selected')
    case = tmp_path / 'case'
    export_incident_case('native_processor_source', before, case, {}, corrected=after)
    later = make_capture(tmp_path / 'later', consumer='selected', label='different numerical observation')
    report = check_incident_candidate(case, later)
    assert report['status'] == 'pass'
    mutate_capture(later, lambda w: w['selected_source'].update(identity='different/source@revision'))
    assert check_incident_candidate(case, later)['status'] == 'error'


def test_equal_values_from_other_frame_cannot_be_correction(tmp_path):
    from policy_doctor.incidents import export_incident_case
    before = make_capture(tmp_path / 'before')
    after = make_capture(tmp_path / 'after', consumer='selected')
    mutate_raw(after, lambda raw: raw['identity'].update(frame_index=3, index=100))
    with pytest.raises(InputError):
        export_incident_case('native_processor_source', before, tmp_path / 'bad-case', {}, corrected=after)


def test_source_selection_diagnostic_is_bounded_not_a_criterion(tmp_path):
    capture = make_capture(tmp_path / 'selection-diagnostic', consumer='selected')
    report = api.audit_native_processor_source(capture)
    assert report['source_selection'] == {'selected_checkpoint': 'community/example@pinned',
                                         'consumer_argument': 'selected', 'reference_argument': 'selected'}
    # Missing/invalid oversized diagnostic values cannot be copied or create a fault.
    def alter(raw):
        raw['routes']['selected']['factory_arguments'] = {'pretrained_path': ['not', 'a', 'source']}
    mutate_raw(capture, alter)
    report = api.audit_native_processor_source(capture)
    assert report['status'] == 'pass'
    assert report['source_selection'] == {'selected_checkpoint': 'community/example@pinned'}
    mutate_raw(capture, lambda raw: raw['routes']['selected']['factory_arguments'].update(pretrained_path='x' * 513))
    assert api.audit_native_processor_source(capture)['source_selection'] == {'selected_checkpoint': 'community/example@pinned'}
    mutate_raw(capture, lambda raw: raw['routes']['selected']['factory_arguments'].update(pretrained_path=None))
    assert api.audit_native_processor_source(capture)['source_selection']['consumer_argument'] is None


def test_native_timestamp_overflow_is_unknown_without_float_coercion(tmp_path):
    capture = make_capture(tmp_path / 'timestamp-overflow', consumer='selected')
    mutate_raw(capture, lambda raw: raw['identity'].update(timestamp=10**400))
    report = api.audit_native_processor_source(capture)
    assert report['status'] == 'unknown'
    assert report['source_criterion']['code'] == 'missing_evidence'
