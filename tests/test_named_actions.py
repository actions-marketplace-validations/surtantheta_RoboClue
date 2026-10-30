"""Generated named boundary packets; saved sources are never executed."""
import hashlib
import json
import shutil

import pytest

from policy_doctor import named_actions, named_rules
from policy_doctor.artifacts import InputError


def commit(root, data):
    raw = json.dumps(data).encode()
    (root / 'capture.json').write_bytes(raw)
    (root / 'READY.json').write_text(json.dumps({'schema_version': 1, 'manifest_sha256': hashlib.sha256(raw).hexdigest()}))


def change(root, mutate):
    data = json.loads((root / 'capture.json').read_text())
    mutate(data)
    commit(root, data)


@pytest.fixture
def packet(tmp_path, monkeypatch):
    def write(name='capture', *, limited=True, offset=0.):
        root = tmp_path / name
        root.mkdir()
        (root / 'sources').mkdir()
        sources = {key: ('controlled pinned source ' + key).encode() for key in named_rules.SOURCE_HASHES}
        sources['driver_source.py'] = b'raise AssertionError("untrusted driver source must never execute")'
        hashes = {key: hashlib.sha256(raw).hexdigest() for key, raw in sources.items()}
        monkeypatch.setattr(named_actions, 'SOURCE_HASHES', {key: hashes[key] for key in named_rules.SOURCE_HASHES})
        for key, raw in sources.items():
            (root / 'sources' / key).write_bytes(raw)
        canonical = {'a.pos': 2. + offset, 'b.pos': -1. + offset}
        returned = {**canonical, 'a.pos': 1. + offset} if limited else dict(canonical)
        tick = {'tick_id': 0, 'completion': 'normal', 'outcome': 'sent',
                'interpolated': [2. + offset, -1. + offset], 'canonical_at_processor': dict(canonical),
                'canonical_at_return': dict(canonical), 'robot_processor_output': dict(canonical),
                'wrapper_request': dict(canonical), 'driver_request': dict(canonical), 'driver_return': returned,
                'wrapper_return': dict(returned),
                'driver_called': True, 'driver_completed': True, 'wrapper_completed': True,
                'ack_count': 1, 'emitted_policy_action': True,
                'relations': {key: True for key in named_actions.RELATIONS}}
        data = {'schema_version': 1, 'kind': 'named_action_capture', 'domain': 'native_named_action_software',
                'source': {'repository': named_rules.REPOSITORY, 'revision': named_rules.REVISION,
                           'files': {key: {'sha256': value} for key, value in hashes.items()}},
                'action_names': ['a.pos', 'b.pos'], 'engine_action_names': ['a.pos', 'b.pos'],
                'driver_namespace': 'generated/software-driver',
                'coverage': {'target': 'one_native_send_next_action_on_owner_thread',
                             'complete_target_call_history': True, 'external_state_mutations': 'untracked'},
                'ticks': [tick], 'files': [{'path': 'sources/' + key, 'sha256': value} for key, value in hashes.items()]}
        commit(root, data)
        return root
    return write


def contract(*, require=False):
    return {'expected_action_names': ['a.pos', 'b.pos'], 'expected_driver_namespace': 'generated/software-driver',
            'require_canonical_matches_driver_return': require}


def test_legal_limiting_observed_without_automatic_fault(packet):
    root = packet()
    report = named_actions.audit_named_actions(root, **contract())
    assert report['status'] == report['binding_status'] == report['numeric_replay_status'] == 'pass'
    assert report['driver_change_events'] == 1
    assert report['ticks'][0]['canonical_matches_driver_return'] is False
    assert report['recorded_action_status'] == report['physical_execution_status'] == 'unknown'
    assert named_actions.audit_named_actions(root)['status'] == 'unknown'


def test_explicit_equality_requirement_with_first_original_locator(packet):
    root = packet()
    report = named_actions.audit_named_actions(root, **contract(require=True))
    assert report['status'] == report['acceptance_contract']['status'] == 'error'
    first = report['acceptance_contract']['violations'][0]
    assert first['tick_id'] == 0 and first['field'] == 'canonical_matches_driver_return'
    assert first['evidence'][0]['sha256'] == hashlib.sha256((root / 'capture.json').read_bytes()).hexdigest()
    assert any('/ticks/0/' in item['pointer'] for item in first['evidence'])
    assert named_actions.audit_named_actions(packet('healthy', limited=False), **contract(require=True))['status'] == 'pass'


def test_name_and_namespace_criteria_are_explicit(packet):
    root = packet()
    change(root, lambda d: d.update(engine_action_names=['b.pos', 'a.pos']))
    report = named_actions.audit_named_actions(root, **contract())
    assert report['acceptance_contract']['status'] == 'error'
    assert report['acceptance_contract']['violations'][0]['field'] == 'engine_action_names'
    assert named_actions.audit_named_actions(root)['status'] == 'unknown'
    expected = contract()
    expected['expected_driver_namespace'] = 'different-driver'
    assert named_actions.audit_named_actions(root, **expected)['status'] == 'error'


def test_canonical_mutation_remains_a_separate_observation(packet):
    root = packet()
    change(root, lambda d: d['ticks'][0]['canonical_at_return'].update({'a.pos': 1.}))
    result = named_actions.audit_named_actions(root, **contract())
    assert result['status'] == 'pass'
    assert result['canonical_mutation_events'] == 1


@pytest.mark.parametrize('mode', ['coverage', 'source', 'relation', 'missing_return', 'not_sent', 'driver_error'])
def test_missing_or_unqualified_stage_never_certifies_equality_fault(packet, monkeypatch, mode):
    root = packet()
    if mode == 'source':
        monkeypatch.setattr(named_actions, 'SOURCE_HASHES', {**named_actions.SOURCE_HASHES, 'core.py': '0' * 64})
    elif mode == 'coverage':
        change(root, lambda d: d['coverage'].update(complete_target_call_history=False))
    elif mode == 'relation':
        change(root, lambda d: d['ticks'][0]['relations'].update(processor_return_to_wrapper_input=False))
    elif mode == 'missing_return':
        change(root, lambda d: d['ticks'][0].update(driver_return=None))
    else:
        def mutate(d):
            t = d['ticks'][0]
            t.update(completion='normal' if mode == 'not_sent' else 'error', outcome='not_sent' if mode == 'not_sent' else 'error',
                     driver_called=mode != 'not_sent', driver_completed=False, wrapper_completed=False, ack_count=0,
                     driver_return=None)
            t['relations'] = dict.fromkeys(named_actions.RELATIONS)
        change(root, mutate)
    report = named_actions.audit_named_actions(root, **contract(require=True))
    assert report['acceptance_contract']['status'] == 'unknown'
    assert report['status'] == 'unknown'


def test_intrinsic_mapping_contradiction_separate_from_consumer_requirement(packet):
    root = packet()
    change(root, lambda d: d['ticks'][0]['canonical_at_processor'].update({'a.pos': 99.}))
    report = named_actions.audit_named_actions(root, **contract(require=True))
    assert report['numeric_replay_status'] == report['status'] == 'error'
    assert report['acceptance_contract']['status'] == 'unknown'


@pytest.mark.parametrize('value', [1, 0, 'true', None, [], {}])
def test_equality_requirement_is_strict_boolean(packet, value):
    with pytest.raises(InputError, match='Boolean'):
        named_actions.audit_named_actions(packet(), require_canonical_matches_driver_return=value)


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(action_names=['a.pos', 'a.pos']),
    lambda d: d['ticks'][0].update(ack_count=True),
    lambda d: d['ticks'][0]['driver_return'].update({'a.pos': True}),
    lambda d: d['ticks'][0].update(tick_id=True),
    lambda d: d['ticks'][0]['relations'].update(interp_to_canonical=1),
    lambda d: d['ticks'][0].update(driver_completed=False),
])
def test_malformed_evidence_refused(packet, mutation):
    root = packet()
    change(root, mutation)
    with pytest.raises(InputError):
        named_actions.audit_named_actions(root)


def test_hash_tree_guards_and_final_drift(packet, monkeypatch, tmp_path):
    root = packet()
    original = named_actions._analyze
    def during(*args, **kwargs):
        report = original(*args, **kwargs)
        change(root, lambda d: d.update(extra='changed during audit'))
        return report
    monkeypatch.setattr(named_actions, '_analyze', during)
    with pytest.raises(InputError, match='changed'):
        named_actions.audit_named_actions(root, **contract())
    monkeypatch.setattr(named_actions, '_analyze', original)
    (root / 'sources' / 'driver_source.py').write_text('tampered')
    with pytest.raises(InputError, match='hash'):
        named_actions.audit_named_actions(root)
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(InputError, match='symlink'):
        named_actions.audit_named_actions(alias)


def test_duplicate_json_keys_and_nonfinite_refused(packet):
    root = packet()
    data = json.loads((root / 'capture.json').read_text())
    data['ticks'][0]['driver_return']['a.pos'] = float('nan')
    commit(root, data)
    with pytest.raises(InputError, match='finite'):
        named_actions.audit_named_actions(root)


def test_same_saved_equality_contract_reused_after_relocation_and_new_values(packet, tmp_path):
    from policy_doctor.incidents import (
        check_incident_candidate,
        export_incident_case,
        reproduce_incident_case,
    )
    before, after = packet('before'), packet('after', limited=False)
    healthy, recurrence = packet('later', limited=False, offset=10.), packet('fault', offset=10.)
    unknown = packet('unknown', limited=False)
    change(unknown, lambda d: d['coverage'].update(complete_target_call_history=False))
    case = tmp_path / 'case'
    result = export_incident_case('native_named_action', before, case,
        {'require_canonical_matches_driver_return': True}, corrected=after,
        controls=[{'name': 'missing', 'source': unknown, 'parameters': {'require_canonical_matches_driver_return': True}, 'expected_status': 'unknown'}])
    assert result['status'] == 'pass'
    manifest = json.loads((case / 'manifest.json').read_text())
    assert manifest['parameters']['require_canonical_matches_driver_return'] is True
    assert manifest['parameters']['expected_action_names'] == ['a.pos', 'b.pos']
    moved = tmp_path / 'moved'
    case.rename(moved)
    shutil.rmtree(before)
    shutil.rmtree(after)
    assert reproduce_incident_case(moved)['status'] == 'pass'
    for path, status in [(healthy, 'pass'), (recurrence, 'error'), (unknown, 'unknown')]:
        assert check_incident_candidate(moved, path)['status'] == status


def test_no_implicit_equality_incident_and_no_weakened_control(packet, tmp_path):
    from policy_doctor.incidents import export_incident_case
    before, after = packet('before'), packet('after', limited=False)
    with pytest.raises(InputError):
        export_incident_case('native_named_action', before, tmp_path / 'no-implicit', {}, corrected=after)
    with pytest.raises(InputError, match='same|identical'):
        export_incident_case('native_named_action', before, tmp_path / 'weak', contract(require=True), corrected=after,
            controls=[{'name': 'weakened', 'source': before, 'parameters': contract(require=False), 'expected_status': 'pass'}])


def test_mutating_decorator_bridge_is_unknown_not_intrinsic_fault(packet):
    root = packet()
    change(root, lambda d: d['ticks'][0]['driver_request'].update({'a.pos': 3.}))
    result = named_actions.audit_named_actions(root, **contract(require=True))
    assert result['status'] == result['numeric_replay_status'] == 'unknown'
    assert result['acceptance_contract']['status'] == 'unknown'
    assert result['first_intrinsic_contradiction'] is None


def test_explicit_robot_transform_semantics_unknown_and_changed_values(packet):
    result = named_actions.audit_named_actions(packet(), **contract())
    assert result['robot_processor_semantics_status'] == 'unknown'
    difference = result['first_driver_change']
    assert difference['changed_channel_count'] == 1
    assert difference['channels'][0] == {'name': 'a.pos', 'driver_request': 2., 'driver_return': 1.}
    assert difference['evidence'][0]['pointer'] == '/ticks/0/driver_request'


def test_existing_cli_audit_and_contract_file_guards(packet, tmp_path, capsys):
    from policy_doctor.cli import human, main
    root = packet()
    path = tmp_path / 'criterion.json'
    path.write_text(json.dumps(contract(require=True)))
    assert main(['audit-capture', str(root), '--named-contract', str(path), '--json']) == 1
    result = json.loads(capsys.readouterr().out)
    assert result['kind'] == 'named_action_audit' and result['acceptance_contract']['status'] == 'error'
    assert 'Legal limiting' in human(result) and 'Software driver namespace' in human(result)
    assert main(['audit-capture', str(root), '--json']) == 3
    capsys.readouterr()
    for value in [[], {'atol': 0.}, {'require_canonical_matches_driver_return': 1}]:
        path.write_text(json.dumps(value))
        output = tmp_path / 'must-not-exist.json'
        assert main(['audit-capture', str(root), '--named-contract', str(path), '--output', str(output)]) == 2
        assert not output.exists()
        capsys.readouterr()


def test_existing_cli_create_relocate_and_later_check(packet, tmp_path, capsys):
    from policy_doctor.cli import human, main
    before, after = packet('before'), packet('after', limited=False)
    later, fault, unknown = packet('later', limited=False, offset=10), packet('fault', offset=10), packet('unknown')
    change(unknown, lambda d: d['coverage'].update(complete_target_call_history=False))
    path = tmp_path / 'criterion.json'
    path.write_text(json.dumps({'require_canonical_matches_driver_return': True}))
    case = tmp_path / 'case'
    assert main(['create-case', 'native-named-action', str(before), str(case), '--corrected', str(after),
                 '--named-contract', str(path), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['checks']['before']['acceptance_contract']['status'] == 'error'
    assert 'Supplied named-action snapshots' in human(result)
    moved = tmp_path / 'colleague'
    case.rename(moved)
    shutil.rmtree(before)
    shutil.rmtree(after)
    assert main(['reproduce-case', str(moved), '--json']) == 0
    capsys.readouterr()
    for candidate, code, status in [(later, 0, 'pass'), (fault, 1, 'error'), (unknown, 3, 'unknown')]:
        assert main(['check-case', str(moved), str(candidate), '--json']) == code
        report = json.loads(capsys.readouterr().out)
        assert report['status'] == status and report['candidate']['kind'] == 'named_action_audit'
        assert 'named' in human(report).lower()


def test_existing_mcp_catalog_and_named_contract_creation_guards(packet, tmp_path):
    pytest.importorskip('mcp')
    from policy_doctor.server import create_server
    before, after = packet('before'), packet('after', limited=False)
    tools = create_server([str(tmp_path)])._tool_manager._tools
    assert len(tools) == 29 and 'create_incident_case' not in tools
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
    audit = tools['audit_runtime_capture'].fn
    assert audit(str(before), named_contract=contract(require=True))['status'] == 'error'
    with pytest.raises(InputError, match='Boolean'):
        audit(str(before), named_contract={'require_canonical_matches_driver_return': 1})
    with pytest.raises(InputError, match='outside'):
        audit('/outside', named_contract=contract())
    output = tmp_path / 'output'
    output.mkdir()
    tools = create_server([str(tmp_path)], case_root=str(output))._tool_manager._tools
    assert len(tools) == 31
    assert {'dataset_health_card', 'triage_flight_recording', 'assess_upgrade_impact', 'check_training_targets'} <= set(tools)
    creator = tools['create_incident_case'].fn
    case = output / 'case'
    assert creator('native_named_action', str(before), str(case), {'require_canonical_matches_driver_return': True}, corrected=str(after))['status'] == 'pass'
    assert tools['reproduce_incident_case'].fn(str(case))['status'] == 'pass'
    assert tools['check_incident_candidate'].fn(str(case), str(after))['status'] == 'pass'
    with pytest.raises(InputError, match='strictly below'):
        creator('native_named_action', str(before), str(tmp_path / 'outside-output'), {}, corrected=str(after))


def test_dispatch_rejects_foreign_criteria_and_preserves_strict_input(packet):
    from policy_doctor.capture import audit_capture
    root = packet()
    assert audit_capture(str(root), named_contract=contract())['status'] == 'pass'
    for arguments in [{'dataset': 'anything'}, {'expected_names': ['a']}, {'max_query_gap': 1},
                      {'command_contract': {}}, {'named_contract': []}, {'named_contract': {'unknown': 1}}]:
        with pytest.raises(InputError):
            audit_capture(str(root), **arguments)


def test_native_integer_precision_is_not_silently_lost(packet):
    root = packet()
    change(root, lambda d: d['ticks'][0]['driver_return'].update({'a.pos': 2 ** 53 + 1}))
    with pytest.raises(InputError, match='finite'):
        named_actions.audit_named_actions(root)


def test_legacy_missing_wrapper_return_cannot_certify_equality(packet):
    root = packet(limited=False)
    change(root, lambda d: d['ticks'][0].pop('wrapper_return'))
    result = named_actions.audit_named_actions(root, **contract(require=True))
    assert result['status'] == result['binding_status'] == 'unknown'
    assert result['acceptance_contract']['status'] == 'unknown'


def test_return_decorator_mutation_is_unknown_not_fault_or_equality_pass(packet):
    root = packet(limited=False)
    change(root, lambda d: d['ticks'][0]['wrapper_return'].update({'a.pos': 1.}))
    result = named_actions.audit_named_actions(root, **contract(require=True))
    assert result['status'] == result['binding_status'] == 'unknown'
    assert result['numeric_replay_status'] == 'unknown'
    assert result['acceptance_contract']['status'] == 'unknown'
    assert result['first_intrinsic_contradiction'] is None
