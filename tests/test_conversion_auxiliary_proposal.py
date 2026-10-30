"""Generated-only expanded reader controls, with unchanged native source fragments."""
import ast
import hashlib
import json
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pytest
from test_conversion_fidelity_reader_proposal import FIXTURES, ROOT, Generated

BINDINGS = [
    ('teleop_base_height_command', 'teleop_base_height_command_name_sim', 'base_height_cmd', 'teleop.base_height_command', 1),
    ('teleop_navigate_command', 'teleop_navigate_command_name_sim', 'navigate_cmd', 'teleop.navigate_command', 3),
    ('teleop_torso_orientation_rpy_command', 'teleop_torso_orientation_rpy_command_name_sim', 'torso_orientation_rpy_cmd', 'teleop.torso_orientation_rpy_command', 3),
]


def native_fragments(config, length):
    """Execute only hash-pinned unchanged helper/formula AST, not module/constructor.

    This is an explicitly invoked test oracle. Product code never executes any
    supplied converter/config. It is not a native whole-converter/model run.
    """
    logical = ROOT + 'lerobot/convert_hdf5_to_lerobot.py'
    expected = json.loads((FIXTURES / 'UPSTREAM.json').read_bytes())['files'][logical]
    raw = (FIXTURES / expected['fixture_path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == expected['sha256']
    tree = ast.parse(raw)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extract_teleop_command')
    assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id != 'data':
            continue
        index = target.slice
        if (isinstance(index, ast.Constant) and index.value in {'timestamp', 'task_index'}) or (
                isinstance(index, ast.Subscript) and isinstance(index.value, ast.Name) and index.value.id == 'annotation_keys'):
            assignments.append(node)
    assert len(assignments) == 3
    namespace = {'np': np, 'h5py': h5py, 'Gr00tDatasetConfig': object, 'Any': Any,
                 'config': SimpleNamespace(**config), 'length': length, 'data': {},
                 'annotation_keys': (config['task_description_lerobot'],)}
    exec(compile(ast.Module(body=[helper, *assignments], type_ignores=[]), logical, 'exec'), namespace)
    return namespace


@pytest.fixture
def generated(tmp_path):
    return Generated(tmp_path)


def test_all_three_auxiliary_fields_and_explicit_native_scope(generated):
    result = generated.check()
    assert result['status'] == result['joint_numeric_association_status'] == result['auxiliary_numeric_association_status'] == 'pass', result
    assert result['requested_scope'] == 'state_action_and_teleop_numeric_conversion'
    assert result['criterion']['schema'] == 'arena43_with_teleop_numeric_v2'
    assert result['coverage']['requested_base_height_navigation_dimensions'] == 4
    assert result['coverage']['requested_additional_torso_dimensions'] == 3
    assert result['native_consumption']['status'] == 'unknown' and not result['native_consumption']['requested']
    assert result['pixel_fidelity']['status'] == 'unknown'
    for _, _, _, column, width in BINDINGS:
        field = result['auxiliary_fields'][column]
        assert field['status'] == 'pass' and field['width'] == width
        assert field['compared_rows'] == field['expected_rows'] == 12
    assert len(result['auxiliary_episode_records']) == 6
    assert all(r['expected_sha256'] == r['observed_sha256'] for r in result['auxiliary_episode_records'])


@pytest.mark.parametrize('binding', BINDINGS)
def test_native_unchanged_helper_exact_trim_and_dtype(generated, binding):
    logical, _, source, column, _ = binding
    config = {**generated.config, 'hdf5_keys': {b[0]: b[2] for b in BINDINGS}}
    oracle = native_fragments(config, 7)
    with h5py.File(generated.original, 'r') as file:
        actual = oracle['extract_teleop_command'](file['data/demo_z'], logical, SimpleNamespace(**config))
    expected = generated.auxiliary[0][source][:-1]
    assert np.asarray(actual).dtype == expected.dtype
    assert np.array_equal(actual, expected)
    table = generated.table(0)
    assert np.array_equal(np.asarray(table[column].to_pylist(), dtype='float32'), actual)


@pytest.mark.parametrize('column', [b[3] for b in BINDINGS])
def test_shape_valid_auxiliary_fault_same_criterion_and_correction(generated, column):
    healthy = generated.check()
    table = generated.table(0)
    values = np.asarray(table[column].to_pylist(), dtype='float32')
    values = np.concatenate([values[1:], values[-1:]])
    generated.write(0, table.set_column(table.schema.get_field_index(column), column, pa.array(values.tolist(), type=pa.list_(pa.float32()))))
    result = generated.check()
    assert result['status'] == result['auxiliary_numeric_association_status'] == 'error', result
    assert result['joint_numeric_association_status'] == 'pass'
    assert result['criterion_sha256'] == healthy['criterion_sha256']
    findings = [f for f in result['findings'] if f['code'] == 'conversion.auxiliary_numeric_association']
    assert len(findings) == 6 and all(len(f['evidence']) == 2 for f in findings)
    assert all('row ' in f['message'] and 'source frame ' in f['message'] for f in findings)
    generated.restore()
    corrected = generated.check()
    assert corrected['status'] == 'pass' and corrected['criterion_sha256'] == result['criterion_sha256']


@pytest.mark.parametrize('change', ['missing', 'width', 'inner_null', 'dtype', 'nonfinite', 'permutation', 'wrong_source'])
def test_auxiliary_export_contract_and_same_width_corruption(generated, change):
    column = 'teleop.navigate_command'
    table = generated.table(0)
    if change == 'missing':
        table = table.drop([column])
    else:
        values = table[column].to_pylist()
        if change == 'width':
            values = [row[:-1] for row in values]
        elif change == 'inner_null':
            values[0][0] = None
        elif change == 'nonfinite':
            values[0][0] = float('nan')
        elif change == 'permutation':
            values = [list(reversed(row)) for row in values]
        elif change == 'wrong_source':
            values = table['teleop.torso_orientation_rpy_command'].to_pylist()
        dtype = pa.float64() if change == 'dtype' else pa.float32()
        table = table.set_column(table.schema.get_field_index(column), column, pa.array(values, type=pa.list_(dtype)))
    generated.write(0, table)
    result = generated.check()
    assert result['status'] == result['auxiliary_numeric_association_status'] == 'error', result
    assert result['joint_numeric_association_status'] == 'pass'
    assert not result['coverage']['whole_numeric_source_to_export']


def test_missing_auxiliary_source_keeps_independent_joint_error(generated):
    with h5py.File(generated.original, 'r+') as file:
        del file['data/demo_z/action/navigate_cmd']
    table = generated.table(0)
    values = np.asarray(table['action'].to_pylist(), dtype='float32') + 1
    generated.write(0, table.set_column(1, 'action', pa.array(values.tolist(), type=pa.list_(pa.float32()))))
    result = generated.check()
    assert result['status'] == result['joint_numeric_association_status'] == 'error', result
    assert result['auxiliary_numeric_association_status'] == 'unknown'
    assert result['auxiliary_fields']['teleop.navigate_command']['status'] == 'unknown'
    assert result['source_membership']['status'] == 'unknown'
    assert result['counts']['unknown'] > 0 and result['counts']['error'] > 0


def test_invalid_source_aux_does_not_hide_export_nonfinite_error(generated):
    with h5py.File(generated.original, 'r+') as file:
        del file['data/demo_z/action/navigate_cmd']
    table = generated.table(0)
    values = table['teleop.navigate_command'].to_pylist()
    values[0][0] = float('inf')
    generated.write(0, table.set_column(table.schema.get_field_index('teleop.navigate_command'), 'teleop.navigate_command', pa.array(values, type=pa.list_(pa.float32()))))
    result = generated.check()
    assert result['status'] == 'error' and result['counts']['unknown'] > 0, result
    assert result['auxiliary_fields']['teleop.navigate_command']['status'] == 'error'


def test_lawful_auxiliary_terminal_drop_does_not_compare_unused_terminal(generated):
    with h5py.File(generated.original, 'r+') as file:
        file['data/demo_z/action/navigate_cmd'][-1, :] = np.nan
    result = generated.check()
    assert result['status'] == 'pass', result


@pytest.mark.parametrize('task,fps,instruction', [(0, 1, 'stack the cups'), (37, 30, '将苹果放到盘子上'), (2**63-1, 1000, 'follow the declared task')])
def test_alternate_task_fps_instruction_matches_native_formulas(generated, task, fps, instruction):
    old = generated.check()['criterion_sha256']
    generated.config.update(task_index=task, fps=fps, language_instruction=instruction)
    generated.write_config()
    for episode in generated.episodes:
        episode['tasks'] = [instruction]
    generated.write_metadata()
    for eid in generated.arrays:
        table = generated.table(eid)
        oracle = native_fragments(generated.config, len(table))
        for name, values in oracle['data'].items():
            expected = table[name].to_numpy()
            assert expected.dtype == values.dtype and np.array_equal(expected, values)
            table = table.set_column(table.schema.get_field_index(name), name, pa.array(values))
        generated.write(eid, table)
    result = generated.check()
    assert result['status'] == 'pass' and result['criterion_sha256'] != old, result
    assert result['criterion']['actual_config']['language_instruction'] == instruction
    assert result['pixel_fidelity']['status'] == result['native_consumption']['status'] == 'unknown'


@pytest.mark.parametrize('change', ['annotation', 'task_index', 'timestamp', 'task_text', 'episode_text', 'info_fps'])
def test_declared_task_and_time_contradictions_remain_errors(generated, change):
    if change in {'annotation', 'task_index', 'timestamp'}:
        name = {'annotation': generated.config['task_description_lerobot'], 'task_index': 'task_index', 'timestamp': 'timestamp'}[change]
        table = generated.table(0)
        generated.write(0, table.set_column(table.schema.get_field_index(name), name, pa.array(table[name].to_numpy() + 1)))
    elif change == 'task_text':
        (generated.export / 'meta/tasks.jsonl').write_text(json.dumps({'task_index': 3, 'task': 'different instruction'}) + '\n')
    elif change == 'episode_text':
        generated.episodes[0]['tasks'] = ['different instruction']
        generated.write_metadata()
    else:
        p = generated.export / 'meta/info.json'
        value = json.loads(p.read_bytes())
        value['fps'] = 30
        p.write_text(json.dumps(value))
    result = generated.check()
    assert result['status'] == 'error' and not result['coverage']['whole_numeric_source_to_export'], result


@pytest.mark.parametrize('field,value', [('task_index', True), ('task_index', -1), ('task_index', 2**63), ('task_index', 1.5), ('fps', True), ('fps', 0), ('fps', 1001), ('fps', 29.97), ('language_instruction', ''), ('language_instruction', '   '), ('language_instruction', '界' * 1366), ('language_instruction', '\ud800')])
def test_unsupported_scalar_values_unknown_before_original_arrays(generated, field, value):
    generated.config[field] = value
    generated.write_config()
    result = generated.check()
    assert result['status'] == 'unknown', result
    assert not any(a.get('kind') == 'original_hdf5' for a in result['artifacts'])


def test_known_joint_error_is_actionable_after96_auxiliary_unknowns(generated):
    with h5py.File(generated.original, 'r+') as file:
        data = file['data']
        for eid in range(3, 33):
            key = 'generated_' + str(eid)
            group = data.create_group(key)
            action = np.arange(2 * 43, dtype='float32').reshape(2, 43) + eid
            state = action + 100
            group.create_group('obs').create_dataset('robot_joint_pos', data=state)
            group.create_dataset('processed_actions', data=action)
            generated.arrays[eid] = (key, state, action)
            generated.auxiliary[eid] = {'base_height_cmd': action[:, :1], 'navigate_cmd': action[:, :3], 'torso_orientation_rpy_cmd': action[:, :3]}
        for key, _, _ in generated.arrays.values():
            if 'action' in data[key]:
                del data[key]['action']
    generated.episodes = [{'episode_index': eid, 'length': len(action) - 1, 'tasks': [generated.config['language_instruction']]}
                          for eid, (_, _, action) in generated.arrays.items()]
    generated.write_metadata()
    generated.restore()
    table = generated.table(32)
    values = np.asarray(table['action'].to_pylist(), dtype='float32') + 1
    generated.write(32, table.set_column(1, 'action', pa.array(values.tolist(), type=pa.list_(pa.float32()))))
    result = generated.check()
    assert result['status'] == result['joint_numeric_association_status'] == 'error'
    assert result['auxiliary_numeric_association_status'] == 'unknown'
    assert result['counts'] == {'error': 1, 'unknown': 96}, result
    assert result['findings_total'] == 97 and result['findings_truncated']
    assert len(result['findings']) == 64
    assert result['findings'][0]['code'] == 'conversion.numeric_joint_association'
    assert len(result['findings'][0]['evidence']) == 2


def test_bounded_collector_retains_diverse_error_witnesses():
    from policy_doctor.conversion_fidelity_worker import _keep_finding
    findings = []
    for row in range(100):
        _keep_finding(findings, {'status': 'error', 'code': 'conversion.auxiliary_numeric_association', 'column': 'teleop.navigate_command', 'row': row})
    joint = {'status': 'error', 'code': 'conversion.numeric_joint_association', 'role': 'action', 'evidence': ['original', 'candidate']}
    _keep_finding(findings, joint)
    assert len(findings) == 64 and findings[1] == joint
