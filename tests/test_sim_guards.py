import hashlib
import json
import os

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from policy_doctor.artifacts import InputError
from policy_doctor.sim_guards import check_sim_provenance, remedy_list, remedy_lists_digest

DELTA_OSC = {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
             'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'control_delta': True}
CTRL = 'robomimic.controller_vs_actions'


def codes(report, status=None):
    return {f['code'] for f in report['findings'] if status is None or f['status'] == status}


def finding(report, code):
    return next(f for f in report['findings'] if f['code'] == code)


def check_contract(report):
    assert set(report) == {'schema_version', 'kind', 'status', 'counts', 'coverage', 'findings', 'remedy_lists', 'scope'}
    assert report['kind'] == 'sim_provenance' and report['schema_version'] == 2
    assert all('_lists' not in v for v in report['coverage'].values() if isinstance(v, dict))  # hoisted
    lists = report['remedy_lists']
    for key, groups in lists.items():  # every group refers to stored name lists
        if key != 'names':
            for g in groups:
                assert g['datasets'] in lists['names'] and g.get('demos', 'all') in (*lists['names'], 'all')
    assert sum(report['counts'].values()) == len(report['findings'])
    order = ['error', 'warning', 'unknown', 'pass']
    assert report['status'] == min((f['status'] for f in report['findings']), key=order.index)
    for f in report['findings']:
        assert set(f) == {'code', 'status', 'message', 'remedy', 'evidence'}
        for e in f['evidence']:
            assert set(e) == {'path', 'sha256', 'pointer', 'value'} and len(e['sha256']) == 64
    json.dumps(report)
    return report


# ---------------------------------------------------------------------------------------------- Isaac Lab

def yaw(deg, order):
    h = np.radians(deg) / 2
    return [np.cos(h), 0, 0, np.sin(h)] if order == 'WXYZ' else [0, 0, np.sin(h), np.cos(h)]


def isaac_file(path, *, order='WXYZ', version=None, robot=None, extra_quat=True, demos=2, attrs=None, orders=None,
               robots=None):
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        if version is not None:
            f.attrs['format_version'] = version
        for key, value in (attrs or {}).items():
            f.attrs[key] = value
        data = f.create_group('data')
        data.attrs['env_args'] = json.dumps({'env_name': 'Isaac-Test-v0', 'type': 2})
        for d in range(demos):
            o = orders[d] if orders else order
            q = np.array(robots[o] if robots else robot if robot is not None else yaw(0, o), dtype=np.float32)
            g = data.create_group(f'demo_{d}')
            g.attrs['num_samples'] = 5
            g.create_dataset('actions', data=rng.normal(size=(5, 8)).astype(np.float32) + d)
            pose = np.concatenate([[0.1 * d, 0, 0], q]).astype(np.float32)
            g.create_dataset('initial_state/articulation/robot/root_pose', data=pose[None])
            g.create_dataset('states/articulation/robot/root_pose', data=np.repeat(pose[None], 5, 0))
            cube = np.concatenate([[0.5, 0, 0], yaw(30 + 10 * d, o)]).astype(np.float32)
            g.create_dataset('initial_state/rigid_object/cube/root_pose', data=cube[None])
            if extra_quat:
                g.create_dataset('obs/eef_quat', data=np.array([yaw(20 * t + d, o) for t in range(5)], dtype=np.float32))
            g.create_dataset('camera_obs/rgb', data=np.zeros((5, 4, 4, 3), np.uint8))
    return path


def copy_without_root_attrs(src, dst):
    """What IsaacLab merge_hdf5_datasets.py L29-41 does: copy data/<demo> and env_args only."""
    with h5py.File(src, 'r') as a, h5py.File(dst, 'w') as b:
        for name in a['data']:
            a.copy(f'data/{name}', b, f'data/{name}')
        b['data'].attrs['env_args'] = a['data'].attrs['env_args']
    return dst


def convert_root_pose_only(src, dst, *, stamp=1, keep_source_attrs=False):
    """What IsaacLab convert_dataset_to_xyzw L298-332 does (root_pose reordered only)."""
    with h5py.File(src, 'r') as a, h5py.File(dst, 'w') as b:
        b.attrs['format_version'] = stamp
        if keep_source_attrs:
            for k, v in a.attrs.items():
                b.attrs[k] = v

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                value = obj[...]
                if name.endswith('root_pose') and value.shape[-1] == 7:
                    value = np.concatenate([value[..., :3], np.roll(value[..., 3:7], -1, axis=-1)], -1)
                b.create_dataset(name, data=value)
        a.visititems(visit)
        b['data'].attrs['env_args'] = a['data'].attrs['env_args']
    return dst


def test_decisive_legacy_file_prescribes_a_whole_file_conversion(tmp_path):
    report = check_contract(check_sim_provenance(isaac_file(tmp_path / 'legacy.hdf5')))
    assert finding(report, 'isaaclab.format_version')['status'] == 'pass'  # robot upright only as WXYZ
    warn = finding(report, 'isaaclab.quaternions_outside_root_pose')
    assert warn['status'] == 'warning' and 'obs/eef_quat' in warn['message'] and 'np.roll' in warn['remedy']
    assert report['coverage']['source']['content_order_inference'] == 'WXYZ'


def test_d1_undecidable_unversioned_file_gets_one_consistent_remedy(tmp_path):
    path = isaac_file(tmp_path / 'merged.hdf5', order='XYZW')  # XYZW identity roots: undecidable, version dropped
    report = check_contract(check_sim_provenance(path))
    fv = finding(report, 'isaaclab.format_version')
    outside = finding(report, 'isaaclab.quaternions_outside_root_pose')
    assert fv['status'] == 'warning' and 'stamp format_version=1' in fv['remedy'] and 'change no data' in fv['remedy']
    assert outside['status'] == 'warning' and outside['message'].startswith('[source] Conditional')
    assert 'np.roll' not in outside['remedy'] and 'Change no data' in outside['remedy']
    assert 'reorder' not in ' '.join(f['remedy'].lower() for f in report['findings'] if f['status'] != 'pass')


def test_d4_versioned_file_with_undecidable_demos_is_unknown_not_pass(tmp_path):
    report = check_contract(check_sim_provenance(isaac_file(tmp_path / 'v1.hdf5', order='XYZW', version=1)))
    f = finding(report, 'isaaclab.format_version')
    assert f['status'] == 'unknown' and 'confirm it only partly or not at all' in f['message']
    assert report['status'] == 'unknown' and report['coverage']['source']['content_order_inference'] == 'undecidable'


def test_versioned_file_passes_when_every_demo_confirms(tmp_path):
    path = isaac_file(tmp_path / 'v1.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'), extra_quat=False)
    f = finding(check_contract(check_sim_provenance(path)), 'isaaclab.format_version')
    assert f['status'] == 'pass' and 'robot roots of all 2 demos agree' in f['message']


def test_unversioned_xyzw_content_gets_a_conditional_stamp_remedy(tmp_path):
    path = isaac_file(tmp_path / 'u.hdf5', order='XYZW', robot=yaw(120, 'XYZW'), extra_quat=False)
    f = finding(check_contract(check_sim_provenance(path)), 'isaaclab.declared_order_vs_content')
    assert f['status'] == 'warning' and 'only as XYZW' in f['message'] and 'mounted tilted' in f['message']
    assert f['remedy'].startswith('If the recording is Isaac Lab >= 3.0, stamp format_version=1')


def test_d6_tilted_robot_in_a_correct_v1_file_gets_no_rotation_claim_or_remedy(tmp_path):
    ceiling_xyzw = [1.0, 0, 0, 0]  # XYZW: 180 deg about x (ceiling-mounted); reads upright only as WXYZ
    path = isaac_file(tmp_path / 'ceiling.hdf5', order='XYZW', version=1, robot=ceiling_xyzw, extra_quat=False)
    f = finding(check_contract(check_sim_provenance(path)), 'isaaclab.declared_order_vs_content')
    assert f['status'] == 'warning' and 'mounted tilted' in f['message'] and 'loads a version-1 file verbatim' in f['message']
    assert 'will load' not in f['message'] and f['remedy'].startswith('Change nothing on this evidence alone')
    assert 'the rest are undecidable' not in f['message']  # every inspected demo decided


DECISIVE = {'XYZW': yaw(120, 'XYZW'), 'WXYZ': yaw(0, 'WXYZ')}  # each reads upright only in its own order


def test_d4_mixed_quaternion_order_across_demos_is_reported(tmp_path):
    path = isaac_file(tmp_path / 'mixed.hdf5', order='XYZW', robots=DECISIVE, extra_quat=False, demos=4,
                      orders=['XYZW', 'XYZW', 'WXYZ', 'WXYZ'])
    report = check_contract(check_sim_provenance(path))
    assert finding(report, 'isaaclab.mixed_order_across_demos')['status'] == 'warning'
    assert 'isaaclab.format_version' not in codes(report, 'pass')
    assert report['coverage']['source']['content_order_inference'] == 'mixed'


def test_d4_partly_decided_unversioned_file_does_not_pass(tmp_path):
    path = isaac_file(tmp_path / 'partly.hdf5', order='XYZW', extra_quat=False, demos=4,
                      orders=['XYZW', 'XYZW', 'WXYZ', 'WXYZ'])  # identity roots: XYZW undecidable, WXYZ decisive
    report = check_contract(check_sim_provenance(path))
    f = finding(report, 'isaaclab.format_version')
    assert f['status'] == 'warning' and '2 of 4 inspected demos' in f['message']


def test_d4_demos_are_sampled_over_the_whole_file(tmp_path):
    path = isaac_file(tmp_path / 'tail.hdf5', order='XYZW', robots=DECISIVE, extra_quat=False, demos=20,
                      orders=['WXYZ'] * 16 + ['XYZW'] * 4)
    report = check_sim_provenance(path, max_demos=8)
    assert report['coverage']['source']['demos_inspected'] == 8
    assert 'isaaclab.mixed_order_across_demos' in codes(report)


@pytest.mark.parametrize('value, status', [('one', 'error'), ('v1', 'error'), (2, 'unknown'), (True, 'pass'),
                                           (1.0, 'pass'), ('1', 'pass'), (np.bytes_(b'1'), 'pass'),
                                           (np.array([1]), 'pass'), (1.5, 'pass')])
def test_d10_format_version_is_read_as_isaac_lab_reads_it(tmp_path, value, status):
    path = isaac_file(tmp_path / 'v.hdf5', order='XYZW', version=value, robot=yaw(120, 'XYZW'))  # roots decide
    f = finding(check_contract(check_sim_provenance(path)), 'isaaclab.format_version')
    assert f['status'] == status


@pytest.mark.parametrize('value', [0, -1])
def test_d10_explicit_versions_below_one_warn_about_double_conversion(tmp_path, value):
    path = isaac_file(tmp_path / 'v.hdf5', version=value, extra_quat=False)
    f = finding(check_contract(check_sim_provenance(path)), 'isaaclab.format_version')
    assert f['status'] == 'warning' and 'Delete the attribute' in f['remedy']


def test_d1_converted_v0_file_gets_no_delete_or_reorder_remedy(tmp_path):
    src = isaac_file(tmp_path / 'v0.hdf5', version=0)
    with h5py.File(src, 'r+') as f:
        f['data'].attrs['isaaclab_version'] = '2.2.0'
    dst = convert_root_pose_only(src, tmp_path / 'v0_out.hdf5', keep_source_attrs=True)  # keeps 0, root now XYZW
    with h5py.File(dst, 'r+') as f:
        f['data'].attrs['isaaclab_version'] = '2.2.0'
    report = check_contract(check_sim_provenance(dst))
    fv = finding(report, 'isaaclab.format_version')
    assert fv['remedy'].startswith('Establish whether root_pose was already converted')
    outside = finding(report, 'isaaclab.quaternions_outside_root_pose')
    assert outside['message'].startswith('[source] Conditional') and 'np.roll' not in outside['remedy']


def test_converted_2x_recording_still_warns_conditionally(tmp_path):
    src = isaac_file(tmp_path / 'legacy.hdf5')
    dst = convert_root_pose_only(src, tmp_path / 'xyzw.hdf5')
    with h5py.File(dst, 'r+') as f:
        f['data'].attrs['isaaclab_version'] = '2.2.0'
    f = finding(check_contract(check_sim_provenance(dst)), 'isaaclab.quaternions_outside_root_pose')
    assert f['status'] == 'warning' and 'recorded by Isaac Lab 2.2.0' in f['message'] and 'np.roll' not in f['remedy']
    assert any(e['pointer'] == '/data@isaaclab_version' for e in f['evidence'])


def test_3x_recording_without_format_version_is_flagged(tmp_path):
    path = isaac_file(tmp_path / 'u.hdf5', order='XYZW', extra_quat=False, attrs={'isaaclab_version': '3.0.0'})
    f = finding(check_contract(check_sim_provenance(path)), 'isaaclab.declared_order_vs_content')
    assert f['status'] == 'warning' and 'Recorded by Isaac Lab 3.0.0' in f['message']
    assert any(e['pointer'] == '/@isaaclab_version' for e in f['evidence'])


def test_merge_that_drops_format_version_is_an_error(tmp_path):
    src = isaac_file(tmp_path / 'src.hdf5', order='XYZW', version=1)
    dst = copy_without_root_attrs(src, tmp_path / 'merged.hdf5')
    report = check_contract(check_sim_provenance(src, converted=dst))
    f = finding(report, 'isaaclab.conversion_quaternion_order')
    assert f['status'] == 'error' and 'lost format_version' in f['message'] and 'change no data' in f['remedy']
    assert report['coverage']['pair']['paired_demos'] == 2


def test_convert_that_leaves_other_quaternions_wxyz_is_an_error(tmp_path):
    src = isaac_file(tmp_path / 'legacy.hdf5')
    dst = convert_root_pose_only(src, tmp_path / 'xyzw.hdf5')
    f = finding(check_contract(check_sim_provenance(src, converted=dst)), 'isaaclab.conversion_quaternion_order')
    assert f['status'] == 'error' and 'obs/eef_quat' in f['message'] and 'unchanged copies' in f['message']


def test_explicit_version_zero_double_conversion_is_an_error(tmp_path):
    src = isaac_file(tmp_path / 'v0.hdf5', version=0, extra_quat=False)
    dst = convert_root_pose_only(src, tmp_path / 'v0_out.hdf5', keep_source_attrs=True)
    f = finding(check_contract(check_sim_provenance(src, converted=dst)), 'isaaclab.conversion_quaternion_order')
    assert f['status'] == 'error' and 'reorder them again' in f['message']


def test_consistent_conversion_passes(tmp_path):
    src = isaac_file(tmp_path / 'legacy.hdf5', extra_quat=False)
    dst = convert_root_pose_only(src, tmp_path / 'xyzw.hdf5')
    f = finding(check_contract(check_sim_provenance(src, converted=dst)), 'isaaclab.conversion_quaternion_order')
    assert f['status'] == 'pass'


def test_pairing_uses_every_source_demo_not_only_the_sample(tmp_path):
    src = isaac_file(tmp_path / 'src.hdf5', order='XYZW', version=1, demos=12, extra_quat=False)
    dst = tmp_path / 'part.hdf5'
    with h5py.File(src, 'r') as a, h5py.File(dst, 'w') as b:  # merge of the last two demos, renumbered
        for i, name in enumerate(('demo_10', 'demo_11')):
            a.copy(f'data/{name}', b, f'data/demo_{i}')
        b['data'].attrs['env_args'] = a['data'].attrs['env_args']
    report = check_sim_provenance(src, converted=dst, max_demos=2)
    assert finding(report, 'isaaclab.conversion_quaternion_order')['status'] == 'error'
    assert report['coverage']['pair']['paired_demos'] == 2


def test_ambiguous_action_anchors_are_not_paired(tmp_path):
    src = isaac_file(tmp_path / 'src.hdf5', order='XYZW', version=1)
    with h5py.File(src, 'r+') as f:
        f['data/demo_1/actions'][...] = f['data/demo_0/actions'][...]
    dst = copy_without_root_attrs(src, tmp_path / 'merged.hdf5')
    f = finding(check_sim_provenance(src, converted=dst), 'isaaclab.conversion_quaternion_order')
    assert f['status'] == 'unknown'


def test_soft_links_images_and_oversized_chunks_are_not_read(tmp_path):
    path = isaac_file(tmp_path / 'l.hdf5', order='XYZW', version=1)
    with h5py.File(path, 'r+') as f:
        f['data/demo_0/obs/linked_quat'] = h5py.SoftLink('/data/demo_1/obs/eef_quat')
        q = f['data/demo_1/obs/eef_quat'][...]
        del f['data/demo_1/obs/eef_quat']
        f['data/demo_1/obs'].create_dataset('eef_quat', data=q, maxshape=(None, 4), chunks=(2_000_000, 4),
                                            compression='gzip')
    report = check_sim_provenance(path)
    skipped = ' '.join(report['coverage']['source']['skipped'])
    assert 'soft/external link' in skipped and 'chunk of 32000000 bytes' in skipped
    assert report['coverage']['values_read'] < 400  # camera arrays and the oversized chunk are never read


def test_d8_corrupt_hdf5_bytes_raise_only_input_errors(tmp_path):
    base = isaac_file(tmp_path / 'base.hdf5', order='XYZW', version=1)
    blob = base.read_bytes()
    rng = np.random.default_rng(7)
    for k in range(40):
        corrupt = bytearray(blob)
        for off in rng.integers(0, min(len(blob), 4096), size=8):
            corrupt[int(off)] = int(rng.integers(0, 256))
        path = tmp_path / f'c{k}.hdf5'
        path.write_bytes(bytes(corrupt))
        try:
            check_sim_provenance(path)
        except InputError:
            pass


# ---------------------------------------------------------------------------------------------- robomimic

def robomimic_file(path, *, kind='delta', control_delta=True, composite=False, controller=None,
                   env_args='default', demos=3, steps=60, kinds=None, arm_extra=None):
    rng = np.random.default_rng(1)
    config = dict(controller or DELTA_OSC, control_delta=control_delta)
    if composite:
        config = {'type': 'BASIC', 'body_parts': {'right': dict(config, gripper={'type': 'GRIP'}, **(arm_extra or {}))}}
    meta = {'env_name': 'Lift', 'type': 1, 'env_version': '1.5.1',
            'env_kwargs': {'robots': ['Panda'], 'control_freq': 20, 'controller_configs': config}}
    with h5py.File(path, 'w') as f:
        data = f.create_group('data')
        if env_args == 'default':
            data.attrs['env_args'] = json.dumps(meta)
        elif env_args is not None:
            data.attrs['env_args'] = env_args
        for d in range(demos):
            g = data.create_group(f'demo_{d}')
            p = np.zeros((steps, 3))
            p[0] = [-0.1, 0.0, 1.0]
            delta = np.clip(np.cumsum(rng.normal(scale=0.3, size=(steps, 3)), 0) / 4, -1, 1)
            for t in range(steps - 1):
                p[t + 1] = p[t] + 0.005 * delta[t] + rng.normal(scale=1e-5, size=3)
            rot = rng.uniform(-0.2, 0.2, size=(steps, 3))
            grip = np.where(np.arange(steps) > steps // 2, 1.0, -1.0)[:, None]
            delta_actions = np.concatenate([delta, rot, grip], 1)
            target = np.vstack([p[1:], p[-1:]]) + rng.normal(scale=2e-3, size=(steps, 3))
            abs_actions = np.concatenate([target, np.tile([3.0, 0.1, 0.0], (steps, 1)), grip], 1)
            this = kinds[d] if kinds else kind
            g.create_dataset('actions', data=delta_actions if this == 'delta' else abs_actions)
            g.create_dataset('actions_abs', data=abs_actions)
            g.create_dataset('obs/robot0_eef_pos', data=p)
            g.create_dataset('obs/robot0_eef_quat', data=np.tile([1.0, 0, 0, 0], (steps, 1)))
            g.create_dataset('states', data=rng.normal(size=(steps, 10)))
    return path


def set_env(path, mutate):
    with h5py.File(path, 'r+') as f:
        meta = json.loads(f['data'].attrs['env_args'])
        mutate(meta)
        f['data'].attrs['env_args'] = json.dumps(meta)


def test_delta_actions_with_delta_controller_pass_and_state_coverage(tmp_path):
    report = check_contract(check_sim_provenance(robomimic_file(tmp_path / 'd.hdf5', demos=5)))
    assert report['status'] == 'pass' and report['coverage']['source']['demos_inspected'] == 5
    assert codes(report) == {'robomimic.env_args', CTRL}
    assert 'all 5 demos look delta' in finding(report, CTRL)['message']


def test_n3_partial_coverage_never_passes(tmp_path):
    report = check_contract(check_sim_provenance(robomimic_file(tmp_path / 'd.hdf5', demos=5), max_demos=3))
    f = finding(report, CTRL)
    assert f['status'] == 'unknown' and 'only 3 of 5 demos were inspected' in f['message']
    assert report['status'] == 'unknown'


def test_absolute_actions_with_delta_env_args_are_an_error(tmp_path):
    report = check_contract(check_sim_provenance(robomimic_file(tmp_path / 'abs.hdf5', kind='absolute')))
    f = finding(report, CTRL)
    assert f['status'] == 'error' and 'control_delta=true' in f['message']
    assert f['evidence'][0]['pointer'] == '/data@env_args/env_kwargs/controller_configs/control_delta'
    assert f['evidence'][1]['value']['verdicts'] == {'absolute': 3}


def test_d2_flat_config_messages_say_robosuite_15_refuses_them(tmp_path):
    f = finding(check_sim_provenance(robomimic_file(tmp_path / 'abs.hdf5', kind='absolute')), CTRL)
    assert 'robosuite 1.5.1 refuses this config loudly' in f['message']
    assert 'would also run' not in f['message'] and 'copied verbatim' not in f['message']


def test_d3_absolute_remedy_names_the_world_frame(tmp_path):
    f = finding(check_sim_provenance(robomimic_file(tmp_path / 'abs.hdf5', kind='absolute')), CTRL)
    assert 'input_type="absolute" with input_ref_frame="world"' in f['remedy']


def test_absolute_positions_inside_the_input_range_are_detected_by_tracking(tmp_path):
    path = robomimic_file(tmp_path / 'abs.hdf5', kind='absolute')
    with h5py.File(path, 'r+') as f:
        for g in f['data'].values():
            a = g['actions'][...]
            a[:, 3:6] = 0.1
            a[:, 2] -= 0.5
            g['actions'][...] = a
            g['obs/robot0_eef_pos'][:, 2] -= 0.5
    f = finding(check_sim_provenance(path), CTRL)
    assert f['status'] == 'error' and f['evidence'][1]['value']['rows_outside_input_range'] == 0


@pytest.mark.parametrize('kind, status', [('absolute', 'pass'), ('delta', 'error')])
def test_flat_control_delta_false_is_decided_for_robosuite_14(tmp_path, kind, status):
    path = robomimic_file(tmp_path / 'f.hdf5', kind=kind, control_delta=False)
    assert finding(check_contract(check_sim_provenance(path)), CTRL)['status'] == status


def test_control_delta_absent_defaults_to_true(tmp_path):
    path = robomimic_file(tmp_path / 'a.hdf5', kind='absolute')
    set_env(path, lambda m: m['env_kwargs']['controller_configs'].pop('control_delta'))
    f = finding(check_sim_provenance(path), CTRL)
    assert f['status'] == 'error' and 'default true' in str(f['evidence'][0]['value'])


@pytest.mark.parametrize('kind, control_delta, extra, status', [
    ('absolute', False, {}, 'error'),  # control_delta ignored on >= 1.5: input_type defaults to delta
    ('delta', False, {}, 'pass'),
    ('absolute', True, {'input_type': 'absolute', 'input_ref_frame': 'world'}, 'pass'),
    ('absolute', True, {'input_type': 'absolute'}, 'error'),  # default frame "base" (D3)
    ('absolute', True, {'input_type': 'absolute', 'input_ref_frame': 'base'}, 'error'),
    ('delta', True, {'input_type': 'absolute', 'input_ref_frame': 'world'}, 'error')])
def test_d2_composite_configs_are_judged_by_input_type(tmp_path, kind, control_delta, extra, status):
    path = robomimic_file(tmp_path / 'c.hdf5', kind=kind, control_delta=control_delta, composite=True, arm_extra=extra)
    f = finding(check_contract(check_sim_provenance(path)), CTRL)
    assert f['status'] == status and 'control_delta, if present, is ignored' in f['message']
    if status == 'error' and kind == 'absolute' and extra.get('input_type') == 'absolute':
        assert 'input_ref_frame="world"' in f['remedy']


def test_d4_mixed_absolute_and_delta_demos_are_an_error(tmp_path):
    path = robomimic_file(tmp_path / 'm.hdf5', demos=5, kinds=['delta', 'absolute', 'delta', 'absolute', 'delta'])
    f = finding(check_contract(check_sim_provenance(path)), CTRL)
    assert f['status'] == 'error' and '2 of 5 inspected demos' in f['message'] and 'mixes conventions' in f['message']


def test_d4_absolute_tail_beyond_max_demos_is_still_inspected(tmp_path):
    path = robomimic_file(tmp_path / 't.hdf5', demos=12, kinds=['delta'] * 8 + ['absolute'] * 4)
    f = finding(check_sim_provenance(path, max_demos=4), CTRL)
    assert f['status'] == 'error'


def test_robosuite_15_v15_style_file_passes_by_input_type_default(tmp_path):
    path = robomimic_file(tmp_path / 'v15.hdf5', composite=True, arm_extra={'input_ref_frame': 'world'})
    f = finding(check_contract(check_sim_provenance(path)), CTRL)
    assert f['status'] == 'pass' and 'input_type="delta"' in f['message']


def test_out_of_range_delta_commands_are_not_called_absolute(tmp_path):
    path = robomimic_file(tmp_path / 'd.hdf5')
    with h5py.File(path, 'r+') as f:
        for g in f['data'].values():
            g['actions'][:, 0:3] = g['actions'][:, 0:3] * 3  # unclipped scripted deltas: out of range, still deltas
    f = finding(check_sim_provenance(path), CTRL)
    assert f['status'] == 'pass' and f['evidence'][1]['value']['rows_outside_input_range'] > 0


@pytest.mark.parametrize('env_args, status', [(None, 'unknown'), ('{not json', 'unknown'),
                                              (json.dumps({'type': 1}), 'unknown')])
def test_missing_or_invalid_env_args_are_unknown(tmp_path, env_args, status):
    report = check_contract(check_sim_provenance(robomimic_file(tmp_path / 'e.hdf5', env_args=env_args)))
    assert finding(report, 'robomimic.env_args')['status'] == status
    assert CTRL not in codes(report) and report['status'] == 'unknown'


@pytest.mark.parametrize('controller', [{'type': 'JOINT_POSITION', 'input_max': 1, 'input_min': -1},
                                        dict(DELTA_OSC, impedance_mode='variable_kp')])
def test_unsupported_controllers_are_unknown(tmp_path, controller):
    path = robomimic_file(tmp_path / 'j.hdf5', controller=controller)
    assert finding(check_sim_provenance(path), CTRL)['status'] == 'unknown'


def test_without_eef_positions_nothing_is_decided(tmp_path):
    path = robomimic_file(tmp_path / 'n.hdf5')
    with h5py.File(path, 'r+') as f:
        for g in f['data'].values():
            del g['obs/robot0_eef_pos']
    assert finding(check_sim_provenance(path), CTRL)['status'] == 'unknown'


def test_d9_gym_type_file_with_nothing_assessed_is_unknown(tmp_path):
    path = tmp_path / 'gym.hdf5'
    with h5py.File(path, 'w') as f:
        f.create_group('data').attrs['env_args'] = json.dumps({'env_name': 'hopper-medium-v2', 'type': 2, 'env_kwargs': {}})
        f.create_dataset('data/demo_0/obs/body_quat', data=np.tile([0.0, 0, 0, 1], (5, 1)))
        f.create_dataset('data/demo_0/actions', data=np.zeros((5, 3)))
    report = check_contract(check_sim_provenance(path))
    assert report['status'] == 'unknown' and 'sim.no_applicable_checks' in codes(report)
    assert report['coverage']['source']['layout'] == 'unrecognized'


# ---------------------------------------------------------------------------------------------- LeRobot

def lerobot(root, *, state_names, state, action=None, episodes=None, stats=None, width=None):
    (root / 'meta').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    features = {'observation.state': {'dtype': 'float32', 'shape': [width or len(state_names)],
                                      'names': {'motors': state_names}}}
    columns = {'observation.state': pa.array(state.astype(np.float32).tolist(), pa.list_(pa.float32()))}
    if action is not None:
        features['action'] = {'dtype': 'float32', 'shape': [action.shape[1]], 'names': None}
        columns['action'] = pa.array(action.astype(np.float32).tolist(), pa.list_(pa.float32()))
        columns['episode_index'] = pa.array(episodes, pa.int64())
    (root / 'meta/info.json').write_text(json.dumps({'codebase_version': 'v3.0', 'fps': 20, 'features': features}))
    if stats is not None:
        (root / 'meta/stats.json').write_text(json.dumps(stats))
    pq.write_table(pa.table(columns), root / 'data/chunk-000/file-000.parquet')
    return root


LIBERO_NAMES = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'rw', 'gripper']


def unit_rows(n, seed=3):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_d6_declared_quaternion_holding_axis_angle_is_a_warning_with_rename_remedy(tmp_path):
    rng = np.random.default_rng(2)
    state = np.concatenate([rng.normal(size=(50, 3)), rng.uniform(2.5, 3.5, (50, 3)), rng.uniform(0, 0.04, (50, 1)),
                            rng.uniform(-0.04, 0, (50, 1))], 1)
    report = check_contract(check_sim_provenance(lerobot(tmp_path / 'libero', state_names=LIBERO_NAMES, state=state)))
    f = finding(report, 'lerobot.declared_quaternion')
    assert f['status'] == 'warning' and "['rx', 'ry', 'rz', 'rw']" in f['message'] and 'far from 1' in f['message']
    assert 'FIRST' in f['message'] and 'does not create' in f['message']  # D12/D6 provenance wording
    assert 'unit quaternions' not in f['remedy'] and 'do not change the values' in f['remedy']


@pytest.mark.parametrize('mutate', ['lerp', 'scale_noise', 'zero_rows'])
def test_d5_near_unit_quaternions_are_a_normalize_warning(tmp_path, mutate):
    q = unit_rows(400)
    if mutate == 'lerp':
        q = 0.5 * (q[:-1] + np.roll(q, -1, 0)[:-1]) / np.linalg.norm(0.5 * (q[:-1] + np.roll(q, -1, 0)[:-1]), axis=1,
                                                                      keepdims=True) * 0.985
    elif mutate == 'scale_noise':
        q = q * (1 + np.random.default_rng(4).normal(scale=0.01, size=(len(q), 1)))
    else:
        q[::33] = 0
    report = check_contract(check_sim_provenance(lerobot(tmp_path / 'n', state_names=['qx', 'qy', 'qz', 'qw'], state=q)))
    f = finding(report, 'lerobot.declared_quaternion')
    assert f['status'] == 'warning' and 'near-unit' in f['message'] and 'Normalize' in f['remedy']


def test_declared_unit_quaternions_pass_without_claiming_order(tmp_path):
    q = unit_rows(40)
    names = ['pos.0', 'pos.1', 'pos.2', 'eef_quat.0', 'eef_quat.1', 'eef_quat.2', 'eef_quat.3']
    report = check_contract(check_sim_provenance(lerobot(tmp_path / 'ok', state_names=names,
                                                         state=np.concatenate([q[:, :3], q], 1))))
    f = finding(report, 'lerobot.declared_quaternion')
    assert f['status'] == 'pass' and 'undeclared' in f['message'] and 'not verifiable' in f['message']


def test_stats_alone_can_refute_a_declared_quaternion(tmp_path):
    root = lerobot(tmp_path / 's', state_names=LIBERO_NAMES, state=np.zeros((0, 8)),
                   stats={'observation.state': {'min': [0, 0, 0, -3.6, -1, 0, 0, 0], 'max': [1, 1, 1, 3.6, 1, 1, 1, 1]}})
    f = finding(check_contract(check_sim_provenance(root)), 'lerobot.declared_quaternion')
    assert f['status'] == 'warning' and 'min/max' in f['message']


def test_rows_override_stale_stats(tmp_path):
    root = lerobot(tmp_path / 'st', state_names=['qx', 'qy', 'qz', 'qw'], state=unit_rows(30, 5),
                   stats={'observation.state': {'min': [-3, -1, -1, -1], 'max': [3, 1, 1, 1]}})
    assert finding(check_sim_provenance(root), 'lerobot.declared_quaternion')['status'] == 'pass'


def test_d7_list_column_wider_than_declared_shape_is_refused_before_reading(tmp_path):
    rng = np.random.default_rng(6)
    wide = rng.normal(size=(20, 4000))
    root = lerobot(tmp_path / 'w', state_names=['qx', 'qy', 'qz', 'qw'], state=wide, width=4)
    report = check_contract(check_sim_provenance(root))
    assert finding(report, 'lerobot.declared_quaternion')['status'] == 'unknown'
    assert 'exceed 20 rows x declared width 4' in report['coverage']['source']['unsupported_columns']['observation.state']
    assert report['coverage']['source']['lerobot_rows_read'] == 0


@pytest.mark.parametrize('names', [['cursor_x', 'cursor_y', 'cursor_z', 'cursor_w'],
                                   ['orientation.0', 'orientation.1', 'orientation.2', 'orientation.3',
                                    'orientation.4', 'orientation.5'],
                                   ['x', 'y', 'z', 'axis_angle1', 'axis_angle2', 'axis_angle3', 'gripper', 'gripper']])
def test_non_quaternion_names_are_not_claimed(tmp_path, names):
    report = check_contract(check_sim_provenance(lerobot(tmp_path / 'n', state_names=names,
                                                         state=np.full((5, len(names)), 3.0))))
    assert codes(report) == {'sim.no_applicable_checks'} and report['status'] == 'unknown'


def converted_from(tmp_path, source, stream, demos=None):
    with h5py.File(source, 'r') as f:
        demos = range(len(f['data'])) if demos is None else demos
        parts = [f[f'data/demo_{d}/{stream}'][...] for d in demos]
    action = np.concatenate(parts)
    episodes = np.concatenate([[i] * len(p) for i, p in enumerate(parts)]).tolist()
    return lerobot(tmp_path / f'conv_{stream}', state_names=['a', 'b'], state=np.zeros((len(action), 2)),
                   action=action, episodes=episodes)


def test_converted_actions_traced_to_delta_stream_pass(tmp_path):
    src = robomimic_file(tmp_path / 'src.hdf5')
    report = check_contract(check_sim_provenance(src, converted=converted_from(tmp_path, src, 'actions')))
    assert report['status'] == 'pass'
    assert "['actions']" in finding(report, 'conversion.action_source')['message']
    assert report['coverage']['converted']['converted_episodes_traced'] == 3


def test_n4_converted_actions_abs_stream_under_delta_env_args_is_a_warning(tmp_path):
    src = robomimic_file(tmp_path / 'src.hdf5')
    report = check_contract(check_sim_provenance(src, converted=converted_from(tmp_path, src, 'actions_abs')))
    traced = [f for f in report['findings'] if f['code'] == CTRL and 'actions_abs' in f['message']]
    assert len(traced) == 1 and traced[0]['status'] == 'warning' and 'absolute by name' in traced[0]['message']
    assert 'input_ref_frame="world"' in traced[0]['remedy'] and report['status'] == 'warning'


def test_n4_canonical_actions_contradicting_env_args_stay_an_error(tmp_path):
    src = robomimic_file(tmp_path / 'src.hdf5', kind='absolute')  # DP low_dim_abs layout: "actions" absolute
    report = check_contract(check_sim_provenance(src, converted=converted_from(tmp_path, src, 'actions')))
    assert [f['status'] for f in report['findings'] if f['code'] == CTRL] == ['error', 'error']


def test_d11_converted_demos_beyond_max_demos_are_traced(tmp_path):
    src = robomimic_file(tmp_path / 'src.hdf5', demos=8)
    report = check_sim_provenance(src, converted=converted_from(tmp_path, src, 'actions', demos=[6, 7]), max_demos=2)
    assert finding(report, 'conversion.action_source')['status'] == 'pass'


def test_d11_untraced_converted_actions_are_unknown_without_asking_for_the_same_source(tmp_path):
    src = robomimic_file(tmp_path / 'src.hdf5')
    conv = lerobot(tmp_path / 'other', state_names=['a'], state=np.zeros((4, 1)), action=np.ones((4, 7)),
                   episodes=[0, 0, 1, 1])
    f = finding(check_sim_provenance(src, converted=conv), 'conversion.action_source')
    assert f['status'] == 'unknown' and '3 indexed source demos (of 3)' in f['message']
    assert 'transforms actions' in f['remedy']


# ---------------------------------------------------------------------------------------------- inputs

def test_read_roots_are_enforced(tmp_path):
    inside = tmp_path / 'inside'
    inside.mkdir()
    path = robomimic_file(inside / 'd.hdf5')
    assert check_sim_provenance(path, read_roots=[str(inside)])['status'] == 'pass'
    with pytest.raises(InputError):
        check_sim_provenance(path, read_roots=[str(tmp_path / 'elsewhere')])
    outside = tmp_path / 'outside'
    outside.mkdir()
    os.symlink(path, outside / 'link.hdf5')  # a link inside the root that resolves outside it
    with pytest.raises(InputError):
        check_sim_provenance(outside / 'link.hdf5', read_roots=[str(outside)])


def test_info_json_alias_cannot_widen_read_roots(tmp_path):
    rng = np.random.default_rng(4)
    root = lerobot(tmp_path / 'ds', state_names=LIBERO_NAMES, state=rng.normal(size=(4, 8)))
    with pytest.raises(InputError):
        check_sim_provenance(root / 'meta/info.json', read_roots=[str(root / 'meta')])
    assert check_sim_provenance(root / 'meta/info.json', read_roots=[str(root)])['status'] == 'warning'


def test_fifo_input_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / 'pipe.hdf5'
    os.mkfifo(fifo)
    with pytest.raises(InputError):
        check_sim_provenance(fifo)


def test_converted_input_that_is_not_compared_is_unknown(tmp_path):
    src = isaac_file(tmp_path / 'v1.hdf5', order='XYZW', version=1)
    conv = lerobot(tmp_path / 'conv', state_names=['a'], state=np.zeros((3, 1)))
    report = check_contract(check_sim_provenance(src, converted=conv))
    assert finding(report, 'conversion.not_compared')['status'] == 'unknown' and report['status'] == 'unknown'


@pytest.mark.parametrize('kwargs', [{'max_demos': 0}, {'max_demos': True}, {'max_rows': 0}])
def test_bounds_are_validated(tmp_path, kwargs):
    with pytest.raises(InputError):
        check_sim_provenance(robomimic_file(tmp_path / 'd.hdf5'), **kwargs)


def test_non_hdf5_and_missing_inputs_are_refused(tmp_path):
    bad = tmp_path / 'x.hdf5'
    bad.write_text('not hdf5')
    with pytest.raises(InputError):
        check_sim_provenance(bad)
    with pytest.raises(InputError):
        check_sim_provenance(tmp_path / 'missing.hdf5')
    with h5py.File(tmp_path / 'nodata.hdf5', 'w') as f:
        f.create_group('other')
    with pytest.raises(InputError):
        check_sim_provenance(tmp_path / 'nodata.hdf5')


def test_max_demos_limits_coverage(tmp_path):
    report = check_sim_provenance(robomimic_file(tmp_path / 'd.hdf5', demos=5), max_demos=2)
    assert report['coverage']['source']['demos_inspected'] == 2 and report['coverage']['source']['demos_total'] == 5


def test_lerobot_source_cannot_have_converted(tmp_path):
    root = lerobot(tmp_path / 'l', state_names=['a'], state=np.zeros((2, 1)))
    with pytest.raises(InputError):
        check_sim_provenance(root, converted=root)


# ---------------------------------------------------------------------------------------------- recheck N1-N5

def legacy_many(path, *, objects=10, cameras=4, eef_pose=False, demos=2):
    """A legacy (WXYZ) recording whose robot root decides WXYZ, with many quaternion-bearing datasets."""
    rng = np.random.default_rng(9)
    with h5py.File(path, 'w') as f:
        d = f.create_group('data')
        d.attrs['env_args'] = json.dumps({'env_name': 'Isaac-Many-v0', 'type': 2})
        for i in range(demos):
            g = d.create_group(f'demo_{i}')
            g.create_dataset('actions', data=rng.normal(size=(3, 7)) + i)
            for where in ('initial_state', 'states'):
                g.create_dataset(f'{where}/articulation/robot/root_pose', data=np.tile([0, 0, 0, 1.0, 0, 0, 0], (3, 1)))
                for o in range(objects):
                    g.create_dataset(f'{where}/rigid_object/obj{o:02d}/root_pose',
                                     data=np.tile(np.r_[[0.1 * o, 0.2, 0.3], yaw(15 + 7 * o, 'WXYZ')], (3, 1)))
            for c in range(cameras):
                g.create_dataset(f'initial_state/cameras/cam{c:02d}/orientation', data=np.array([yaw(10 * c + 5, 'WXYZ')] * 3))
            if eef_pose:
                g.create_dataset('obs/eef_pose', data=np.tile(np.r_[[0.4, -0.1, 0.9], yaw(33, 'WXYZ')], (3, 1)))
    return path


def apply_entries(path, entries, *, root_pose=False, stamp=None):
    """Apply a remedy literally: roll the listed quaternion columns of each dataset, in every demo."""
    with h5py.File(path, 'r+') as f:
        for demo in f['data']:
            g = f[f'data/{demo}']
            targets = [(e['dataset'], e['quaternion_columns']) for e in entries if e['dataset'] in g]
            if root_pose:
                g.visititems(lambda n, o, targets=targets: targets.append((n, '3:7')) if isinstance(o, h5py.Dataset)
                             and n.rsplit('/', 1)[-1] == 'root_pose' and o.shape[-1] == 7 else None)
            for name, cols in dict(targets).items():
                a, b = (int(x) for x in cols.split(':'))
                v = g[name][...]
                v[..., a:b] = np.roll(v[..., a:b], -1, axis=-1)
                g[name][...] = v
        if stamp is not None:
            f.attrs['format_version'] = stamp


def wrong_after(original_wxyz, remedied):
    """Datasets whose quaternion part is not the XYZW reading of the original or whose positions moved."""
    bad = []
    with h5py.File(original_wxyz, 'r') as a, h5py.File(remedied, 'r') as b:
        def visit(name, obj):
            if not isinstance(obj, h5py.Dataset) or obj.shape[-1] not in (4, 7) or name.endswith('actions'):
                return
            src, dst = obj[...], b[name][...]
            q = slice(3, 7) if obj.shape[-1] == 7 else slice(0, 4)
            ok = np.allclose(np.roll(src[..., q], -1, axis=-1), dst[..., q]) and (
                obj.shape[-1] == 4 or np.allclose(src[..., :3], dst[..., :3]))
            if not ok:
                bad.append(name)
        a.visititems(visit)
    return bad


def copy_file(src, dst, *, version='absent'):
    with h5py.File(src, 'r') as a, h5py.File(dst, 'w') as b:
        a.copy('data', b)
        if version != 'absent':
            b.attrs['format_version'] = version
    return dst


def apply_fixes(path, entries, *, stamp=None):
    """Apply remedy_list() entries literally: np.roll(q, roll) on the listed columns, only in the listed demos."""
    with h5py.File(path, 'r+') as f:
        for e in entries:
            a, b = (int(x) for x in e['quaternion_columns'].split(':'))
            for demo in (list(f['data']) if e['demos'] == 'all' else e['demos']):
                ds = f[f'data/{demo}/{e["dataset"]}']
                v = ds[...]
                v[..., a:b] = np.roll(v[..., a:b], e['roll'], axis=-1)
                ds[...] = v
        if stamp is not None:
            f.attrs['format_version'] = stamp


def pair_finding(src, dst):
    report = check_contract(check_sim_provenance(src, converted=dst))
    return report, finding(report, 'isaaclab.conversion_quaternion_order')


def test_n1_pair_remedy_lists_every_contradicting_dataset_and_fixes_all(tmp_path):
    src = legacy_many(tmp_path / 'legacy.hdf5')  # 1 robot + 10 objects root_pose x2 + 4 cameras = 26 datasets
    stamped = copy_file(src, tmp_path / 'stamped.hdf5', version=1)  # format_version=1 without converting any data
    report, f = pair_finding(src, stamped)
    entries = remedy_list(report, 'pair.fixes')
    assert f['status'] == 'error' and len(entries) == 26 and "remedy_lists['pair.fixes']" in f['message']
    assert 'exactly' not in f['remedy'] and "remedy_lists['pair.fixes']" in f['remedy']
    assert {e['roll'] for e in entries} == {-1} and {e['demos'] for e in entries} == {'all'}
    apply_fixes(stamped, entries)
    assert wrong_after(src, stamped) == []
    assert pair_finding(src, stamped)[1]['status'] == 'pass'


def test_n1_double_conversion_pair_lists_the_unconverted_datasets_and_fixes_all(tmp_path):
    src = legacy_many(tmp_path / 'v0.hdf5', objects=2, cameras=9, eef_pose=True)  # robot roots decide WXYZ
    with h5py.File(src, 'r+') as f:
        f.attrs['format_version'] = 0
    dst = convert_root_pose_only(src, tmp_path / 'v0_out.hdf5', keep_source_attrs=True)  # roots reordered, keeps 0
    report, f = pair_finding(src, dst)
    entries = remedy_list(report, 'pair.fixes')
    assert f['status'] == 'error' and 'reorder them again' in f['message']
    assert len(entries) == 10 and all(not e['dataset'].endswith('root_pose') for e in entries)
    assert {'dataset': 'obs/eef_pose', 'quaternion_columns': '3:7', 'relation': 'identity', 'roll': -1,
            'demos': 'all'} in entries
    # recheck3 P1: roots that verify WXYZ in every paired demo make it decisive, with the mount caveat and its branch
    assert f['remedy'].startswith('Only if the source is an Isaac Lab 2.x (WXYZ) recording;')
    caveat = f['remedy'].split(' Caveat: ', 1)[1]
    assert 'ceiling or wall' in caveat and "remedy_lists['pair.fixes_if_source_xyzw']" in caveat
    assert report['coverage']['source']['content_order_verified'] == 'WXYZ'
    stamp_only = copy_file(dst, tmp_path / 'stamp_only.hdf5', version=1)  # the earlier remedy: roots right, copies not
    assert len(wrong_after(src, stamp_only)) == 20  # the 10 datasets in each of the 2 demos
    apply_fixes(dst, entries, stamp=1)  # the decisive remedy read literally
    assert wrong_after(src, dst) == []


def test_n1_double_conversion_remedy_is_conditional_when_source_roots_do_not_decide(tmp_path):
    src = isaac_file(tmp_path / 'v0.hdf5', version=0, robot=[0, 0, 0, 1.0])  # yaw 180: upright under both orders
    dst = convert_root_pose_only(src, tmp_path / 'v0_out.hdf5', keep_source_attrs=True)
    report, f = pair_finding(src, dst)
    assert report['coverage']['source']['content_order_verified'] is None
    assert [e['dataset'] for e in remedy_list(report, 'pair.fixes')] == ['obs/eef_quat']
    assert {e['roll'] for e in remedy_list(report, 'pair.fixes_if_source_xyzw')} == {1}  # undo the root reorder
    assert f['remedy'].startswith('First establish that the source is an Isaac Lab 2.x (WXYZ) recording')
    assert 'its robot roots do not decide it' in f['remedy'] and 'fixes_if_source_xyzw' in f['remedy']


def test_n1_converted_v0_file_alone_points_to_the_pair_check(tmp_path):
    src = isaac_file(tmp_path / 'v0.hdf5', version=0)
    dst = convert_root_pose_only(src, tmp_path / 'v0_out.hdf5', keep_source_attrs=True)  # roots reordered, keeps 0
    report = check_contract(check_sim_provenance(dst))
    outside = finding(report, 'isaaclab.quaternions_outside_root_pose')
    assert outside['message'].startswith('[source] Conditional') and 'still WXYZ' in outside['message']
    assert 'convert_dataset_to_xyzw() converted it' in outside['message']
    assert outside['remedy'].startswith('Change no data on these values alone') and 'np.roll' not in outside['remedy']
    assert 'pass its input as source' in outside['remedy']
    fv = finding(report, 'isaaclab.format_version')
    assert 'isaaclab.quaternions_outside_root_pose' in fv['remedy'] and 'Change no root_pose data' in fv['remedy']


def test_n2_whole_file_remedy_lists_every_dataset_and_rolls_only_pose_quaternion_columns(tmp_path):
    src = legacy_many(tmp_path / 'legacy.hdf5', objects=0, cameras=12, eef_pose=True)
    report = check_contract(check_sim_provenance(src))
    f = finding(report, 'isaaclab.quaternions_outside_root_pose')
    entries = remedy_list(report, 'source.non_root_quaternion_datasets')
    assert len(entries) == 13 and {'dataset': 'obs/eef_pose', 'quaternion_columns': '3:7'} in entries
    assert "all 13 in remedy_lists['source.non_root_quaternion_datasets']" in f['message'] and '3:7' in f['remedy']
    assert f['remedy'].startswith('Only if the robots are known to start upright on a floor or table')
    assert 'ceiling or wall' in f['remedy']
    remedied = copy_file(src, tmp_path / 'remedied.hdf5')
    apply_entries(remedied, entries, root_pose=True, stamp=1)
    assert wrong_after(src, remedied) == []


# ---------------------------------------------------------------------------------------------- recheck2 R1-R4

def partial_conversion(tmp_path, *, version, demos=4, converted=(0, 1)):
    """Validator recheck2 F/F2/F3: explicit-0 legacy source; root_pose of some demos reordered (convert_dataset_to_xyzw
    on a subset), the rest copied, merged (root attributes dropped), then stored with `version`."""
    src = legacy_many(tmp_path / 'src_v0.hdf5', objects=1, cameras=2, eef_pose=True, demos=demos)
    with h5py.File(src, 'r+') as f:
        f.attrs['format_version'] = 0
    dst = copy_file(src, tmp_path / f'partial_{version}.hdf5', version=version)
    with h5py.File(dst, 'r+') as f:
        for i in converted:
            f[f'data/demo_{i}'].visititems(lambda n, o: o.__setitem__(Ellipsis, np.concatenate(
                [o[...][..., :3], np.roll(o[...][..., 3:7], -1, axis=-1)], -1)) if isinstance(o, h5py.Dataset)
                and n.endswith('root_pose') else None)
    return src, dst


def test_r1_partial_conversion_stamped_1_gets_per_demo_fixes_and_converges(tmp_path):
    src, dst = partial_conversion(tmp_path, version=1)  # F2
    report, f = pair_finding(src, dst)
    entries = remedy_list(report, 'pair.fixes')
    roots = [e for e in entries if e['dataset'].endswith('root_pose')]
    assert f['status'] == 'error' and 'the relation differs between demos' in f['message']
    assert roots and all(e['demos'] == ['demo_2', 'demo_3'] and e['roll'] == -1 for e in roots)  # never demos 0-1
    assert all(e['demos'] == 'all' for e in entries if not e['dataset'].endswith('root_pose'))
    assert report['coverage']['pair']['relations_per_dataset_and_demo'] == {'WXYZ->XYZW': 8, 'identity': 20}
    double = copy_file(dst, tmp_path / 'double.hdf5', version=1)  # the 93a23b04 remedy: identity roots in every demo
    apply_fixes(double, [dict(e, demos='all') for e in entries])
    report2, f2 = pair_finding(src, double)
    assert f2['status'] == 'error' and 'reordered twice' in f2['message']  # roll-by-2 named, never a pass
    apply_fixes(double, remedy_list(report2, 'pair.fixes'))
    assert wrong_after(src, double) == [] and pair_finding(src, double)[1]['status'] == 'pass'
    apply_fixes(dst, entries)
    assert wrong_after(src, dst) == [] and pair_finding(src, dst)[1]['status'] == 'pass'


@pytest.mark.parametrize('version', ['absent', 0])
def test_r1_partial_conversion_declared_wxyz_lists_both_branches_and_converges(tmp_path, version):
    src, dst = partial_conversion(tmp_path, version=version)  # F (merge dropped the attribute) and F3 (stored 0)
    report, f = pair_finding(src, dst)  # roots verify WXYZ: decisive with the mount caveat (recheck3 P1)
    assert f['status'] == 'error' and f['remedy'].startswith('Only if the source is an Isaac Lab 2.x (WXYZ) recording;')
    assert "If the source is nevertheless an Isaac Lab >= 3.0 (XYZW) recording" in f['remedy']
    fixes = remedy_list(report, 'pair.fixes')
    assert {tuple(e['demos']) for e in fixes if e['dataset'].endswith('root_pose')} == {('demo_2', 'demo_3')}
    alternative = remedy_list(report, 'pair.fixes_if_source_xyzw')
    assert {tuple(e['demos']) for e in alternative} == {('demo_0', 'demo_1')} and {e['roll'] for e in alternative} == {1}
    apply_fixes(dst, fixes, stamp=1)  # the source is 2.x: "If it is" branch
    assert wrong_after(src, dst) == [] and pair_finding(src, dst)[1]['status'] == 'pass'


def test_r1_values_changed_beyond_a_reorder_never_pass(tmp_path):
    src = isaac_file(tmp_path / 'v1.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'))
    dst = copy_file(src, tmp_path / 'changed.hdf5', version=1)
    with h5py.File(dst, 'r+') as f:
        for demo in f['data']:
            f[f'data/{demo}/obs/eef_quat'][...] = np.array([yaw(20 * t + 45, 'XYZW') for t in range(5)])
    report, f = pair_finding(src, dst)
    assert f['status'] == 'warning' and 'changed beyond a reorder' in f['message']
    assert report['coverage']['pair']['paired_datasets_changed_beyond_reorder'] == 2
    assert {e['dataset'] for e in remedy_list(report, 'pair.changed_beyond_reorder')} == {'obs/eef_quat'}
    with h5py.File(dst, 'r+') as f:  # plus a real contradiction: regenerate, no per-demo patch
        f['data/demo_0/obs/eef_quat'][...] = np.array([yaw(20 * t, 'XYZW') for t in range(5)])
        ds = f['data/demo_1/states/articulation/robot/root_pose']
        ds[...] = np.concatenate([ds[...][..., :3], np.roll(ds[...][..., 3:7], -1, axis=-1)], -1)
    report, f = pair_finding(src, dst)
    assert f['status'] == 'error' and f['remedy'].startswith('Regenerate') and 'pair.fixes' not in report['remedy_lists']


def test_r1_sign_flip_is_the_same_rotation_and_roll_by_two_is_named(tmp_path):
    src = isaac_file(tmp_path / 'v1.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'), extra_quat=False)
    for label, fn, status in (('neg', lambda q: -q, 'pass'), ('roll2', lambda q: np.roll(q, -2, axis=-1), 'error')):
        dst = copy_file(src, tmp_path / f'{label}.hdf5', version=1)
        with h5py.File(dst, 'r+') as f:
            f['data'].visititems(lambda n, o, fn=fn: o.__setitem__(Ellipsis, np.concatenate(
                [o[...][..., :3], fn(o[...][..., 3:7])], -1)) if isinstance(o, h5py.Dataset) and n.endswith('root_pose')
                else None)
        report, f = pair_finding(src, dst)
        assert f['status'] == status
        if status == 'error':
            assert 'reordered twice' in f['message'] and {e['roll'] for e in remedy_list(report, 'pair.fixes')} == {2}


def test_r1_unpaired_converted_demos_never_pass(tmp_path):
    src = isaac_file(tmp_path / 'src.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'), demos=2)
    dst = isaac_file(tmp_path / 'dst.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'), demos=3)  # demo_2 is extra
    report, f = pair_finding(src, dst)
    assert f['status'] == 'unknown' and '1 of 3 inspected converted demos could not be paired' in f['message']
    assert '1 (demo_2) match no source demo by byte-identical "actions"' in f['message']


def test_r2_ceiling_mounted_xyzw_recording_stored_with_0_gets_a_mount_conditional_remedy(tmp_path):
    ceiling = [1.0, 0, 0, 0]  # XYZW 180 deg about x: reads upright only as WXYZ
    src = isaac_file(tmp_path / 'xyzw_v0.hdf5', order='XYZW', version=0, robot=ceiling)
    dst = convert_root_pose_only(src, tmp_path / 'conv.hdf5', keep_source_attrs=True)
    report, f = pair_finding(src, dst)
    assert report['coverage']['source']['content_order_verified'] == 'WXYZ'  # the upright assumption misreads it
    # recheck3 P1: decisive for upright mounts; the caveat carries the ceiling/wall (>= 3.0) branch
    assert f['remedy'].startswith('Only if the source is an Isaac Lab 2.x (WXYZ) recording;')
    caveat = f['remedy'].split(' Caveat: ', 1)[1]
    assert 'ceiling or wall' in caveat and 'stamp format_version=1 on the source file' in caveat
    apply_fixes(dst, remedy_list(report, 'pair.fixes_if_source_xyzw'), stamp=1)  # the source is >= 3.0: the caveat
    with h5py.File(src, 'r') as a, h5py.File(dst, 'r') as b:
        a['data'].visititems(lambda n, o: np.testing.assert_array_equal(o[...], b[f'data/{n}'][...])
                             if isinstance(o, h5py.Dataset) else None)  # back to the XYZW recording, stamped 1
    outside = finding(report, 'isaaclab.quaternions_outside_root_pose')
    assert outside['remedy'].startswith('Only if the robots are known to start upright on a floor or table')
    with h5py.File(src, 'r+') as s:  # the caveat branch also fixes the source
        s.attrs['format_version'] = 1
    report2, f2 = pair_finding(src, dst)  # a v1 source whose roots read only as WXYZ is never trusted for a pass
    assert f2['status'] == 'warning' and 'contradict' in f2['message']
    assert f2['remedy'].endswith('If it is not (an Isaac Lab >= 3.0 recording, as its format_version=1 declares): '
                                 'keep format_version=1 and change no data.')
    assert 'also delete format_version from the source file' in f2['remedy'] and remedy_list(report2, 'pair.fixes')


def test_r2_source_whose_roots_contradict_its_declaration_gets_no_decisive_roll(tmp_path):
    src = isaac_file(tmp_path / 'xyzw_v0.hdf5', order='XYZW', version=0, robot=yaw(120, 'XYZW'))  # >= 3.0 stored 0
    report, f = pair_finding(src, copy_file(src, tmp_path / 'stamped.hdf5', version=1))  # a correct XYZW copy
    assert report['coverage']['source']['content_order_verified'] == 'XYZW'
    assert f['status'] == 'error' and f['remedy'].startswith('First establish that the source is an Isaac Lab 2.x')
    assert remedy_list(report, 'pair.fixes_if_source_xyzw') == [] and 'keep format_version=1 and change no data' in f['remedy']


def test_r3_lists_are_stored_once_and_digest_is_bounded(tmp_path):
    src = legacy_many(tmp_path / 'legacy.hdf5', objects=0, cameras=60)
    stamped = copy_file(src, tmp_path / 'stamped.hdf5', version=1)
    report = check_contract(check_sim_provenance(src, converted=stamped))
    text = json.dumps(report)
    assert text.count('cam47/orientation') == 1  # source, converted and pair lists share one stored name list
    assert 'non_root_quaternion_datasets' not in json.dumps(report['coverage'])
    lists = report['remedy_lists']
    assert lists['source.non_root_quaternion_datasets'] == lists['converted.non_root_quaternion_datasets']
    assert len(remedy_list(report, 'pair.fixes')) == 62 and len(remedy_list(report, 'converted.non_root_quaternion_datasets')) == 60
    digest = remedy_lists_digest(report, max_names=3)
    assert len(json.dumps(digest)) < 2000 and digest['lists']['pair.fixes']['datasets'] == 62
    assert digest['sha256'] == hashlib.sha256(json.dumps(lists, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def test_r4_no_pair_and_no_comparable_dataset_are_worded_apart(tmp_path):
    src = isaac_file(tmp_path / 'src.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'), extra_quat=False)
    other = isaac_file(tmp_path / 'other.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'), extra_quat=False)
    with h5py.File(other, 'r+') as f:
        for demo in f['data']:
            f[f'data/{demo}/actions'][...] += 100.0  # no demo matches by actions
    assert 'No converted demo could be paired' in pair_finding(src, other)[1]['message']
    sym = isaac_file(tmp_path / 'sym.hdf5', order='XYZW', version=1, robot=[0.5, 0.5, 0.5, 0.5], extra_quat=False)
    with h5py.File(sym, 'r+') as f:
        for demo in f['data']:
            f[f'data/{demo}/initial_state/rigid_object/cube/root_pose'][..., 3:7] = 0.5
    f = pair_finding(sym, copy_file(sym, tmp_path / 'sym_copy.hdf5', version=1))[1]
    assert f['status'] == 'unknown' and 'read the same under every reorder' in f['message']


def test_n3_default_inspects_every_demo_and_catches_a_single_bad_one(tmp_path):
    kinds = ['delta'] * 100
    kinds[37] = 'absolute'
    path = robomimic_file(tmp_path / 'one.hdf5', demos=100, kinds=kinds, steps=20)
    report = check_sim_provenance(path)
    assert report['coverage']['source']['demos_inspected'] == 100
    f = finding(report, CTRL)
    assert f['status'] == 'error' and '1 of 100 inspected demos' in f['message']
    partial = finding(check_sim_provenance(path, max_demos=10), CTRL)
    assert partial['status'] in ('error', 'unknown')


def test_n3_isaac_block_of_other_order_is_found_by_default(tmp_path):
    orders = ['WXYZ'] * 80
    orders[41:45] = ['XYZW'] * 4
    path = isaac_file(tmp_path / 'block.hdf5', robots=DECISIVE, extra_quat=False, demos=80, orders=orders)
    report = check_sim_provenance(path)
    assert 'isaaclab.mixed_order_across_demos' in codes(report) and report['coverage']['source']['demos_inspected'] == 80
    sampled = check_sim_provenance(path, max_demos=8)  # partial coverage never passes
    assert 'isaaclab.format_version' not in codes(sampled, 'pass')


def test_n5_parquet_value_budget_bounds_honest_wide_columns(tmp_path, monkeypatch):
    import policy_doctor.sim_guards as sg
    monkeypatch.setattr(sg, 'MAX_PARQUET_VALUES', 4000)
    width = 1000
    state = np.zeros((100, width))
    state[:, :4] = unit_rows(100)
    names = ['qx', 'qy', 'qz', 'qw'] + [f'p{i}' for i in range(width - 4)]
    root = lerobot(tmp_path / 'wide', state_names=names, state=state)
    report = check_contract(check_sim_provenance(root))
    cov = report['coverage']['source']
    assert cov['parquet_value_budget_exhausted'] is True and cov['lerobot_rows_read'] == 4 and cov['lerobot_rows_total'] == 100
    f = finding(report, 'lerobot.declared_quaternion')
    assert f['status'] == 'unknown' and 'only 4 of 100 rows were read' in f['message']


# ---------------------------------------------------------------------------------------------- recheck3 P1, P2, W1

def roll_all(path, shift):
    """Roll the quaternion columns of every quaternion-bearing dataset (a whole-file conversion when shift=-1)."""
    with h5py.File(path, 'r+') as f:
        def visit(name, o):
            if isinstance(o, h5py.Dataset) and o.dtype.kind == 'f' and o.shape[-1] in (4, 7):
                v = o[...]
                a = 3 if v.shape[-1] == 7 else 0
                v[..., a:a + 4] = np.roll(v[..., a:a + 4], shift, axis=-1)
                o[...] = v
        f['data'].visititems(visit)
    return path


def assert_same_data(a, b):
    with h5py.File(a, 'r') as x, h5py.File(b, 'r') as y:
        x['data'].visititems(lambda n, o: np.testing.assert_array_equal(o[...], y[f'data/{n}'][...])
                             if isinstance(o, h5py.Dataset) else None)


def stamp_source(path, version):
    with h5py.File(path, 'r+') as f:
        if version is None:
            del f.attrs['format_version']
        else:
            f.attrs['format_version'] = version


def test_p1_xyzw_recording_that_lost_format_version_and_was_converted_is_conditional(tmp_path):
    rec = isaac_file(tmp_path / 'rec_v1.hdf5', order='XYZW', version=1, robot=yaw(20, 'XYZW'))  # roots undecidable
    merged = copy_without_root_attrs(rec, tmp_path / 'merged.hdf5')  # merge_hdf5_datasets.py drops format_version
    conv = convert_root_pose_only(merged, tmp_path / 'conv.hdf5')  # convert_dataset_to_xyzw(): root_pose rolled again
    report, f = pair_finding(merged, conv)
    assert f['status'] == 'error' and report['coverage']['pair']['source_declaration'] == 'not established'
    assert f['remedy'].startswith('First establish that the source is an Isaac Lab 2.x') and 'Caveat' not in f['remedy']
    assert 'If it is not (an Isaac Lab >= 3.0 recording that lost format_version)' in f['remedy']
    assert 'also stamp format_version=1 on the source file, whose missing value is wrong' in f['remedy']
    apply_fixes(conv, remedy_list(report, 'pair.fixes_if_source_xyzw'))  # the >= 3.0 branch read literally
    stamp_source(merged, 1)
    assert_same_data(rec, conv)
    assert pair_finding(merged, conv)[1]['status'] == 'pass'  # v1 source not contradicted by its roots


def test_p1_consistent_relation_with_an_unestablished_declaration_never_passes(tmp_path):
    rec = isaac_file(tmp_path / 'rec_v1.hdf5', order='XYZW', version=1, extra_quat=False)  # XYZW identity roots
    merged = copy_without_root_attrs(rec, tmp_path / 'merged.hdf5')
    conv = convert_root_pose_only(merged, tmp_path / 'conv.hdf5')  # only root_pose datasets: relation 1 as declared
    report, f = pair_finding(merged, conv)
    assert f['status'] == 'warning' and 'changed as the declared orders require' in f['message']
    assert 'Arena G1' in f['message'] and f['remedy'].startswith('First establish')
    assert remedy_list(report, 'pair.fixes') == [] and 'If it is: keep format_version=1 and change no data' in f['remedy']
    apply_fixes(conv, remedy_list(report, 'pair.fixes_if_source_xyzw'))
    assert_same_data(rec, conv)
    legacy = isaac_file(tmp_path / 'legacy.hdf5', extra_quat=False)  # the same pipeline on a 2.x recording passes
    assert pair_finding(legacy, convert_root_pose_only(legacy, tmp_path / 'lc.hdf5'))[1]['status'] == 'pass'


def test_p1_explicit_zero_xyzw_recording_rolled_by_a_whole_file_conversion_is_conditional(tmp_path):
    src = isaac_file(tmp_path / 'xyzw_v0.hdf5', order='XYZW', version=0, robot=yaw(20, 'XYZW'))  # >= 3.0 stored 0
    whole = roll_all(copy_file(src, tmp_path / 'whole.hdf5', version=1), -1)
    report, f = pair_finding(src, whole)
    assert f['status'] == 'warning' and 'stored with format_version=0' in f['remedy']
    apply_fixes(whole, remedy_list(report, 'pair.fixes_if_source_xyzw'))  # the >= 3.0 branch: undo the roll
    assert_same_data(src, whole)
    stamp_source(src, 1)
    assert pair_finding(src, whole)[1]['status'] == 'pass'


def test_p1_ceiling_recording_that_lost_format_version_gets_a_mount_caveat(tmp_path):
    rec = isaac_file(tmp_path / 'ceiling_v1.hdf5', order='XYZW', version=1, robot=[1.0, 0, 0, 0])
    merged = copy_without_root_attrs(rec, tmp_path / 'merged.hdf5')
    conv = convert_root_pose_only(merged, tmp_path / 'conv.hdf5')
    report, f = pair_finding(merged, conv)
    assert f['status'] == 'error' and report['coverage']['pair']['source_declaration'] == 'corroborated by the robot roots'
    caveat = f['remedy'].split(' Caveat: ', 1)[1]
    assert "If the source is nevertheless an Isaac Lab >= 3.0 (XYZW) recording (robots mounted on a ceiling or wall)" in caveat
    assert "remedy_lists['pair.fixes_if_source_xyzw']" in caveat and 'on the source file' in caveat
    apply_fixes(conv, remedy_list(report, 'pair.fixes_if_source_xyzw'))  # the user knows the robots hang from the ceiling
    stamp_source(merged, 1)
    assert_same_data(rec, conv)
    assert pair_finding(merged, conv)[1]['status'] == 'warning'  # the v1 source's roots contradict it: no pass


def test_p1_source_stamped_1_without_conversion_does_not_override_a_correct_conversion(tmp_path):
    legacy = isaac_file(tmp_path / 'legacy.hdf5')  # WXYZ identity roots read upright only as WXYZ
    stamped = copy_file(legacy, tmp_path / 'stamped1.hdf5', version=1)  # the source mis-stamped 1, data unconverted
    good = roll_all(copy_file(legacy, tmp_path / 'good.hdf5', version=1), -1)  # a correct whole-file conversion
    before = good.read_bytes()
    report, f = pair_finding(stamped, good)
    assert f['status'] == 'error' and f['remedy'].startswith('First establish') and 'contradict that' in f['remedy']
    assert remedy_list(report, 'pair.fixes') == []
    assert ('If it is: keep format_version=1 and change no data; also delete format_version from the source file'
            in f['remedy'])
    stamp_source(stamped, None)  # the 2.x branch: no data change, the source's stamp removed
    assert good.read_bytes() == before and pair_finding(stamped, good)[1]['status'] == 'pass'


def test_p1_every_decisive_pair_remedy_states_its_condition(tmp_path):
    v1 = isaac_file(tmp_path / 'v1.hdf5', order='XYZW', version=1)  # roots undecidable: rests on the declaration
    f = pair_finding(v1, copy_without_root_attrs(v1, tmp_path / 'm.hdf5'))[1]
    assert f['status'] == 'error' and "Caveat: this rests on the source's format_version=1" in f['remedy']
    assert ('stamped 1 without conversion: keep the format_version as it is and change no data; also delete '
            'format_version from the source file') in f['remedy']
    v1u = isaac_file(tmp_path / 'v1u.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'))  # roots confirm XYZW
    f = pair_finding(v1u, copy_without_root_attrs(v1u, tmp_path / 'mu.hdf5'))[1]
    assert 'which its robot roots confirm' in f['remedy'] and '(robots mounted on a ceiling or wall): ' in f['remedy']
    legacy = isaac_file(tmp_path / 'legacy.hdf5')
    f = pair_finding(legacy, convert_root_pose_only(legacy, tmp_path / 'c.hdf5'))[1]
    assert "Caveat: the source's robot roots read upright only as WXYZ in every paired demo" in f['remedy']


def test_p2_non_finite_rows_are_compared_on_the_finite_rows(tmp_path):
    legacy = isaac_file(tmp_path / 'legacy.hdf5')
    with h5py.File(legacy, 'r+') as f:
        f['data/demo_0/obs/eef_quat'][3] = np.nan
    conv = convert_root_pose_only(legacy, tmp_path / 'conv.hdf5')  # eef_quat left WXYZ, NaN row kept
    report, f = pair_finding(legacy, conv)
    entries = remedy_list(report, 'pair.fixes')
    assert f['status'] == 'error' and {'dataset': 'obs/eef_quat', 'quaternion_columns': '0:4', 'relation': 'identity',
                                        'roll': -1, 'demos': 'all'} in entries
    apply_fixes(conv, entries)
    assert pair_finding(legacy, conv)[1]['status'] == 'pass'
    with h5py.File(conv, 'r+') as f:  # a NaN row only in the converted file is a change beyond a reorder
        f['data/demo_1/obs/eef_quat'][2] = np.nan
    report, f = pair_finding(legacy, conv)
    assert f['status'] == 'warning' and report['coverage']['pair']['paired_datasets_changed_beyond_reorder'] == 1


def oversized_chunks(path, rel, demo='demo_1'):
    with h5py.File(path, 'r+') as f:
        q = f[f'data/{demo}/{rel}'][...]
        del f[f'data/{demo}/{rel}']
        f[f'data/{demo}'].create_dataset(rel, data=q, maxshape=(None, 4), chunks=(2_000_000, 4), compression='gzip')


@pytest.mark.parametrize('case', ['shape', 'zeroed', 'garbage', 'non_unit_both', 'unreadable'])
def test_p2_datasets_in_both_files_that_cannot_be_verified_are_named_and_block_a_pass(tmp_path, case):
    src = isaac_file(tmp_path / 'v1.hdf5', order='XYZW', version=1, robot=yaw(120, 'XYZW'))
    dst = copy_file(src, tmp_path / 'copy.hdf5', version=1)
    assert pair_finding(src, dst)[1]['status'] == 'pass'
    rel = 'obs/eef_quat'
    with h5py.File(dst, 'r+') as f:
        if case == 'shape':  # validator AT8a: reordered and one row shorter
            v = f[f'data/demo_1/{rel}'][...]
            del f[f'data/demo_1/{rel}']
            f['data/demo_1'].create_dataset(rel, data=np.roll(v, 1, axis=-1)[:-1])
        elif case == 'zeroed':  # AT8b
            rel = 'initial_state/rigid_object/cube/root_pose'
            v = f[f'data/demo_1/{rel}'][...]
            v[..., 3:7] = 0.0
            f[f'data/demo_1/{rel}'][...] = v
        elif case == 'garbage':  # AT8c
            f[f'data/demo_1/{rel}'][...] = np.random.default_rng(5).normal(size=(5, 4)) * 3
    if case == 'non_unit_both':
        for path in (src, dst):
            with h5py.File(path, 'r+') as f:
                f[f'data/demo_1/{rel}'][...] = f[f'data/demo_1/{rel}'][...] * 3
    if case == 'unreadable':
        oversized_chunks(dst, rel)
    report, f = pair_finding(src, dst)
    assert f['status'] != 'pass' and report['status'] != 'pass'
    if case in ('shape', 'zeroed', 'garbage'):
        listed = remedy_list(report, 'pair.changed_beyond_reorder')
        assert f['status'] == 'warning' and [e['dataset'] for e in listed] == [rel] and listed[0]['demos'] == ['demo_1']
        assert listed[0]['relation'] == ('shape changed' if case == 'shape' else 'changed beyond a reorder')
    else:
        listed = remedy_list(report, 'pair.not_compared')
        assert f['status'] == 'unknown' and 'were not compared' in f['message']
        assert [e['dataset'] for e in listed] == [rel] and listed[0]['demos'] == ['demo_1']
        assert listed[0]['reason'] == ('not unit quaternions in either file' if case == 'non_unit_both'
                                       else f'chunk larger than {16 * 1024 ** 2} bytes')


def test_w1_converted_demos_beyond_max_demos_are_not_inspected_and_get_no_patch(tmp_path):
    src, dst = partial_conversion(tmp_path, version=1)  # 4 converted demos, 2 inspected
    report = check_contract(check_sim_provenance(src, converted=dst, max_demos=2))
    f = finding(report, 'isaaclab.conversion_quaternion_order')
    assert f['status'] == 'error' and '2 of 4 converted demos were not inspected' in f['message']
    assert 'could not be paired' not in f['message'] and 'Regenerate' not in f['remedy']
    assert f['remedy'].startswith('Inspect every converted demo before changing data') and 'max_demos=None' in f['remedy']
    assert 'pair.fixes' not in report['remedy_lists']


def test_w1_a_converted_demo_from_another_recording_gets_per_demo_fixes_and_a_note(tmp_path):
    legacy = isaac_file(tmp_path / 'legacy.hdf5')
    conv = convert_root_pose_only(legacy, tmp_path / 'conv.hdf5')
    other = isaac_file(tmp_path / 'other.hdf5', order='XYZW', version=1, demos=3)
    with h5py.File(other, 'r') as a, h5py.File(conv, 'r+') as b:
        a.copy('data/demo_2', b['data'], 'demo_9')  # its actions match no source demo
        unrelated = b['data/demo_9/obs/eef_quat'][...]
    report, f = pair_finding(legacy, conv)
    entries = remedy_list(report, 'pair.fixes')
    assert f['status'] == 'error' and not f['remedy'].startswith('Regenerate')
    assert 'Not covered by these fixes: 1 converted demo(s) (demo_9) match no source demo' in f['remedy']
    assert entries and all(e['demos'] == ['demo_0', 'demo_1'] for e in entries)  # never 'all'
    apply_fixes(conv, entries)
    with h5py.File(conv, 'r') as b:
        np.testing.assert_array_equal(b['data/demo_9/obs/eef_quat'][...], unrelated)
    f2 = pair_finding(legacy, conv)[1]
    assert f2['status'] == 'unknown' and 'match no source demo by byte-identical "actions"' in f2['message']
    dup = isaac_file(tmp_path / 'dup.hdf5', demos=3)  # demos sharing actions in the source still regenerate
    with h5py.File(dup, 'r+') as f:
        f['data/demo_1/actions'][...] = f['data/demo_0/actions'][...]
    f3 = pair_finding(dup, convert_root_pose_only(dup, tmp_path / 'dup_c.hdf5'))[1]
    assert f3['remedy'].startswith('Regenerate') and 'match several source demos' in f3['remedy']


def test_p2_contradictions_beside_uncompared_datasets_patch_only_what_was_verified(tmp_path):
    legacy = isaac_file(tmp_path / 'legacy.hdf5')
    with h5py.File(legacy, 'r+') as f:  # not unit in either file: named, never rolled
        f['data/demo_1/obs/eef_quat'][...] = f['data/demo_1/obs/eef_quat'][...] * 3
    conv = convert_root_pose_only(legacy, tmp_path / 'conv.hdf5')
    with h5py.File(conv, 'r') as f:
        scaled = f['data/demo_1/obs/eef_quat'][...]
    report, f = pair_finding(legacy, conv)
    entries = remedy_list(report, 'pair.fixes')
    assert f['status'] == 'error' and 'Not covered either:' in f['remedy'] and 'not unit quaternions' in f['remedy']
    assert [(e['dataset'], e['demos']) for e in entries] == [('obs/eef_quat', ['demo_0'])]
    apply_fixes(conv, entries)
    with h5py.File(conv, 'r') as f:
        np.testing.assert_array_equal(f['data/demo_1/obs/eef_quat'][...], scaled)
    assert pair_finding(legacy, conv)[1]['status'] == 'unknown'  # still named, never a pass
    plain = isaac_file(tmp_path / 'plain.hdf5')
    unreadable = convert_root_pose_only(plain, tmp_path / 'unreadable.hdf5')
    oversized_chunks(unreadable, 'obs/eef_quat')  # an instance of this conversion the check cannot read
    f = pair_finding(plain, unreadable)[1]
    assert f['remedy'].startswith('Regenerate') and 'chunk larger than' in f['remedy']


@pytest.mark.parametrize('source_version', [None, 0, 1])
@pytest.mark.parametrize('converted_kind', ['copy', 'convert', 'xyzw_demos_only'])
def test_d1_mixed_version_source_has_no_whole_file_fix_or_restamp(tmp_path, source_version, converted_kind):
    source = isaac_file(tmp_path / 'mixed.hdf5', version=source_version, demos=4, robots=DECISIVE,
                        orders=['WXYZ', 'WXYZ', 'XYZW', 'XYZW'])
    if converted_kind == 'convert':
        converted = convert_root_pose_only(source, tmp_path / 'converted.hdf5')
    else:
        converted = copy_file(source, tmp_path / 'converted.hdf5', version=1)
        if converted_kind == 'xyzw_demos_only':
            with h5py.File(converted, 'r+') as f:
                del f['data/demo_0']
                del f['data/demo_1']
    original_bytes = source.read_bytes(), converted.read_bytes()
    report, f = pair_finding(source, converted)
    assert f['status'] == 'warning' and 'conflicting' in f['message']
    assert f['remedy'].startswith('Change no data or format_version on either file')
    assert 'per-version originals' in f['remedy']
    assert not any(key.startswith('pair.fix') for key in report['remedy_lists'])
    assert report['coverage']['pair']['source_declaration'] == 'conflicting robot-root votes; no whole-file remedy'
    assert (source.read_bytes(), converted.read_bytes()) == original_bytes


def test_d2_wall_mounted_roots_only_pass_names_the_general_mount_condition(tmp_path):
    # XB4d: a native >=3.0 wall recording can be indistinguishable from a correct upright legacy conversion.
    wall = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    recording = isaac_file(tmp_path / 'wall.hdf5', order='XYZW', version=1, robot=wall, extra_quat=False, demos=4)
    merged = copy_without_root_attrs(recording, tmp_path / 'merged.hdf5')
    converted = convert_root_pose_only(merged, tmp_path / 'converted.hdf5')
    report, f = pair_finding(merged, converted)
    assert f['status'] == 'pass'
    assert 'Ceiling or wall mounts' in report['scope'][:600] and 'including an overall PASS' in report['scope'][:600]
    assert f['message'].startswith('Only under the stated source-version and upright floor/table-mount conditions')
    assert 'ceiling or wall mount' in f['remedy']
    for item in report['findings']:
        if item['code'] == 'isaaclab.format_version' and item['status'] == 'pass':
            assert 'known to start upright on a floor or table only' in item['message'][:150]


@pytest.mark.parametrize('version', [None, 0, 1, 'bad', 2])
@pytest.mark.parametrize('recorder', [None, '3.0.0'])
@pytest.mark.parametrize('paired', [False, True])
def test_d1_all_mixed_source_branches_refuse_file_wide_attribute_advice(tmp_path, version, recorder, paired):
    source = isaac_file(tmp_path / 'mixed.hdf5', version=version, demos=4, robots=DECISIVE,
                        orders=['WXYZ', 'WXYZ', 'XYZW', 'XYZW'],
                        attrs={'isaaclab_version': recorder} if recorder else None)
    converted = copy_file(source, tmp_path / 'converted.hdf5', version=1) if paired else None
    before = source.read_bytes()
    report = check_contract(check_sim_provenance(source, converted=converted))
    source_findings = [f for f in report['findings'] if f['message'].startswith('[source]')]
    assert source_findings and all(f['remedy'].startswith('Change no data or format_version in this file')
                                   for f in source_findings)
    assert not any(k.startswith(('source.non_root', 'pair.fix')) for k in report['remedy_lists'])
    assert ('isaaclab.format_version' in codes(report, 'error')) == (version == 'bad')
    if paired:
        assert finding(report, 'isaaclab.conversion_quaternion_order')['remedy'].startswith(
            'Change no data or format_version on either file')
    assert source.read_bytes() == before
