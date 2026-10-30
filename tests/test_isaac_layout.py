"""Producer guards using disclosed Python objects, never claimed native sensors."""
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from policy_doctor.artifacts import InputError


def fake_term(*args, **kwargs):
    raise AssertionError('The producer must not rerun the observation term')


class LiveNamesOnlySensor:
    def __init__(self):
        self.cfg = SimpleNamespace(prim_path='/World/envs/env_.*/Robot')
        self.name_reads = 0

    @property
    def body_names(self):
        self.name_reads += 1
        return ['right_foot', 'left_foot']

    @property
    def data(self):
        raise AssertionError('The producer must never reread sensor.data')


def capture(monkeypatch, **changes):
    module = importlib.import_module('policy_doctor.isaac_layout')
    sensor = LiveNamesOnlySensor()
    force = np.array([[[1., 2., 3.], [11., 12., 13.]]], dtype=np.float32)
    torque = force + np.float32(3.)
    data = SimpleNamespace(force=SimpleNamespace(torch=force), torque=SimpleNamespace(torch=torque))
    cfg = SimpleNamespace(body_ids=[1, 0], body_names=['left_foot', 'right_foot'], preserve_order=True,
                          name='joint_wrench')
    raw = np.concatenate((force[:, [1, 0]], torque[:, [1, 0]]), axis=-1).reshape(1, 12)
    args = {'sensor': sensor, 'sensor_cfg': cfg, 'cached_data': data, 'raw_term_output': raw,
            'term_function': fake_term, 'term_cfg': SimpleNamespace(scale=.01, modifiers=[], clip=None, history_length=0)}
    args.update(changes)

    def writer(output, metadata, arrays, *, sources):
        return {'output': output, 'metadata': metadata, 'arrays': arrays, 'sources': sources}

    monkeypatch.setattr(module, '_write_capture', writer)
    result = module.capture_isaac_joint_wrench_layout('/unused-new-case', **args)
    return result, sensor, force, torque, raw


def test_captures_observed_names_ids_and_cached_buffers_without_sensor_update(monkeypatch):
    result, sensor, force, torque, raw = capture(monkeypatch)
    assert sensor.name_reads == 1
    assert result['metadata']['runtime']['sensor_body_names'] == ['right_foot', 'left_foot']
    assert result['metadata']['runtime']['resolved_body_ids'] == [1, 0]
    assert result['metadata']['runtime']['sensor_identity'] == {'prim_path': '/World/envs/env_.*/Robot'}
    np.testing.assert_array_equal(result['arrays']['source_force'], force)
    np.testing.assert_array_equal(result['arrays']['source_torque'], torque)
    np.testing.assert_array_equal(result['arrays']['raw_term_output'], raw)
    assert 'component_names' not in result['metadata']
    assert result['metadata']['provenance']['physical_sample_identity'] == 'unknown'


def test_snapshots_are_independent_of_later_buffer_mutation(monkeypatch):
    result, _, force, torque, raw = capture(monkeypatch)
    force[:] = torque[:] = raw[:] = 999.
    assert result['arrays']['source_force'][0, 0, 0] == 1.
    assert result['arrays']['source_torque'][0, 0, 0] == 4.
    assert result['arrays']['raw_term_output'][0, 0] == 11.


def test_preserves_actual_wrong_output_for_auditor_instead_of_recomputing_it(monkeypatch):
    wrong = np.full((1, 12), 42., dtype=np.float32)
    result, *_ = capture(monkeypatch, raw_term_output=wrong)
    np.testing.assert_array_equal(result['arrays']['raw_term_output'], wrong)


def test_missing_cached_buffer_remains_explicit(monkeypatch):
    result, *_ = capture(monkeypatch, cached_data=SimpleNamespace(force=None, torque=None))
    assert 'source_force' not in result['arrays']
    assert 'source_torque' not in result['arrays']
    assert result['metadata']['provenance']['missing_buffers'] == ['source_force', 'source_torque']


def test_slice_ids_expand_using_actual_native_order(monkeypatch):
    cfg = SimpleNamespace(body_ids=slice(None), body_names=['.*foot'], preserve_order=False, name='joint_wrench')
    result, *_ = capture(monkeypatch, sensor_cfg=cfg)
    assert result['metadata']['runtime']['resolved_body_ids'] == [0, 1]


def test_missing_sensor_namespace_is_not_invented(monkeypatch):
    sensor = LiveNamesOnlySensor()
    sensor.cfg = SimpleNamespace()
    result, *_ = capture(monkeypatch, sensor=sensor)
    assert result['metadata']['runtime']['sensor_identity'] is None


def test_processed_output_is_optional_and_scale_is_from_actual_cfg(monkeypatch):
    result, *_ = capture(monkeypatch, term_cfg=None)
    assert result['metadata']['processing']['scale'] is None
    assert 'processed_term_output' not in result['arrays']


def test_absent_processing_stage_evidence_is_unknown(monkeypatch):
    result, *_ = capture(monkeypatch, term_cfg=SimpleNamespace(scale=.01))
    for key in ('modifiers', 'clip', 'history', 'noise', 'delay'):
        assert result['metadata']['processing'][key] == 'unknown'


def test_explicit_disabled_processing_stages_are_preserved(monkeypatch):
    cfg = SimpleNamespace(scale=.01, modifiers=[], clip=None, history_length=0, noise=None,
                          delay_min_lag=0, delay_max_lag=0, delay_hold_prob=0.)
    result, *_ = capture(monkeypatch, term_cfg=cfg)
    processing = result['metadata']['processing']
    assert processing['noise'] is None and processing['delay'] is None
    assert processing['modifiers'] == [] and processing['history'] == 0


def test_large_batch_requires_explicit_bounded_environment_selection(monkeypatch):
    force = np.arange(64 * 6, dtype=np.float32).reshape(64, 2, 3)
    data = SimpleNamespace(force=SimpleNamespace(torch=force), torque=SimpleNamespace(torch=force + 1))
    raw = np.concatenate((force[:, [1, 0]], (force + 1)[:, [1, 0]]), axis=-1).reshape(64, 12)
    with pytest.raises(InputError, match='env_indices'):
        capture(monkeypatch, cached_data=data, raw_term_output=raw)
    result, *_ = capture(monkeypatch, cached_data=data, raw_term_output=raw, env_indices=[0, 63])
    np.testing.assert_array_equal(result['arrays']['source_force'], force[[0, 63]])
    np.testing.assert_array_equal(result['arrays']['raw_term_output'], raw[[0, 63]])
    assert result['metadata']['runtime']['source_num_envs'] == 64
    assert result['metadata']['runtime']['env_indices'] == [0, 63]
    assert result['metadata']['runtime']['omitted_env_count'] == 62


@pytest.mark.parametrize('selection', [[True], [-1], [1], [0, 0], [], list(range(33)), '0'])
def test_invalid_environment_selection_refused(monkeypatch, selection):
    with pytest.raises(InputError, match='env_indices'):
        capture(monkeypatch, env_indices=selection)


@pytest.mark.parametrize('ids', [[True], [-1], [2], [0, 0], [], ['1']])
def test_invalid_resolved_ids_refused(monkeypatch, ids):
    cfg = SimpleNamespace(body_ids=ids, body_names=['.*foot'], preserve_order=False, name='joint_wrench')
    with pytest.raises(InputError):
        capture(monkeypatch, sensor_cfg=cfg)


@pytest.mark.parametrize('value', [[[1.]], np.array([[float('nan')] * 12]), np.ones((1, 2, 6)),
                                  np.array([['x'] * 12]), np.array([[object()] * 12], dtype=object)])
def test_unsafe_raw_output_or_implicit_array_conversion_refused(monkeypatch, value):
    with pytest.raises(InputError):
        capture(monkeypatch, raw_term_output=value)


def test_source_identity_records_real_python_objects_without_native_relabelling(monkeypatch):
    result, *_ = capture(monkeypatch)
    source = result['metadata']['source']
    assert source['term']['module'] == __name__
    assert source['term']['qualname'] == 'fake_term'
    assert source['sensor_class']['qualname'] == 'LiveNamesOnlySensor'
    assert source['term']['source_file'] in result['sources']
    assert source['files'][source['term']['source_file']]['sha256'] == source['term']['sha256']


def profiled_input(monkeypatch):
    module = importlib.import_module('policy_doctor.isaac_layout')
    data = SimpleNamespace(force=SimpleNamespace(torch=np.ones((1, 2, 3), dtype=np.float32)),
                           torque=SimpleNamespace(torch=np.full((1, 2, 3), 2., dtype=np.float32)))

    class CountingSensor(LiveNamesOnlySensor):
        reads = 0

        @property
        def data(self):
            self.reads += 1
            return data

    sensor = CountingSensor()
    cfg = SimpleNamespace(name='joint_wrench', body_ids=[0, 1], body_names=['.*foot'], preserve_order=False)
    env = SimpleNamespace(scene=SimpleNamespace(sensors={'joint_wrench': sensor}), num_envs=1)
    publications = []

    def publish(output, metadata, arrays, *, sources):
        publications.append((metadata, arrays))
        return {'published': True}

    monkeypatch.setattr(module, '_write_capture', publish)
    return module, env, cfg, sensor, data, publications


def normal_numpy_term(env, sensor_cfg):
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_data = sensor.data
    force_data, torque_data = sensor_data.force, sensor_data.torque
    force = force_data.torch[:, sensor_cfg.body_ids]
    torque = torque_data.torch[:, sensor_cfg.body_ids]
    return np.concatenate((force, torque), axis=-1).reshape(env.num_envs, -1)


def generated_torch_term(env, sensor_cfg):
    torch = importlib.import_module('torch')
    sensor = env.scene.sensors[sensor_cfg.name]
    sensor_data = sensor.data
    force_data, torque_data = sensor_data.force, sensor_data.torque
    force = force_data.torch[:, sensor_cfg.body_ids]
    torque = torque_data.torch[:, sensor_cfg.body_ids]
    result = torch.cat((force, torque), dim=-1).view(env.num_envs, -1)
    env.original_result = result
    return result


@pytest.mark.parametrize('count,selection', [(64, [0, 63]), (4096, [0, 4095]), (4096, [4095, 0])])
def test_torch_batch_selection_precedes_cpu_copy_and_preserves_full_return(monkeypatch, count, selection):
    torch = pytest.importorskip('torch')
    module, env, cfg, sensor, data, publications = profiled_input(monkeypatch)
    env.num_envs = count
    force = torch.arange(count * 6, dtype=torch.float32).view(count, 2, 3)
    data.force.torch, data.torque.torch = force, force + 3
    copied_shapes = []
    original_cpu = torch.Tensor.cpu

    def cpu(value, *args, **kwargs):
        copied_shapes.append(tuple(value.shape))
        return original_cpu(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'cpu', cpu)
    result, receipt = module.capture_isaac_joint_wrench_call(
        '/unused', term_function=generated_torch_term, env=env, sensor_cfg=cfg,
        allow_tensor_snapshot=True, env_indices=selection)
    assert receipt == {'published': True} and sensor.reads == 1
    assert result is env.original_result
    assert tuple(result.shape) == (count, 12)
    assert copied_shapes == [(2, 2, 3), (2, 2, 3), (2, 12)]
    metadata, arrays = publications[0]
    assert metadata['runtime']['source_num_envs'] == count
    assert metadata['runtime']['env_indices'] == selection
    assert metadata['runtime']['omitted_env_count'] == count - 2
    assert arrays['source_force'].dtype == np.float32
    np.testing.assert_array_equal(arrays['source_force'], force[selection].numpy())
    np.testing.assert_array_equal(arrays['raw_term_output'], result[selection].numpy())


def test_invalid_torch_row_selection_copies_no_tensor(monkeypatch):
    torch = pytest.importorskip('torch')
    module, env, cfg, sensor, data, publications = profiled_input(monkeypatch)
    data.force.torch = torch.ones((1, 2, 3))
    data.torque.torch = torch.ones((1, 2, 3))

    def forbidden_cpu(*args, **kwargs):
        raise AssertionError('Invalid row selection must be refused before any CPU copy')

    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden_cpu)
    result, receipt = module.capture_isaac_joint_wrench_call(
        '/unused', term_function=generated_torch_term, env=env, sensor_cfg=cfg,
        allow_tensor_snapshot=True, env_indices=[True])
    assert tuple(result.shape) == (1, 12) and sensor.reads == 1
    assert receipt['published'] is False and 'env_indices' in receipt['reason']
    assert publications == []


def test_explicit_native_torch_long_body_ids_are_copied_without_mutation(monkeypatch):
    torch = pytest.importorskip('torch')
    selector = torch.tensor([1, 0], dtype=torch.long)
    cfg = SimpleNamespace(name='wrench', body_ids=selector, body_names=['left_foot', 'right_foot'],
                          preserve_order=True)
    original = selector.clone()
    result, *_ = capture(monkeypatch, sensor_cfg=cfg, allow_tensor_snapshot=True)
    assert result['metadata']['runtime']['resolved_body_ids'] == [1, 0]
    assert cfg.body_ids is selector and torch.equal(selector, original)


def test_torch_body_ids_require_explicit_copy_opt_in(monkeypatch):
    torch = pytest.importorskip('torch')
    cfg = SimpleNamespace(body_ids=torch.tensor([1, 0]), preserve_order=True, body_names=None)
    with pytest.raises(InputError, match='allow_tensor_snapshot'):
        capture(monkeypatch, sensor_cfg=cfg)


@pytest.mark.parametrize('case', ['rank', 'float', 'bool', 'empty', 'too_many'])
def test_torch_body_id_type_shape_limits_precede_cpu_copy(monkeypatch, case):
    torch = pytest.importorskip('torch')
    selector = {'rank': torch.ones((1, 2), dtype=torch.long), 'float': torch.tensor([1., 0.]),
                'bool': torch.tensor([True, False]), 'empty': torch.tensor([], dtype=torch.long),
                'too_many': torch.arange(257)}[case]
    cfg = SimpleNamespace(body_ids=selector, preserve_order=True, body_names=None)

    def forbidden_cpu(*args, **kwargs):
        raise AssertionError('Selector rank/dtype/count must be checked before CPU copy')

    monkeypatch.setattr(torch.Tensor, 'cpu', forbidden_cpu)
    with pytest.raises(InputError):
        capture(monkeypatch, sensor_cfg=cfg, allow_tensor_snapshot=True)


@pytest.mark.parametrize('values', [[0, 0], [-1], [2], [2**63 - 1]])
def test_invalid_bounded_torch_body_id_values_refused(monkeypatch, values):
    torch = pytest.importorskip('torch')
    cfg = SimpleNamespace(body_ids=torch.tensor(values, dtype=torch.long), preserve_order=True, body_names=None)
    with pytest.raises(InputError):
        capture(monkeypatch, sensor_cfg=cfg, allow_tensor_snapshot=True)


def test_one_call_with_finalized_tensor_ids_and_row_selection_preserves_shared_objects(monkeypatch):
    torch = pytest.importorskip('torch')
    module, env, cfg, sensor, data, publications = profiled_input(monkeypatch)
    env.num_envs = 4096
    data.force.torch = torch.arange(4096 * 6, dtype=torch.float32).view(4096, 2, 3)
    data.torque.torch = data.force.torch + 3
    selector = torch.tensor([1, 0], dtype=torch.long)
    cfg.body_ids = selector
    result, receipt = module.capture_isaac_joint_wrench_call(
        '/unused', term_function=generated_torch_term, env=env, sensor_cfg=cfg,
        allow_tensor_snapshot=True, env_indices=[0, 4095])
    assert receipt == {'published': True} and sensor.reads == 1
    assert result is env.original_result and list(result.shape) == [4096, 12]
    assert cfg.body_ids is selector and selector.tolist() == [1, 0]
    metadata, arrays = publications[0]
    assert metadata['runtime']['resolved_body_ids'] == [1, 0]
    np.testing.assert_array_equal(arrays['raw_term_output'], result[[0, 4095]].numpy())


def test_one_call_wrapper_keeps_actual_return_and_reads_sensor_once(monkeypatch):
    module, env, cfg, sensor, data, publications = profiled_input(monkeypatch)
    previous = sys.getprofile()
    value, receipt = module.capture_isaac_joint_wrench_call('/unused', term_function=normal_numpy_term,
                                                          env=env, sensor_cfg=cfg)
    assert sys.getprofile() is previous
    assert sensor.reads == 1
    assert receipt == {'published': True}
    assert len(publications) == 1
    assert publications[0][0]['provenance']['capture_method'] == 'profiled_exact_function_return'
    np.testing.assert_array_equal(publications[0][1]['raw_term_output'], value)
    data.force.torch[:] = 9.
    assert publications[0][1]['source_force'][0, 0, 0] == 1.


def test_one_call_wrapper_refuses_existing_profiler_without_invocation(monkeypatch):
    module, env, cfg, sensor, _, publications = profiled_input(monkeypatch)

    def profiler(frame, event, arg):
        pass

    previous = sys.getprofile()
    try:
        sys.setprofile(profiler)
        with pytest.raises(InputError, match='profiler'):
            module.capture_isaac_joint_wrench_call('/unused', term_function=normal_numpy_term,
                                                   env=env, sensor_cfg=cfg)
        assert sys.getprofile() is profiler
    finally:
        sys.setprofile(previous)
    assert sensor.reads == 0 and publications == []


def test_one_call_wrapper_preserves_term_exception_and_publishes_nothing(monkeypatch):
    module, env, cfg, sensor, _, publications = profiled_input(monkeypatch)
    sentinel = RuntimeError('original term exception')

    def failing(env, sensor_cfg):
        sensor = env.scene.sensors[sensor_cfg.name]
        sensor_data = sensor.data
        assert sensor_data is not None
        raise sentinel

    previous = sys.getprofile()
    with pytest.raises(RuntimeError, match='original term exception') as caught:
        module.capture_isaac_joint_wrench_call('/unused', term_function=failing, env=env, sensor_cfg=cfg)
    assert caught.value is sentinel
    assert sys.getprofile() is previous
    assert sensor.reads == 1 and publications == []


def test_one_call_wrapper_restores_profiler_after_bad_snapshot(monkeypatch):
    module, env, cfg, sensor, data, publications = profiled_input(monkeypatch)
    data.force.torch[:] = np.nan
    previous = sys.getprofile()
    result, receipt = module.capture_isaac_joint_wrench_call('/unused', term_function=normal_numpy_term,
                                                            env=env, sensor_cfg=cfg)
    assert sys.getprofile() is previous
    assert np.isnan(result).any()
    assert receipt['status'] == 'unknown' and receipt['published'] is False
    assert sensor.reads == 1 and publications == []


def test_one_call_wrapper_does_not_overwrite_callee_profiler(monkeypatch):
    module, env, cfg, _, _, publications = profiled_input(monkeypatch)

    def foreign(frame, event, arg):
        pass

    def installs_profiler(env, sensor_cfg):
        sys.setprofile(foreign)
        return 'native-result'

    previous = sys.getprofile()
    try:
        result, receipt = module.capture_isaac_joint_wrench_call('/unused', term_function=installs_profiler,
                                                                env=env, sensor_cfg=cfg)
        assert sys.getprofile() is foreign
    finally:
        sys.setprofile(previous)
    assert result == 'native-result'
    assert receipt['status'] == 'unknown' and receipt['published'] is False
    assert publications == []


def test_recursive_native_function_still_runs_but_capture_is_not_published(monkeypatch):
    module, env, cfg, _, _, publications = profiled_input(monkeypatch)
    counter = []

    def recursive(env, sensor_cfg):
        counter.append(1)
        if len(counter) == 1:
            return recursive(env, sensor_cfg)
        return 'native-result'

    result, receipt = module.capture_isaac_joint_wrench_call('/unused', term_function=recursive,
                                                            env=env, sensor_cfg=cfg)
    assert len(counter) == 2 and result == 'native-result'
    assert receipt['status'] == 'unknown' and receipt['published'] is False
    assert publications == []


def test_normal_none_return_is_preserved_without_publication(monkeypatch):
    module, env, cfg, _, _, publications = profiled_input(monkeypatch)

    def returns_none(env, sensor_cfg):
        return None

    result, receipt = module.capture_isaac_joint_wrench_call('/unused', term_function=returns_none,
                                                            env=env, sensor_cfg=cfg)
    assert result is None and receipt['published'] is False
    assert receipt['status'] == 'unknown' and publications == []


def test_publication_failure_does_not_discard_successful_native_result(monkeypatch):
    module, env, cfg, sensor, _, _ = profiled_input(monkeypatch)

    def failing_writer(*args, **kwargs):
        raise InputError('publication blocked')

    monkeypatch.setattr(module, '_write_capture', failing_writer)
    result, receipt = module.capture_isaac_joint_wrench_call('/unused', term_function=normal_numpy_term,
                                                            env=env, sensor_cfg=cfg)
    assert result.shape == (1, 12) and sensor.reads == 1
    assert receipt['published'] is False and 'publication blocked' in receipt['reason']
