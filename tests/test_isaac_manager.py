"""Controlled manager-call mechanics; these classes are not native IsaacLab."""
import gc
import hashlib
import importlib
import sys
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_isaac_layout import normal_numpy_term, profiled_input

from policy_doctor.artifacts import InputError


class ControlledManager:
    def __init__(self, env, sensor_cfg):
        self._env = env
        term = SimpleNamespace(func=normal_numpy_term, params={'sensor_cfg': sensor_cfg},
                               scale=.01, modifiers=None, clip=None, history_length=0,
                               noise=None, delay_min_lag=0, delay_max_lag=0, delay_hold_prob=0.)
        self.cfg = SimpleNamespace(policy=SimpleNamespace(wrench=term))
        self.active_terms = {'policy': ['wrench']}
        self.raw_results = {}
        self.results = {}
        self.raise_after = None
        self.fail_term = None

    def compute_group(self, group_name):
        group_obs = {}
        for term_name in self.active_terms[group_name]:
            term_cfg = getattr(getattr(self.cfg, group_name), term_name)
            if self.fail_term is not None:
                raise self.fail_term
            obs = term_cfg.func(self._env, **term_cfg.params)
            self.raw_results[term_name] = obs
            group_obs[term_name] = obs * term_cfg.scale
        if self.raise_after is not None:
            raise self.raise_after
        result = np.concatenate(list(group_obs.values()), axis=-1)
        self.results[group_name] = result
        return result

    def compute(self):
        return {name: self.compute_group(name) for name in self.active_terms}


@pytest.fixture
def controlled(monkeypatch):
    producer, env, cfg, sensor, data, publications = profiled_input(monkeypatch)
    module = importlib.import_module('policy_doctor.isaac_manager')
    manager = ControlledManager(env, cfg)
    # Only the exact native-source preflight is bypassed for mechanical tests.
    # Real object/source identities remain captured; no native semantic pass is asserted.
    monkeypatch.setattr(module, '_manager_definition', lambda obj: ControlledManager.compute_group)
    monkeypatch.setattr(module, 'MANAGER_SHA256', hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return module, producer, manager, sensor, data, publications


def start(module, manager, **kwargs):
    return module.capture_isaac_observation(manager, 'unused-controlled-destination',
                                          group='policy', term='wrench', **kwargs)


def test_normal_manager_return_and_actual_processed_term_are_preserved(controlled):
    module, _, manager, sensor, _, publications = controlled
    # Constructor/probe-like calls precede installation and are not captured.
    manager.compute()
    term_cfg = manager.cfg.policy.wrench
    original_function = term_cfg.func
    before = sensor.reads
    with start(module, manager) as capture:
        result = manager.compute()
        assert not publications
    assert result['policy'] is manager.results['policy']
    assert term_cfg.func is original_function
    assert sensor.reads == before + 1
    assert capture.receipt['published'] is True
    metadata, arrays = publications[0]
    np.testing.assert_array_equal(arrays['raw_term_output'], manager.raw_results['wrench'])
    np.testing.assert_array_equal(arrays['processed_term_output'], result['policy'])
    assert metadata['processing']['scale'] == .01
    assert metadata['provenance']['manager_observation']['group'] == 'policy'


def test_sibling_term_using_same_function_is_not_misbound(controlled):
    module, _, manager, sensor, _, publications = controlled
    manager.active_terms['policy'] = ['other', 'wrench']
    manager.cfg.policy.other = SimpleNamespace(**vars(manager.cfg.policy.wrench))
    manager.cfg.policy.other.scale = .5
    with start(module, manager) as capture:
        result = manager.compute_group('policy')
    assert capture.receipt['published'] and sensor.reads == 2
    assert result.shape == (1, 24)
    np.testing.assert_array_equal(publications[0][1]['processed_term_output'], result[:, 12:])


def test_unrelated_group_is_ignored(controlled):
    module, _, manager, _, _, publications = controlled
    manager.active_terms['other'] = ['wrench']
    manager.cfg.other = SimpleNamespace(wrench=manager.cfg.policy.wrench)
    with start(module, manager) as capture:
        manager.compute()
    assert capture.receipt['published'] and len(publications) == 1


@pytest.mark.parametrize('calls', [0, 2])
def test_requires_one_selected_normal_manager_call(controlled, calls):
    module, _, manager, _, _, publications = controlled
    with start(module, manager) as capture:
        for _ in range(calls):
            manager.compute()
    assert not capture.receipt['published'] and not publications


def test_outer_exception_discards_staged_capture_and_restores_profile(controlled):
    module, _, manager, _, _, publications = controlled
    error = RuntimeError('outer failure')
    previous = sys.getprofile()
    with pytest.raises(RuntimeError) as caught, start(module, manager) as capture:
        manager.compute()
        raise error
    assert caught.value is error and not publications
    assert not capture.receipt['published'] and sys.getprofile() is previous


@pytest.mark.parametrize('location', ['fail_term', 'raise_after'])
def test_normal_manager_exceptions_keep_identity(controlled, location):
    module, _, manager, _, _, publications = controlled
    error = RuntimeError('native sentinel')
    setattr(manager, location, error)
    with pytest.raises(RuntimeError) as caught, start(module, manager) as capture:
        manager.compute()
    assert caught.value is error and not publications and not capture.receipt['published']
    assert sys.getprofile() is None


def test_caught_group_failure_is_not_published(controlled):
    module, _, manager, _, _, publications = controlled
    manager.raise_after = RuntimeError('caught inside caller')
    with start(module, manager) as capture:
        with pytest.raises(RuntimeError):
            manager.compute()
    assert not capture.receipt['published'] and not publications


def test_exception_inside_actual_term_preserves_original_exception(controlled, monkeypatch):
    module, _, manager, sensor, _, publications = controlled
    error = RuntimeError('sensor read sentinel')
    def fail(self):
        raise error
    monkeypatch.setattr(type(sensor), 'data', property(fail))
    with pytest.raises(RuntimeError) as caught, start(module, manager) as capture:
        manager.compute()
    assert caught.value is error and not publications and not capture.receipt['published']
    assert sys.getprofile() is None


@pytest.mark.parametrize('rows', [[], [True], [0, 0], [-1], [1.], list(range(33))])
def test_invalid_row_selectors_refuse_before_native_call(controlled, rows):
    module, _, manager, sensor, _, publications = controlled
    with pytest.raises(InputError):
        with start(module, manager, env_indices=rows):
            manager.compute()
    assert sensor.reads == 0 and not publications and sys.getprofile() is None


def test_manager_source_change_during_capture_preserves_return_but_not_publication(controlled, monkeypatch):
    module, _, manager, _, _, publications = controlled
    with start(module, manager) as capture:
        monkeypatch.setattr(module, 'MANAGER_SHA256', '0' * 64)
        result = manager.compute()
    assert result['policy'] is manager.results['policy']
    assert not capture.receipt['published'] and not publications


def test_existing_profiler_refuses_before_read(controlled):
    module, _, manager, sensor, _, publications = controlled
    def profiler(*args):
        return None
    previous = sys.getprofile()
    try:
        sys.setprofile(profiler)
        with pytest.raises(InputError, match='profiler'):
            with start(module, manager):
                manager.compute()
        assert sys.getprofile() is profiler and sensor.reads == 0 and not publications
    finally:
        sys.setprofile(previous)


def test_foreign_profiler_is_preserved(controlled):
    module, _, manager, _, _, publications = controlled
    def foreign(*args):
        return None
    previous = sys.getprofile()
    try:
        with start(module, manager) as capture:
            manager.compute()
            sys.setprofile(foreign)
        assert sys.getprofile() is foreign and not capture.receipt['published'] and not publications
    finally:
        sys.setprofile(previous)


def test_nested_capture_refuses_without_disabling_outer(controlled):
    module, _, manager, _, _, publications = controlled
    with start(module, manager) as capture:
        with pytest.raises(InputError, match='active|Recursive|nested'):
            with start(module, manager):
                pass
        manager.compute()
    assert capture.receipt['published'] and len(publications) == 1


def test_snapshot_failure_preserves_normal_manager_result(controlled):
    module, _, manager, _, data, publications = controlled
    data.force.torch[:] = np.nan
    with start(module, manager) as capture:
        result = manager.compute()
    assert np.isnan(result['policy']).any()
    assert not capture.receipt['published'] and not publications


def test_publication_failure_preserves_normal_manager_result(controlled, monkeypatch):
    module, producer, manager, _, _, _ = controlled
    def fail(*args, **kwargs):
        raise InputError('publication blocked')
    monkeypatch.setattr(producer, '_write_capture', fail)
    with start(module, manager) as capture:
        result = manager.compute()
    assert result['policy'] is manager.results['policy'] and not capture.receipt['published']


def test_actual_writer_success_explicitly_distinguishes_publication_from_audit_status(controlled, monkeypatch, tmp_path):
    module, producer, manager, _, _, _ = controlled
    from policy_doctor.layouts import write_layout_capture
    monkeypatch.setattr(producer, '_write_capture', write_layout_capture)
    output = tmp_path / 'actual-capture'
    with module.capture_isaac_observation(manager, output, group='policy', term='wrench') as capture:
        result = manager.compute()
        assert not output.exists()
    assert output.is_dir() and result['policy'] is manager.results['policy']
    assert capture.receipt['published'] is True
    assert capture.receipt['kind'] == 'resolved_layout_audit'
    # These are generated lookalike classes, so publication does not imply native semantics.
    assert capture.receipt['status'] == 'unknown'


def test_explicit_rows_bound_capture_but_leave_full_group_result(controlled):
    module, _, manager, sensor, data, publications = controlled
    manager._env.num_envs = 4096
    data.force.torch = np.arange(4096 * 6, dtype=np.float32).reshape(4096, 2, 3)
    data.torque.torch = np.ones((4096, 2, 3), dtype=np.float32)
    with start(module, manager, env_indices=[4095, 0]) as capture:
        result = manager.compute()
    assert result['policy'].shape == (4096, 12) and sensor.reads == 1
    metadata, arrays = publications[0]
    assert capture.receipt['published'] and metadata['runtime']['env_indices'] == [4095, 0]
    np.testing.assert_array_equal(arrays['processed_term_output'], result['policy'][[4095, 0]])


def test_exited_context_does_not_retain_manager_or_environment(controlled):
    module, _, manager, _, _, _ = controlled
    manager = ControlledManager(manager._env, manager.cfg.policy.wrench.params['sensor_cfg'])
    ref = weakref.ref(manager)
    with start(module, manager) as capture:
        manager.compute()
    del controlled, manager
    gc.collect()
    assert ref() is None and capture.receipt['published']


def test_thread_exit_must_restore_on_owning_thread(controlled):
    module, _, manager, _, _, _ = controlled
    capture = start(module, manager)
    capture.__enter__()
    failures = []
    def foreign_exit():
        try:
            capture.__exit__(None, None, None)
        except InputError as exc:
            failures.append(str(exc))
    worker = threading.Thread(target=foreign_exit)
    try:
        worker.start()
        worker.join()
        assert failures
        manager.compute()
    finally:
        capture.__exit__(None, None, None)
    assert sys.getprofile() is None


def test_unrecognized_manager_is_refused_without_profile_or_read(monkeypatch):
    _, env, cfg, sensor, _, publications = profiled_input(monkeypatch)
    module = importlib.import_module('policy_doctor.isaac_manager')
    with pytest.raises(InputError, match='supported|source|native'):
        with start(module, ControlledManager(env, cfg)):
            pass
    assert sensor.reads == 0 and not publications and sys.getprofile() is None


@pytest.mark.parametrize('value', [.01, -2., 0.])
def test_actual_manager_scalar_torch_scale_is_captured_without_mutation(monkeypatch, value):
    torch = pytest.importorskip('torch')
    producer, _, _, _, _, _ = profiled_input(monkeypatch)
    tensor = torch.tensor(value, dtype=torch.float32)
    original = tensor.clone()
    assert producer._scale(tensor, 12, True) == tensor.item()
    assert torch.equal(tensor, original)
    with pytest.raises(InputError):
        producer._scale(tensor, 12, False)


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_scalar_scale_is_not_sanitized(monkeypatch, value):
    torch = pytest.importorskip('torch')
    producer, _, _, _, _, _ = profiled_input(monkeypatch)
    with pytest.raises(InputError):
        producer._scale(torch.tensor(value), 12, True)
