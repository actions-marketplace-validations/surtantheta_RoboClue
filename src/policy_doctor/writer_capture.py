"""Observe one fresh native numeric episode; never call a writer or save data."""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path

from .act_capture import _bytes, _normal_return, _sha
from .artifacts import InputError
from .command_capture import _method
from .layouts import _safe_root
from .named_rules import MAX_BYTES, MAX_TICKS, RECORDING_SOURCE_HASHES, RECORDING_SOURCE_MODULES

DATASET_PATHS = {
    'meta/info.json', 'meta/stats.json', 'meta/tasks.parquet',
    'meta/episodes/chunk-000/file-000.parquet', 'data/chunk-000/file-000.parquet',
}


def _dataset_tree(root):
    allowed_directories = {str(p) for n in DATASET_PATHS for p in Path(n).parents if str(p) != '.'}
    stack, files, directories, entries = [root], set(), set(), 0
    while stack:
        directory = stack.pop()
        with os.scandir(directory) as children:
            for child in children:
                entries += 1
                if entries > 32 or child.is_symlink():
                    raise InputError('Unsafe or over-budget native dataset traversal')
                relative = str(Path(child.path).relative_to(root))
                if child.is_dir(follow_symlinks=False):
                    if relative not in allowed_directories:
                        raise InputError('Unmodeled native dataset directory cannot be silently omitted')
                    directories.add(relative)
                    stack.append(Path(child.path))
                elif child.is_file(follow_symlinks=False) and relative in DATASET_PATHS:
                    files.add(relative)
                else:
                    raise InputError('Unsafe or unsupported native dataset payload')
    if files != DATASET_PATHS:
        raise InputError('Incomplete native dataset after observed save/finalize')
    return files, directories


class RecordingObservation:
    def __init__(self, parent):
        from .named_capture import _code, _names

        self.parent = parent
        definition = parent._definition
        modules = {}
        for name, module_name in RECORDING_SOURCE_MODULES.items():
            module = sys.modules.get(module_name)
            if module is None or not getattr(module, '__file__', None):
                raise InputError(f'Load the complete native writer module first: {module_name}')
            path = Path(module.__file__)
            raw = _bytes(path)
            if _sha(raw) != RECORDING_SOURCE_HASHES[name]:
                raise InputError(f'Unsupported native writer source: {name}')
            definition['sources'][name], definition['paths'][name] = raw, path
            modules[name] = module
        self.dataset = parent.ctx.data.dataset
        if type(self.dataset) is not modules['lerobot_dataset.py'].LeRobotDataset:
            raise InputError('Recording requires the actual writable native LeRobotDataset')
        self.writer = self.dataset.writer
        if type(self.writer) is not modules['dataset_writer.py'].DatasetWriter:
            raise InputError('Recording requires the actual original DatasetWriter instance')
        if (self.writer._meta.total_episodes != 0 or self.writer._meta.total_frames != 0
                or self.writer.episode_buffer is not None and self.writer.episode_buffer.get('size') != 0
                or getattr(self.dataset, '_is_finalized', False) or getattr(self.writer, '_finalized', False)):
            raise InputError('Recording supports only one fresh empty native numeric episode')
        self.names = _names(self.dataset.features['action']['names'])
        features = self.dataset.features
        if (features['action'].get('dtype') != 'float32'
                or any(f.get('dtype') in ('image', 'video') for f in features.values())):
            raise InputError('Only bounded numeric native episodes are supported')
        self.root = _safe_root(self.dataset.root)
        self.loop_cls = modules['episodic.py'].EpisodicStrategy
        self.functions = {
            'record_loop': self.loop_cls._policy_loop,
            'record_build': modules['feature_utils.py'].build_dataset_frame,
        }
        self.methods = []
        for prefix, owner, cls in (
                ('record_facade', self.dataset, modules['lerobot_dataset.py'].LeRobotDataset),
                ('record_writer', self.writer, modules['dataset_writer.py'].DatasetWriter)):
            for suffix, method in (('add', 'add_frame'), ('save', 'save_episode'), ('finalize', 'finalize')):
                self.functions[prefix + '_' + suffix] = _method(owner, method, cls)
                self.methods.append((owner, method, getattr(owner, method)))
        for function in self.functions.values():
            name = next(n for n, m in RECORDING_SOURCE_MODULES.items() if m == function.__module__)
            _code(function, definition['sources'][name])
            definition['loaded_bindings'].append((function.__module__, sys.modules[function.__module__],
                                                  function.__qualname__.split('.'), function))
        definition['functions'].update(self.functions)
        self.codes = {role: fn.__code__ for role, fn in self.functions.items()}
        self.loop = self.strategy = self.last_tick = self.canonical = None
        self.frames, self.refs, self.rows, self.buffer_refs = {}, {}, [], []
        self.row = None
        self.lifecycle = dict.fromkeys((
            'loop_completed', 'facade_save_completed', 'writer_save_completed',
            'facade_finalize_completed', 'writer_finalize_completed', 'save_buffer_continuity'), False)
        self.lifecycle.update(save_episode_count=0, finalize_count=0)
        self.loop_count = 0

    def check(self):
        if (self.parent.ctx.data.dataset is not self.dataset or self.dataset.writer is not self.writer
                or self.dataset.root != self.root or self.dataset.features['action']['names'] != self.names):
            raise InputError('Observed native dataset/writer/root or action names changed')

    def inside_loop(self, frame):
        return self.loop is not None and self.inside(frame, self.loop)

    @staticmethod
    def inside(frame, target):
        parent = frame.f_back
        for _ in range(40):
            if parent is None:
                return False
            if parent is target:
                return True
            parent = parent.f_back
        return False

    def tick_return(self, tick, value, frame, normal):
        if self.loop is not None and frame.f_back is self.loop:
            self.last_tick = tick
            self.canonical = value if normal else None

    def vector(self, value):
        # Native builder output has the same supported width as its actual names.
        if len(self.names) != len(self.parent.names):
            raise InputError('Writer and canonical action widths differ')
        return self.parent._vector(value)

    def profile(self, frame, event, value):
        role = next((r for r, code in self.codes.items() if frame.f_code is code), None)
        if role is None:
            return False
        if role == 'record_loop':
            if frame.f_locals.get('ctx') is not self.parent.ctx or frame.f_locals.get('dataset') is not self.dataset:
                return True
            if event == 'call':
                self.check()
                if self.loop_count or type(frame.f_locals.get('self')) is not self.loop_cls:
                    raise InputError('Only one original episodic strategy invocation is supported')
                self.loop_count += 1
                self.loop, self.strategy = frame, frame.f_locals['self']
                _method(self.strategy, '_policy_loop', self.loop_cls)
            elif frame is self.loop:
                self.lifecycle['loop_completed'] = _normal_return(frame)
                self.loop = None
            return True
        if role == 'record_build':
            if frame.f_locals.get('prefix') != 'action':
                return True
            if not self.inside_loop(frame) or frame.f_back is not self.loop:
                return True
        else:
            owner = self.dataset if role.startswith('record_facade') else self.writer
            if frame.f_locals.get('self') is not owner:
                return True
        if event == 'call':
            if role in self.frames:
                raise InputError('Recursive writer boundary is unsupported')
            self.check()
            self.frames[role] = frame
            self.entry(role, frame)
        elif frame is self.frames.get(role):
            self.returned(role, frame, value)
            self.frames.pop(role, None)
        return True

    def entry(self, role, frame):
        if role == 'record_build':
            if (self.row is not None or self.last_tick is None or self.canonical is None
                    or not self.last_tick['emitted_policy_action'] or self.last_tick['outcome'] != 'sent'
                    or frame.f_locals['values'] is not self.canonical
                    or frame.f_locals['ds_features'] is not self.loop.f_locals['features']):
                raise InputError('Writer builder is not joined to the observed native canonical endpoint')
            if self.parent._mapping(self.canonical) != self.last_tick['canonical_at_return']:
                raise InputError('Canonical values changed before the actual frame builder')
            if frame.f_locals['ds_features']['action']['names'] != self.names:
                raise InputError('Builder feature names disagree with actual writer names')
            if len(self.rows) >= MAX_TICKS:
                raise InputError('Writer row budget exceeded')
            self.row = {'tick_id': self.last_tick['tick_id'], 'episode_index': 0,
                        'frame_index': len(self.rows), 'index': len(self.rows), 'action_names': list(self.names),
                        'source_role': 'canonical_return', 'canonical_at_frame': self.parent._mapping(self.canonical),
                        'frame_action': None, 'action': None, 'buffer_action': None,
                        'facade_completed': False, 'writer_completed': False,
                        'buffer_size_before': len(self.rows), 'buffer_size_after': None,
                        'relations': dict.fromkeys(('canonical_to_build_frame', 'build_to_facade',
                                                   'facade_to_writer', 'writer_to_buffer'), False)}
            self.row['relations']['canonical_to_build_frame'] = True
        elif role == 'record_facade_add':
            if self.row is None or not self.inside_loop(frame) or frame.f_back is not self.loop:
                raise InputError('Dataset append outside the observed original episodic loop')
            data = frame.f_locals['frame']
            action = data.get('action') if type(data) is dict else None
            self.row['action'] = self.vector(action)
            if self.row['action'] != self.row['frame_action']:
                raise InputError('Action array values changed between builder and facade entry')
            self.row['relations']['build_to_facade'] = action is self.refs.get('built_action')
            self.refs['facade_frame'], self.refs['action'] = data, action
        elif role == 'record_writer_add':
            if self.row is None or not self.inside(frame, self.frames.get('record_facade_add')):
                raise InputError('Writer append outside the observed dataset facade')
            data = frame.f_locals['frame']
            self.row['relations']['facade_to_writer'] = data is self.refs.get('facade_frame')
            if data.get('action') is not self.refs.get('action'):
                raise InputError('Actual writer received a different action array')
            if self.vector(data['action']) != self.row['action']:
                raise InputError('Action values changed between facade and actual writer entry')
            buffer = self.writer.episode_buffer
            if buffer is not None and (buffer.get('size') != len(self.rows) or buffer.get('episode_index') != 0):
                raise InputError('Writer episode buffer identity/index continuity changed')
        elif role in ('record_facade_save', 'record_writer_save'):
            if self.row is not None or not self.rows or not self.lifecycle['loop_completed']:
                raise InputError('Save must follow the completed observed episode')
            if frame.f_locals.get('episode_data') is not None:
                raise InputError('Externally supplied episode data cannot establish buffer save lineage')
            if role == 'record_writer_save':
                if not self.inside(frame, self.frames.get('record_facade_save')):
                    raise InputError('Writer save did not originate from the observed facade')
                buffer = self.writer.episode_buffer
                if (type(buffer) is not dict or buffer.get('size') != len(self.rows)
                        or len(buffer.get('action', ())) != len(self.buffer_refs)):
                    raise InputError('Observed writer buffer changed before saving')
                for array, ref, row in zip(buffer['action'], self.buffer_refs, self.rows, strict=True):
                    if array is not ref or self.vector(array) != row['buffer_action']:
                        raise InputError('Observed buffer array identity or values changed before saving')
                self.lifecycle['save_buffer_continuity'] = True
        elif role == 'record_writer_finalize':
            if not self.inside(frame, self.frames.get('record_facade_finalize')):
                raise InputError('Writer finalize did not originate from the observed facade')

    def returned(self, role, frame, value):
        normal = _normal_return(frame)
        if role == 'record_build' and normal:
            if type(value) is not dict or 'action' not in value:
                raise InputError('Native action frame is missing its action array')
            self.refs['built_action'] = value['action']
            self.row['frame_action'] = self.vector(value['action'])
        elif role == 'record_writer_add' and normal:
            buffer = self.writer.episode_buffer
            if (type(buffer) is not dict or buffer.get('size') != len(self.rows) + 1
                    or buffer.get('episode_index') != 0 or buffer['frame_index'][-1] != len(self.rows)):
                raise InputError('Native writer did not append exactly the observed frame')
            array = buffer['action'][-1]
            self.row['buffer_action'] = self.vector(array)
            if self.row['buffer_action'] != self.row['action']:
                raise InputError('Action values changed during actual buffer append')
            self.row['relations']['writer_to_buffer'] = array is self.refs['action']
            self.row['buffer_size_after'] = buffer['size']
            self.row['writer_completed'] = True
            self.buffer_refs.append(array)
        elif role == 'record_facade_add' and normal:
            if self.row is None or not self.row['writer_completed']:
                raise InputError('Dataset facade returned without the observed writer completion')
            self.row['facade_completed'] = True
            self.rows.append(self.row)
            self.row = None
            self.refs.clear()
        elif role in ('record_facade_save', 'record_writer_save') and normal:
            self.lifecycle['facade_save_completed' if role == 'record_facade_save' else 'writer_save_completed'] = True
            if role == 'record_facade_save':
                self.lifecycle['save_episode_count'] += 1
                if self.lifecycle['save_episode_count'] != 1:
                    raise InputError('Multiple saves are outside fresh single-episode capture scope')
        elif role in ('record_facade_finalize', 'record_writer_finalize') and normal:
            self.lifecycle['facade_finalize_completed' if role == 'record_facade_finalize' else 'writer_finalize_completed'] = True
            if role == 'record_facade_finalize':
                self.lifecycle['finalize_count'] += 1
                if self.lifecycle['finalize_count'] != 1:
                    raise InputError('Multiple finalizations are outside capture scope')
        elif not normal:
            raise InputError('Incomplete writer operation prevents recording publication')

    def snapshot(self, stage):
        self.check()
        packet = {'coverage': {'target': 'same_native_episodic_writer', 'complete_target_call_history': True},
                  'lifecycle': dict(self.lifecycle), 'rows': deepcopy(self.rows), 'dataset_files': []}
        complete = (self.rows and self.row is None and all(v is True for k, v in self.lifecycle.items()
                    if k not in ('save_episode_count', 'finalize_count'))
                    and self.lifecycle['save_episode_count'] == self.lifecycle['finalize_count'] == 1)
        if not complete:
            return packet
        root = _safe_root(self.root)
        tree = _dataset_tree(root)
        payloads, total = {}, 0
        for name in sorted(tree[0]):
            path = root / name
            if path.stat().st_size > MAX_BYTES // 4:
                raise InputError('Oversized original numeric dataset payload')
            raw = path.read_bytes()
            total += len(raw)
            if total > MAX_BYTES // 2:
                raise InputError('Native dataset exceeds bounded capture budget')
            payloads[name] = raw
        for name, raw in sorted(payloads.items()):
            path = stage / 'dataset' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            packet['dataset_files'].append({'path': 'dataset/' + name, 'sha256': _sha(raw)})
        if (_safe_root(self.root) != root or _dataset_tree(root) != tree
                or any((root / name).read_bytes() != raw for name, raw in payloads.items())):
            raise InputError('Native dataset changed during evidence copying')
        return packet

    def clear(self):
        self.frames.clear()
        self.refs.clear()
        self.buffer_refs.clear()
        self.parent = self.dataset = self.writer = self.loop = self.strategy = self.last_tick = self.canonical = None
