"""Bounded weak-reference witnesses for local ACT call/view bindings.

No numeric equality establishes identity here. Tokens identify live Python objects
within one capture only; historical state mutation and runtime authenticity remain
unknown. This module never imports Torch, retains tensors, or stores addresses in
its wire records. Native Tensor metadata is used only when Torch is already loaded.
"""
from __future__ import annotations

import sys
import weakref

MAX_TOKENS = 4096
MAX_INDEX = 2**53 - 1


def _weak(value):
    try:
        return weakref.ref(value)
    except TypeError:
        return None


class OpaqueTokens:
    """Never reuse an expired object's opaque token, even when its id is reused."""

    def __init__(self, max_tokens=MAX_TOKENS):
        self._limit = min(max_tokens, MAX_TOKENS)
        self._by_identity = {}
        self._refs = {}
        self._next = 0

    @staticmethod
    def _identity(value):
        return id(value)

    def token(self, value):
        reference = _weak(value)
        if reference is None:
            return None
        identity = self._identity(value)
        previous = self._by_identity.get(identity)
        if previous is not None and self.matches(previous, value):
            return previous
        if self._next >= self._limit:
            return None
        token = f'o{self._next:06d}'
        self._next += 1
        self._by_identity[identity] = token
        self._refs[token] = reference
        return token

    def matches(self, token, value):
        reference = self._refs.get(token)
        return reference is not None and reference() is not None and reference() is value

    def live(self, token):
        reference = self._refs.get(token)
        return reference is not None and reference() is not None

    def clear(self):
        self._by_identity.clear()
        self._refs.clear()


def _nonoverlapping(shape, stride):
    span = 1
    for step, length in sorted((s, n) for n, s in zip(shape, stride, strict=True) if n > 1):
        if step < span:
            return False
        span += (length - 1) * step
        if span > MAX_INDEX:
            return False
    return True


def _view(value, tokens):
    """Return wire metadata and weak live-base evidence; no tensor/data copy."""
    torch = sys.modules.get('torch')
    if torch is None or type(value) is not getattr(torch, 'Tensor', None):
        return None
    try:
        if (value.device.type != 'cpu' or value.layout != torch.strided
                or value.dtype not in (torch.float16, torch.float32, torch.float64)):
            return None
        shape, stride = tuple(value.shape), tuple(value.stride())
        offset, size = value.storage_offset(), value.element_size()
        if (len(shape) not in (2, 3) or any(type(n) is not int or not 1 <= n <= 1_000_000 for n in shape)
                or shape[-1] > 256 or (len(shape) == 3 and shape[1] > 256)
                or len(stride) != len(shape)
                or any(type(n) is not int or not 0 <= n <= MAX_INDEX for n in stride)
                or type(offset) is not int or not 0 <= offset <= MAX_INDEX
                or size not in (2, 4, 8) or not _nonoverlapping(shape, stride)):
            return None
        root = value
        seen = set()
        for _ in range(16):
            if id(root) in seen:
                return None
            seen.add(id(root))
            parent = root._base
            if parent is None:
                break
            if type(parent) is not torch.Tensor:
                return None
            root = parent
        else:
            return None
        storage = value.untyped_storage()
        root_storage = root.untyped_storage()
        # Runtime-only identity checks: neither identifier nor address enters JSON.
        if storage._cdata != root_storage._cdata:
            return None
        extent = offset + sum((n - 1) * s for n, s in zip(shape, stride, strict=True))
        if extent > MAX_INDEX or (extent + 1) * size > storage.nbytes():
            return None
        token = tokens.token(root)
        if token is None:
            return None
        return {'wire': {'base_token': token, 'shape': list(shape), 'stride': list(stride),
                         'storage_offset': offset, 'element_size': size},
                'root': weakref.ref(root), 'storage_identity': root_storage._cdata}
    except (AttributeError, RuntimeError, TypeError, ValueError, OverflowError):
        return None


def _same_live_base(left, right):
    if left is None or right is None:
        return None
    root = left['root']()
    if root is None or right['root']() is not root:
        return False
    try:
        return (left['storage_identity'] == right['storage_identity']
                == root.untyped_storage()._cdata)
    except (AttributeError, RuntimeError):
        return False


class BindingTracker:
    """Track one policy's active local operation using weak references only."""

    def __init__(self):
        self.tokens = OpaqueTokens()
        self._active = None
        self._off_path = 0

    def start(self, event, *, queue_head=None, helper=None):
        self._active = {'event': event, 'head': _weak(queue_head),
                        'head_token': self.tokens.token(queue_head),
                        'helper': _weak(helper), 'helper_token': self.tokens.token(helper),
                        'predictions': [], 'update_count': 0, 'update': None}

    def prediction(self, value, query):
        if self._active is not None:
            self._active['predictions'].append({'object': _weak(value), 'token': self.tokens.token(value),
                                                'query': dict(query), 'view': _view(value, self.tokens)})

    def update_start(self, helper, actions, *, inside_select):
        active = self._active
        if not inside_select or active is None or active['event']['pre_state']['mode'] != 'ensemble':
            self._off_path += 1
            return
        active['update_count'] += 1
        prediction = active['predictions'][-1] if len(active['predictions']) == 1 else None
        expected_helper = active['helper']() if active['helper'] is not None else None
        matches = (prediction is not None and prediction['object'] is not None
                   and prediction['object']() is actions and expected_helper is helper)
        active['update'] = {'helper_token': self.tokens.token(helper),
                            'actions_token': self.tokens.token(actions),
                            'prediction_token': None if prediction is None else prediction['token'],
                            'return_token': None, 'return_ref': None,
                            'actions_match': matches}

    def update_return(self, value):
        if self._active is not None and self._active['update'] is not None:
            self._active['update']['return_token'] = self.tokens.token(value)
            self._active['update']['return_ref'] = _weak(value)

    def finish(self, event, selected):
        active, self._active = self._active, None
        selected_token = self.tokens.token(selected)
        selected_view = _view(selected, self.tokens)
        result = {'schema_version': 1, 'status': 'unknown', 'mode': event['pre_state']['mode'],
                  'basis': 'unobserved', 'selected_array': event['selected'],
                  'selected_token': selected_token, 'queue_head': None, 'query': None,
                  'selected_view': None if selected_view is None else selected_view['wire'],
                  'query_view': None, 'query_offset': None, 'update': None,
                  'relations': {'selected_is_pre_head': None, 'same_live_base': None,
                                'actions_is_prediction': None, 'selected_is_update_return': None},
                  'reason': 'No supported live local binding observed', 'history': 'unknown'}
        if active is None or active['event'] is not event or selected_token is None:
            return result
        if result['mode'] == 'queue':
            if event['pre_state']['queue']:
                head = active['head']
                same = (head is not None and head() is selected
                        and active['head_token'] is not None)
                if active['head_token'] is not None:
                    result['queue_head'] = {'array': event['pre_state']['queue'][0],
                                            'token': active['head_token']}
                result['relations']['selected_is_pre_head'] = same
                if same:
                    result.update(status='pass', basis='pre_queue_head_identity', reason=None)
            elif len(active['predictions']) == 1:
                prediction = active['predictions'][0]
                if prediction['token'] is not None:
                    result['query'] = dict(prediction['query'], token=prediction['token'])
                query_view = prediction['view']
                result['query_view'] = None if query_view is None else query_view['wire']
                same_base = _same_live_base(query_view, selected_view)
                result['relations']['same_live_base'] = same_base
                if same_base and prediction['token'] is not None:
                    q, s = query_view['wire'], selected_view['wire']
                    if (len(q['shape']) == 3 and len(s['shape']) == 2
                            and s['shape'] == [q['shape'][0], q['shape'][2]]
                            and s['stride'] == [q['stride'][0], q['stride'][2]]
                            and s['element_size'] == q['element_size']):
                        offsets = [i for i in range(q['shape'][1])
                                   if s['storage_offset'] == q['storage_offset'] + i * q['stride'][1]]
                        first = max(0, min(event['settings']['n_action_steps'], q['shape'][1])
                                    - event['pre_state']['queue_maxlen'])
                        if offsets == [first]:
                            result.update(status='pass', basis='query_view_mapping',
                                          query_offset=first, reason=None)
        else:
            prediction = active['predictions'][0] if len(active['predictions']) == 1 else None
            if prediction is not None and prediction['token'] is not None:
                result['query'] = dict(prediction['query'], token=prediction['token'])
            update = active['update']
            if update is not None:
                same_return = (update['return_ref'] is not None and update['return_ref']() is selected)
                result['relations']['actions_is_prediction'] = update['actions_match']
                result['relations']['selected_is_update_return'] = same_return
                keys = ('helper_token', 'actions_token', 'prediction_token', 'return_token')
                if active['update_count'] == 1 and all(update[k] is not None for k in keys):
                    result['update'] = {k: update[k] for k in keys}
                    result['update'].update(selected_token=selected_token, call_count=1)
                    if update['actions_match'] and same_return:
                        result.update(status='pass', basis='ensemble_argument_return_identity', reason=None)
        if self._off_path:
            result.update(status='unknown', reason='Off-path helper update prevents local coverage qualification')
        return result

    def finalize(self, events):
        if self._off_path:
            for event in events:
                binding = event.get('local_binding')
                if binding is not None:
                    binding.update(status='unknown', reason='Off-path helper update prevents local coverage qualification')

    def scope(self):
        return {'schema_version': 1, 'token_scope': 'one_capture_owner_thread',
                'off_path_update_calls': self._off_path, 'external_state_mutations': 'untracked'}

    def clear(self):
        self._active = None
        self.tokens.clear()
