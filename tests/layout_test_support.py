"""Disclosed generated native-rule controls; these are not live IsaacLab captures."""
import hashlib
from pathlib import Path

import numpy as np


def layout_input(order=('right_foot', 'left_foot'), *, identity=True):
    """Return metadata, arrays and exact pinned whole-module source bytes."""
    root = Path(__file__).parent / 'fixtures' / 'isaac_layout_sources'
    sources = {
        'observations.py': (root / 'observations.py.txt').read_bytes(),
        'base_joint_wrench_sensor.py': (root / 'base_joint_wrench_sensor.py.txt').read_bytes(),
        'joint_wrench_sensor_newton.py': (root / 'joint_wrench_sensor_newton.py.txt').read_bytes(),
        'LICENSE.txt': (root / 'LICENSE.txt').read_bytes(),
    }
    hashes = {key: hashlib.sha256(value).hexdigest() for key, value in sources.items()}

    def source_identity(module, qualname, key):
        return {'module': module, 'qualname': qualname, 'source_file': key, 'sha256': hashes[key]}

    sensor = source_identity('isaaclab_newton.sensors.joint_wrench.joint_wrench_sensor',
                             'JointWrenchSensor', 'joint_wrench_sensor_newton.py')
    base = source_identity('isaaclab.sensors.joint_wrench.base_joint_wrench_sensor',
                           'BaseJointWrenchSensor', 'base_joint_wrench_sensor.py')
    metadata = {
        'schema_version': 1,
        'kind': 'isaaclab_joint_wrench_layout',
        'source': {
            'repository': 'https://github.com/isaac-sim/IsaacLab',
            'revision': '32a1f330511130ae75ffeda96304c1bf569b5acb',
            'rule': 'body_incoming_wrench.force_torque_per_body.v1',
            'term': source_identity('isaaclab.envs.mdp.observations', 'body_incoming_wrench', 'observations.py'),
            'sensor_class': sensor,
            'sensor_mro': [sensor.copy(), base],
            'files': {key: {'sha256': value} for key, value in hashes.items()},
        },
        'runtime': {
            'sensor_body_names': list(order),
            'requested_body_names': ['left_foot', 'right_foot'],
            'resolved_body_ids': list(range(len(order))),
            'preserve_order': False,
            'sensor_identity': {'prim_path': '/World/envs/env_.*/Robot'} if identity else None,
        },
        'processing': {'scale': .01, 'observation_slice': None, 'modifiers': [], 'clip': None,
                       'history': 0, 'noise': None, 'delay': None},
        'provenance': {
            'origin': 'generated_native_rule_control',
            'physical_sample_identity': 'unknown',
            'authenticity': 'Generated asymmetric numeric control; no sensor, simulator or physical capture.',
            'source_fields': {'force': 'cached_data.force.torch', 'torque': 'cached_data.torque.torch'},
        },
    }
    force_by_name = {'right_foot': [1., 2., 3.], 'left_foot': [11., 12., 13.]}
    torque_by_name = {'right_foot': [4., 5., 6.], 'left_foot': [14., 15., 16.]}
    force = np.array([[force_by_name[name] for name in order]], dtype=np.float32)
    torque = np.array([[torque_by_name[name] for name in order]], dtype=np.float32)
    raw = np.concatenate((force, torque), axis=-1).reshape(1, -1)
    arrays = {'source_force': force, 'source_torque': torque,
              'raw_term_output': raw, 'processed_term_output': raw * np.float32(.01)}
    return metadata, arrays, sources
