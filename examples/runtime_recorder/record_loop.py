"""Call from an existing, already-connected synchronous inference application."""
from policy_doctor.runtime_recorder import RuntimeRecorder


def record_existing_loop(robot, infer_action, *, path, steps, action_spec, returned_spec,
                         observation_spec, provenance, camera_keys=()):
    """infer_action is the caller's existing observation->flat action function.

    It can include the application's original policy pre/postprocessors. This
    function invokes those existing interfaces only because its caller runs it.
    Robot connection, stopping and exception handling remain application-owned.
    """
    with RuntimeRecorder(robot, path, action_spec=action_spec, returned_spec=returned_spec,
                         observation_spec=observation_spec, policy_spec=action_spec,
                         camera_keys=camera_keys, provenance=provenance) as recording:
        for _ in range(steps):
            observation = recording.get_observation()  # Association remains unknown.
            action = infer_action(observation)
            policy_id = recording.record_policy_command(action)
            recording.send_action(action, policy_command_id=policy_id)
