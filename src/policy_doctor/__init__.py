"""Local, evidence-backed robot policy diagnostics."""

__version__ = "0.1.0"


def capture_native_actions(ctx, interp, output, *, driver_namespace, allow_tensor_snapshot=False,
                           observe_recording=False):
    """Opt in to observing an ordinary native named-action software call."""
    from .named_capture import capture_native_actions as capture

    return capture(ctx, interp, output, driver_namespace=driver_namespace,
                   allow_tensor_snapshot=allow_tensor_snapshot, observe_recording=observe_recording)
