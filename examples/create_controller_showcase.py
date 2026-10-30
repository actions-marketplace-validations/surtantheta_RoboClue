"""Generate a native ROS trajectory investigation; also available as policy-doctor demo."""
from __future__ import annotations

import argparse
from pathlib import Path

from policy_doctor.cli import main
from policy_doctor.demo import create_controller_demo

# Preserve the documented example API while keeping the installed demo identical.
create_showcase = create_controller_demo


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New directory; never overwritten')
    raise SystemExit(main(['demo', str(parser.parse_args().destination), '--json']))
