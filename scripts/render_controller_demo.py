"""Optional documentation figure; requires matplotlib, never runs a controller."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from policy_doctor.trajectory import sample_joint_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('demo', type=Path)
    parser.add_argument('destination', type=Path, help='New .svg or .png')
    args = parser.parse_args()
    if args.destination.exists():
        raise SystemExit('Figure destination exists')
    times = np.linspace(0, 1, 501).tolist()
    fig, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    axis.axhspan(0., 1., color='#edf2f7', label='Fixture position bounds')
    for name, color, label in [('reference', '#1463b8', 'Positions only: linear'),
                                ('candidate', '#c64232', 'Same positions + endpoint velocities: cubic')]:
        report = sample_joint_trajectory(str(args.demo / f'{name}.json'), times)
        axis.plot(times, [row['positions'][0] for row in report['samples']], color=color, linewidth=2.4, label=label)
    axis.scatter([0., 1.], [0., 1.], color='#172b4d', s=65, zorder=4, label='Same waypoint positions')
    axis.scatter([.75], [1.03125], color='#c64232', zorder=5)
    axis.annotate('1.03125 at 0.75 s', (.75, 1.03125), xytext=(.46, 1.16),
                  arrowprops={'arrowstyle': '->', 'color': '#c64232'}, color='#9b3027')
    axis.set(xlabel='Time from trajectory start (s)', ylabel='Sampled position (fixture units)', ylim=(-.05, 1.28),
             title='The waypoints match. The interpolated references differ.')
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.2)
    axis.legend(loc='lower right', fontsize=9)
    fig.text(.01, -.04, 'Pinned ros2_control interpolation rules. Generated fixture; no ROS controller or robot execution.', fontsize=9)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.destination, bbox_inches='tight', metadata={'Creator': 'Policy Deployment Doctor'})
    plt.close(fig)
    print(args.destination)


if __name__ == '__main__':
    main()
