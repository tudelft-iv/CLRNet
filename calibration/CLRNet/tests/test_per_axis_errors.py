"""Tests for the per-axis (X/Y/Z, roll/pitch/yaw) error statistics.

The errors are collected as one 3-vector per test frame, shape (N, 3); per-axis statistics must be taken
over the columns. These tests do not need a GPU.

    python3 -m pytest calibration/CLRNet/tests/test_per_axis_errors.py -v
    # or, with no pytest available:
    python3 calibration/CLRNet/tests/test_per_axis_errors.py
"""

import os
import re
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from CLRNet.utils.utils import write_results_to_txt


def _parse(txt, label):
    """Pull 'Mean/Median/Std <label> Error: <value>' out of the written report."""
    m = re.search(rf'Mean {label} Error:\s*(-?[\d.]+)', txt)
    assert m, f'no "Mean {label} Error" in report'
    return float(m.group(1))


def _write_and_read(errors_t2, errors_rpy, n):
    errors_t = [torch.stack([v.norm() for v in errors_t2[0]])]
    errors_r = [torch.zeros(n)]
    with tempfile.TemporaryDirectory() as d:
        write_results_to_txt(d, errors_t, errors_r, errors_t2, errors_rpy, 0)
        with open(os.path.join(d, 'results.txt')) as f:
            return f.read()


def test_per_axis_uses_columns_not_samples():
    """Axes with deliberately distinct magnitudes must be reported distinctly and correctly."""
    n = 400
    torch.manual_seed(0)
    # X ~ 10, Y ~ 20, Z ~ 30 (cm before the *100 scaling is applied by the caller).
    per_sample = [torch.tensor([10.0, 20.0, 30.0]) for _ in range(n)]
    # Make the first three samples very different, so indexing samples instead of columns is detected.
    per_sample[0] = torch.tensor([999.0, 999.0, 999.0])
    per_sample[1] = torch.tensor([888.0, 888.0, 888.0])
    per_sample[2] = torch.tensor([777.0, 777.0, 777.0])

    rpy = [torch.tensor([1.0, 2.0, 3.0]) for _ in range(n)]
    rpy[0] = torch.tensor([99.0, 99.0, 99.0])
    rpy[1] = torch.tensor([88.0, 88.0, 88.0])
    rpy[2] = torch.tensor([77.0, 77.0, 77.0])

    txt = _write_and_read([per_sample], [rpy], n)

    x = _parse(txt, 'Translation X')
    y = _parse(txt, 'Translation Y')
    z = _parse(txt, 'Translation Z')

    # True column means: (397*10 + 999+888+777)/400 etc.
    expected_x = (397 * 10.0 + 999 + 888 + 777) / 400
    expected_y = (397 * 20.0 + 999 + 888 + 777) / 400
    expected_z = (397 * 30.0 + 999 + 888 + 777) / 400

    assert abs(x - expected_x) < 1e-3, f'X: got {x}, expected {expected_x}'
    assert abs(y - expected_y) < 1e-3, f'Y: got {y}, expected {expected_y}'
    assert abs(z - expected_z) < 1e-3, f'Z: got {z}, expected {expected_z}'

    # Indexing samples instead of columns would return 999 / 888 / 777.
    assert abs(x - 999.0) > 1.0, 'X still reports sample 0 rather than the X column'
    assert abs(y - 888.0) > 1.0, 'Y still reports sample 1 rather than the Y column'
    assert abs(z - 777.0) > 1.0, 'Z still reports sample 2 rather than the Z column'

    r = _parse(txt, 'Rotation Roll')
    p = _parse(txt, 'Rotation Pitch')
    yaw = _parse(txt, 'Rotation Yaw')
    assert abs(r - (397 * 1.0 + 99 + 88 + 77) / 400) < 1e-3
    assert abs(p - (397 * 2.0 + 99 + 88 + 77) / 400) < 1e-3
    assert abs(yaw - (397 * 3.0 + 99 + 88 + 77) / 400) < 1e-3


def test_iteration_zero_recovers_the_sampling_distribution():
    """Miscalibration translation is drawn i.i.d. uniform on [-max_t, max_t] per axis, so the mean
    absolute error per axis at iteration 0 must be max_t/2, equally on all three axes. With
    max_t = 1 m = 100 cm that is 50 cm each.
    """
    n = 20000
    torch.manual_seed(1)
    max_t_cm = 100.0
    samples = (torch.rand(n, 3) * 2 - 1) * max_t_cm
    per_sample = [samples[k].abs() for k in range(n)]
    rpy = [torch.zeros(3) for _ in range(n)]

    txt = _write_and_read([per_sample], [rpy], n)

    for label in ('Translation X', 'Translation Y', 'Translation Z'):
        v = _parse(txt, label)
        assert abs(v - max_t_cm / 2) < 2.0, (
            f'{label} mean {v:.2f} cm deviates from the expected {max_t_cm / 2:.0f} cm '
            f'implied by uniform per-axis sampling')


if __name__ == '__main__':
    test_per_axis_uses_columns_not_samples()
    test_iteration_zero_recovers_the_sampling_distribution()
    print('per-axis error reporting: all tests passed')
