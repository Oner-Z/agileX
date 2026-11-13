"""Grasp planner: cap-center aim, hidden round widths and neighbor-aware block grasps."""
from __future__ import annotations

import math

import numpy as np
import pytest

from cube_task.core.transforms import wrap_pi
from cube_task.perception.grasp import FINGER_HALF_BAND, GraspPlanner
from test_task_fakes import make_cfg, make_det


def _axis_diff(a, b):
    """Smallest difference of two closing axes (period pi), rad."""
    d = (a - b) % math.pi
    return min(d, math.pi - d)


def _square(cx, cy, side, step=0.002):
    g = np.arange(-side / 2.0, side / 2.0 + 1e-9, step)
    gx, gy = np.meshgrid(g, g)
    return (np.column_stack([gx.ravel(), gy.ravel()]) + [cx, cy]).astype(np.float32)


# --------------------------------------------------------------------------- cap center


def test_cap_center_is_the_grasp_point_when_close_to_the_footprint():
    gp = GraspPlanner(make_cfg(), 'L2')
    d = make_det('apple', 0.30, 0.05, size=0.048, length=0.056, height=0.062)
    d.extra['cap_center_xy'] = [0.305, 0.046]
    p = gp.plan(d)
    assert p.T_grasp[:2, 3] == pytest.approx([0.305, 0.046])
    assert p.T_pre[:2, 3] == pytest.approx([0.305, 0.046])
    # Farther than half the width from the footprint center: ignored.
    d.extra['cap_center_xy'] = [0.33, 0.05]
    assert gp.plan(d).T_grasp[:2, 3] == pytest.approx([0.30, 0.05])
    for bad in ([float('nan'), 0.05], [0.3], None):
        d.extra['cap_center_xy'] = bad
        assert gp.plan(d).T_grasp[:2, 3] == pytest.approx([0.30, 0.05])
    d.extra['cap_center_xy'] = [0.305, 0.046]
    off = GraspPlanner(make_cfg(grasp__use_cap_center=False), 'L2')
    assert off.plan(d).T_grasp[:2, 3] == pytest.approx([0.30, 0.05])


def test_hidden_round_cross_section_uses_the_height():
    gp = GraspPlanner(make_cfg(), 'L2')
    # Apple (r = 31 mm) seen obliquely: 47 x 54 mm footprint, 62 mm high.
    apple = gp.plan(make_det('apple', 0.30, 0.0, size=0.047, length=0.054, height=0.062))
    assert apple.expected_width == pytest.approx(0.054)
    assert apple.open_width == pytest.approx(0.070)             # clipped at max_width
    # Carrot lying down (30 mm thick): the height restores the hidden width.
    carrot = gp.plan(make_det('carrot', 0.30, 0.0, size=0.023, length=0.13, height=0.030))
    assert carrot.expected_width == pytest.approx(0.030)
    assert carrot.open_width == pytest.approx(0.050)
    # Flat objects keep the measured width; blocks never use the height.
    flat = gp.plan(make_det('banana', 0.30, 0.0, size=0.040, length=0.14, height=0.030))
    assert flat.expected_width == pytest.approx(0.040)
    blk = GraspPlanner(make_cfg(), 'L1').plan(
        make_det('red', 0.30, 0.0, size=0.030, length=0.045, height=0.040))
    assert blk.expected_width == pytest.approx(0.030)
    # The height estimate can make an object too wide for the gripper.
    big = gp.plan(make_det('orange', 0.30, 0.0, size=0.050, length=0.068, height=0.068))
    assert not big.feasible and big.reason.startswith('too_wide')


# --------------------------------------------------------------------------- neighbors


def _block_with_neighbor(x, y, side, nx, ny, yaw=0.0, length=None):
    d = make_det('red', x, y, size=side, height=side, yaw=yaw, length=length)
    d.extra['_neighbor_xy'] = _square(nx, ny, 0.035)
    return d


def test_square_block_turns_the_fingers_away_from_a_touching_neighbor():
    gp = GraspPlanner(make_cfg(), 'L1')
    x, y = 0.30, 0.05
    free = gp.plan(make_det('red', x, y, size=0.035, height=0.035))
    # Put the neighbor where the default fingers would go, then on the other axis.
    a = free.closing_angle
    for k, expect_turn in ((0, True), (1, False)):
        ang = a + k * math.pi / 2.0
        nx, ny = x + 0.035 * math.cos(ang), y + 0.035 * math.sin(ang)
        d = _block_with_neighbor(x, y, 0.035, nx, ny)
        p = gp.plan(d)
        assert p.feasible
        assert gp.finger_hits(d, p.closing_angle, x, y, p.expected_width) == 0
        turned = _axis_diff(p.closing_angle, a) > math.radians(45)
        assert turned == expect_turn
        assert abs(wrap_pi(p.closing_angle - free.closing_angle)) <= math.pi / 2 + 1e-9
    # Neighbors on both axes: nothing better, keep the default.
    d = make_det('red', x, y, size=0.035, height=0.035)
    d.extra['_neighbor_xy'] = np.vstack([
        _square(x + 0.035 * math.cos(a), y + 0.035 * math.sin(a), 0.035),
        _square(x - 0.035 * math.sin(a), y + 0.035 * math.cos(a), 0.035)])
    assert gp.plan(d).closing_angle == pytest.approx(free.closing_angle)


def test_partly_hidden_block_closes_along_its_visible_extent():
    # A 30 mm block behind a taller one shows 30 x 24 mm; the minor axis points at
    # the neighbor, so the fingers close along the major axis with that width.
    gp = GraspPlanner(make_cfg(), 'L1')
    x, y = 0.30, 0.10
    d = _block_with_neighbor(x, y, 0.024, x + 0.0315, y, yaw=math.pi / 2, length=0.030)
    p = gp.plan(d)
    assert p.feasible and p.expected_width == pytest.approx(0.030)
    assert _axis_diff(p.closing_angle, math.pi / 2) < 1e-9
    assert (gp.finger_hits(d, p.closing_angle, x, y, 0.030)
            < gp.finger_hits(d, 0.0, x, y, 0.024))
    # Same detection without the neighbor: across the minor axis.
    del d.extra['_neighbor_xy']
    q = gp.plan(d)
    assert q.expected_width == pytest.approx(0.024) and _axis_diff(q.closing_angle, 0.0) < 1e-9


def test_neighbors_do_not_change_non_block_grasps():
    gp = GraspPlanner(make_cfg(), 'L2')
    d = make_det('banana', 0.30, 0.0, size=0.035, length=0.15, yaw=0.0)
    ref = gp.plan(d).closing_angle
    d.extra['_neighbor_xy'] = _square(0.30, 0.04, 0.035)
    assert gp.plan(d).closing_angle == pytest.approx(ref)


def test_finger_hits_band():
    gp = GraspPlanner(make_cfg(grasp__open_full_tasks=[]), 'L1')
    d = make_det('red', 0.0, 0.0)
    assert gp.finger_hits(d, 0.0, 0.0, 0.0, 0.03) == 0              # no neighbor key
    d.extra['_neighbor_xy'] = np.zeros((0, 2))
    assert gp.finger_hits(d, 0.0, 0.0, 0.0, 0.03) == 0
    # Closing axis = base x; open width 0.05 -> fingers up to 0.025 + 0.015 m.
    d.extra['_neighbor_xy'] = np.array([[0.02, 0.0], [-0.039, 0.01], [0.045, 0.0],
                                        [0.02, FINGER_HALF_BAND + 0.001], [0.016, 0.0]])
    assert gp.finger_hits(d, 0.0, 0.0, 0.0, 0.03) == 2


def test_l1_opens_fully_and_its_finger_band_reaches_that_far():
    gp = GraspPlanner(make_cfg(), 'L1')
    d = make_det('red', 0.0, 0.0)
    assert gp.plan(d).open_width == pytest.approx(gp.max_width)
    # Open to max_width (0.07 here) -> fingers reach 0.035 + FINGER_OUTER: 0.045 m is hit.
    d.extra['_neighbor_xy'] = np.array([[0.045, 0.0]])
    assert gp.finger_hits(d, 0.0, 0.0, 0.0, 0.03) == 1
    assert GraspPlanner(make_cfg(grasp__open_full_tasks=[]), 'L1').finger_hits(
        d, 0.0, 0.0, 0.0, 0.03) == 0
    assert GraspPlanner(make_cfg(), 'L2').plan(make_det('apple', 0.0, 0.0)).open_width < \
        gp.max_width


def test_block_is_not_turned_onto_an_axis_too_wide_for_the_gripper():
    gp = GraspPlanner(make_cfg(), 'L1')
    x, y = 0.30, 0.10
    d = _block_with_neighbor(x, y, 0.030, x + 0.033, y, yaw=math.pi / 2, length=0.068)
    p = gp.plan(d)
    assert p.feasible and p.expected_width == pytest.approx(0.030)
    assert _axis_diff(p.closing_angle, 0.0) < 1e-9


def test_round_fruit_turns_the_fingers_away_from_a_touching_neighbor():
    gp = GraspPlanner(make_cfg(), 'L2')
    x, y = 0.30, 0.05
    free = gp.plan(make_det('pear', x, y, size=0.055, height=0.06))
    a = free.closing_angle
    d = make_det('pear', x, y, size=0.055, height=0.06)
    d.extra['_neighbor_xy'] = _square(x + 0.05 * math.cos(a), y + 0.05 * math.sin(a), 0.035)
    p = gp.plan(d)
    assert p.feasible
    assert gp.finger_hits(d, free.closing_angle, x, y, free.expected_width) > 0
    assert gp.finger_hits(d, p.closing_angle, x, y, p.expected_width) == 0


def test_fit_width_per_label_lets_a_wide_pear_be_grasped():
    cfg = make_cfg('L2')
    cfg.set('robot.gripper.max_width', 0.100)
    cfg.set('grasp.fit_width', {'pear': 0.098})
    gp = GraspPlanner(cfg, 'L2')
    assert gp._fit('pear') == pytest.approx(0.098)
    assert gp._max_object('pear') == pytest.approx(0.098)
    assert gp._fit('potato') == pytest.approx(0.100 - 2.0 * gp.finger_clearance)
    cfg.set('grasp.fit_width', {'pear': 0.2})               # wider than the gripper: ignored
    assert GraspPlanner(cfg, 'L2')._fit('pear') == pytest.approx(gp._fit('potato'))
    cfg.set('grasp.fit_width', {'pear': 0.0})
    with pytest.raises(ValueError):
        GraspPlanner(cfg, 'L2')


def test_open_full_labels_open_the_gripper_fully_for_those_labels_only():
    cfg = make_cfg('L2')
    cfg.set('grasp.open_full_tasks', [])
    cfg.set('grasp.open_full_labels', ['bell_pepper'])
    cfg.set('robot.gripper.max_width', 0.100)
    gp = GraspPlanner(cfg, 'L2')
    assert gp._opens_full('bell_pepper') and not gp._opens_full('persimmon')
    det = make_det('bell_pepper', 0.30, 0.0, size=0.05)
    plan = gp.plan(det)
    assert plan.open_width == pytest.approx(gp.max_width)
    det2 = make_det('persimmon', 0.30, 0.0, size=0.05)
    assert gp.plan(det2).open_width < gp.max_width
