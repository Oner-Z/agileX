"""Flat 200 mm zones, long objects, reach lowering, free-space tilt and gripper force."""
from __future__ import annotations

import math

from cube_task.core.clock import SimClock
from cube_task.core.transforms import neutral_closing_angle, top_down_T, wrap_pi
from cube_task.core.zones import parse_zone, parse_zones
from cube_task.perception.grasp import (capsule_edge_margin, footprint_capsule, GraspPlanner,
                                        held_offset_of, PlacePlanner)
from cube_task.robot.factory import make_arm
from cube_task.robot.safety import SafetyGuard
from cube_task.task.reach import adjust_plan_poses, free_tilt_limit
from cube_task.task.runner import held_shape, line_obstacle_top
from cube_task.task.skills import gripper_force_for, PickPlaceSkill
import numpy as np
import pytest
from test_plan_official_shapes import make_object_det
from test_task_fakes import (FakeArm, FakeSafeArm, FakeWorld, install_fake_modules, make_cfg,
                             make_det, Rig)

CARD = {'center': [0.25, 0.30, 0.0], 'size': [0.20, 0.20]}
CUCUMBER = dict(length=0.26, width=0.04, elongated=True, grip_width=0.036, grip_z=0.01)
PEAR = dict(length=0.085, width=0.085, elongated=False, grip_width=0.056, grip_z=0.086)


def _card(name='zone_green', **kw):
    d = dict(CARD)
    d.update(kw)
    return parse_zone(name, d)


def _inside_along(zone, x, y, axis, length):
    """Return the fraction of a length-long segment centered at x, y on axis inside zone."""
    ts = np.linspace(-length / 2.0, length / 2.0, 261)
    px = x + ts * math.cos(axis) - zone.center[0]
    py = y + ts * math.sin(axis) - zone.center[1]
    c, s_ = math.cos(-zone.yaw), math.sin(-zone.yaw)
    u, v = c * px - s_ * py, s_ * px + c * py
    return float(np.mean((abs(u) <= zone.size[0] / 2.0) & (abs(v) <= zone.size[1] / 2.0)))


def _row(pp, plan, shape, height):
    i = pp.last_info
    return (i['xy'][0], i['xy'][1], shape['length'], height, shape['length'], shape['width'],
            i['axis'])


# ------------------------------------------------------------------ flat zones
def test_flat_card_release_height_and_walled_bins_unchanged():
    pp = PlacePlanner(make_cfg('L2'))
    p = pp.plan(_card(), 0.02, 0.2, held=PEAR)
    assert pp.release_height_flat == pytest.approx(0.01)       # low: fruit does not roll
    assert p.T_release[2, 3] == pytest.approx(0.01 + 0.02)
    bin_ = _card('bin', wall_height=0.05)
    q = pp.plan(bin_, 0.02, 0.2)
    assert q.T_release[2, 3] == pytest.approx(0.05 + 0.02 + 0.02)
    assert pp.plan(_card(release_height=0.03), 0.02, 0.2).T_release[2, 3] == pytest.approx(0.05)


def test_long_object_lies_along_a_card_axis_on_the_side_nearer_the_robot():
    z = _card()
    pp = PlacePlanner(make_cfg('L2'))
    p = pp.plan(z, 0.01, 0.2, held=CUCUMBER, center_first=False)
    info = pp.last_info
    assert info['tier'] == 'long_strip'
    # The card (0.25, 0.30) lies at 50 deg from the base: its y axis is nearer.
    assert abs(abs(wrap_pi(info['axis'])) - math.pi / 2.0) < 1e-4       # rounded in info
    assert info['xy'] == pytest.approx([0.25 - (0.10 - 0.02 - pp.slot_margin), 0.30])
    assert _inside_along(z, info['xy'][0], info['xy'][1], info['axis'], 0.26) >= 0.75
    # The fingers close across the long axis, at the equivalent angle nearest the neutral one.
    x, y = p.T_release[:2, 3]
    closing = math.atan2(p.T_release[1, 1], p.T_release[0, 1])
    assert abs(math.cos(closing - info['axis'])) < 1e-3
    assert abs(wrap_pi(closing - neutral_closing_angle(x, y))) <= math.pi / 2.0 + 1e-9
    assert p.T_release[2, 3] == pytest.approx(0.01 + 0.01)


@pytest.mark.parametrize('pear_first', [False, True])
def test_green_zone_holds_cucumber_and_pear(pear_first):
    z = _card()
    pp = PlacePlanner(make_cfg('L2'))
    seq = [(CUCUMBER, 0.04, [(0.085, 0.085)]), (PEAR, 0.114, [])]
    if pear_first:
        seq = [(PEAR, 0.114, [(0.26, 0.04)]), (CUCUMBER, 0.04, [])]
    occ, caps = [], []
    for shape, h, pending in seq:
        p = pp.plan(z, shape['grip_z'], 0.2, occupied=occ, center_first=False, held=shape,
                    pending=pending)
        info = pp.last_info
        cap = footprint_capsule(info['xy'][0], info['xy'][1], shape['length'], shape['width'],
                                info['axis'])
        if shape is CUCUMBER:                                      # 2/3 inside is enough
            assert _inside_along(z, info['xy'][0], info['xy'][1], info['axis'], 0.26) >= 0.75
        else:
            assert capsule_edge_margin(z, cap) >= 0.0              # wholly inside the card
        assert info['footprint_clearance'] is None or info['footprint_clearance'] >= 0.0
        assert p.T_release[2, 3] - shape['grip_z'] == pytest.approx(0.01)    # not stacked
        occ.append(_row(pp, p, shape, h))
        caps.append(cap)
    assert pp.last_info['tier'] in ('slot', 'grid', 'best_effort', 'long_strip')


def test_two_cubes_spread_on_a_card():
    z = _card('zone_red')
    pp = PlacePlanner(make_cfg('L1'))
    cube = dict(length=0.05, width=0.05, elongated=False, grip_width=0.05, grip_z=0.02)
    p1 = pp.plan(z, 0.02, 0.2, held=cube)
    assert np.allclose(p1.T_release[:2, 3], [0.25, 0.30])           # center first
    p2 = pp.plan(z, 0.02, 0.2, held=cube, occupied=[_row(pp, p1, cube, 0.05)])
    assert np.linalg.norm(p2.T_release[:2, 3] - p1.T_release[:2, 3]) >= 0.05
    assert p2.T_release[2, 3] == pytest.approx(0.03)
    assert p2.open_width == pytest.approx(0.07)


# ------------------------------------------------------------- reach lowering
def _pear_plans(x=0.40):
    cfg = make_cfg('L2')
    det = make_object_det('pear', x=x, y=0.0)
    gp = GraspPlanner(cfg, 'L2').plan(det)
    pp = PlacePlanner(cfg).plan(_card(), held_offset_of(gp), 0.30, held=held_shape(det, gp))
    return cfg, det, gp, pp


def test_unreachable_free_space_poses_are_lowered_to_their_floors():
    _cfg, det, gp, pp = _pear_plans()
    arm = FakeSafeArm(SimClock())
    arm.reach_fn = lambda T: (T[2, 3] <= 0.16 + 1e-9, 'too high')
    floors = {'pre_grasp': det.top_z + 0.02, 'lift': 0.15, 'pre_place': 0.15, 'retreat': 0.12}
    report = {}
    ok, why, _tilts = adjust_plan_poses(arm, gp, pp, floors, None, report)
    assert ok, why
    # Lowered in 1 cm steps to the first reachable height, never below the floor.
    for T, floor in ((gp.T_pre, det.top_z + 0.02), (gp.T_lift, 0.15), (pp.T_pre, 0.15)):
        assert floor - 1e-9 <= T[2, 3] <= 0.16 + 1e-9
    assert set(report['lowered']) >= {'pre_grasp', 'lift', 'pre_place'}
    # Floors above what is reachable: refused, plans unchanged.
    _cfg, det, gp, pp = _pear_plans()
    before = gp.T_pre.copy()
    ok, why, _ = adjust_plan_poses(arm, gp, pp, {'pre_grasp': 0.18}, None)
    assert not ok and why.startswith('pre_grasp') and np.allclose(gp.T_pre, before)


def test_retreat_goes_straight_up_from_the_release():
    _cfg, _det, gp, pp = _pear_plans()
    arm = FakeSafeArm(SimClock())
    assert adjust_plan_poses(arm, gp, pp)[0]
    assert np.allclose(pp.T_retreat[:3, :3], pp.T_release[:3, :3])
    assert np.allclose(pp.T_retreat[:2, 3], pp.T_release[:2, 3])
    assert np.allclose(gp.T_lift[:3, :3], gp.T_grasp[:3, :3])


# ------------------------------------------------------------- free-space tilt
def test_guard_tilt_limit_is_capped_by_max_tilt_free():
    cfg = make_cfg()
    g = SafetyGuard(cfg)
    assert math.degrees(g.tilt_limit()) == pytest.approx(30.0)
    assert math.degrees(g.tilt_limit(40.0)) == pytest.approx(40.0)
    assert math.degrees(g.tilt_limit(60.0)) == pytest.approx(40.0)
    assert g.tilt_limit(float('nan')) == 0.0
    cfg.set('robot.workspace.max_tilt_free_deg', 10.0)
    assert math.degrees(SafetyGuard(cfg).tilt_limit(40.0)) == pytest.approx(30.0)
    cfg.set('robot.workspace.max_tilt_free_deg', 'x')
    with pytest.raises(ValueError):
        SafetyGuard(cfg)


def test_safe_arm_accepts_a_per_call_tilt_limit():
    cfg = make_cfg()
    arm = make_arm(cfg, SimClock())
    arm.connect()
    assert arm.enable()
    assert free_tilt_limit(arm, cfg) == pytest.approx(40.0)
    T = top_down_T(0.30, 0.0, 0.30, neutral_closing_angle(0.30, 0.0))
    T30, _t, _why = arm.find_reachable_tilt(T)
    T40, tilt, why = arm.find_reachable_tilt(T, max_tilt_deg=40.0)
    assert T30 is None and T40 is not None, why
    assert math.degrees(tilt) > 30.0
    assert not arm.check_reachable(T40)[0]
    assert arm.check_reachable(T40, max_tilt_deg=40.0)[0]
    assert arm.move_tcp(T40).error == 'rejected'
    assert arm.move_tcp(T40, max_tilt_deg=40.0).ok
    # A request above max_tilt_free_deg is capped.
    assert arm.find_reachable_tilt(top_down_T(0.30, 0.0, 0.36, 0.0), max_tilt_deg=80.0)[0] is None
    assert free_tilt_limit(FakeSafeArm(SimClock()), cfg) is None


def test_skill_passes_the_free_tilt_only_for_free_space_moves():
    cfg = make_cfg('L2')
    clock = SimClock()
    seen = []

    class TiltArm(FakeArm):
        def move_tcp(self, T_base_tcp, linear=False, timeout=None, max_tilt_deg=None):
            seen.append(max_tilt_deg)
            return super().move_tcp(T_base_tcp, linear, timeout)

        def check_reachable(self, T, q_seed=None, max_tilt_deg=None):
            return True, ''

    arm = TiltArm(clock)
    skill = PickPlaceSkill(cfg, arm, clock, task='L2')
    assert skill.free_tilt == pytest.approx(40.0)
    for stage, expect in (('pre', 40.0), ('descend', None), ('release', None),
                          ('transit', 40.0), ('retreat', 40.0)):
        seen.clear()
        skill._move(type('R', (), {'detail': {'steps': []}, 'label': 'x', 'zone': 'z'})(),
                    stage, top_down_T(0.3, 0.0, 0.1, 0.0), False)
        assert seen == [expect], stage


# ------------------------------------------------------------------ transit line
def test_line_obstacle_top_only_counts_objects_near_the_line():
    det = make_det('cucumber', 0.30, -0.10, size=0.04, length=0.26, height=0.04)
    near = make_det('pear', 0.27, 0.10, size=0.085, height=0.115)
    far = make_det('pear', 0.55, 0.05, size=0.085, height=0.20)
    top = line_obstacle_top(det, [det, near, far], (0.25, 0.30))
    assert top == pytest.approx(0.115)
    assert line_obstacle_top(det, [det, far], (0.25, 0.30)) is None


# ------------------------------------------------------------------ gripper force
def test_task_gripper_force_defaults_overrides_and_clamp():
    cfg = make_cfg()
    assert gripper_force_for(cfg, 'L1') == pytest.approx(1.5)
    assert gripper_force_for(cfg, 'L2') == pytest.approx(2.5)
    assert gripper_force_for(cfg, None) == pytest.approx(1.5)        # robot.gripper.force
    cfg.set('tasks.L2.gripper_force', 2.0)
    assert gripper_force_for(cfg, 'L2') == pytest.approx(2.0)
    cfg.set('tasks.L1.gripper_force', 5.0)
    assert gripper_force_for(cfg, 'L1') == pytest.approx(3.0)
    clock = SimClock()
    assert PickPlaceSkill(cfg, FakeArm(clock), clock, task='L2').force == pytest.approx(2.0)
    assert PickPlaceSkill(cfg, FakeArm(clock), clock, force=9.0).force == pytest.approx(3.0)


# ------------------------------------------------------------ runner, green card
def _green_rig(monkeypatch, order=()):
    install_fake_modules(monkeypatch)
    rig = Rig('L2', zones={'zone_green': dict(CARD, label='green card')})
    rig.cfg.set('robot.gripper.max_width', 0.10)
    rig.cfg.set('tasks.L2.labels', ['cucumber', 'pear'])
    rig.cfg.set('tasks.L2.class_to_zone', {'cucumber': 'zone_green', 'pear': 'zone_green'})
    rig.cfg.set('tasks.L2.required', 2)
    rig.cfg.set('tasks.L2.order', list(order))
    rig.world = FakeWorld(parse_zones(rig.cfg))
    rig.arm.world = rig.perception.world = rig.world
    rig.arm.max_width = 0.10
    # The pear is nearer to the base: 'nearest' alone would pick it first.
    rig.world.add(make_det('pear', 0.25, -0.05, size=0.085, height=0.115))
    rig.world.add(make_det('cucumber', 0.36, -0.12, size=0.04, length=0.26, height=0.04))
    return rig


def test_runner_releases_the_long_cucumber_first_then_the_pear_beside_it(monkeypatch):
    rig = _green_rig(monkeypatch)
    rep = rig.runner().run()
    assert rep.ok and sorted(rig.world.placed['zone_green']) == ['cucumber', 'pear']
    picks = [e for e in rig.logger.kinds('decision') if e['action'] == 'pick']
    assert [e['label'] for e in picks] == ['cucumber', 'pear']
    cuc, pear = (e['place'] for e in picks)
    assert cuc['tier'] == 'long_strip'                                # along a card axis
    assert abs(math.sin(2.0 * cuc['axis'])) < 1e-4
    zone = parse_zones(rig.cfg)['zone_green']
    assert _inside_along(zone, cuc['xy'][0], cuc['xy'][1], cuc['axis'], 0.26) >= 0.75
    pear_cap = footprint_capsule(pear['xy'][0], pear['xy'][1], 0.085, 0.085, None)
    assert capsule_edge_margin(zone, pear_cap) >= 0.0
    assert pear['footprint_clearance'] >= 0.0
    # Both released place.release_height_flat above the card (bottom = TCP - held offset),
    # the pear place.release_lower lower.
    for e in picks:
        low = 0.005 if e['label'] == 'pear' else 0.0
        assert e['release_xyz'][2] - e['grasp_xyz'][2] == pytest.approx(0.01 - low)
    closes = {c[2] for c in rig.arm.commands if c[0] == 'gripper' and c[1] < 0.01}
    assert closes == {2.5}                                            # tasks.L2 default


def test_a_fixed_order_is_kept(monkeypatch):
    rig = _green_rig(monkeypatch, order=('pear', 'cucumber'))
    rig.runner().run()
    picks = [e['label'] for e in rig.logger.kinds('decision') if e['action'] == 'pick']
    assert picks[0] == 'pear'
