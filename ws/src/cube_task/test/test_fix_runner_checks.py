"""Table floor (expected_z), self-check zones / expected_z, place slots and release faults."""
from __future__ import annotations

import io

from cube_task.core.clock import SimClock
from cube_task.core.types import GripperResult, PerceptionResult, TablePlane
from cube_task.core.zones import parse_zone, parse_zones
from cube_task.perception.grasp import (GraspPlanner, held_offset_of, PlacePlanner,
                                        release_slots, SLOT_MARGIN, table_floor_z,
                                        table_z_warning)
from cube_task.task.selfcheck import self_check
from cube_task.task.skills import PickPlaceSkill
from cube_task.tools import calib_verify as cv
from cube_task.tools.ui import Ui
import numpy as np
import pytest
from test_task_fakes import (default_camera_T, FakeArm, FakeCamera, FakeExt, FakeLogger,
                             FakeSafeArm, FakeWorld, install_fake_modules, make_cfg, make_det,
                             Rig)


# ------------------------------------------------------------------- table floor
def test_table_floor_helpers():
    assert table_floor_z(-0.02, None) == -0.02
    assert table_floor_z(-0.02, 0.0) == 0.0
    assert table_floor_z(0.01, 0.0) == 0.01
    assert table_z_warning(-0.005, 0.0) == ''
    assert '-25 mm' in table_z_warning(-0.025, 0.0)
    assert table_z_warning(-0.025, None) == ''


def test_grasp_floor_never_below_the_taught_table():
    cfg = make_cfg('L1')
    # Camera z offset: the cube (35 mm) and the table both look 25 mm low.
    det = make_det('red', 0.30, 0.0, size=0.035, height=0.035, table_z=-0.025)
    z_cam = float(GraspPlanner(cfg, 'L1').plan(det).T_grasp[2, 3])
    assert z_cam < 0.0                                  # into the real table without the bound
    cfg.set('perception.table.expected_z', 0.0)
    plan = GraspPlanner(cfg, 'L1').plan(det)
    clearance = float(cfg.get('robot.workspace.min_table_clearance'))
    assert plan.feasible
    assert float(plan.T_grasp[2, 3]) >= 0.0 + clearance
    # A table seen higher than taught keeps the perceived floor.
    det2 = make_det('red', 0.30, 0.0, size=0.035, height=0.035, table_z=0.01)
    assert float(GraspPlanner(cfg, 'L1').plan(det2).T_grasp[2, 3]) >= 0.01 + clearance


def test_expected_z_must_be_a_number():
    cfg = make_cfg('L1')
    cfg.set('perception.table.expected_z', 'abc')
    with pytest.raises(ValueError):
        GraspPlanner(cfg, 'L1')


class _TablePerception:
    """Report one block on a table plane at z = table (base frame)."""

    def __init__(self, clock, world, table):
        self.clock, self.world, self.table = clock, world, table

    def process(self, frame, T_base_camera, zones=None):
        self.clock.sleep(0.2)
        import copy
        dets = [copy.deepcopy(d) for d in self.world.objects]
        plane = TablePlane(normal=np.array([0.0, 0.0, 1.0]), d=-self.table, frame='base')
        return PerceptionResult(detections=dets, frame_stamp=frame.stamp, table=plane)


def test_runner_safety_floor_uses_expected_z_and_warns(monkeypatch):
    install_fake_modules(monkeypatch)
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 1)
    rig.cfg.set('perception.table.expected_z', 0.0)
    rig.world.add(make_det('red', 0.25, -0.10, table_z=-0.02))
    rig.perception = _TablePerception(rig.clock, rig.world, -0.02)
    rep = rig.runner().run()
    assert rep.ok
    assert rig.arm.table_z == pytest.approx(0.0)
    obs = [e for e in rig.logger.kinds('observation') if e['ok']]
    assert any('table z differs from the taught table' in w and '-20 mm' in w
               for w in obs[0]['warnings'])


def test_runner_safety_floor_without_expected_z_is_the_perceived_table(monkeypatch):
    install_fake_modules(monkeypatch)
    rig = Rig('L1')
    rig.cfg.set('tasks.L1.required', 1)
    rig.world.add(make_det('red', 0.25, -0.10, table_z=-0.02))
    rig.perception = _TablePerception(rig.clock, rig.world, -0.02)
    rep = rig.runner().run()
    assert rep.ok and rig.arm.table_z == pytest.approx(-0.02)


def test_calib_verify_safety_floor_uses_expected_z(tmp_path, monkeypatch):
    box = {}
    real = cv.open_session

    def wrapper(*a, **k):
        box['s'] = real(*a, **k)
        return box['s']
    monkeypatch.setattr(cv, 'open_session', wrapper)
    from cube_task.robot.safety import SafeArm
    floors = []
    real_set = SafeArm.set_table_z

    def recording(self, z):
        floors.append(z)
        return real_set(self, z)
    monkeypatch.setattr(SafeArm, 'set_table_z', recording)
    out = io.StringIO()
    ui = Ui(input_fn=lambda prompt: 'q' if prompt.startswith('next: ') else (
        'y' if '(y/n)' in prompt else ''), headless=True,
        preview_dir=str(tmp_path / 'preview'), output=out)
    code = cv.main(['--mode', 'sim', '--task', 'L1', '--headless', '--log-dir', str(tmp_path),
                    '--set', 'perception.table.expected_z=0.015'], ui=ui)
    text = out.getvalue()
    assert code == cv.EXIT_CANCELLED, text                  # 'q' at the first object
    assert 'WARNING: table z differs from the taught table' in text
    assert '15.0 mm used by the motion safety checks' in text
    assert floors == [pytest.approx(0.015)]


# ------------------------------------------------------------------ self check
@pytest.fixture
def fakes(monkeypatch):
    install_fake_modules(monkeypatch)


def _warns(msgs, text):
    return [m for m in msgs if m.startswith('[WARN]') and text in m]


def test_zone_overlap_uses_the_exclusion_margin(fakes):
    cfg = make_cfg()
    # Center outside the pick polygon (x 0.15..0.45, y -0.22..0.22); the zone edge
    # (y 0.235) plus the 2 cm margin reaches into it.
    cfg.set('zones.zone_near', {'center': [0.30, 0.285, 0.0], 'size': [0.10, 0.10]})
    cfg.set('tasks.L1.class_to_zone.purple', 'zone_near')
    ok, msgs = self_check(cfg, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert _warns(msgs, 'zone zone_near grown by perception.roi.zone_exclude_margin')
    assert not _warns(msgs, 'zone zone_red grown')
    cfg.set('perception.roi.zone_exclude_margin', 0.0)
    ok, msgs = self_check(cfg, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert not _warns(msgs, 'zone zone_near grown')


def test_zone_crossing_the_pick_polygon_without_a_vertex_inside(fakes):
    cfg = make_cfg()
    # A thin diagonal zone cutting the polygon's corner (0.45, 0.22): its center and
    # corners are outside the polygon and no polygon corner is inside it.
    cfg.set('zones.zone_bar', {'center': [0.46, 0.19, 0.0], 'size': [0.20, 0.01],
                               'yaw': -0.7853981634})
    cfg.set('perception.roi.zone_exclude_margin', 0.0)
    cfg.set('tasks.L1.class_to_zone.purple', 'zone_bar')
    ok, msgs = self_check(cfg, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert _warns(msgs, 'zone zone_bar grown')


def test_unreferenced_zone_warns(fakes):
    cfg = make_cfg()
    ok, msgs = self_check(cfg, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert ok
    # zone_yellow is used by L2 only: still referenced.
    assert not _warns(msgs, 'is not used by any tasks')
    cfg.set('zones.veg_bin', {'center': [0.45, -0.30, 0.0]})
    ok, msgs = self_check(cfg, 'L1', mode='sim', extrinsics=FakeExt(default_camera_T()))
    assert ok and _warns(msgs, 'zone veg_bin is not used by any tasks')


def test_expected_z_missing_warns_in_real_mode_only(fakes):
    clock = SimClock()
    cfg = make_cfg()
    kw = dict(arm=FakeSafeArm(clock), camera=FakeCamera(clock),
              extrinsics=FakeExt(default_camera_T()), clock=clock)
    ok, msgs = self_check(cfg, 'L1', mode='real', **kw)
    assert ok and _warns(msgs, 'perception.table.expected_z is not set')
    ok, msgs = self_check(cfg, 'L1', mode='sim', **kw)
    assert not _warns(msgs, 'expected_z is not set')
    cfg.set('perception.table.expected_z', 0.0)
    ok, msgs = self_check(cfg, 'L1', mode='real', **kw)
    assert ok and not _warns(msgs, 'expected_z is not set')


# ----------------------------------------------------------------- place slots
def _bin(size=0.14, yaw=0.0, wall=0.05):
    return parse_zone('bin', {'center': [0.25, 0.30, 0.0], 'size': [size, size], 'yaw': yaw,
                              'wall_height': wall})


def test_release_slots_fit_the_footprint_inside_the_zone():
    z = _bin()
    slots = release_slots(z, 0.065)
    assert len(slots) == 4
    h = (0.14 - 0.065) / 2.0 - SLOT_MARGIN
    # diagonal pair first, the slot farther from the base first (the next release is nearer)
    assert np.allclose(slots[0], [0.25 + h, 0.30 + h])
    assert np.allclose(slots[1], [0.25 - h, 0.30 - h])
    for a, b in (slots[0:2], slots[2:4]):
        assert np.hypot(*a) >= np.hypot(*b)
    assert release_slots(z, 0.13) == [(0.25, 0.30)]                    # too big: center only
    zr = _bin(yaw=np.pi / 2)
    for x, y in release_slots(zr, 0.065):
        assert zr.contains_xy(x, y, margin=-(0.065 / 2.0 + SLOT_MARGIN) + 1e-9)
    long_zone = parse_zone('mat', {'center': [0.25, 0.30, 0.0], 'size': [0.20, 0.08]})
    assert len(release_slots(long_zone, 0.05)) == 2


def test_place_plan_spreads_and_clears_objects_already_in_the_zone():
    cfg = make_cfg('L2')
    cfg.set('place.slot_margin', 0.015)          # room for the slot grid in a 0.14 m zone
    pp = PlacePlanner(cfg)
    z = _bin(wall=0.0)
    first = pp.plan(z, 0.03, 0.2, footprint=0.065, center_first=False)
    assert np.allclose(first.T_release[:2, 3], release_slots(z, 0.065, 0.015)[0])
    base = pp.plan(z, 0.03, 0.2)                                       # old behavior: center
    assert np.allclose(base.T_release[:2, 3], [0.25, 0.30])
    assert first.T_release[2, 3] == pytest.approx(base.T_release[2, 3])
    occ = [(float(first.T_release[0, 3]), float(first.T_release[1, 3]), 0.065, 0.08)]
    second = pp.plan(z, 0.03, 0.2, footprint=0.065, occupied=occ, center_first=False)
    # No slot of a 14 cm bin is fully clear of a 65 mm object: the one with the most
    # clearance (the diagonal opposite) wins.
    assert np.allclose(second.T_release[:2, 3], release_slots(z, 0.065, 0.015)[1])
    occ_small = [(occ[0][0], occ[0][1], 0.05, 0.08)]
    third = pp.plan(z, 0.03, 0.2, footprint=0.05, occupied=occ_small, center_first=False)
    d = np.linalg.norm(third.T_release[:2, 3] - first.T_release[:2, 3])
    assert d >= 0.05 - 1e-9                                 # first fully clear slot
    assert second.T_release[2, 3] == pytest.approx(0.0 + 0.08 + 0.01 + 0.03)
    assert second.T_pre[2, 3] >= second.T_release[2, 3]
    # Center first for single-object zones; an occupied center moves the next one off it.
    c1 = pp.plan(z, 0.03, 0.2, footprint=0.05)
    assert np.allclose(c1.T_release[:2, 3], [0.25, 0.30])
    c2 = pp.plan(z, 0.03, 0.2, footprint=0.05, occupied=[(0.25, 0.30, 0.05, 0.03)])
    assert np.linalg.norm(c2.T_release[:2, 3] - [0.25, 0.30]) > 0.02


# ------------------------------------------------------------ release faults
class _ShortOpenArm(FakeArm):
    """Gripper whose release opening stops short (finger on the bin) at a given width."""

    def __init__(self, *a, stop_at=0.05, **kw):
        super().__init__(*a, **kw)
        self.stop_at = stop_at

    def gripper(self, width, force=None, wait=True):
        if self.world.held is not None and width >= 0.01:
            super().gripper(width, force, wait)
            self.width = self.stop_at
            return GripperResult(ok=False, width=self.stop_at, error='fault',
                                 detail='opening stopped at %.4f m, commanded %.4f m'
                                 % (self.stop_at, width))
        return super().gripper(width, force, wait)


def _skill_setup(stop_at):
    cfg = make_cfg('L1')
    clock = SimClock()
    zones = parse_zones(cfg)
    world = FakeWorld(zones)
    det = world.add(make_det('red', 0.30, 0.05))
    arm = _ShortOpenArm(clock, world, stop_at=stop_at)
    skill = PickPlaceSkill(cfg, arm, clock, FakeLogger())
    g = GraspPlanner(cfg, 'L1').plan(det)
    p = PlacePlanner(cfg).plan(zones['zone_red'], held_offset_of(g), 0.2)
    return skill, g, p, world, arm


def test_release_opening_stopped_short_but_past_the_held_width_is_released():
    skill, g, p, world, arm = _skill_setup(stop_at=0.03 + 0.012)       # held 30 mm
    r = skill.execute(g, p)
    assert r.ok and not r.detail['fault'] and r.stage == 'done', (r.stage, r.error)
    assert 'opening stopped' in r.detail['opening_short']
    assert world.placed == {'zone_red': ['red']}
    assert arm.motions()[-1][1][2, 3] == pytest.approx(p.T_retreat[2, 3])   # retreated


def test_release_opening_stopped_at_the_held_width_is_still_a_fault():
    skill, g, p, world, arm = _skill_setup(stop_at=0.03 + 0.005)
    r = skill.execute(g, p)
    assert not r.ok and r.detail['fault'] and r.error == 'gripper_fault'


def test_pair_offset_keeps_zone_mates_near_the_card_center_far_one_first():
    cfg = make_cfg('L2')
    z = parse_zone('zone_red', {'center': [0.60, 0.30, 0.0], 'size': [0.20, 0.20]})
    cfg.set('place.pair_offset', {'zone_red': [0.035, 0.025]})
    pp = PlacePlanner(cfg)
    slots = pp._slots(z, 0.075)
    assert len(slots) == 2
    (x1, y1), (x2, y2) = slots
    assert np.hypot(x1, y1) > np.hypot(x2, y2)                         # far one first
    assert np.hypot(x1 - 0.60, y1 - 0.30) == pytest.approx(np.hypot(0.035, 0.025))
    assert np.allclose([x1 + x2, y1 + y2], [1.20, 0.60])               # mirrored
    for x, y in slots:
        assert z.contains_xy(x, y, margin=-(0.075 / 2.0 + SLOT_MARGIN) + 1e-9)
    other = parse_zone('zone_green', {'center': [0.60, -0.30, 0.0], 'size': [0.20, 0.20]})
    assert pp._slots(other, 0.075) == release_slots(other, 0.075, pp.slot_margin)
    cfg.set('place.pair_offset', {'zone_red': [-0.01, 0.0]})
    with pytest.raises(ValueError):
        PlacePlanner(cfg)


def test_near_center_release_goes_as_close_to_the_center_as_the_zone_mates_allow():
    cfg = make_cfg('L2')
    cfg.set('robot.gripper.max_width', 0.100)
    z = parse_zone('zone_green', {'center': [0.60, -0.30, 0.0], 'size': [0.20, 0.20]})
    pp = PlacePlanner(cfg)
    h = {'length': 0.085, 'width': 0.085, 'grip_width': 0.08, 'grip_z': 0.03}
    alone = pp.plan(z, 0.03, 0.2, footprint=0.085, held=h, center_first=False,
                    near_center=True)
    assert pp.last_info['tier'] == 'near_center'
    assert np.allclose(alone.T_release[:2, 3], [0.60, -0.30], atol=1e-6)
    occ = [(0.60, -0.245, 0.26, 0.08, 0.26, 0.04, 0.0)]              # cucumber on the strip
    slot = pp.plan(z, 0.03, 0.2, footprint=0.085, held=h, center_first=False, occupied=occ)
    near = pp.plan(z, 0.03, 0.2, footprint=0.085, held=h, center_first=False, occupied=occ,
                   near_center=True)
    info = pp.last_info
    assert info['tier'] == 'near_center'
    assert info['edge_margin'] >= pp.slot_margin - 1e-9 and info['clearance'] >= 0.0
    d = [np.hypot(p.T_release[0, 3] - 0.60, p.T_release[1, 3] + 0.30) for p in (slot, near)]
    assert d[1] < d[0]
