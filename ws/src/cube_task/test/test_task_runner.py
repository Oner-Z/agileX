"""Integration tests for task/runner.py with local fakes (no hardware, SimClock)."""
from __future__ import annotations

from cube_task.core.types import MotionResult
from cube_task.task.operator import Operator
import numpy as np
import pytest
from test_task_fakes import install_fake_modules, make_det, Rig


@pytest.fixture(autouse=True)
def fake_modules(monkeypatch):
    install_fake_modules(monkeypatch)


L1_BLOCKS = [('red', 0.25, -0.10), ('green', 0.30, 0.05), ('blue', 0.38, 0.12)]
L2_ITEMS = [('apple', 0.22, -0.15, 0.06, 0.065), ('banana', 0.30, -0.05, 0.035, 0.16),
            ('carrot', 0.40, 0.00, 0.03, 0.14), ('eggplant', 0.25, 0.10, 0.05, 0.12),
            ('orange', 0.35, 0.15, 0.055, 0.058), ('corn', 0.42, -0.15, 0.045, 0.15)]


def _l1(rig, blocks=L1_BLOCKS):
    return [rig.world.add(make_det(lbl, x, y)) for lbl, x, y in blocks]


def _commands_after(arm, n):
    return arm.commands[n:]


def test_l1_places_three_within_limit():
    rig = Rig('L1', runner__return_home_at_end=False)
    _l1(rig)
    rep = rig.runner().run()
    assert rep.stop_reason == 'done' and rep.ok, (rep.stop_reason, rig.logger.events[-1])
    assert rep.placed == 3 and rep.attempts == 3
    assert 0 < rep.elapsed < 180
    assert rig.world.placed == {'zone_red': ['red'], 'zone_green': ['green'],
                                'zone_blue': ['blue']}
    # One observe move per pick (the wrist camera captures there), no return home.
    assert len([c for c in rig.arm.commands if c[0] == 'move_joints']) == 3
    assert rig.arm.commands[-1][0] == 'move_tcp'
    kinds = [c[0] for c in rig.arm.commands]
    first_tcp = kinds.index('move_tcp')
    assert kinds[:first_tcp] == ['move_joints', 'gripper', 'gripper']
    assert len(rig.logger.images) == 3
    assert len([e for e in rig.logger.kinds('decision') if e['action'] == 'pick']) == 3
    assert rig.logger.json['report.json']['placed'] == 3
    assert rig.arm.table_z == pytest.approx(0.0)
    assert rig.arm.reach_checks == 4 * 2 + 3 * 6      # self check (4 zones) + 6 per pick


def test_l2_sorts_six_into_category_zones():
    rig = Rig('L2')
    for lbl, x, y, w, ln in L2_ITEMS:
        rig.world.add(make_det(lbl, x, y, size=w, length=ln, height=0.05, yaw=0.3))
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 6, (rep.stop_reason, [p.error for p in rep.picks])
    assert rep.elapsed < 300
    assert sorted(rig.world.placed['zone_red']) == ['apple', 'orange']
    assert sorted(rig.world.placed['zone_yellow']) == ['banana', 'corn']
    assert rig.world.placed['zone_green'] == ['carrot']
    assert rig.world.placed['zone_blue'] == ['eggplant']


def test_grasp_miss_then_retry_succeeds():
    rig = Rig('L1')
    _l1(rig)
    rig.world.miss_next = 1
    rep = rig.runner().run()
    assert rep.ok and rep.placed == 3 and rep.attempts == 4
    assert rep.picks[0].stage == 'grasp_miss'


def test_repeated_failures_blacklist_and_move_on():
    rig = Rig('L1')
    _l1(rig)
    rig.world.always_miss = {'green'}
    rep = rig.runner().run()
    assert rep.placed == 2 and rep.stop_reason == 'no_targets' and not rep.ok
    green = [p for p in rep.picks if p.label == 'green']
    assert len(green) == 3 and all(p.stage == 'grasp_miss' for p in green)
    bl = [e for e in rig.logger.kinds('decision') if e['action'] == 'blacklist']
    assert len(bl) == 1 and bl[0]['label'] == 'green'


def test_unknown_and_unmapped_labels_are_never_picked():
    rig = Rig('L1')
    _l1(rig, [('red', 0.25, -0.10)])
    rig.world.add(make_det('unknown', 0.20, 0.0, score=0.2))
    rig.world.add(make_det('purple', 0.22, 0.05))          # classified, but no zone
    rig.cfg.set('tasks.L1.required', 3)
    rep = rig.runner().run()
    assert rep.placed == 1 and rep.stop_reason == 'no_targets'
    assert [p.label for p in rep.picks] == ['red']
    for c in rig.arm.motions():
        if c[0] == 'move_tcp':
            xy = c[1][:2, 3]
            assert np.linalg.norm(xy - [0.20, 0.0]) > 0.01
            assert np.linalg.norm(xy - [0.22, 0.05]) > 0.01
    reasons = {s['reason'] for e in rig.logger.kinds('decision') if e['action'] == 'skip'
               for s in e['skipped']}
    assert reasons == {'unknown_label', 'unmapped_label'}


def test_targets_restrict_and_order_labels():
    rig = Rig('L1')
    _l1(rig)
    rig.cfg.set('tasks.L1.required', 2)
    rep = rig.runner(targets=['blue', 'red']).run()
    assert rep.ok and [p.label for p in rep.picks] == ['blue', 'red']


def test_deadline_stops_new_picks():
    rig = Rig('L1')
    _l1(rig)
    rig.cfg.set('tasks.L1.time_limit_s', 25)
    rep = rig.runner().run()
    assert rep.stop_reason == 'timeout' and not rep.ok
    assert 1 <= rep.placed < 3
    picks = [e for e in rig.logger.kinds('decision') if e['action'] == 'pick']
    assert all(e['remaining'] >= 12.0 for e in picks)
    assert rep.elapsed <= 25


def test_comm_fault_stops_immediately_without_more_commands():
    rig = Rig('L1')
    _l1(rig)
    state = {}

    def fn(kind, target, linear, idx):
        if (kind == 'move_tcp' and linear and target[2, 3] < 0.03
                and np.linalg.norm(target[:2, 3] - [0.30, 0.05]) < 0.01):
            state['n'] = len(rig.arm.commands)
            return MotionResult(ok=False, error='comm')
        return None
    rig.arm.fail_fn = fn
    rep = rig.runner().run()
    assert rep.stop_reason == 'fault' and rep.placed == 1
    assert len(rig.arm.commands) == state['n']          # nothing after the failed command
    assert rep.picks[-1].detail['fault']


def test_unhealthy_arm_between_picks_stops_as_fault():
    rig = Rig('L1')
    _l1(rig)
    rig.world.on_place = lambda zone, label: setattr(rig.arm, 'healthy', False)
    rep = rig.runner().run()
    assert rep.stop_reason == 'fault' and rep.placed == 1
    assert rig.arm.commands[-1][0] == 'move_tcp'       # the retreat; no observe / home move


def test_operator_abort_during_run_stops_motion():
    rig = Rig('L1')
    _l1(rig)
    op = Operator(enabled=False)
    rig.world.on_place = lambda zone, label: op.abort('test')
    rep = rig.runner(operator=op).run()
    assert rep.stop_reason == 'aborted' and rep.placed == 1
    # Abort arrived while the gripper opened at the release pose: no command after it.
    assert rig.arm.commands[-1][0] == 'gripper'
    assert rep.picks[-1].ok and rep.picks[-1].error == 'retreat_aborted'
    assert len([c for c in rig.arm.commands if c[0] == 'move_joints']) == 1


def test_operator_abort_before_start():
    import io
    rig = Rig('L1')
    _l1(rig)
    op = Operator(enabled=True, stdin=io.StringIO('p'), out=io.StringIO())
    rep = rig.runner(operator=op).run()
    op.close()
    assert rep.stop_reason == 'aborted' and rep.placed == 0 and rep.attempts == 0
    # The key reader runs from the first motion on, so the 'p' stops the preparation
    # wherever it is read: during the observe move, the gripper open or the prompt.
    kinds = [c[0] for c in rig.arm.commands]
    assert kinds in (['move_joints'], ['move_joints', 'gripper']), kinds


def test_dry_run_never_closes_gripper():
    rig = Rig('L1')
    _l1(rig)
    rep = rig.runner(dry_run=True).run()
    assert rep.placed == 0 and not rep.ok and rep.stop_reason == 'no_targets'
    assert rep.attempts == 3 and all(p.stage == 'dry_run_done' for p in rep.picks)
    assert min(rig.arm.gripper_cmds()) >= 0.04
    assert len(rig.world.objects) == 3


def test_real_mode_refuses_when_self_check_fails():
    rig = Rig('L1')
    _l1(rig)
    rig.cfg.set('robot.backend', 'pyagxarm')
    rig.cfg.set('zones', {})
    rep = rig.runner().run()
    assert rep.stop_reason == 'self_check' and rig.arm.commands == []


def test_unreachable_target_is_blacklisted_without_motion():
    rig = Rig('L1')
    _l1(rig)
    rig.arm.reach_fn = lambda T: ((False, 'ik failed')
                                  if np.linalg.norm(T[:2, 3] - [0.30, 0.05]) < 0.01
                                  else (True, ''))
    rep = rig.runner().run()
    assert rep.placed == 2 and 'green' not in [p.label for p in rep.picks]
    bl = [e for e in rig.logger.kinds('decision') if e['action'] == 'blacklist']
    assert bl and bl[0]['reason'].startswith('unreachable')


@pytest.mark.parametrize('mode', ['stale', 'none'])
def test_bad_frames_are_retried_until_the_deadline(mode):
    rig = Rig('L1')
    _l1(rig)
    if mode == 'stale':
        rig.camera.stale = True
    else:
        rig.camera.none_frames = 1000
    runner = rig.runner()
    rep = runner.run()
    # Camera failures are not 'nothing to pick': the round keeps trying until the
    # deadline and the report names the camera.
    assert rep.stop_reason == 'timeout' and rep.attempts == 0
    assert rep.elapsed >= 180 - 12.0 - 1e-6
    reason = 'stale frame' if mode == 'stale' else 'no frame'
    assert runner.fault_detail.startswith('observation failures: %s' % reason)
    assert not [c for c in rig.arm.commands if c[0] == 'move_tcp']
    assert rig.perception.calls == []


def test_observe_move_failure_is_a_fault():
    rig = Rig('L1')
    _l1(rig)
    rig.arm.fail_fn = lambda kind, T, lin, idx: (MotionResult(ok=False, error='timeout')
                                                 if kind == 'move_joints' and idx > 0
                                                 else None)
    rep = rig.runner().run()
    assert rep.stop_reason == 'fault' and rep.placed == 1


def test_eye_in_hand_uses_flange_pose_at_capture():
    from test_task_fakes import FakeExt
    from cube_task.core.transforms import top_down_T
    rig = Rig('L1', camera__mount='eye_in_hand', runner__return_home_at_end=False)
    _l1(rig)
    T_fc = np.eye(4)
    T_fc[:3, 3] = [0.05, 0.0, 0.03]
    rig.ext = FakeExt(T_fc, 'eye_in_hand')
    rep = rig.runner().run()
    assert rep.ok
    flange_at_observe = top_down_T(0.05, 0.25, 0.30, 0.0)    # FakeArm's observe pose
    assert all(np.allclose(T, flange_at_observe @ T_fc) for T in rig.perception.calls)
    assert len([c for c in rig.arm.commands if c[0] == 'move_joints']) == 3


def test_start_on_enter_false_does_not_wait():
    import io
    rig = Rig('L1')
    _l1(rig, L1_BLOCKS[:1])
    rig.cfg.set('tasks.L1.required', 1)
    rig.cfg.set('runner.start_on_enter', False)
    op = Operator(enabled=True, stdin=io.StringIO(''), out=io.StringIO())
    rep = rig.runner(operator=op).run()
    op.close()
    assert rep.ok and rep.placed == 1


def test_held_object_without_fault_flag_still_stops(monkeypatch):
    from cube_task.core.types import PickResult
    from cube_task.task.skills import PickPlaceSkill
    rig = Rig('L1')
    _l1(rig)
    monkeypatch.setattr(PickPlaceSkill, 'execute', lambda self, g, p, dry_run=False: PickResult(
        ok=False, stage='transit', error='timeout', label=g.target.label,
        detail={'fault': False, 'holding': True}))
    rep = rig.runner().run()
    assert rep.stop_reason == 'fault' and rep.attempts == 1


def test_zone_order_places_the_earlier_label_first_and_falls_back_when_it_is_gone():
    rig = Rig('L2')
    rig.cfg.set('runner.zone_order', {'zone_red': ['orange', 'apple']})
    # apple is nearer the base, so without zone_order it would be picked first
    rig.world.add(make_det('apple', 0.22, -0.15, size=0.06, length=0.065, height=0.05))
    rig.world.add(make_det('orange', 0.35, 0.15, size=0.055, length=0.058, height=0.05))
    rep = rig.runner().run()
    assert rep.placed == 2, (rep.stop_reason, [p.error for p in rep.picks])
    assert rig.world.placed['zone_red'] == ['orange', 'apple']
    rig2 = Rig('L2')
    rig2.cfg.set('runner.zone_order', {'zone_red': ['orange', 'apple']})
    rig2.cfg.set('tasks.L2.required', 1)
    rig2.world.add(make_det('apple', 0.22, -0.15, size=0.06, length=0.065, height=0.05))
    rep2 = rig2.runner().run()                                         # no orange in view
    assert rig2.world.placed.get('zone_red') == ['apple'], rep2.stop_reason
