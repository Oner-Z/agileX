"""
Pick-and-place skill: one top-down grasp of one detection into one zone.

All poses are TCP poses in the robot base frame (m, rad). Gripper widths in m.
Durations use the injected Clock.

Failure policy (the runner never auto-restarts, see the competition rules):

- Motion errors 'comm' / 'arm_error' / 'aborted', gripper errors 'comm' /
  'fault' / 'aborted', missing gripper feedback and unknown error strings are
  faults: return at once with ``detail['fault'] = True`` and send no more motion.
  Exception: at the release, an opening that stopped short (finger against the
  bin or an earlier object) once open past held width + 10 mm counts as released.
- 'timeout' / 'stalled' / 'unreachable' / 'rejected' are recoverable. If the
  arm is healthy, one recovery is tried and the pick reports failure:
  before the object is held, a linear retreat up to the pre-grasp pose; while
  holding over the pick area or in transit, the object is put back where it was
  (back to T_lift, down to T_grasp, open, up to T_pre). The gripper is never
  opened over a bin except at the release pose. If a held object cannot be put
  down, the result is a fault ('holding_object').
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.types import GraspPlan, MotionResult, PickResult, PlacePlan
from ..perception.grasp import RELEASE_OPEN_EXTRA, max_gripper_width

_LOG = logging.getLogger('cube_task.task.skills')

FAULT_MOTION_ERRORS = ('comm', 'arm_error', 'aborted')
RECOVERABLE_MOTION_ERRORS = ('timeout', 'stalled', 'unreachable', 'rejected')
FAULT_GRIPPER_ERRORS = ('comm', 'fault', 'aborted')
#: measured opening must exceed the expected object width by this before descending.
OPEN_CLEARANCE = 0.002
#: a stopped release move still counts as "at release" within these limits (m).
RELEASE_XY_TOL = 0.02
RELEASE_Z_ABOVE_TOL = 0.08
RELEASE_Z_BELOW_TOL = 0.01
#: at the release the fingers must open this far past the held width (m).
RELEASE_OPEN_MARGIN = 0.01
#: ... but never more than the commanded opening minus this (m): a gripper opened to
#: its full width reports a few um to a few 0.1 mm less (MuJoCo: 0.06995 for 0.07),
#: which must not turn a release into a 'holding_object' fault.
RELEASE_FULL_OPEN_TOL = 0.002
#: detail text of a gripper 'fault' whose opening stopped short (robot/pyagxarm_arm.py).
OPENING_SHORT = 'opening stopped'
#: AGX gripper force range (N); a configured force above it is clamped.
MAX_GRIPPER_FORCE = 3.0
#: default tasks.<T>.gripper_force (N): light foam cubes (L1), 40-100 g models (L2).
TASK_GRIPPER_FORCE = {'L1': 1.5, 'L2': 2.5}
#: stages whose target is a free-space pose (may lean up to max_tilt_free_deg).
FREE_STAGES = ('pre', 'lift', 'transit', 'retreat', 'recover_retreat', 'recover_up',
               'putback_lift', 'putback_retreat', 'up')


def _force_value(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f >= 0.0 else None


def gripper_force_for(cfg, task: Optional[str] = None) -> Optional[float]:
    """
    Return the grasp force (N) for task, clamped to MAX_GRIPPER_FORCE.

    tasks.<task>.gripper_force, else TASK_GRIPPER_FORCE, else
    robot.gripper.force. None when none of them is a number >= 0.
    """
    f = None
    if task:
        f = _force_value(cfg.get('tasks.%s.gripper_force' % task, None))
        if f is None:
            f = _force_value(TASK_GRIPPER_FORCE.get(task))
    if f is None:
        f = _force_value(cfg.get('robot.gripper.force', None))
    if f is None:
        return None
    if f > MAX_GRIPPER_FORCE:
        _LOG.warning('gripper force %.2f N above %.1f N: clamped', f, MAX_GRIPPER_FORCE)
    return min(f, MAX_GRIPPER_FORCE)


class _StepFailed(Exception):
    def __init__(self, stage: str, error: str, detail: str = '', fault: bool = False):
        super().__init__('%s: %s %s' % (stage, error, detail))
        self.stage = stage
        self.error = error or 'unknown'
        self.detail = detail
        self.fault = fault


class _Lost(Exception):
    """The gripper closed on nothing after holding: the object fell."""

    def __init__(self, stage: str, width: float):
        super().__init__(stage)
        self.stage = stage
        self.width = width


def _xyz(T: np.ndarray) -> List[float]:
    return [round(float(v), 4) for v in np.asarray(T)[:3, 3]]


def is_fault_motion_error(error: str) -> bool:
    """Return True when a MotionResult error must stop the run (no more motion)."""
    return error not in RECOVERABLE_MOTION_ERRORS


class PickPlaceSkill:
    """
    Execute GraspPlan + PlacePlan on an ArmInterface.

    logger: core.logger.RunLogger-like object with ``event(kind, **fields)``, or
    None. Config: robot.gripper.*, grasp.linear_descent, grasp.verify_grasp,
    runner.dry_run_hover, robot.motion.feedback_max_age_s. task: grasp force
    from gripper_force_for(cfg, task) (None = robot.gripper.force, clamped);
    force: explicit force (N) that wins over both. Moves to free-space poses
    (FREE_STAGES) pass robot.workspace.max_tilt_free_deg to an arm that takes
    a per-call tilt limit (task.reach.free_tilt_limit).
    """

    def __init__(self, cfg, arm, clock, logger=None, task: Optional[str] = None,
                 force: Optional[float] = None):
        self.arm = arm
        self.clock = clock
        self.logger = logger
        self.max_width = max_gripper_width(cfg)
        ew = cfg.get('robot.gripper.empty_width', None)
        if ew is None or not math.isfinite(float(ew)) or float(ew) < 0.0:
            raise ValueError('robot.gripper.empty_width must be a number >= 0, got %r' % (ew,))
        self.empty_width = float(ew)
        f = _force_value(force)
        self.force = gripper_force_for(cfg, task) if f is None else min(f, MAX_GRIPPER_FORCE)
        from .reach import free_tilt_limit
        self.free_tilt = free_tilt_limit(arm, cfg)
        self.width_tolerance = float(cfg.get('robot.gripper.width_tolerance', 0.015))
        self.linear_descent = bool(cfg.get('grasp.linear_descent', True))
        # robot.motion.linear_moves false: short vertical moves (lift, release,
        # retreat, recovery) go as P2P, e.g. if move_l spins the wrist across the
        # +-180 deg roll wrap on the real controller.
        from ..robot.pyagxarm_arm import linear_moves_enabled
        self.linear_moves = linear_moves_enabled(cfg)
        self.verify = bool(cfg.get('grasp.verify_grasp', True))
        self.hover = float(cfg.get('runner.dry_run_hover', 0.05))
        self.feedback_max_age = float(cfg.get('robot.motion.feedback_max_age_s', 0.5))
        #: called with the stage name before every move (task.live watcher); None = off.
        self.on_stage = None

    # ------------------------------------------------------------------ public
    def execute(self, grasp: GraspPlan, place: PlacePlan, dry_run: bool = False) -> PickResult:
        """Run the pick-and-place. Never raises for arm failures; see module docstring."""
        t0 = self.clock.now()
        r = PickResult(ok=False, stage='start', label=grasp.target.label, zone=place.zone,
                       detail={'fault': False, 'holding': False, 'dry_run': bool(dry_run),
                               'steps': []})
        if not grasp.feasible or not place.feasible:
            r.stage = 'plan'
            r.error = 'infeasible: %s' % (grasp.reason if not grasp.feasible else place.reason)
        else:
            try:
                if dry_run:
                    self._dry_run(grasp, place, r)
                else:
                    self._pick_place(grasp, place, r)
            except Exception as e:  # noqa: B902 - an arm backend bug must not crash the run
                _LOG.exception('pick-place raised')
                r.ok = False
                r.error = 'exception: %s' % e
                r.detail['fault'] = True
                try:
                    self.arm.stop()
                except Exception:  # noqa: B902
                    _LOG.exception('arm.stop() raised')
        r.duration = self.clock.now() - t0
        self._event('pick_result', ok=r.ok, stage=r.stage, error=r.error, label=r.label,
                    zone=r.zone, duration=round(r.duration, 3), grasp_width=r.grasp_width,
                    fault=r.detail.get('fault', False), holding=r.detail.get('holding', False))
        return r

    # -------------------------------------------------------------- primitives
    def _event(self, kind: str, **fields: Any) -> None:
        if self.logger is None:
            return
        try:
            self.logger.event(kind, **fields)
        except Exception:  # noqa: B902 - logging must never stop the arm logic
            _LOG.exception('logger.event failed')

    def _record(self, r: PickResult, stage: str, ok: bool, error: str, t_start: float,
                **extra: Any) -> None:
        dt = self.clock.now() - t_start
        step: Dict[str, Any] = {'stage': stage, 'ok': bool(ok), 'error': error,
                                'duration': round(dt, 3)}
        step.update(extra)
        r.detail['steps'].append(step)
        _LOG.info('step %-14s ok=%s %s %.2fs', stage, ok, error, dt)
        self._event('skill_step', label=r.label, zone=r.zone, **step)

    def _check_abort(self, stage: str) -> None:
        try:
            aborted = self.arm.abort_event.is_set()
        except Exception:  # noqa: B902
            aborted = False
        if aborted:
            raise _StepFailed(stage, 'aborted', 'abort event set', fault=True)

    def _healthy(self) -> bool:
        try:
            h = self.arm.health()
        except Exception:  # noqa: B902
            _LOG.exception('arm.health() raised')
            return False
        return bool(h is not None and h.ok)

    def _move(self, r: PickResult, stage: str, T: np.ndarray, linear: bool) -> MotionResult:
        self._check_abort(stage)
        if self.on_stage is not None:
            try:
                self.on_stage(stage)
            except Exception:  # noqa: B902 - an observer never changes the motion
                _LOG.exception('on_stage hook failed')
        t = self.clock.now()
        if self.free_tilt is not None and stage in FREE_STAGES:
            m = self.arm.move_tcp(T, linear=linear, max_tilt_deg=self.free_tilt)
        else:
            m = self.arm.move_tcp(T, linear=linear)
        self._record(r, stage, m.ok, m.error, t, target=_xyz(T), linear=bool(linear),
                     pos_err=m.pos_err)
        if not m.ok:
            err = m.error or 'unknown'
            raise _StepFailed(stage, err, m.detail, fault=is_fault_motion_error(err))
        return m

    def _read_width(self) -> Optional[float]:
        """Return a fresh measured gripper width (m) or None (missing / stale / NaN)."""
        try:
            s = self.arm.gripper_width()
        except Exception:  # noqa: B902
            _LOG.exception('gripper_width() raised')
            return None
        if s is None or s.value is None:
            return None
        if self.clock.now() - float(s.stamp) > self.feedback_max_age:
            return None
        w = float(s.value)
        return w if math.isfinite(w) else None

    def _gripper(self, r: PickResult, stage: str, width: float,
                 short_ok: Optional[float] = None) -> Optional[float]:
        """
        Command the gripper; return the measured width (m) or None if unknown.

        short_ok: an opening that stopped short of the command (gripper error
        'fault', detail 'opening stopped ...') is accepted when the measured
        width reaches short_ok (m); used at the release, where the fingers may
        touch the bin or an object already in it.
        """
        self._check_abort(stage)
        t = self.clock.now()
        g = self.arm.gripper(width, force=self.force, wait=True)
        w = g.width if (g.width is not None and math.isfinite(float(g.width))) else None
        if w is None:
            w = self._read_width()
        self._record(r, stage, g.ok, g.error, t, command=round(float(width), 4),
                     width=None if w is None else round(w, 4))
        if (short_ok is not None and g.error == 'fault' and OPENING_SHORT in (g.detail or '')
                and w is not None and w >= short_ok):
            _LOG.warning('%s: %s; open %.4f m >= %.4f m counts as released', stage, g.detail,
                         w, short_ok)
            r.detail['opening_short'] = g.detail
            return w
        if g.error and g.error != 'timeout':  # FAULT_GRIPPER_ERRORS and unknown ones
            raise _StepFailed(stage, 'gripper_' + g.error, g.detail, fault=True)
        return w

    def _open(self, r: PickResult, stage: str, width: float, need: Optional[float],
              short_ok: Optional[float] = None) -> None:
        w = self._gripper(r, stage, width, short_ok)
        if need is None:
            return
        if w is None:
            raise _StepFailed(stage, 'no_gripper_feedback', 'after opening', fault=True)
        if w < need:
            raise _StepFailed(stage, 'gripper_not_open',
                              'measured %.4f m < needed %.4f m' % (w, need))

    def _verify_held(self, r: PickResult, stage: str) -> None:
        if not self.verify:
            return
        w = self._read_width()
        if w is None:
            raise _StepFailed('verify_' + stage, 'no_gripper_feedback', 'while holding',
                              fault=True)
        if w < self.empty_width:
            raise _Lost(stage, w)

    # -------------------------------------------------------------------- flow
    def _pick_place(self, g: GraspPlan, p: PlacePlan, r: PickResult) -> None:
        holding = False
        try:
            self._open(r, 'open', g.open_width, need=g.expected_width + OPEN_CLEARANCE)
            self._move(r, 'pre', g.T_pre, False)
            self._move(r, 'descend', g.T_grasp, self.linear_descent)
            w = self._gripper(r, 'close', g.close_width)
            r.grasp_width = w
            if w is None and self.verify:
                raise _StepFailed('close', 'no_gripper_feedback', 'after closing', fault=True)
            if self.verify and w < self.empty_width:
                r.stage = r.error = 'grasp_miss'
                self._retreat_empty(r, g, 'grasp_miss')
                return
            if w is not None and abs(w - g.expected_width) > self.width_tolerance:
                _LOG.warning('held width %.4f m differs from expected %.4f m', w,
                             g.expected_width)
            holding = True
            r.detail['holding'] = True
            for stage, T, lin in (('lift', g.T_lift, self.linear_moves),
                                  ('transit', p.T_pre, False)):
                self._move(r, stage, T, lin)
                self._verify_held(r, 'dropped' if stage == 'lift' else 'dropped_in_transit')
            self._move(r, 'release', p.T_release, self.linear_moves)
            self._release_and_retreat(r, g, p)
        except _Lost as e:
            r.detail['holding'] = False
            r.stage = r.error = e.stage
            r.detail['lost_width'] = e.width
        except _StepFailed as e:
            r.stage = e.stage
            r.error = e.error
            if e.fault:
                r.detail['fault'] = True
                return
            if not holding:
                if e.stage in ('descend',):
                    self._retreat_empty(r, g, e.error)
                return
            if e.stage == 'release':
                self._recover_release(r, g, p)
            elif e.stage in ('lift', 'transit'):
                self._put_back(r, g, from_transit=(e.stage == 'transit'))

    def _release_and_retreat(self, r: PickResult, g: GraspPlan, p: PlacePlan) -> None:
        held = r.grasp_width or 0.0
        width = p.open_width
        if r.grasp_width is not None:
            # The plan sized the opening from the detected width; an object that is
            # wider between the fingers would stay held by it.
            width = min(self.max_width, max(width, held + RELEASE_OPEN_EXTRA))
        need = min(width - RELEASE_FULL_OPEN_TOL, held + RELEASE_OPEN_MARGIN)
        try:
            # A finger stopped by the bin or an earlier object is not a fault once
            # the fingers are open past the held width: the object is released.
            self._open(r, 'open_release', width,
                       need=need if self.verify else None,
                       short_ok=need if r.grasp_width is not None else None)
        except _StepFailed as e:
            r.stage, r.error = e.stage, e.error
            if not e.fault:
                e = _StepFailed('open_release', 'holding_object', e.error, fault=True)
                r.error = e.error
            r.detail['fault'] = True
            return
        r.detail['holding'] = False
        r.ok = True
        r.stage = 'done'
        try:
            self._move(r, 'retreat', p.T_retreat, self.linear_moves)
        except _StepFailed as e:
            # The object is in the zone; report the failed retreat but keep ok=True.
            r.error = 'retreat_' + e.error
            r.detail['fault'] = bool(e.fault)
            r.detail['retreat_failed'] = True

    def _retreat_empty(self, r: PickResult, g: GraspPlan, why: str) -> None:
        """Nothing held: open the gripper and go straight up to T_pre (one try)."""
        if not self._healthy():
            r.detail['fault'] = True
            r.detail['recovery'] = 'skipped: arm unhealthy'
            return
        try:
            if why == 'grasp_miss':
                try:
                    self._open(r, 'recover_open', g.open_width, need=None)
                except _StepFailed as e:
                    if e.fault:
                        raise
            self._move(r, 'recover_retreat', g.T_pre, self.linear_moves)
            r.detail['recovery'] = 'retreated'
        except _StepFailed as e:
            # Stuck low next to the objects: further P2P motion is not safe.
            r.detail['recovery'] = 'failed: %s' % e.error
            r.detail['fault'] = True

    def _near_release(self, p: PlacePlan) -> bool:
        try:
            s = self.arm.get_tcp_pose()
        except Exception:  # noqa: B902
            return False
        if s is None or s.value is None:
            return False
        if self.clock.now() - float(s.stamp) > self.feedback_max_age:
            return False
        cur = np.asarray(s.value, dtype=float)[:3, 3]
        if not np.all(np.isfinite(cur)):
            return False
        tgt = np.asarray(p.T_release)[:3, 3]
        dz = float(cur[2] - tgt[2])
        return (float(np.linalg.norm(cur[:2] - tgt[:2])) <= RELEASE_XY_TOL
                and -RELEASE_Z_BELOW_TOL <= dz <= RELEASE_Z_ABOVE_TOL)

    def _recover_release(self, r: PickResult, g: GraspPlan, p: PlacePlan) -> None:
        """Descent into the zone stopped while holding the object."""
        if not self._healthy():
            r.detail['fault'] = True
            r.detail['recovery'] = 'skipped: arm unhealthy'
            return
        if self._near_release(p):
            r.detail['recovery'] = 'released near the release pose'
            self._release_and_retreat(r, g, p)
            return
        try:
            self._move(r, 'recover_up', p.T_pre, self.linear_moves)
        except _StepFailed as e:
            r.detail['fault'] = True
            r.error = 'holding_object'
            r.detail['recovery'] = 'failed: %s' % e.error
            return
        self._put_back(r, g, from_transit=True)

    def _put_back(self, r: PickResult, g: GraspPlan, from_transit: bool) -> None:
        """Put a held object back where it was picked; fault if that fails."""
        if not self._healthy():
            r.detail['fault'] = True
            r.detail['recovery'] = 'skipped: arm unhealthy'
            return
        try:
            if from_transit:
                self._move(r, 'putback_lift', g.T_lift, False)
            self._move(r, 'putback_down', g.T_grasp, self.linear_moves)
            self._open(r, 'putback_open', g.open_width, need=None)
        except _StepFailed as e:
            r.detail['fault'] = True
            r.error = 'holding_object'
            r.detail['recovery'] = 'put back failed: %s %s' % (e.stage, e.error)
            return
        r.detail['holding'] = False
        try:
            self._move(r, 'putback_retreat', g.T_pre, self.linear_moves)
            r.detail['recovery'] = 'put back'
        except _StepFailed as e:
            r.detail['fault'] = True
            r.detail['recovery'] = 'put back, retreat failed: %s' % e.error

    def _dry_run(self, g: GraspPlan, p: PlacePlan, r: PickResult) -> None:
        """Visit every pose of the plan without closing the gripper."""
        from .reach import reach_adjust
        hover_g = np.array(g.T_grasp, dtype=float)
        hover_g[2, 3] = min(hover_g[2, 3] + self.hover, g.T_pre[2, 3])
        hover_r = np.array(p.T_release, dtype=float)
        hover_r[2, 3] = min(hover_r[2, 3] + self.hover, p.T_pre[2, 3])
        # Raised poses can exceed joint5's vertical reach: lean them like planned poses.
        for name, T in (('hover_grasp', hover_g), ('hover_release', hover_r)):
            T2, _tilt, why = reach_adjust(self.arm, T)
            if T2 is None:
                r.stage, r.error = name, 'unreachable'
                r.detail['reason'] = why
                return
            T[:, :] = T2
        try:
            self._open(r, 'open', g.open_width, need=None)
            self._move(r, 'pre', g.T_pre, False)
            try:
                self._move(r, 'hover_grasp', hover_g, self.linear_descent)
            except _StepFailed as e:
                if not e.fault:
                    self._retreat_empty(r, g, e.error)
                raise
            self._move(r, 'up', g.T_pre, self.linear_moves)
            self._move(r, 'transit', p.T_pre, False)
            self._move(r, 'hover_release', hover_r, self.linear_moves)
            self._move(r, 'retreat', p.T_pre, self.linear_moves)
        except _StepFailed as e:
            r.stage, r.error = e.stage, e.error
            r.detail['fault'] = r.detail['fault'] or e.fault
            return
        r.ok = False
        r.stage = 'dry_run_done'
        r.detail['hover_grasp'] = _xyz(hover_g)
        r.detail['hover_release'] = _xyz(hover_r)
