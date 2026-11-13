"""
Watch the pick area with the wrist camera while a pick moves (runner.live_watch).

The arm path is not changed: a background thread grabs frames during the moves
after the gripper closed (LIVE_STAGES) and keeps the detections of frames the
arm hardly moved around (flange pose read before the frame's exposure and after
its receipt differ by at most max_motion_m / max_rot_deg), so the camera pose
belongs to the image. Detections near the held object are dropped.

apply_live then updates the multi-view scene: a live detection whose object
moved more than move_m, or that matches no known object, replaces / adds it
(scene key LIVE = number of views); one that agrees with a known object only
confirms it (no re-observation needed for it).
"""
from __future__ import annotations

import logging
import math
import threading
from typing import Dict, List, Optional, Sequence, Set

import numpy as np

from ..core.types import Detection

_LOG = logging.getLogger('cube_task.task.live')

#: skill stages during which frames are used (the object is held or released).
LIVE_STAGES = ('lift', 'transit', 'release', 'retreat')
#: of those, the stages where the held object is under the camera.
HELD_STAGES = ('lift', 'transit', 'release')


def _xy(d: Detection):
    c = d.center_base
    if c is None:
        return None
    x, y = float(c[0]), float(c[1])
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


def _rot_deg(A: np.ndarray, B: np.ndarray) -> float:
    R = A[:3, :3].T @ B[:3, :3]
    c = max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))
    return math.degrees(math.acos(c))


class LiveSettings:
    def __init__(self, cfg):
        k = 'runner.live_watch.'
        self.enabled = bool(cfg.get(k + 'enabled', False))
        self.max_motion = float(cfg.get(k + 'max_motion_m', 0.01))
        self.max_rot = float(cfg.get(k + 'max_rot_deg', 2.0))
        self.latency = float(cfg.get(k + 'latency_s', 0.1))
        self.move = float(cfg.get(k + 'move_m', 0.03))
        self.match = float(cfg.get(k + 'match_m', 0.15))
        self.held_margin = float(cfg.get(k + 'held_margin_m', 0.02))
        self.join_s = float(cfg.get(k + 'join_s', 3.0))


class LiveWatcher:
    """
    Background capture during one pick; see the module docstring.

    stage(name) is the skill's on_stage hook (called before every move);
    finish() stops the thread (waits for the frame in progress, so perception
    is never used by two threads) and returns the detections, newest per object.
    held_radius: m around the TCP xy where detections are the held object.
    """

    def __init__(self, s: LiveSettings, arm, camera, perception, ext, T_flange_tcp,
                 zones, clock, held_radius: float, merge_dist: float):
        self.s = s
        self.arm, self.camera, self.perception = arm, camera, perception
        self.ext, self.T_ft, self.zones, self.clock = ext, T_flange_tcp, zones, clock
        self.held_radius = float(held_radius)
        self.merge_dist = float(merge_dist)
        self._stage = ''
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames: List[List[Detection]] = []
        self.stats = {'frames': 0, 'used': 0, 'moving': 0, 'errors': 0}

    def stage(self, name: str) -> None:
        self._stage = str(name)
        if self._thread is None and self._stage in LIVE_STAGES:
            self._thread = threading.Thread(target=self._run, name='live_watch', daemon=True)
            self._thread.start()

    def finish(self) -> List[Detection]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.s.join_s)
            if self._thread.is_alive():
                _LOG.warning('live watch: frame still in progress, waiting for it')
                self._thread.join()
        out: List[Detection] = []
        for dets in reversed(self._frames):            # newest frame first
            for d in dets:
                p = _xy(d)
                if p is not None and all(math.hypot(p[0] - q[0], p[1] - q[1]) > self.merge_dist
                                         for q in map(_xy, out)):
                    out.append(d)
        return out

    def _pose(self):
        s = self.arm.get_flange_pose()
        return None if s is None or s.value is None else np.asarray(s.value, dtype=float)

    def _run(self) -> None:
        from ..core.extrinsics import camera_pose_in_base
        while not self._stop.is_set():
            try:
                st1 = self._stage
                if st1 not in LIVE_STAGES:
                    self._stop.wait(0.05)
                    continue
                P1, t1 = self._pose(), self.clock.now()
                frame = self.camera.get_frame(timeout=0.5, newer_than=t1 + self.s.latency)
                P2, st2 = self._pose(), self._stage
                if frame is None or P1 is None or P2 is None or st2 not in LIVE_STAGES:
                    continue
                self.stats['frames'] += 1
                if (float(np.linalg.norm(P2[:3, 3] - P1[:3, 3])) > self.s.max_motion
                        or _rot_deg(P1, P2) > self.s.max_rot):
                    self.stats['moving'] += 1        # camera pose unknown at exposure
                    continue
                T_bc = camera_pose_in_base(self.ext, P2, self.T_ft)
                res = self.perception.process(frame, T_bc, self.zones)
                dets = res.valid_detections()
                if st1 in HELD_STAGES or st2 in HELD_STAGES:
                    tcp = (P2 @ self.T_ft)[:2, 3]
                    dets = [d for d in dets if _xy(d) is not None and math.hypot(
                        _xy(d)[0] - float(tcp[0]), _xy(d)[1] - float(tcp[1])) > self.held_radius]
                for d in dets:
                    d.extra = dict(d.extra or {})
                    d.extra['live_stamp'] = float(frame.stamp)
                self._frames.append(dets)
                self.stats['used'] += 1
            except Exception:  # noqa: B902 - live watching never stops the pick
                self.stats['errors'] += 1
                if self.stats['errors'] == 1:
                    _LOG.exception('live watch frame failed')
                self._stop.wait(0.1)


def apply_live(scene: Dict[int, List[Detection]], live: Sequence[Detection], live_key: int,
               mark: int, move: float, match: float, merge_dist: float) -> Dict[str, list]:
    """
    Update scene (view -> detections) with live detections; return what changed.

    For each live detection, the known object it is: the nearest of the same
    label within match (m), else of any label within merge_dist. None ->
    added ('new'); farther than move (m) -> the known one is replaced
    ('moved'); else the known one is kept and confirmed ('same'). Added and
    confirmed detections get extra['live_ok'] = mark.
    """
    out: Dict[str, list] = {'new': [], 'moved': [], 'same': []}
    scene.setdefault(live_key, [])
    live = [d for d in live if _xy(d) is not None]
    known = [(v, k) for v, dets in scene.items() for k in dets if _xy(k) is not None]
    pairs = []
    for i, d in enumerate(live):
        p = _xy(d)
        for j, (_v, k) in enumerate(known):
            q = _xy(k)
            dist = math.hypot(p[0] - q[0], p[1] - q[1])
            same = k.label == d.label
            if dist <= (match if same else merge_dist):
                pairs.append(((0 if same else 1, dist), i, j))
    pairs.sort()
    got: Dict[int, int] = {}
    used: Set[int] = set()
    for _key, i, j in pairs:                   # one-to-one, nearest same-label pairs first
        if i not in got and j not in used:
            got[i] = j
            used.add(j)
    for i, d in enumerate(live):
        p = _xy(d)
        d.extra = dict(d.extra or {})
        d.extra['view'] = int(live_key)
        row = [d.label, round(p[0], 4), round(p[1], 4)]
        j = got.get(i)
        if j is None:
            d.extra['live_ok'] = mark
            scene[live_key].append(d)
            out['new'].append(row)
            continue
        v, k = known[j]
        q = _xy(k)
        dist = math.hypot(p[0] - q[0], p[1] - q[1])
        if dist > move:
            scene[v] = [o for o in scene[v] if o is not k]
            d.extra['live_ok'] = mark
            scene[live_key].append(d)
            out['moved'].append(row + [round(dist, 4)])
        else:
            k.extra = dict(k.extra or {})
            k.extra['live_ok'] = mark
            out['same'].append(row)
    return out


def drop_unconfirmed_live(scene: Dict[int, List[Detection]], live_key: int, mark: int,
                          xy: Sequence[float], radius: float) -> int:
    """Drop live detections near xy not confirmed by pick `mark` (they may have been touched)."""
    dets = scene.get(live_key, [])
    keep = [d for d in dets if (d.extra or {}).get('live_ok') == mark or _xy(d) is None
            or math.hypot(_xy(d)[0] - float(xy[0]), _xy(d)[1] - float(xy[1])) > radius]
    scene[live_key] = keep
    return len(dets) - len(keep)


def drop_live_seen(scene: Dict[int, List[Detection]], live_key: int, view: int,
                   dist: float) -> None:
    """Drop the live detections of the objects a fresh scan of view has seen again."""
    fresh = [_xy(d) for d in scene.get(view, []) if _xy(d) is not None]
    scene[live_key] = [d for d in scene.get(live_key, []) if _xy(d) is None or all(
        math.hypot(_xy(d)[0] - q[0], _xy(d)[1] - q[1]) > dist for q in fresh)]
