"""
Top-down grasp and place planning.

All poses are TCP poses (fingertip center) in the robot base frame, built with
``core.transforms.top_down_T`` (tool z = base -Z, tool y = finger closing axis at
``closing_angle`` from base +X). Units: meters and radians.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..core.transforms import (choose_closing_angle, neutral_closing_angle, top_down_T,
                               wrap_pi)
from ..core.types import Detection, GraspPlan, PlacePlan, Zone

_LOG = logging.getLogger('cube_task.perception.grasp')

#: length / width below this ratio counts as round (closing direction is free).
ROUND_RATIO = 1.06  # below this the minor axis is too uncertain to aim at
#: length / width below this ratio counts as a square block (L1, 90 deg symmetric).
SQUARE_RATIO = 1.15
#: TCP stays this far above table z + robot.workspace.min_table_clearance.
TABLE_MARGIN = 0.001
#: transit TCP height above the highest zone wall top.
WALL_TOP_CLEARANCE = 0.08
#: the bottom of a held object passes this far above every obstacle in transit.
HELD_CLEARANCE = 0.05
#: release over a walled bin with the object bottom this far above the rim.
WALL_RELEASE_CLEARANCE = 0.02
#: finger band for neighbor checks (m): half width across the closing axis (as
#: task.selection.FINGER_HALF_BAND), reach beyond the open finger face, and the
#: gap beyond the object faces that is ignored (noise of touching footprints).
FINGER_HALF_BAND = 0.015
FINGER_OUTER = 0.015
NEIGHBOR_CLEARANCE = 0.002
#: perceived table z farther than this from perception.table.expected_z is reported (m).
TABLE_Z_WARN_TOL = 0.01
#: a released object's bottom passes this far above the tallest object already in the zone.
STACK_CLEARANCE = 0.01
#: near_center releases (place.center_labels): grid step (m) searched outward from the
#: zone center, and the clearance (m) kept to the objects already there.
NEAR_CENTER_STEP = 0.0025
NEAR_CENTER_GAP = 0.008
#: release slots keep the held object's footprint this far inside the zone edge (m); the
#: rules count an object only when it is wholly inside the card (cucumber: 2/3).
SLOT_MARGIN = 0.025
#: a zone axis gets two release slots only when they are at least this far off center (m).
SLOT_MIN_OFFSET = 0.01
#: cross-section profile (det.extra['_points_base']): height bin (m), points a bin needs
#: to be measured, points an instance needs for the profile (else the footprint rule),
#: robust extent percentiles, top percentile of the profile.
PROFILE_BIN = 0.005
PROFILE_MIN_BIN_POINTS = 4
PROFILE_MIN_POINTS = 30
PROFILE_PCT = (2.0, 98.0)
PROFILE_TOP_PCT = 99.5
#: grasp heights are searched on this grid (m).
GRASP_Z_STEP = 0.0025
#: default finger pad half height around the TCP (m), grasp.pad_half.
PAD_HALF = 0.015
#: an object longer than this fraction of a zone's smaller side is released along the
#: zone diagonal.
LONG_ZONE_RATIO = 0.9
#: length / width at or above this: the released object is a capsule along its long axis.
CAPSULE_RATIO = 1.3
#: profile points farther than this from the grasp point across the closing axis are
#: ignored (the finger pads only pass the local section; a banana's bow is not in it), m.
SECTION_HALF_BAND = 0.02
#: only objects at least this long / wide are centered on their local section (a near-round
#: one seen obliquely would be shifted toward the camera).
SHIFT_RATIO = 2.0
#: ... using the points above this fraction of the object height (its crest).
SHIFT_CREST = 0.85
#: ... and only when the crest is at least this far (m) off the cap center: a curved banana's
#: bow (~10-15 mm), not the few mm an oblique or close wrist camera biases the crest.
SHIFT_MIN = 0.008
#: default grasp.finger_clearance (m): each finger passes the object this far off.
FINGER_CLEARANCE = 0.005
#: a grasp above the widest section within reach (neck) needs the section to taper by
#: at most this half angle over the lower half of the finger pads (deg, friction ~0.36);
#: steeper is a dome that slips.
NECK_MAX_TAPER_DEG = 20.0
#: default grasp.min_grasp_height (m): profile grasps keep the TCP this far above the table.
MIN_GRASP_HEIGHT = 0.008
#: default place.release_height_flat (m): object bottom above a flat zone (wall_height 0).
RELEASE_HEIGHT_FLAT = 0.015
#: spread releases: grid step (m) of the fallback search, smallest edge margin it accepts.
SLOT_GRID_STEP = 0.005
SLOT_MIN_EDGE = 0.0
#: a release xy found unreachable bans the release positions this close (m) to it.
AVOID_RADIUS = 0.012
#: a release keeps a free zone diagonal this far (m) from a long object still to come.
RESERVE_CLEARANCE = 0.005
#: at a shaped release the fingers open to the grip width + this (m).
RELEASE_OPEN_EXTRA = 0.02
#: ... or only this much (m) when the wider opening would touch an object in the zone.
RELEASE_OPEN_MIN_EXTRA = 0.012


def _required_float(cfg, key: str) -> float:
    """Read a safety-relevant number; raise ValueError when missing or not finite."""
    v = cfg.get(key, None)
    if v is None:
        raise ValueError('config key %s is required' % key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError('config key %s must be a number, got %r' % (key, v))
    if not math.isfinite(f):
        raise ValueError('config key %s must be finite, got %r' % (key, v))
    return f


def _float(cfg, key: str, default: float) -> float:
    """Read a tuning number with a default equal to the one in default.yaml."""
    v = cfg.get(key, default)
    if v is None:
        v = default
    f = float(v)
    if not math.isfinite(f):
        raise ValueError('config key %s must be finite, got %r' % (key, v))
    return f


def _finite(v) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def max_gripper_width(cfg) -> float:
    """Return robot.gripper.max_width (m); raise ValueError if missing or <= 0."""
    w = _required_float(cfg, 'robot.gripper.max_width')
    if w <= 0.0:
        raise ValueError('robot.gripper.max_width must be > 0, got %r' % w)
    return w


def expected_table_z(cfg) -> Optional[float]:
    """
    Return perception.table.expected_z (m, base z of the table taught by TCP touches).

    None when it is not set; ValueError when it is set but not a finite number.
    """
    v = cfg.get('perception.table.expected_z', None)
    if v is None:
        return None
    try:
        z = float(v)
    except (TypeError, ValueError):
        raise ValueError('perception.table.expected_z must be a number or null, got %r' % (v,))
    if not math.isfinite(z):
        raise ValueError('perception.table.expected_z must be finite, got %r' % (v,))
    return z


def table_floor_z(perceived_z: float, expected_z: Optional[float]) -> float:
    """
    Return the table z to keep the fingertips above (m, base frame).

    The taught table (expected_z) is a lower bound: a camera that sees the table
    too low (extrinsics z offset) must not lower the floor below the real table.
    """
    if expected_z is None:
        return float(perceived_z)
    return max(float(perceived_z), float(expected_z))


def table_z_warning(perceived_z: Optional[float], expected_z: Optional[float],
                    tol: float = TABLE_Z_WARN_TOL) -> str:
    """Return a warning when the perceived table z is off the taught one by > tol, else ''."""
    if perceived_z is None or expected_z is None or not _finite(perceived_z):
        return ''
    d = float(perceived_z) - float(expected_z)
    if abs(d) <= tol:
        return ''
    return ('table z differs from the taught table (perception.table.expected_z) by %+.0f mm: '
            'extrinsics z offset?' % (d * 1e3))


def detection_table_z(det: Detection) -> Optional[float]:
    """
    Table z (base frame, m) under a detection, or None when unknown.

    Uses ``det.extra['table_z']``; falls back to ``top_z - height``.
    """
    tz = (det.extra or {}).get('table_z', None)
    if _finite(tz):
        return float(tz)
    top = det.top_z
    if top is None and det.center_base is not None and len(det.center_base) >= 3:
        top = det.center_base[2]
    if _finite(top) and _finite(det.height):
        return float(top) - float(det.height)
    return None


def transit_height(cfg, zones: Iterable[Zone], table_z: float, held_offset: float = 0.0,
                   obstacles_top_z: Optional[float] = None) -> float:
    """
    Return the TCP z (base frame, m) for moves between the pick area and a zone.

    max of: grasp.transit_z_min; table_z + grasp.transit_clearance; highest zone
    wall top + 0.08; highest obstacle (table, wall tops, obstacles_top_z) +
    held_offset + 0.05, so the bottom of the held object (held_offset below the
    TCP) clears everything by 5 cm.
    """
    if not _finite(table_z):
        raise ValueError('transit_height: table_z must be finite, got %r' % (table_z,))
    if not _finite(held_offset) or held_offset < 0.0:
        raise ValueError('transit_height: held_offset must be >= 0, got %r' % (held_offset,))
    z = max(_float(cfg, 'grasp.transit_z_min', 0.15),
            float(table_z) + _float(cfg, 'grasp.transit_clearance', 0.12))
    highest = float(table_z)
    for zone in zones:
        top = float(zone.center[2]) + max(0.0, float(zone.wall_height))
        highest = max(highest, top)
        if zone.wall_height > 0.0:
            z = max(z, top + WALL_TOP_CLEARANCE)
    if _finite(obstacles_top_z):
        highest = max(highest, float(obstacles_top_z))
    return max(z, highest + float(held_offset) + HELD_CLEARANCE)


def section_profile(points: np.ndarray, center_xy: Sequence[float], closing: float,
                    table_z: float, half_band: float = SECTION_HALF_BAND,
                    bin_m: float = PROFILE_BIN) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Return (z_edges, widths) of an object's cross-section along the closing axis.

    points: Nx3 base-frame points of the instance (det.extra['_points_base']);
    center_xy: grasp point (m); closing: closing axis angle (rad); table_z: table
    z under the object. Only points within half_band of the grasp point across
    the closing axis count. Bin i spans z_edges[i]..z_edges[i + 1] (bin_m high,
    from table_z up to the PROFILE_TOP_PCT percentile of the points); its width
    is the 2-98 percentile extent along the closing axis, completed by symmetry
    about the grasp point (the far side is hidden from an oblique camera). A bin with
    too few points lies under the object (hidden from the camera) and gets the
    widest measured width above it. None when there are too few points.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.shape[0] < PROFILE_MIN_POINTS or not (_finite(table_z) and bin_m > 0.0):
        return None
    c, sn = math.cos(closing), math.sin(closing)
    rel = P[:, :2] - np.asarray(center_xy, dtype=np.float64)[:2]
    along = rel @ np.array([c, sn])
    keep = np.abs(rel @ np.array([-sn, c])) <= half_band
    along, h = along[keep], P[keep, 2] - float(table_z)
    if along.size < PROFILE_MIN_POINTS:
        return None
    top = float(np.percentile(h, PROFILE_TOP_PCT))
    if top <= 0.0:
        return None
    n = max(1, int(math.ceil(top / bin_m)))
    idx = np.clip(np.floor(h / bin_m).astype(int), 0, n - 1)
    widths = np.full(n, np.nan)
    for i in range(n):
        sel = along[idx == i]
        if sel.size >= PROFILE_MIN_BIN_POINTS:
            lo, hi = np.percentile(sel, PROFILE_PCT)
            widths[i] = max(hi - lo, 2.0 * max(hi, -lo), 0.0)
    run = 0.0
    for i in range(n - 1, -1, -1):
        if not np.isfinite(widths[i]):
            widths[i] = run
        run = max(run, widths[i])
    edges = float(table_z) + bin_m * np.arange(n + 1)
    return edges, widths


def section_width(profile: Tuple[np.ndarray, np.ndarray], z_lo: float, z_hi: float) -> float:
    """Return the widest section (m) of profile between base z z_lo and z_hi (0 if none)."""
    edges, widths = profile
    sel = (edges[1:] > z_lo + 1e-9) & (edges[:-1] < z_hi - 1e-9)
    if z_hi - z_lo <= 1e-9:
        sel = (edges[1:] >= z_lo) & (edges[:-1] <= z_lo)
    return float(widths[sel].max()) if np.any(sel) else 0.0


def section_center_shift(points: np.ndarray, center_xy: Sequence[float], closing: float,
                         z_min: float, half_band: float = SECTION_HALF_BAND) -> Optional[float]:
    """
    Return the offset (m) of the local section center along the closing axis.

    Uses the points above z_min within half_band across the closing axis; the
    middle of their 2-98 percentile extent. None with too few points.
    """
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1) & (P[:, 2] >= z_min)] if P.size else P
    if P.shape[0] < PROFILE_MIN_POINTS:
        return None
    c, sn = math.cos(closing), math.sin(closing)
    rel = P[:, :2] - np.asarray(center_xy, dtype=np.float64)[:2]
    keep = np.abs(rel @ np.array([-sn, c])) <= half_band
    if np.count_nonzero(keep) < PROFILE_MIN_POINTS:
        return None
    lo, hi = np.percentile(rel[keep] @ np.array([c, sn]), PROFILE_PCT)
    return float(lo + hi) / 2.0


def _identity_plan(det: Detection, reason: str, close_width: float) -> GraspPlan:
    eye = np.eye(4)
    return GraspPlan(target=det, T_pre=eye.copy(), T_grasp=eye.copy(), T_lift=eye.copy(),
                     open_width=0.0, close_width=close_width, expected_width=0.0,
                     closing_angle=0.0, feasible=False, reason=reason)


class GraspPlanner:
    """
    Plan a vertical top-down grasp for one detection.

    Closing axis across the object's minor axis (``yaw + pi/2``), turned by the
    object's symmetry (pi, or pi/2 for square blocks: every detection in L1 and
    detections labeled with a tasks.L1 label in other tasks, length / width <
    1.15) to the equivalent angle closest to the neutral wrist angle. Other
    near-round objects (length / width < 1.2) use the neutral angle directly.
    Blocks with touching neighbors (det.extra['_neighbor_xy'], set by the
    perception pipeline) close along the other face pair when that keeps the
    fingers off the neighbors (finger_hits); the expected width is then the
    extent along that axis.
    """

    def __init__(self, cfg, task: str):
        self.task = str(task)
        self.max_width = max_gripper_width(cfg)
        mow = cfg.get('grasp.max_object_width', None)
        self.max_object_width = (self.max_width - 0.005 if mow is None else float(mow))
        if not (0.0 < self.max_object_width <= self.max_width):
            raise ValueError('grasp.max_object_width must be in (0, max_width], got %r'
                             % self.max_object_width)
        self.open_margin = _float(cfg, 'robot.gripper.open_margin', 0.02)
        # Open to max_width before every grasp of these tasks (a turned block fits).
        self.open_full = self.task in [str(t) for t in
                                       (cfg.get('grasp.open_full_tasks', None) or [])]
        #: ... and before every grasp of these labels (grasp.open_full_labels)
        self.open_full_labels = set(str(x) for x in
                                    (cfg.get('grasp.open_full_labels', None) or []))
        self.close_width = _float(cfg, 'robot.gripper.close_width', 0.0)
        self.table_clearance = _required_float(cfg, 'robot.workspace.min_table_clearance')
        # The taught table height bounds the grasp floor from below (extrinsics z offset).
        self.expected_table_z = expected_table_z(cfg)
        self.approach_height = _float(cfg, 'grasp.approach_height', 0.08)
        self.lift_height = _float(cfg, 'grasp.lift_height', 0.10)
        self.depth_ratio = _float(cfg, 'grasp.grasp_depth_ratio', 0.5)
        self.min_depth = _float(cfg, 'grasp.min_grasp_depth', 0.008)
        self.max_depth = _float(cfg, 'grasp.max_grasp_depth', 0.030)
        if self.min_depth > self.max_depth:
            raise ValueError('grasp.min_grasp_depth > grasp.max_grasp_depth')
        self.square_blocks = bool(cfg.get('grasp.block_labels_square', True))
        self.use_cap_center = bool(cfg.get('grasp.use_cap_center', True))
        self.block_labels = set(str(x) for x in (cfg.get('tasks.L1.labels', []) or []))
        self.use_profile = bool(cfg.get('grasp.section_profile', True))
        self.finger_clearance = _float(cfg, 'grasp.finger_clearance', FINGER_CLEARANCE)
        self.pad_half = _float(cfg, 'grasp.pad_half', PAD_HALF)
        self.min_grasp_height = _float(cfg, 'grasp.min_grasp_height', MIN_GRASP_HEIGHT)
        if self.finger_clearance < 0.0 or self.pad_half < 0.0:
            raise ValueError('grasp.finger_clearance and grasp.pad_half must be >= 0')
        #: label -> widest measured section (m) still grasped (grasp.fit_width): the depth
        #: points of a round fruit bulge a few mm past its real outline.
        self.fit_width: Dict[str, float] = {}
        for k, v in (cfg.get('grasp.fit_width', {}) or {}).items():
            if v is None:
                continue
            if not float(v) > 0.0:
                raise ValueError('grasp.fit_width.%s must be > 0, got %r' % (k, v))
            if float(v) > self.max_width:
                # set for the 100 mm gripper; a narrower one keeps its default rule
                _LOG.debug('grasp.fit_width.%s %.3f m > max_width %.3f m: ignored',
                           k, float(v), self.max_width)
                continue
            self.fit_width[str(k)] = float(v)

    def _opens_full(self, label: Optional[str]) -> bool:
        """True when the gripper opens to max_width before grasping label."""
        return self.open_full or str(label) in self.open_full_labels

    def _fit(self, label: Optional[str]) -> float:
        """Widest section (m) grasped for label: grasp.fit_width or max_width - 2 x clearance."""
        return self.fit_width.get(str(label), self.max_width - 2.0 * self.finger_clearance)

    def _max_object(self, label: Optional[str]) -> float:
        """Widest footprint (m) grasped for label: grasp.fit_width or max_object_width."""
        return self.fit_width.get(str(label), self.max_object_width)

    def plan(self, det: Detection) -> GraspPlan:
        """Return a GraspPlan; feasible=False with a reason when it cannot be done."""
        missing = [n for n in ('yaw', 'length', 'width', 'height') if not _finite(getattr(det, n))]
        c = det.center_base
        if c is None or len(c) < 2 or not (_finite(c[0]) and _finite(c[1])):
            missing.insert(0, 'center_base')
        top = det.top_z
        if not _finite(top) and c is not None and len(c) >= 3:
            top = c[2]
        if not _finite(top):
            missing.append('top_z')
        table_z = detection_table_z(det)
        if table_z is None:
            missing.append('table_z')
        if missing:
            return _identity_plan(det, 'missing_geometry: ' + ','.join(missing), self.close_width)
        x, y, top = float(c[0]), float(c[1]), float(top)
        length, width = float(det.length), float(det.width)
        height = float(det.height)
        if width <= 0.0 or length <= 0.0 or height <= 0.0:
            return _identity_plan(det, 'bad_geometry: length=%.4f width=%.4f height=%.4f'
                                  % (length, width, height), self.close_width)
        if length < width:
            length, width = width, length
        # Grasp at the center of the topmost points when perception provides it.
        # For round objects seen by an oblique camera the hidden far side biases the
        # footprint center toward the camera (about 0.15 r), while the top cap sits
        # straight above the true center.
        cap = det.extra.get('cap_center_xy') if det.extra else None
        if (self.use_cap_center and cap is not None and len(cap) >= 2
                and _finite(cap[0]) and _finite(cap[1])
                and math.hypot(float(cap[0]) - x, float(cap[1]) - y) <= 0.5 * width):
            x, y = float(cap[0]), float(cap[1])
        ratio = length / width
        neutral = neutral_closing_angle(x, y)
        is_block = self.task == 'L1' or det.label in self.block_labels
        if self.square_blocks and is_block and ratio < SQUARE_RATIO:
            # Square block: grip across two faces, any of the 4 face pairs works.
            closing = choose_closing_angle(float(det.yaw) + math.pi / 2.0, x, y, math.pi / 2.0)
            expected = width
            kind = 'square'
            closing, expected = self._avoid_neighbors(det, closing, x, y, expected, length)
        elif ratio < ROUND_RATIO:
            closing = neutral
            expected = length
            kind = 'round'
            closing = self._round_closing(det, closing, x, y, expected)
        else:
            closing = choose_closing_angle(float(det.yaw) + math.pi / 2.0, x, y, math.pi)
            # A partly hidden round cross-section (fruit lying on its side) makes the
            # visible width too small; the height is a second estimate of it.
            expected = width if is_block else max(width, min(length, height))
            kind = 'elongated'
            if is_block:
                # A block next to a taller one looks elongated (strip hidden behind it).
                closing, expected = self._avoid_neighbors(det, closing, x, y, expected, length)
        z_floor = (table_floor_z(table_z, self.expected_table_z) + self.table_clearance
                   + TABLE_MARGIN)
        pts = det.extra.get('_points_base') if (det.extra and self.use_profile) else None
        prof = None
        if pts is not None:
            if kind == 'elongated' and not is_block and ratio >= SHIFT_RATIO:
                # Center the fingers on the local section (a curved banana's footprint
                # center lies inside its bow). Only the crest counts: an oblique camera
                # sees both sides of it, but only the near side lower down.
                shift = section_center_shift(pts, (x, y), closing,
                                             table_z + SHIFT_CREST * height)
                if shift is not None and SHIFT_MIN <= abs(shift) <= 0.5 * width:
                    x += shift * math.cos(closing)
                    y += shift * math.sin(closing)
            prof = section_profile(pts, (x, y), closing, table_z)
        info = {'kind': kind, 'mode': 'footprint' if prof is None else 'profile'}
        reason = ''
        if prof is None:
            depth = min(max(self.depth_ratio * height, self.min_depth), self.max_depth)
            z_grasp = max(top - depth, z_floor)
            open_w = min(self.max_width, expected + self.open_margin)
            if expected > self._max_object(det.label):
                reason = 'too_wide: %.4f m > %.4f m' % (expected, self._max_object(det.label))
        else:
            z_grasp, expected, open_w, reason, more = self._profile_grasp(
                det, prof, top, z_floor)
            info.update(more)
        if self._opens_full(det.label):
            open_w = self.max_width
        if not reason and z_grasp >= top:
            reason = 'too_flat: top %.4f m is not above the lowest TCP z %.4f m' % (top, z_floor)
        info['z_rel'] = round(z_grasp - table_z, 4)
        det.extra['grasp'] = info
        plan = GraspPlan(
            target=det,
            T_pre=top_down_T(x, y, top + self.approach_height, closing),
            T_grasp=top_down_T(x, y, z_grasp, closing),
            T_lift=top_down_T(x, y, z_grasp + self.lift_height, closing),
            open_width=open_w,
            close_width=self.close_width,
            expected_width=expected,
            closing_angle=closing)
        if reason:
            plan.feasible = False
            plan.reason = reason
        _LOG.debug('grasp %s kind=%s mode=%s closing=%.3f z=%.4f feasible=%s %s', det.label,
                   kind, info['mode'], closing, z_grasp, plan.feasible, plan.reason)
        return plan

    def _profile_grasp(self, det: Detection, prof, top: float, z_floor: float):
        """
        Choose the grasp height from a section profile (see section_profile).

        Candidates: TCP z in [max(z_floor, table + grasp.min_grasp_height),
        top - min_grasp_depth], at most max_grasp_depth below the top, where
        every section the open fingers pass (z - pad_half up to the top) fits
        max_width - 2 * finger_clearance. The lowest candidate wins when it is
        at or below the widest section within finger reach (stable grip below
        the equator). Above it the shape must narrow gently under the lower
        half of the finger pads (a neck, at most NECK_MAX_TAPER_DEG half angle): the lowest such
        candidate wins; a dome (steeper) is too_wide. Returns (z, expected
        width, open width, reason, info).
        """
        edges, widths = prof
        top_pts = float(edges[-1])
        fit = self._fit(det.label)
        lo = max(z_floor, float(edges[0]) + self.min_grasp_height, top - self.max_depth)
        hi = top - self.min_depth
        n = int(math.floor((hi - lo) / GRASP_Z_STEP + 1e-9)) + 1 if hi >= lo else 1
        zs = [lo + k * GRASP_Z_STEP for k in range(max(1, n))]
        passed = [section_width(prof, z - self.pad_half, top_pts) for z in zs]
        widest = section_width(prof, float(edges[0]), top_pts)
        info = {'section_max': round(widest, 4), 'fit': round(fit, 4)}
        ok = [i for i, w in enumerate(passed) if w <= fit + 1e-9]
        label = det.label or 'object'
        if not ok:
            reason = ('too_wide: %s section %.3f m > %.3f m at every grasp height within the '
                      'finger reach (%.3f m below the top; widest %.3f m; gripper max_width '
                      '%.3f m)' % (label, min(passed), fit, self.max_depth, widest,
                                   self.max_width))
            return zs[0], min(passed), self.max_width, reason, info
        centers = (edges[:-1] + edges[1:]) / 2.0
        reach = (edges[1:] > lo + 1e-9) & (edges[:-1] < top_pts)
        z_wide = lo
        if np.any(reach):
            w_reach = widths[reach].max()
            z_wide = float(centers[reach][widths[reach] >= w_reach - 0.001].max())
        pick = ok[0]
        reason = ''
        if zs[pick] > z_wide + PROFILE_BIN / 2.0:
            pick = None
            taper = 90.0
            for i in ok:
                z = zs[i]
                # A neck is held where it is widest under the pads (their lower half).
                w_lo = section_width(prof, z - self.pad_half, z - self.pad_half)
                w_mid = section_width(prof, z, z)
                t = math.degrees(math.atan2(max(0.0, w_lo - w_mid) / 2.0,
                                            max(self.pad_half, 1e-6)))
                taper = min(taper, t)
                if t <= NECK_MAX_TAPER_DEG:
                    pick = i
                    break
            info['neck_taper_deg'] = round(taper, 1)
            if pick is None:
                pick = ok[0]
                reason = ('too_wide: %s widest section %.3f m > %.3f m (gripper max_width '
                          '%.3f m) and the narrower part above it is a dome (taper %.0f deg '
                          '> %.0f)' % (label, section_width(prof, z_wide, z_wide), fit,
                                       self.max_width, taper, NECK_MAX_TAPER_DEG))
            else:
                info['neck'] = True
        z = zs[pick]
        expected = section_width(prof, z - self.pad_half, z + self.pad_half)
        open_w = min(self.max_width, passed[pick] + 2.0 * self.finger_clearance
                     + self.open_margin)
        return z, expected, open_w, reason, info

    def finger_hits(self, det: Detection, closing: float, x: float, y: float,
                    expected: float) -> int:
        """
        Count neighbor footprint points in the band the open fingers sweep.

        Neighbor points: det.extra['_neighbor_xy'] (base XY, set by the
        perception pipeline for instances next to det). The band reaches
        NEIGHBOR_CLEARANCE beyond the object faces up to the outer finger
        side (open width / 2 + FINGER_OUTER) along the closing axis and
        FINGER_HALF_BAND across it, on both sides.
        """
        nb = det.extra.get('_neighbor_xy') if det.extra else None
        if nb is None:
            return 0
        nb = np.asarray(nb, dtype=np.float64).reshape(-1, 2)
        if nb.shape[0] == 0:
            return 0
        open_w = (self.max_width if self._opens_full(det.label)
                  else min(self.max_width, expected + self.open_margin))
        ax = np.array([math.cos(closing), math.sin(closing)])
        rel = nb - np.array([x, y])
        along = np.abs(rel @ ax)
        across = np.abs(rel @ np.array([-ax[1], ax[0]]))
        band = ((across <= FINGER_HALF_BAND) & (along >= expected / 2.0 + NEIGHBOR_CLEARANCE)
                & (along <= open_w / 2.0 + FINGER_OUTER))
        return int(np.count_nonzero(band))

    def _round_closing(self, det: Detection, neutral: float, x: float, y: float,
                       expected: float) -> float:
        """
        Return the closing angle of a round object that keeps the fingers off neighbors.

        Any angle grips a round object, so the neutral one is turned by 45 / 90 / 135 deg
        only when that sweeps fewer neighbor points (finger_hits); ties keep neutral.
        """
        best_hits = self.finger_hits(det, neutral, x, y, expected)
        best = neutral
        if best_hits == 0:
            return neutral
        for k in (2, 1, 3):
            alt = choose_closing_angle(neutral + k * math.pi / 4.0, x, y, math.pi)
            hits = self.finger_hits(det, alt, x, y, expected)
            if hits < best_hits:
                best, best_hits = alt, hits
                if hits == 0:
                    break
        return best

    def _avoid_neighbors(self, det: Detection, closing: float, x: float, y: float,
                         expected: float, alt_expected: float) -> Tuple[float, float]:
        """
        Turn a block's closing axis by 90 deg when that frees the fingers.

        expected / alt_expected: object extent along the current / the turned
        closing axis. Returns (closing angle, expected width): the turned axis
        (closest to the neutral wrist angle) when it has fewer neighbor points
        in the finger band (finger_hits) and alt_expected fits the gripper,
        else the inputs unchanged.
        """
        hits = self.finger_hits(det, closing, x, y, expected)
        if hits == 0 or alt_expected > self._max_object(det.label):
            return closing, expected
        alt = choose_closing_angle(closing + math.pi / 2.0, x, y, math.pi)
        alt_hits = self.finger_hits(det, alt, x, y, alt_expected)
        if alt_hits < hits:
            _LOG.debug('grasp %s: closing %.3f -> %.3f rad avoids a neighbor (%d -> %d points)',
                       det.label, closing, alt, hits, alt_hits)
            return alt, alt_expected
        return closing, expected


def held_offset_of(plan: GraspPlan) -> float:
    """Distance (m) from the grasp TCP down to the object bottom (table z)."""
    tz = detection_table_z(plan.target)
    if tz is None:
        raise ValueError('held_offset_of: table z of the target is unknown')
    return max(0.0, float(plan.T_grasp[2, 3]) - tz)


def release_slots(zone: Zone, footprint: float, margin: float = SLOT_MARGIN
                  ) -> List[Tuple[float, float]]:
    """
    Return release xy positions (base frame, m) inside zone for one object.

    footprint: the object's largest horizontal extent (m). Up to 2 x 2
    positions along the zone axes, as far apart as the zone allows while the
    footprint stays margin inside the zone edge; diagonal pairs first, and in
    each pair the position farther from the base first: the next release then
    comes down on the near side and the gripper never passes over the object
    already there. An axis too short for two positions SLOT_MIN_OFFSET or more
    off center keeps the zone center on that axis.
    """
    offs = []
    for size in zone.size[:2]:
        h = (float(size) - float(footprint)) / 2.0 - float(margin)
        offs.append(h if h >= SLOT_MIN_OFFSET else 0.0)
    hx, hy = offs
    if hx > 0.0 and hy > 0.0:
        loc = [(-hx, -hy), (hx, hy), (-hx, hy), (hx, -hy)]
    elif hx > 0.0:
        loc = [(-hx, 0.0), (hx, 0.0)]
    elif hy > 0.0:
        loc = [(0.0, -hy), (0.0, hy)]
    else:
        loc = [(0.0, 0.0)]
    c, s = math.cos(zone.yaw), math.sin(zone.yaw)
    cx, cy = float(zone.center[0]), float(zone.center[1])
    pts = [(cx + c * a - s * b, cy + s * a + c * b) for a, b in loc]
    out: List[Tuple[float, float]] = []
    for pair in (pts[k:k + 2] for k in range(0, len(pts), 2)):
        out.extend(sorted(pair, key=lambda q: -math.hypot(q[0], q[1])))
    return out


def _seg_dist(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Return the distance (m) between 2D segments p1-q1 and p2-q2."""
    def pt_seg(p, a, b):
        d = b - a
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-18 else min(1.0, max(0.0, float((p - a) @ d) / L2))
        return float(np.linalg.norm(p - (a + t * d)))

    d1, d2, r = q1 - p1, q2 - p2, p2 - p1
    den = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(den) > 1e-15:
        t = float(r[0] * d2[1] - r[1] * d2[0]) / den
        u = float(r[0] * d1[1] - r[1] * d1[0]) / den
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0
    return min(pt_seg(p1, p2, q2), pt_seg(q1, p2, q2), pt_seg(p2, p1, q1), pt_seg(q2, p1, q1))


def footprint_capsule(x: float, y: float, length: float, width: float,
                      axis: Optional[float]) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Return (end a, end b, radius) of an object's footprint capsule in base XY (m).

    length / width: footprint extents; axis: long-axis angle (rad) or None.
    length / width >= CAPSULE_RATIO with a known axis: a capsule of radius
    width / 2 along the axis; otherwise a circle of diameter length.
    """
    c = np.array([float(x), float(y)])
    L, W = max(float(length), float(width)), min(float(length), float(width))
    if axis is None or W <= 0.0 or L / W < CAPSULE_RATIO:
        return c, c.copy(), L / 2.0
    h = (L - W) / 2.0
    d = h * np.array([math.cos(axis), math.sin(axis)])
    return c - d, c + d, W / 2.0


def _occupied_capsule(o: Sequence[float]) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    if len(o) < 3 or not all(_finite(v) for v in o[:3]):
        return None
    if len(o) >= 7 and all(_finite(v) for v in o[4:6]) and o[6] is not None and _finite(o[6]):
        return footprint_capsule(o[0], o[1], o[4], o[5], float(o[6]))
    return footprint_capsule(o[0], o[1], o[2], o[2], None)


def capsule_edge_margin(zone: Zone, cap: Tuple[np.ndarray, np.ndarray, float]) -> float:
    """Return how far (m) a footprint capsule stays inside the zone edge (< 0: outside)."""
    c, s = math.cos(-zone.yaw), math.sin(-zone.yaw)
    hx, hy = float(zone.size[0]) / 2.0, float(zone.size[1]) / 2.0
    out = math.inf
    for p in cap[:2]:
        dx, dy = float(p[0]) - float(zone.center[0]), float(p[1]) - float(zone.center[1])
        lx, ly = c * dx - s * dy, s * dx + c * dy
        out = min(out, hx - abs(lx), hy - abs(ly))
    return out - cap[2]


class PlacePlanner:
    """
    Plan the release above a place zone.

    The object bottom is released ``zone.release_height`` (else
    ``place.release_height`` over a walled bin, ``place.release_height_flat``
    over a flat zone) above the zone floor, at least 2 cm above the wall rim
    of a walled bin and at least STACK_CLEARANCE above the tallest object
    already released into the zone. The TCP is ``held_offset`` above the
    object bottom. With the held object's footprint and the objects already in
    the zone (``occupied``), the release xy is the first free position of the
    zone (center first when ``center_first``, then release_slots), so later
    objects do not land on earlier ones (place.spread_releases, default true).

    With ``held`` (the held object's shape, see plan) the footprints are
    capsules: an object longer than LONG_ZONE_RATIO x the zone's smaller side
    lies along a zone axis on the side nearer the robot (long_strip; both ends
    stick out equally, the rules count a cucumber with 2/3 inside), other elongated
    objects keep the neutral wrist angle when they fit inside the zone, and
    the release xy is searched on a grid when no slot keeps the footprint and
    the open fingers clear of the objects already there. Over a flat zone a
    release whose footprint is clear of them drops from the empty-zone height
    (no stacking rule), and the fingers open only to the grip width +
    RELEASE_OPEN_EXTRA.
    """

    def __init__(self, cfg):
        self.max_width = max_gripper_width(cfg)
        self.release_height = _float(cfg, 'place.release_height', 0.06)
        self.release_height_flat = _float(cfg, 'place.release_height_flat',
                                          RELEASE_HEIGHT_FLAT)
        self.retreat_height = _float(cfg, 'place.retreat_height', 0.10)
        ow = cfg.get('place.open_width', None)
        self.open_width = self.max_width if ow is None else min(float(ow), self.max_width)
        self.spread = bool(cfg.get('place.spread_releases', True))
        self.slot_margin = _float(cfg, 'place.slot_margin', SLOT_MARGIN)
        #: zone name -> (toward the base, across) m off the zone center of its two
        #: zone-mate slots (far first, the near one mirrored); other zones use the 2 x 2
        #: corner slots of release_slots
        self.pair_offset: Dict[str, Tuple[float, float]] = {}
        for zn, po in (cfg.get('place.pair_offset', {}) or {}).items():
            if po is None:
                continue
            vals = [float(v) for v in (po if isinstance(po, (list, tuple)) else [po, 0.0])]
            if len(vals) != 2 or not all(v >= 0.0 for v in vals):
                raise ValueError('place.pair_offset.%s must be [radial, across] >= 0 m, got %r'
                                 % (zn, po))
            self.pair_offset[str(zn)] = (vals[0], vals[1])
        #: details of the last shaped release (plan with held): xy, axis, margins.
        self.last_info: dict = {}

    def release_xy(self, zone: Zone, footprint: Optional[float] = None,
                   occupied: Sequence[Sequence[float]] = (), center_first: bool = True
                   ) -> Tuple[float, float]:
        """
        Return the release xy (base frame, m) for an object of this footprint.

        occupied: (x, y, footprint, height) of the objects already released into
        zone. The first candidate (zone center when center_first, then
        release_slots) whose footprint does not overlap an occupied one wins;
        when all overlap, the one with the most clearance. Without a footprint
        (or with place.spread_releases false) the zone center.
        """
        center = (float(zone.center[0]), float(zone.center[1]))
        if not self.spread or not _finite(footprint) or float(footprint) <= 0.0:
            return center
        fp = float(footprint)
        cands = ([center] if center_first else []) + self._slots(zone, fp)
        occ = [o for o in occupied if len(o) >= 3 and all(_finite(v) for v in o[:3])]
        if not occ:
            return cands[0]
        best, best_clear = cands[0], -math.inf
        for x, y in cands:
            clear = min(math.hypot(x - float(o[0]), y - float(o[1]))
                        - (fp + float(o[2])) / 2.0 for o in occ)
            if clear >= 0.0:
                return x, y
            if clear > best_clear:
                best, best_clear = (x, y), clear
        return best

    def _near_center(self, zone: Zone, center: Tuple[float, float], evaluate, skip,
                     reserve_ok=lambda x, y, ax: True):
        """Grid release nearest the zone center with the edge margin and NEAR_CENTER_GAP."""
        ca, sa = math.cos(zone.yaw), math.sin(zone.yaw)
        nx = int(math.floor(float(zone.size[0]) / 2.0 / NEAR_CENTER_STEP))
        ny = int(math.floor(float(zone.size[1]) / 2.0 / NEAR_CENTER_STEP))
        cells = sorted(((i * NEAR_CENTER_STEP, j * NEAR_CENTER_STEP)
                        for i in range(-nx, nx + 1) for j in range(-ny, ny + 1)),
                       key=lambda q: (math.hypot(q[0], q[1]), -q[0], -q[1]))
        rows = []
        for a, b in cells:
            x, y = center[0] + ca * a - sa * b, center[1] + sa * a + ca * b
            if not skip(x, y):
                rows.extend((x, y) + r[1:] for r in evaluate(x, y))
        for gap in (NEAR_CENTER_GAP, 0.0):     # a clear gap first, else just not touching
            for x, y, ax, closing, e, cl, more in rows:
                if e >= self.slot_margin - 1e-9 and cl >= gap and reserve_ok(x, y, ax):
                    return x, y, ax, closing, e, cl, more
        return None

    def _slots(self, zone: Zone, footprint: float) -> List[Tuple[float, float]]:
        """
        Release slots for an object of this footprint (m): with place.pair_offset for
        this zone (a, b) two mirrored slots a along the zone axis toward the base and b across
        it off the center (each at most as far as the card allows), the far one
        first, so zone-mates sit near the middle of the card; else release_slots.
        """
        off = self.pair_offset.get(zone.name)
        if off is None:
            return release_slots(zone, footprint, self.slot_margin)
        ax = strip_axis(zone)
        k = 0 if abs(wrap_pi(2.0 * (ax - float(zone.yaw)))) < 1e-6 else 1
        lim = [(float(zone.size[i]) - float(footprint)) / 2.0 - self.slot_margin
               for i in (k, 1 - k)]
        a, b = [max(0.0, min(o, m)) for o, m in zip(off, lim)]
        cx, cy = float(zone.center[0]), float(zone.center[1])
        if math.hypot(a, b) < SLOT_MIN_OFFSET:
            return [(cx, cy)]
        ux, uy = math.cos(ax), math.sin(ax)
        dx, dy = a * ux - b * uy, a * uy + b * ux
        pts = [(cx + dx, cy + dy), (cx - dx, cy - dy)]
        return sorted(pts, key=lambda q: -math.hypot(q[0], q[1]))

    def _axes(self, zone: Zone, x: float, y: float, length: float, width: float,
              elongated: bool) -> List[Optional[float]]:
        """Long-axis angles (rad) to try at (x, y), preferred first; [None] = round."""
        if not elongated or width <= 0.0 or length / width < CAPSULE_RATIO:
            return [None]
        diag = math.atan2(float(zone.size[1]), float(zone.size[0]))
        diagonals = [zone.yaw + diag, zone.yaw - diag]
        if length > LONG_ZONE_RATIO * min(float(zone.size[0]), float(zone.size[1])):
            return [strip_axis(zone)]
        return [neutral_closing_angle(x, y) - math.pi / 2.0, zone.yaw,
                zone.yaw + math.pi / 2.0] + diagonals

    def long_strip(self, zone: Zone, width: float) -> Tuple[float, float, float]:
        """
        Return (x, y, long-axis angle) of an object longer than the zone (long_strip).

        The axis is the zone axis nearer the base-to-zone direction (the wrist keeps
        its neutral angle); the object lies place.slot_margin inside the zone edge
        on the side nearer the robot, so the other side is left to its zone-mates.
        """
        ax = strip_axis(zone)
        cx, cy = float(zone.center[0]), float(zone.center[1])
        k = 0 if abs(wrap_pi(2.0 * (ax - zone.yaw))) < 1e-9 else 1
        across = float(zone.size[1 - k])
        u = np.array([-math.sin(ax), math.cos(ax)])
        off = max(0.0, across / 2.0 - float(width) / 2.0 - self.slot_margin)
        if float(u @ np.array([cx, cy])) > 0.0:
            off = -off
        return cx + off * float(u[0]), cy + off * float(u[1]), ax

    def _clearance(self, cap, closing: float, finger_half: float, finger_floor: float,
                   occ: Sequence[Sequence[float]]) -> Tuple[float, float]:
        """Return the smallest gaps (m) of the held footprint / the open fingers to occupied."""
        clear = math.inf
        fingers = math.inf
        c = (cap[0] + cap[1]) / 2.0
        d = finger_half * np.array([math.cos(closing), math.sin(closing)])
        for o in occ:
            oc = _occupied_capsule(o)
            if oc is None:
                continue
            clear = min(clear, _seg_dist(cap[0], cap[1], oc[0], oc[1]) - cap[2] - oc[2])
            h = float(o[3]) if len(o) >= 4 and _finite(o[3]) else math.inf
            if finger_half > 0.0 and h > finger_floor:
                fingers = min(fingers, _seg_dist(c - d, c + d, oc[0], oc[1]) - FINGER_HALF_BAND
                              - oc[2])
        return clear, fingers

    def _reserved(self, zone: Zone, pending: Sequence[Sequence[float]]) -> list:
        """Return the long_strip capsule of the longest pending long object of the zone."""
        side = min(float(zone.size[0]), float(zone.size[1]))
        longs = [(max(float(a), float(b)), min(float(a), float(b))) for a, b in
                 (tuple(p)[:2] for p in pending if len(p) >= 2 and _finite(p[0]) and _finite(p[1]))
                 if max(float(a), float(b)) > LONG_ZONE_RATIO * side]
        if not longs:
            return []
        L, W = max(longs)
        x, y, ax = self.long_strip(zone, W)
        return [footprint_capsule(x, y, L, W, ax)]

    def release_pose(self, zone: Zone, held: dict, occupied: Sequence[Sequence[float]] = (),
                     center_first: bool = True, pending: Sequence[Sequence[float]] = (),
                     center_only: bool = False, avoid: Sequence[Sequence[float]] = (),
                     lower: float = 0.0, near_center: bool = False
                     ) -> Tuple[float, float, float, dict]:
        """
        Return (x, y, closing angle, info) of a shaped release (see plan, held).

        Tiers: the first slot candidate (as release_xy) that keeps the footprint
        place.slot_margin inside the zone and clear of the objects and finger
        band; else the grid position clear of them with the largest edge margin
        (>= SLOT_MIN_EDGE); else the grid position that keeps the footprint on
        the card, then has the most clearance.
        info: axis (rad or None), edge_margin, clearance (min of footprint_clearance
        and finger_clearance), tier, open_width (grip + RELEASE_OPEN_EXTRA, or
        + RELEASE_OPEN_MIN_EXTRA when that clears the fingers better). pending: (length,
        width) of objects still to come into this zone; when one of them is long
        (LONG_ZONE_RATIO), a short held object keeps one zone diagonal free for it
        (RESERVE_CLEARANCE). center_only: the zone center (best axis there).
        avoid: release xy already found unreachable; positions within
        AVOID_RADIUS of them are skipped (the center too, unless center_only).
        near_center: instead of the slots, the grid position nearest the zone center
        that keeps place.slot_margin to the edge, NEAR_CENTER_GAP (else 0) to the
        objects and finger band, and room for the fingers of a pending long object on
        its strip (tier near_center); the usual tiers when none does.
        """
        banned = [(float(a[0]), float(a[1])) for a in avoid]

        def skip(x: float, y: float) -> bool:
            return any(math.hypot(x - a, y - b) <= AVOID_RADIUS for a, b in banned)

        L = float(held.get('length', 0.0) or 0.0)
        W = float(held.get('width', 0.0) or 0.0)
        L, W = max(L, W), min(L, W)
        elongated = bool(held.get('elongated', False))
        grip = float(held.get('grip_width', W) or W)
        open_w = min(self.open_width, grip + RELEASE_OPEN_EXTRA)
        open_min = min(open_w, grip + RELEASE_OPEN_MIN_EXTRA)
        # Finger bottom above the zone floor at the release (the stack rule is off).
        finger_floor = (self.release_height_flat - float(lower)
                        + float(held.get('grip_z', 0.0) or 0.0) - PAD_HALF)
        occ = [o for o in occupied if _occupied_capsule(o) is not None]
        center = (float(zone.center[0]), float(zone.center[1]))
        side = min(float(zone.size[0]), float(zone.size[1]))
        reserve = [] if L > LONG_ZONE_RATIO * side else self._reserved(zone, pending)
        legacy = (([center] if center_first else [])
                  + (self._slots(zone, L) if self.spread else []))
        if not legacy:
            legacy = [center]

        def evaluate(x, y):
            out = []
            for k, ax in enumerate(self._axes(zone, x, y, L, W, elongated)):
                cap = footprint_capsule(x, y, L, W, ax)
                closing = (neutral_closing_angle(x, y) if ax is None else
                           choose_closing_angle(ax + math.pi / 2.0, x, y, math.pi))
                e = capsule_edge_margin(zone, cap)
                fp, fg = self._clearance(cap, closing, open_w / 2.0 + FINGER_OUTER
                                         - FINGER_HALF_BAND, finger_floor, occ)
                ow = open_w
                if fg < 0.0 and open_min < open_w:
                    fg2 = self._clearance(cap, closing, open_min / 2.0 + FINGER_OUTER
                                          - FINGER_HALF_BAND, finger_floor, occ)[1]
                    if fg2 > fg:
                        fg, ow = fg2, open_min
                if reserve:
                    free = max(_seg_dist(cap[0], cap[1], r[0], r[1]) - cap[2] - r[2]
                               for r in reserve)
                    fp = min(fp, free - RESERVE_CLEARANCE)
                out.append((k, ax, closing, e, min(fp, fg), (fp, fg, ow)))
            return out

        def done(x, y, ax, closing, e, cl, more, tier):
            fp, fg, ow = more
            info = {'xy': [round(x, 4), round(y, 4)], 'reserved_diagonal': bool(reserve),
                    'axis': None if ax is None else round(wrap_pi(ax), 4),
                    'edge_margin': round(e, 4),
                    'clearance': None if not math.isfinite(cl) else round(cl, 4),
                    'footprint_clearance': None if not math.isfinite(fp) else round(fp, 4),
                    'finger_clearance': None if not math.isfinite(fg) else round(fg, 4),
                    'tier': tier, 'open_width': round(ow, 4)}
            return x, y, closing, info

        if L > LONG_ZONE_RATIO * side and W > 0.0:
            x, y, ax = self.long_strip(zone, W)
            if center_only or skip(x, y):
                x, y = center
            cap = footprint_capsule(x, y, L, W, ax)
            closing = choose_closing_angle(ax + math.pi / 2.0, x, y, math.pi)
            fp, fg = self._clearance(cap, closing, open_w / 2.0 + FINGER_OUTER - FINGER_HALF_BAND,
                                     finger_floor, occ)
            return done(x, y, ax, closing, capsule_edge_margin(zone, cap), min(fp, fg),
                        (fp, fg, open_w), 'long_strip')
        if near_center and not center_only:
            def reserve_ok(x, y, ax):
                # a long object still to come is released on its strip with the fingers
                # open across it: keep them clear of this one too
                cap = footprint_capsule(x, y, L, W, ax)
                return all(_seg_dist(cap[0], cap[1], r[0], r[1]) - cap[2] - r[2]
                           >= RELEASE_OPEN_EXTRA / 2.0 + FINGER_OUTER for r in reserve)
            hit = self._near_center(zone, center, evaluate, skip, reserve_ok)
            if hit is not None:
                return done(*hit, tier='near_center')
        for x, y in ([] if center_only else legacy):
            if skip(x, y):
                continue
            for _k, ax, closing, e, cl, more in evaluate(x, y):
                if e >= self.slot_margin - 1e-9 and cl >= 0.0:
                    return done(x, y, ax, closing, e, cl, more, 'slot')
        if center_only or (not self.spread and not skip(*center)):
            x, y = center
            _k, ax, closing, e, cl, more = max(evaluate(x, y), key=lambda r: (r[3], -r[0]))
            return done(x, y, ax, closing, e, cl, more, 'center')
        hx, hy = float(zone.size[0]) / 2.0, float(zone.size[1]) / 2.0
        ca, sa = math.cos(zone.yaw), math.sin(zone.yaw)
        nx = int(math.floor(hx / SLOT_GRID_STEP))
        ny = int(math.floor(hy / SLOT_GRID_STEP))
        best_b, best_c = None, None
        for i in range(-nx, nx + 1):
            for j in range(-ny, ny + 1):
                a, b = i * SLOT_GRID_STEP, j * SLOT_GRID_STEP
                x, y = center[0] + ca * a - sa * b, center[1] + sa * a + ca * b
                if skip(x, y):
                    continue
                for k, ax, closing, e, cl, more in evaluate(x, y):
                    cand = (x, y, ax, closing, e, cl, more)
                    if cl >= 0.0 and e >= SLOT_MIN_EDGE:
                        key = (min(e, self.slot_margin), min(cl, 0.02), -k)
                        if best_b is None or key > best_b[0]:
                            best_b = (key, cand)
                    key = (min(e - SLOT_MIN_EDGE, 0.0), min(cl, 0.0), min(e, cl), -k)
                    if best_c is None or key > best_c[0]:
                        best_c = (key, cand)
        if best_b is not None:
            return done(*best_b[1], tier='grid')
        if best_c is None:                      # every grid position avoided
            x, y = center
            _k, ax, closing, e, cl, more = max(evaluate(x, y), key=lambda r: (r[3], -r[0]))
            return done(x, y, ax, closing, e, cl, more, 'center')
        return done(*best_c[1], tier='best_effort')

    def plan(self, zone: Optional[Zone], held_offset: float, transit_z: float,
             footprint: Optional[float] = None, occupied: Sequence[Sequence[float]] = (),
             center_first: bool = True, held: Optional[dict] = None,
             pending: Sequence[Sequence[float]] = (), center_only: bool = False,
             avoid: Sequence[Sequence[float]] = (), lower: float = 0.0,
             near_center: bool = False) -> PlacePlan:
        """
        Return a PlacePlan for releasing into zone (poses in the base frame, m).

        footprint / occupied / center_first: see release_xy; the defaults give
        the zone center at the empty-zone height. occupied rows may carry
        (length, width, long-axis angle) as items 4-6 (capsule footprints).
        held: shape of the held object for a shaped release (release_pose):
        length, width (footprint, m), elongated (long axis perpendicular to the
        closing axis), grip_width (m), grip_z (grasp TCP above the object
        bottom, m). None = the footprint rule above. pending / center_only: see
        release_pose (center_only also applies without held). avoid: see release_pose.
        near_center: see release_pose (place.center_labels).
        lower: m the release goes below the release height (place.release_lower
        per label); the wall rim and stack clearances still apply.
        """
        eye = np.eye(4)
        self.last_info = {}
        if zone is None:
            return PlacePlan('', eye, eye.copy(), eye.copy(), self.open_width, False, 'no_zone')
        if not _finite(held_offset) or held_offset < 0.0:
            return PlacePlan(zone.name, eye, eye.copy(), eye.copy(), self.open_width, False,
                             'bad_held_offset: %r' % (held_offset,))
        if not _finite(transit_z):
            return PlacePlan(zone.name, eye, eye.copy(), eye.copy(), self.open_width, False,
                             'bad_transit_z: %r' % (transit_z,))
        floor = float(zone.center[2])
        if zone.release_height is not None:
            rh = float(zone.release_height)
        elif zone.wall_height > 0.0:
            rh = self.release_height
        else:
            rh = self.release_height_flat
        lower = float(lower) if _finite(lower) else 0.0
        z_rel = floor + rh - lower + held_offset
        if zone.wall_height > 0.0:
            z_rel = max(z_rel, floor + zone.wall_height + WALL_RELEASE_CLEARANCE + held_offset)
        open_w = self.open_width
        stack = True
        L = W = 0.0
        if held is not None:
            L = float(held.get('length', 0.0) or 0.0)
            W = float(held.get('width', 0.0) or 0.0)
        if held is not None and _finite(L) and _finite(W) and max(L, W) > 0.0:
            x, y, ang, info = self.release_pose(zone, held, occupied, center_first, pending,
                                                center_only, avoid, lower, near_center)
            open_w = info['open_width']
            self.last_info = info
            # On a flat card a footprint clear of the objects already there drops from the
            # empty-zone height; in a walled bin they may have rolled anywhere.
            fpc = info['footprint_clearance']
            stack = zone.wall_height > 0.0 or (fpc is not None and fpc < 0.0)
        else:
            x, y = self.release_xy(zone, None if center_only else footprint, occupied,
                                   center_first)
            ang = neutral_closing_angle(x, y)
        tallest = [float(o[3]) for o in occupied if len(o) >= 4 and _finite(o[3])]
        if tallest and stack:
            # Objects already in the zone may have rolled anywhere: clear the tallest.
            z_rel = max(z_rel, floor + max(tallest) + STACK_CLEARANCE + held_offset)
        z_pre = max(float(transit_z), z_rel)
        return PlacePlan(zone=zone.name,
                         T_pre=top_down_T(x, y, z_pre, ang),
                         T_release=top_down_T(x, y, z_rel, ang),
                         T_retreat=top_down_T(x, y, z_rel + self.retreat_height, ang),
                         open_width=open_w)


def strip_axis(zone: Zone) -> float:
    """Return the zone axis (zone.yaw or + pi/2) nearer the direction from the base to it."""
    radial = math.atan2(float(zone.center[1]), float(zone.center[0]))
    return min((float(zone.yaw), float(zone.yaw) + math.pi / 2.0),
               key=lambda a: abs(wrap_pi(2.0 * (a - radial))))


def zones_list(zones) -> list:
    """Return zones as a list whether given as a dict or an iterable."""
    if isinstance(zones, dict):
        return list(zones.values())
    return list(zones or [])
