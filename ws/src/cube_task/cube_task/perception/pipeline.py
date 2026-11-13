"""
Perception pipeline: RGB-D frame -> class-agnostic instances -> labels.

Stages: depth validity -> back-projection -> base frame -> table plane
(optionally reused for perception.table.reuse_s seconds of frame time) ->
segmentation + geometry (segment.py) -> color split -> optional 2D detector
association -> instance classifier -> min_score filter -> merged-block check ->
debug image. process() never raises; problems are reported in
PerceptionResult.warnings.

With a ColorClassifier (L1 blocks), two extra steps handle touching blocks
that depth alone cannot separate (equal heights, flush faces):

- Color split (perception.segmentation.color_split): an instance at least
  COLOR_SPLIT_MIN_LENGTH block sizes long whose upper pixels hold two or
  more classifier colors, each covering color_split_min_fraction, is split
  into per-color parts (segment.split_by_classes), each with recomputed
  geometry and reject checks. Parts carry extra['split'] = 'color'.
- Merged blocks (perception.segmentation.merged_blocks): a valid instance
  with a block label that is elongated (length >= 1.4 width and >= 1.4
  block sizes, width about one perception.l1_block_size) or larger than 1.8
  block areas is cut into square blocks along its major axis in mode
  'split' (segment.find_block_cuts, extra['split'] = 'blocks'). When no cut
  into square blocks exists, or in mode 'reject', it becomes invalid with
  reject_reason 'merged' if its length and width are close to n >= 2 by 1
  block sizes (MERGED_MULTIPLE_TOL). Other suspects stay valid (an object
  that is not a cube, or a single block larger than configured, which gets
  extra['oversize'] = True when above 1.8 block areas).
- Every detection next to another instance gets that instance's footprint
  points in extra['_neighbor_xy'] (annotate_neighbors); the grasp planner
  keeps the fingers of block grasps off them.

The strings in PerceptionResult.warnings:

- 'no_depth': frame without depth or without any valid depth pixel.
- 'no_extrinsics' / 'bad_extrinsics': T_base_camera missing / not a rigid 4x4.
- 'depth_mismatch' / 'color_mismatch': image sizes disagree with the intrinsics.
- 'no_table' plus 'table:<reason>' (see table.TableFit, or 'low_coverage' when the
  plane covers less than perception.table.min_coverage of the visible pick area).
- 'no_color', 'detector_error', 'classifier_error', 'too_many_blobs',
  'error:<ExceptionType>' (unexpected failure, logged with traceback).
- 'split_error' (a split step failed; the instance is kept unsplit),
  'merged_blocks' (some instance was rejected as merged), 'oversize_block'
  (some block-labeled footprint is larger than 1.8 block areas and could
  not be cut into blocks: check perception.l1_block_size).

Timings are wall-clock milliseconds per stage (time.perf_counter, for
diagnostics only; no decision depends on them).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .classify_color import ColorClassifier
from .features import upper_mask
from .geometry import (backproject, check_depth, parse_polygon, points_in_polygon,
                       project_points, transform_cloud)
from .segment import (find_block_cuts, labels_from_cuts, mark_invalid,
                      segment_objects, SegmentParams, Segmentation, split_by_classes,
                      split_instance)
from .table import fit_table_plane_detailed, TableParams
from .visualize import draw_result
from ..core.interfaces import Detector2D, InstanceClassifier
from ..core.transforms import is_rotation
from ..core.types import Detection, Frame, PerceptionResult, TablePlane
from ..core.zones import parse_zones

LOG = logging.getLogger('cube_task.perception.pipeline')

#: color split only for instances at least this many block sizes long.
COLOR_SPLIT_MIN_LENGTH = 1.3
#: a color split seed component needs this fraction of the instance's upper pixels.
COLOR_SPLIT_MIN_SEED = 0.1
#: merged-block suspects: length / width at least this (1.4, not 1.6: two
#: blocks touching with a lateral offset of 30 % of their size give 1.5) ...
MERGED_MIN_RATIO = 1.4
#: ... with the length at least this many block sizes and the width within
#: this range of block sizes,
MERGED_MIN_LENGTH = 1.4
MERGED_WIDTH_RANGE = (0.6, 1.5)
#: or a footprint area above this many block areas.
MERGED_MIN_AREA = 1.8
#: a suspect that cannot be cut into square blocks is rejected as 'merged' only when
#: its length and width are this close (in block sizes) to n >= 2 blocks x 1 block.
MERGED_MULTIPLE_TOL = 0.25
#: annotate_neighbors: neighbors whose bounding circles come this close (m) ...
NEIGHBOR_MARGIN = 0.03
#: ... contribute at most this many footprint points each.
NEIGHBOR_MAX_POINTS = 300


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 2)


def _finite(v: Any) -> bool:
    try:
        return v is not None and math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def mask_in_box_overlap(mask: np.ndarray, box: Any) -> float:
    """Return the fraction of the mask's pixels inside box (x0, y0, x1, y1 pixels)."""
    n = int(np.count_nonzero(mask))
    if n == 0:
        return 0.0
    H, W = mask.shape
    x0, y0, x1, y1 = [float(v) for v in box]
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        return 0.0
    xa, xb = int(max(0, math.floor(min(x0, x1)))), int(min(W, math.ceil(max(x0, x1))))
    ya, yb = int(max(0, math.floor(min(y0, y1)))), int(min(H, math.ceil(max(y0, y1))))
    if xb <= xa or yb <= ya:
        return 0.0
    return float(np.count_nonzero(mask[ya:yb, xa:xb])) / n


def table_coverage(plane: TablePlane, points_base: np.ndarray, polygon: np.ndarray,
                   T_base_camera: np.ndarray, intrinsics: Any, thresh: float,
                   cell: float = 0.01) -> float:
    """
    Fraction of the in-view pick-polygon area covered by points on the plane.

    The polygon (base XY, m) is cut into cell x cell squares; a square is in
    view when its center on the plane projects inside the image, and covered
    when some point of points_base (Nx3, base) inside it lies within thresh
    of the plane. A flat object top mistaken for the table (for example when
    the table itself returns no depth) covers only a small part.
    """
    poly = np.asarray(polygon, dtype=float)
    lo = poly.min(axis=0)
    n = np.ceil((poly.max(axis=0) - lo) / cell).astype(int) + 1
    gx, gy = np.meshgrid(np.arange(n[0]), np.arange(n[1]), indexing='ij')
    centers = lo + (np.stack([gx.ravel(), gy.ravel()], axis=1) + 0.5) * cell
    inside = points_in_polygon(centers, poly)
    if not np.any(inside):
        return 0.0
    c = centers[inside]
    zc = -(plane.normal[0] * c[:, 0] + plane.normal[1] * c[:, 1] + plane.d) / plane.normal[2]
    uv = project_points(np.column_stack([c, zc]), T_base_camera, intrinsics)
    in_view = (np.all(np.isfinite(uv), axis=1) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
               & (uv[:, 0] <= intrinsics.width - 1) & (uv[:, 1] <= intrinsics.height - 1))
    if not np.any(in_view):
        return 0.0
    p = np.asarray(points_base, dtype=float).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    p = p[np.abs(p @ plane.normal + plane.d) < thresh]
    idx = np.floor((p[:, :2] - lo) / cell).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < n), axis=1)
    covered = np.zeros(int(n[0] * n[1]), dtype=bool)
    covered[idx[ok, 0] * n[1] + idx[ok, 1]] = True
    cell_ids = (np.floor((c - lo) / cell).astype(np.int64) @ np.array([n[1], 1]))
    return float(np.count_nonzero(covered[cell_ids] & in_view)) / float(np.count_nonzero(in_view))


def annotate_neighbors(dets: List[Detection], margin: float = NEIGHBOR_MARGIN,
                       max_points: int = NEIGHBOR_MAX_POINTS) -> None:
    """
    Give each detection the footprint points of the instances next to it.

    For every detection with a center and extra['_points_xy'], the points
    (subsampled to at most max_points per neighbor) of every other instance,
    valid or not, whose bounding circle comes within margin (m) of its own
    are stored as extra['_neighbor_xy'] (Mx2 float32 base XY). The grasp
    planner uses them to keep the fingers off touching neighbors. Detections
    without close neighbors get no key.
    """
    items = []
    for d in dets:
        pxy = d.extra.get('_points_xy')
        c = d.center_base
        if pxy is None or c is None or not (_finite(c[0]) and _finite(c[1])):
            continue
        pxy = np.asarray(pxy, dtype=np.float32).reshape(-1, 2)
        if pxy.shape[0] == 0:
            continue
        cxy = np.array([float(c[0]), float(c[1])])
        r = float(np.max(np.hypot(pxy[:, 0] - cxy[0], pxy[:, 1] - cxy[1])))
        step = max(1, int(math.ceil(pxy.shape[0] / float(max_points))))
        items.append((d, cxy, r, pxy[::step]))
    for d, c, r, _ in items:
        near = [q for e, ce, re, q in items
                if e is not d and float(np.hypot(*(ce - c))) < r + re + margin]
        if near:
            d.extra['_neighbor_xy'] = np.concatenate(near, axis=0)


class Perception:
    """
    Detect, localize and classify objects on the table.

    cfg: core.config.Config (reads perception.* and zones). classifier:
    InstanceClassifier for instances not labeled by the detector. detector:
    Detector2D whose boxes label instances whose mask lies at least
    perception.classifier.onnx.min_mask_overlap inside a box. Construction
    validates the configuration and raises ValueError on bad values.
    """

    def __init__(self, cfg: Any, classifier: Optional[InstanceClassifier] = None,
                 detector: Optional[Detector2D] = None):
        self.cfg = cfg
        self.classifier = classifier
        self.detector = detector
        self.table_params = TableParams.from_cfg(cfg)
        self.seg_params = SegmentParams.from_cfg(cfg)
        self.pick_polygon = parse_polygon(cfg.get('perception.roi.pick_polygon', None))
        self.zmin = float(cfg.get('perception.depth.min_z', 0.15))
        #: label -> HSV ranges of its stem (perception.stem): the stem center is found in the
        #: color image inside the instance mask and stored as extra['stem_xy'] (base m)
        self.stems: Dict[str, List[Any]] = {}
        for lbl, rngs in (cfg.get('perception.stem.labels', {}) or {}).items():
            if rngs:
                self.stems[str(lbl)] = [(np.array(lo, np.uint8), np.array(hi, np.uint8))
                                        for lo, hi in rngs]
        self.stem_min_px = int(cfg.get('perception.stem.min_pixels', 30))
        self.zmax = float(cfg.get('perception.depth.max_z', 3.0))
        if not (0.0 < self.zmin < self.zmax):
            raise ValueError('perception.depth: need 0 < min_z < max_z')
        self.reuse_s = float(cfg.get('perception.table.reuse_s', 0.0))
        self.min_score = float(cfg.get('perception.classifier.min_score', 0.55))
        self.min_overlap = float(cfg.get('perception.classifier.onnx.min_mask_overlap', 0.3))
        if not (0.0 <= self.min_score <= 1.0) or not (0.0 < self.min_overlap <= 1.0):
            raise ValueError('perception.classifier: min_score in [0, 1], '
                             'onnx.min_mask_overlap in (0, 1]')
        self.label_map: Dict[str, str] = {
            str(k): str(v) for k, v in
            (cfg.get('perception.classifier.onnx.label_map', {}) or {}).items()}
        self.table_sample_points = int(cfg.get('perception.table.sample_points', 20000))
        self.min_coverage = float(cfg.get('perception.table.min_coverage', 0.25))
        if not (0.0 <= self.min_coverage <= 1.0):
            raise ValueError('perception.table.min_coverage must be in [0, 1]')
        self.seed = int(cfg.get('perception.table.seed', 0))
        self.draw = bool(cfg.get('perception.debug_image', True))
        self.default_zones = parse_zones(cfg)
        if self.pick_polygon is None:
            LOG.warning('perception.roi.pick_polygon is empty: objects anywhere in view '
                        'are candidates')
        self._table: Optional[TablePlane] = None
        self._table_stamp: Optional[float] = None

    def reset_table(self) -> None:
        """Forget the cached table plane (next frame refits)."""
        self._table = None
        self._table_stamp = None

    # ------------------------------------------------------------------ main entry

    def process(self, frame: Frame, T_base_camera: Optional[np.ndarray],
                zones: Any = None) -> PerceptionResult:
        """
        Run the pipeline on one frame.

        T_base_camera: 4x4 camera pose in the base frame at capture time.
        zones: dict name -> core Zone (None = zones from the config) used for
        exclusion and drawing. Returns a PerceptionResult; never raises.
        """
        t0 = time.perf_counter()
        res = PerceptionResult(detections=[], frame_stamp=float(getattr(frame, 'stamp', 0.0)))
        zones_used = self.default_zones if zones is None else zones
        T = None
        try:
            T = self._run(frame, T_base_camera, zones_used, res)
        except Exception as e:  # never let perception crash the task loop
            LOG.exception('perception failed')
            res.detections = []
            res.warnings.append('error:%s' % type(e).__name__)
        res.timings['total'] = _ms(t0)
        if self.draw:
            t1 = time.perf_counter()
            try:
                res.debug_image = draw_result(frame, res, T, zones_used, self.pick_polygon)
            except Exception:
                LOG.exception('debug image failed')
            res.timings['draw'] = _ms(t1)
        for w in res.warnings:
            LOG.debug('perception warning: %s', w)
        return res

    def _check_T(self, T_base_camera: Any, res: PerceptionResult) -> Optional[np.ndarray]:
        if T_base_camera is None:
            res.warnings.append('no_extrinsics')
            return None
        T = np.asarray(T_base_camera, dtype=np.float64)
        if (T.shape != (4, 4) or not np.all(np.isfinite(T)) or not is_rotation(T[:3, :3], 1e-3)
                or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])):
            res.warnings.append('bad_extrinsics')
            return None
        return T

    def _run(self, frame: Frame, T_base_camera: Any, zones: Any,
             res: PerceptionResult) -> Optional[np.ndarray]:
        if frame.depth is None:
            res.warnings.append('no_depth')
            return self._check_T(T_base_camera, res)
        T = self._check_T(T_base_camera, res)
        if T is None:
            return None
        intr = frame.intrinsics
        depth = np.asarray(frame.depth)
        try:
            check_depth(depth, intr)
        except ValueError as e:
            LOG.warning('unusable depth: %s', e)
            res.warnings.append('depth_mismatch')
            return T
        color_ok = (frame.color is not None and frame.color.ndim == 3
                    and frame.color.shape[2] == 3)
        if not color_ok:
            res.warnings.append('no_color')
        elif frame.color.shape[:2] != depth.shape:
            res.warnings.append('color_mismatch')
            return T

        t = time.perf_counter()
        pts_cam = backproject(depth, intr, self.zmin, self.zmax)
        if not np.any(np.isfinite(pts_cam[..., 2])):
            res.warnings.append('no_depth')
            res.timings['backproject'] = _ms(t)
            return T
        pts = transform_cloud(pts_cam, T)
        res.timings['backproject'] = _ms(t)

        t = time.perf_counter()
        plane = self._table_for(frame, pts, zones, T, res)
        res.timings['table'] = _ms(t)
        if plane is None:
            return T
        res.table = plane

        t = time.perf_counter()
        seg = segment_objects(pts, depth, plane, intr, T, self.seg_params, self.pick_polygon,
                              zones)
        res.warnings.extend(seg.warnings)
        res.timings['segment'] = _ms(t)
        dets = seg.detections
        blocks = self.classifier if isinstance(self.classifier, ColorClassifier) else None
        if blocks is not None and color_ok and self.seg_params.color_split:
            t = time.perf_counter()
            dets = self._color_split(frame, seg, dets, blocks, res)
            res.timings['color_split'] = _ms(t)
        self._label(frame, dets, res, color_ok)
        if self.stems and color_ok:
            self._find_stems(frame, dets, pts)
        if blocks is not None and self.seg_params.merged_blocks != 'off':
            t = time.perf_counter()
            dets = self._merged_blocks(frame, seg, dets, blocks, res)
            res.timings['merged_blocks'] = _ms(t)
        annotate_neighbors(dets)
        res.detections = dets
        return T

    def _table_for(self, frame: Frame, pts: np.ndarray, zones: Any, T_base_camera: np.ndarray,
                   res: PerceptionResult) -> Optional[TablePlane]:
        stamp = float(frame.stamp)
        if (self.reuse_s > 0.0 and self._table is not None and self._table_stamp is not None
                and 0.0 <= stamp - self._table_stamp <= self.reuse_s):
            return self._table
        H, W = pts.shape[:2]
        stride = max(1, int(math.sqrt(H * W / float(max(self.table_sample_points, 100)))))
        sample = pts[::stride, ::stride].reshape(-1, 3)
        sample = sample[np.isfinite(sample[:, 2])]
        fit = fit_table_plane_detailed(sample, self.table_params,
                                       rng=np.random.default_rng(self.seed),
                                       pick_polygon=self.pick_polygon, zones=zones)
        if fit.plane is not None and self.pick_polygon is not None and self.min_coverage > 0:
            # Stride 2 keeps point spacing well below the 1 cm coverage cells
            # even at long range or grazing view angles.
            cov = table_coverage(fit.plane, pts[::2, ::2], self.pick_polygon, T_base_camera,
                                 frame.intrinsics, self.table_params.inlier_thresh)
            if cov < self.min_coverage:
                LOG.warning('table plane covers only %.0f%% of the visible pick area',
                            cov * 100.0)
                fit.candidate, fit.plane, fit.reason = fit.plane, None, 'low_coverage'
        if fit.plane is None:
            res.warnings.append('no_table')
            res.warnings.append('table:%s' % fit.reason)
            LOG.warning('table fit failed: %s (%d points, tilt %.1f deg, z %.4f)',
                        fit.reason, fit.n_points, fit.tilt_deg, fit.z_ref)
            return None
        self._table, self._table_stamp = fit.plane, stamp
        return fit.plane

    # ------------------------------------------------------------------ labels

    def _label(self, frame: Frame, dets: List[Detection], res: PerceptionResult,
               color_ok: bool) -> None:
        for d in dets:
            d.label, d.score, d.score_kind = 'unknown', 0.0, 'none'
        if not dets or not color_ok:
            return
        rest = list(dets)
        if self.detector is not None:
            t = time.perf_counter()
            try:
                boxes = list(self.detector.detect(frame.color))
            except Exception:
                LOG.exception('detector failed')
                res.warnings.append('detector_error')
                boxes = []
            rest = self._associate(dets, boxes)
            res.timings['detector'] = _ms(t)
        if rest and self.classifier is not None:
            t = time.perf_counter()
            self._classify(frame, rest, res)
            res.timings['classify'] = _ms(t)
        self._apply_min_score(dets)

    def _find_stems(self, frame: Frame, dets: List[Detection], pts: np.ndarray) -> None:
        """
        Set extra['stem_xy'] / ['stem_px'] / ['stem_pixels'] of the perception.stem labels.

        The stem is the largest blob of stem-colored (HSV) pixels inside the instance
        mask, grown by a few pixels (a stem on the rim sits on the mask border); its
        base-frame xy is the median of the valid 3D points of the blob.
        """
        hsv = None
        for d in dets:
            rngs = self.stems.get(d.label)
            if not rngs or d.mask is None or d.mask.shape[:2] != frame.color.shape[:2]:
                continue
            if hsv is None:
                hsv = cv2.cvtColor(frame.color, cv2.COLOR_BGR2HSV)
            m = d.mask.astype(np.uint8)
            m = cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0
            col = np.zeros(m.shape, bool)
            for lo, hi in rngs:
                col |= cv2.inRange(hsv, lo, hi) > 0
            col &= m
            n, lab, stats, cent = cv2.connectedComponentsWithStats(col.astype(np.uint8), 8)
            if n <= 1:
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[k, cv2.CC_STAT_AREA])
            d.extra['stem_pixels'] = area
            if area < self.stem_min_px:
                continue
            p3 = pts[lab == k]
            p3 = p3[np.all(np.isfinite(p3), axis=1)]
            if p3.shape[0] < max(5, area // 10):
                continue
            xy = np.median(p3[:, :2], axis=0)
            d.extra['stem_xy'] = [float(xy[0]), float(xy[1])]
            d.extra['stem_px'] = [float(cent[k][0]), float(cent[k][1])]

    def _classify(self, frame: Frame, dets: List[Detection], res: PerceptionResult) -> None:
        try:
            self.classifier.classify(frame, dets)
        except Exception:
            LOG.exception('classifier failed')
            if 'classifier_error' not in res.warnings:
                res.warnings.append('classifier_error')
            for d in dets:
                d.label, d.score, d.score_kind = 'unknown', 0.0, 'none'

    def _apply_min_score(self, dets: List[Detection]) -> None:
        for d in dets:
            if not (isinstance(d.score, (int, float)) and math.isfinite(d.score)):
                d.score = 0.0
            if d.label != 'unknown' and d.score < self.min_score:
                d.extra['raw_label'] = d.label
                d.extra['raw_score'] = float(d.score)
                d.label = 'unknown'

    # ------------------------------------------------------------------ touching blocks

    def _color_split(self, frame: Frame, seg: Segmentation, dets: List[Detection],
                     clf: ColorClassifier, res: PerceptionResult) -> List[Detection]:
        """Split instances that hold several block colors (see the module docstring)."""
        p = self.seg_params
        min_len = COLOR_SPLIT_MIN_LENGTH * p.block_size
        out: List[Detection] = []
        for d in dets:
            parts = None
            if (d.mask is not None and _finite(d.length) and d.length >= min_len
                    and seg.ctx is not None):
                try:
                    parts = self._color_parts(frame, seg, d, clf)
                except Exception:
                    LOG.exception('color split failed')
                    if 'split_error' not in res.warnings:
                        res.warnings.append('split_error')
            out.extend(parts or [d])
        return out

    def _color_parts(self, frame: Frame, seg: Segmentation, d: Detection,
                     clf: ColorClassifier) -> Optional[List[Detection]]:
        p = self.seg_params
        x0, y0, x1, y1 = d.bbox
        mask = np.asarray(d.mask[y0:y1, x0:x1], dtype=bool)
        up = upper_mask(d)
        crop = np.ascontiguousarray(frame.color[y0:y1, x0:x1], dtype=np.uint8)
        if up is None or crop.shape[:2] != mask.shape:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        cls = clf.pixel_classes(hsv.reshape(-1, 3)).reshape(mask.shape)
        min_seed = max(p.min_blob_px, int(COLOR_SPLIT_MIN_SEED * np.count_nonzero(up)))
        found = split_by_classes(mask, cls, up, p.color_split_min_fraction, min_seed)
        if found is None:
            return None
        labels, part_class = found
        parts = split_instance(seg.ctx, d, labels, 'color')
        if len(parts) < 2:
            return None
        LOG.debug('color split of a %.3f x %.3f m instance into %s', d.length, d.width,
                  [clf.labels[k] for k in part_class])
        return parts

    def _merged_blocks(self, frame: Frame, seg: Segmentation, dets: List[Detection],
                       clf: ColorClassifier, res: PerceptionResult) -> List[Detection]:
        """Cut or reject block-labeled instances that hold several blocks."""
        p = self.seg_params
        s = p.block_size
        labels = set(clf.labels)
        out: List[Detection] = []
        for d in dets:
            if not (d.valid and d.label in labels and _finite(d.length) and _finite(d.width)
                    and _finite(d.yaw) and d.width > 0.0):
                out.append(d)
                continue
            L, W = float(d.length), float(d.width)
            elongated = (L >= MERGED_MIN_RATIO * W and L >= MERGED_MIN_LENGTH * s
                         and MERGED_WIDTH_RANGE[0] * s <= W <= MERGED_WIDTH_RANGE[1] * s)
            if not (elongated or L * W >= MERGED_MIN_AREA * s * s):
                out.append(d)
                continue
            parts = None
            if p.merged_blocks == 'split' and seg.ctx is not None:
                try:
                    parts = self._block_parts(frame, seg, d, res)
                except Exception:
                    LOG.exception('block split failed')
                    if 'split_error' not in res.warnings:
                        res.warnings.append('split_error')
            if parts:
                out.extend(parts)
                continue
            n = int(round(L / s))
            multiple = (n >= 2 and abs(L / s - n) <= MERGED_MULTIPLE_TOL
                        and abs(W / s - 1.0) <= MERGED_MULTIPLE_TOL)
            if multiple:
                mark_invalid(d, 'merged')
                if 'merged_blocks' not in res.warnings:
                    res.warnings.append('merged_blocks')
                LOG.info('%s instance %.3f x %.3f m looks like %d touching blocks: not picked',
                         d.label, L, W, n)
            elif L * W >= MERGED_MIN_AREA * s * s:
                d.extra['oversize'] = True
                if 'oversize_block' not in res.warnings:
                    res.warnings.append('oversize_block')
                LOG.warning('%s object %.3f x %.3f m is larger than 1.8 blocks of perception.'
                            'l1_block_size (%.3f m): check the setting', d.label, L, W, s)
            out.append(d)
        return out

    def _block_parts(self, frame: Frame, seg: Segmentation, d: Detection,
                     res: PerceptionResult) -> Optional[List[Detection]]:
        x0, y0, x1, y1 = d.bbox
        pts = seg.ctx.points[y0:y1, x0:x1]
        use = (np.asarray(d.mask[y0:y1, x0:x1], dtype=bool) & seg.ctx.cand[y0:y1, x0:x1]
               & ~seg.ctx.edges[y0:y1, x0:x1])
        xyz = pts[use].astype(np.float64)
        cuts = find_block_cuts(xyz[:, :2], float(d.yaw), self.seg_params.block_size,
                               self.seg_params.extent_trim_pct, z=xyz[:, 2])
        if cuts is None:
            return None
        labels = labels_from_cuts(seg.ctx, d, float(d.yaw), cuts[0])
        parts = split_instance(seg.ctx, d, labels, 'blocks')
        if len(parts) < 2:
            return None
        for part in parts:
            part.extra['block_cut_error'] = round(cuts[1], 3)
            part.label, part.score, part.score_kind = 'unknown', 0.0, 'none'
        self._classify(frame, parts, res)
        self._apply_min_score(parts)
        LOG.debug('cut a %s instance %.3f x %.3f m into %d blocks (squareness error %.2f)',
                  d.label, d.length, d.width, len(parts), cuts[1])
        return parts

    def _associate(self, dets: List[Detection], boxes: list) -> List[Detection]:
        """Label detections from detector boxes; return the detections left unlabeled."""
        clean = []
        for b in boxes:
            try:
                box, lab, prob = b
                box = tuple(float(v) for v in box)
                prob = float(prob)
                if len(box) != 4 or not math.isfinite(prob):
                    raise ValueError('bad box')
                clean.append((box, str(lab), min(max(prob, 0.0), 1.0)))
            except (TypeError, ValueError):
                LOG.warning('ignoring malformed detector output %r', (b,))
        rest = []
        for d in dets:
            best = None
            if d.mask is not None:
                for box, lab, prob in clean:
                    ov = mask_in_box_overlap(d.mask, box)
                    if ov >= self.min_overlap and (best is None or (ov, prob) > best[:2]):
                        best = (ov, prob, lab, box)
            if best is None:
                rest.append(d)
                continue
            ov, prob, lab, box = best
            d.label = self.label_map.get(lab, lab)
            d.score = prob
            d.score_kind = 'prob'
            d.extra['detector_label'] = lab
            d.extra['detector_box'] = [round(v, 1) for v in box]
            d.extra['detector_overlap'] = round(ov, 3)
        return rest
