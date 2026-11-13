"""task.live: scene updates from detections seen during a pick, and the watcher thread."""
import numpy as np

from cube_task.core.types import Detection, PerceptionResult
from cube_task.task.live import (LiveSettings, LiveWatcher, apply_live, drop_live_seen,
                                 drop_unconfirmed_live)


def _det(label, x, y):
    return Detection(label=label, score=1.0, center_base=np.array([x, y, 0.03]),
                     n_points=200, extra={})


def test_apply_live_new_moved_same():
    scene = {0: [_det('pear', 0.40, 0.00), _det('potato', 0.40, 0.10)], 1: []}
    live = [_det('pear', 0.46, 0.00),            # pushed 6 cm
            _det('potato', 0.41, 0.10),          # 1 cm: measurement noise
            _det('banana', 0.30, -0.10)]         # not known yet
    ch = apply_live(scene, live, 3, 1, move=0.03, match=0.15, merge_dist=0.04)
    assert [r[0] for r in ch['moved']] == ['pear']
    assert [r[0] for r in ch['same']] == ['potato']
    assert [r[0] for r in ch['new']] == ['banana']
    assert [d.label for d in scene[0]] == ['potato']                 # old pear position gone
    assert scene[0][0].extra['live_ok'] == 1
    assert scene[0][0].center_base[0] == 0.40                        # kept, not replaced
    assert sorted(d.label for d in scene[3]) == ['banana', 'pear']
    assert all(d.extra['view'] == 3 and d.extra['live_ok'] == 1 for d in scene[3])


def test_apply_live_one_to_one_matching():
    scene = {0: [_det('pear', 0.40, 0.00), _det('pear', 0.40, 0.10)]}
    live = [_det('pear', 0.40, 0.01), _det('pear', 0.40, 0.11)]
    ch = apply_live(scene, live, 2, 1, move=0.03, match=0.15, merge_dist=0.04)
    assert len(ch['same']) == 2 and not ch['moved'] and not ch['new']


def test_drop_unconfirmed_and_seen():
    scene = {0: [_det('pear', 0.40, 0.00)],
             2: [_det('potato', 0.42, 0.05), _det('banana', 0.10, 0.30)]}
    scene[2][0].extra['live_ok'] = 1
    scene[2][1].extra['live_ok'] = 1
    assert drop_unconfirmed_live(scene, 2, 2, (0.40, 0.0), 0.12) == 1   # potato near the pick
    assert [d.label for d in scene[2]] == ['banana']
    scene[0] = [_det('banana', 0.11, 0.30)]
    drop_live_seen(scene, 2, 0, 0.04)
    assert scene[2] == []


class _Cfg(dict):
    def get(self, k, d=None):
        return super().get(k, d)


class _Clock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        self.t += 0.01
        return self.t


class _Stamped:
    def __init__(self, v):
        self.value, self.stamp = v, 0.0


class _Arm:
    def __init__(self):
        self.moving = False
        self.k = 0

    def get_flange_pose(self):
        self.k += 1
        T = np.eye(4)
        T[:3, 3] = [0.4 + (0.05 * self.k if self.moving else 0.0), 0.0, 0.3]
        return _Stamped(T)


class _Frame:
    stamp = 1.0


class _Cam:
    def get_frame(self, timeout=1.0, newer_than=None):
        return _Frame()


class _Perc:
    def __init__(self):
        self.calls = 0

    def process(self, frame, T_bc, zones):
        self.calls += 1
        return PerceptionResult(detections=[_det('pear', 0.40, 0.01),     # under the TCP
                                            _det('potato', 0.20, 0.20)])


def _watcher(arm, perc):
    import cube_task.core.extrinsics as ex
    s = LiveSettings(_Cfg({'runner.live_watch.enabled': True, 'runner.live_watch.latency_s': 0.0}))
    w = LiveWatcher(s, arm, _Cam(), perc, None, np.eye(4), {}, _Clock(), 0.06, 0.04)
    return w, ex


def test_watcher_drops_held_object_and_moving_frames(monkeypatch):
    import time
    arm, perc = _Arm(), _Perc()
    w, ex = _watcher(arm, perc)
    monkeypatch.setattr(ex, 'camera_pose_in_base', lambda e, T, t: T)
    w.stage('descend')                       # before closing: no thread
    assert w._thread is None
    w.stage('lift')
    t0 = time.monotonic()
    while perc.calls == 0 and time.monotonic() - t0 < 2.0:
        time.sleep(0.01)
    dets = w.finish()
    assert [d.label for d in dets] == ['potato']
    arm2, perc2 = _Arm(), _Perc()
    arm2.moving = True
    w2, _ = _watcher(arm2, perc2)
    w2.stage('transit')
    time.sleep(0.1)
    assert w2.finish() == [] and perc2.calls == 0 and w2.stats['moving'] > 0
