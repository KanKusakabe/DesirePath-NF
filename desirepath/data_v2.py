"""DesirePath-NF v2 data extraction (SDD).

For each pedestrian step we build, from the auto-generated walkable-surface
masks in data/sdd/{scene}/:

  y          : egocentric one-step increment (d_forward, d_lateral)   [target]
  hist       : last HIST egocentric increments                        [GRU]
  patch      : heading-aligned local-terrain crop, 3 x P x P          [CNN]
               channels = [is_paved, dist_to_obstacle, dist_to_paved]
  stat_geom  : [cos(goal_bearing), sin(goal_bearing), rem_dist, crowd] [transfer-safe]
  pos_norm   : normalized absolute position (x,y) in [-1,1]            [v1-abs baseline only]

Everything is egocentric / rotation-aligned so the flow's grammar binds to
*terrain*, not absolute coordinates — the fix for v1's hotel-type transfer
collapse.  px/m scale is unknown for SDD; all lengths are in pixels and the
target is normalized downstream, so NLL lives in a consistent normalized space.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

HERE = Path(__file__).resolve().parent.parent
SDD = HERE / "data" / "sdd"

SCENES = ["hyang", "little", "quad", "gates"]   # LOSO folds (order fixed)
HIST = 8            # history length for the GRU
DIM = 2             # (d_forward, d_lateral)
PATCH = 32          # crop resolution (cells per side)
# px/m differs per video (drone pose/zoom); calibrate each video by its median
# step length s so speed/scale is equalized across scenes. All spatial extents
# are then expressed in *median-step* units -> removes the pure-scale confound
# that would otherwise corrupt the leave-one-scene-out transfer test.
WIN_STEPS = 20.0    # crop window spans this many median-steps (~real 8-10 m)
CROWD_STEPS = 12.0  # crowd-density radius in median-steps
REM_STEPS = 40.0    # remaining-distance normalizer (median-steps)
FUT = 8             # horizon (steps) for the K-step-ahead occupancy target y_fut
MIN_LEN = HIST + FUT + 2   # need history AND a K-step future
FPS_DS = 12         # 30 fps -> 2.5 Hz


def _rot(dx, dy, heading):
    c, s = math.cos(-heading), math.sin(-heading)
    return c * dx - s * dy, s * dx + c * dy


def _load_video_arrays(d):
    import matplotlib.image as mpimg
    pav = mpimg.imread(d / "paved_mask.png")
    if pav.ndim == 3:
        pav = pav[..., 0]
    paved = (pav > 0.5).astype(np.float32)
    dist_obs = np.load(d / "dist_to_obstacle.npy")
    dist_pav = np.load(d / "dist_to_paved.npy")
    z = np.load(d / "peds.npz")
    return paved, dist_obs, dist_pav, z


def _per_ped_tracks(z):
    """Group rows into per-pedestrian frame-sorted, 2.5 Hz-downsampled tracks."""
    ids, frame, cx, cy = z["ids"], z["frame"], z["cx"], z["cy"]
    tracks = []
    for pid in np.unique(ids):
        m = ids == pid
        f, X, Y = frame[m], cx[m], cy[m]
        o = np.argsort(f)
        f, X, Y = f[o], X[o], Y[o]
        sel = np.unique(f // FPS_DS, return_index=True)[1]
        f, X, Y = f[sel], X[sel], Y[sel]
        if len(X) >= MIN_LEN:
            tracks.append((f, X, Y))
    return tracks


def _frame_index(z):
    """frame -> (Nx2 positions) for crowd density."""
    fr = {}
    for f, x, y in zip(z["frame"], z["cx"], z["cy"]):
        fr.setdefault(int(f), []).append((x, y))
    return {k: np.array(v) for k, v in fr.items()}


def _build_video(d):
    paved, dist_obs, dist_pav, z = _load_video_arrays(d)
    H, W = paved.shape
    frames = _frame_index(z)
    tracks = _per_ped_tracks(z)
    # per-video scale = median pixel step length (calibrate speed/scale)
    allmag = np.concatenate([np.hypot(*np.diff(np.stack([X, Yc], 1), axis=0).T)
                             for _, X, Yc in tracks]) if tracks else np.array([1.0])
    s = float(max(np.median(allmag[allmag > 1e-6]) if (allmag > 1e-6).any() else 1.0, 1e-3))
    win_px = WIN_STEPS * s
    crowd_r = CROWD_STEPS * s
    # distance fields normalized in median-step units (comparable across videos)
    do_n = np.clip(dist_obs / (WIN_STEPS * s), 0, 1).astype(np.float32)
    dp_n = np.clip(dist_pav / (WIN_STEPS * s), 0, 1).astype(np.float32)

    # base crop grid (centered, in pixels); row a (up = -a) aligns to heading
    half = (PATCH - 1) / 2.0
    step = win_px / PATCH
    aa, bb = np.meshgrid(np.arange(PATCH) - half, np.arange(PATCH) - half, indexing="ij")
    aa = aa.ravel() * step   # row offset (down positive)
    bb = bb.ravel() * step   # col offset (right positive)

    Hh, Pp, Sg, Y, Yf, Pos, Frm = [], [], [], [], [], [], []
    for f, X, Yc in tracks:
        pts = np.stack([X, Yc], 1)                       # (n,2) pixel (x,y)
        goal = pts[-1]
        dpx = np.diff(pts, axis=0)                       # (n-1,2)
        head = np.arctan2(dpx[:, 1], dpx[:, 0])          # pixel-space heading
        ego = np.zeros_like(dpx)
        for k in range(len(dpx)):
            h = head[k - 1] if k > 0 else head[k]
            fx, fy = _rot(dpx[k, 0], dpx[k, 1], h)
            ego[k] = (fx, fy)
        for k in range(HIST, len(dpx) - FUT + 1):
            if np.hypot(*dpx[k]) > 8 * s:          # tracking glitch / non-walk jump
                continue
            h = head[k - 1]
            pos = pts[k]
            # K-step-ahead endpoint offset in the heading-aligned frame (occupancy target)
            fxf, fyf = _rot(pts[k + FUT][0] - pos[0], pts[k + FUT][1] - pos[1], h)
            # goal bearing (egocentric)
            gx, gy = _rot(goal[0] - pos[0], goal[1] - pos[1], h)
            gb = math.atan2(gy, gx)
            rem = min(np.hypot(*(goal - pos)) / s / REM_STEPS, 3.0)
            # crowd density at this frame within radius
            fr_pts = frames.get(int(f[k]), None)
            if fr_pts is not None:
                dcnt = (np.hypot(fr_pts[:, 0] - pos[0], fr_pts[:, 1] - pos[1]) < crowd_r).sum() - 1
            else:
                dcnt = 0
            crowd = min(max(dcnt, 0) / 8.0, 3.0)
            # heading-aligned sample coordinates (up = heading)
            fwd = np.array([math.cos(h), math.sin(h)])     # up direction
            rgt = np.array([math.sin(h), -math.cos(h)])    # right direction
            wx = pos[0] + bb * rgt[0] + (-aa) * fwd[0]
            wy = pos[1] + bb * rgt[1] + (-aa) * fwd[1]
            coords = np.stack([wy, wx])                     # (2, P*P) -> (row,col)
            ch_pav = map_coordinates(paved, coords, order=1, mode="nearest").reshape(PATCH, PATCH)
            ch_obs = map_coordinates(do_n, coords, order=1, mode="nearest").reshape(PATCH, PATCH)
            ch_pav2 = map_coordinates(dp_n, coords, order=1, mode="nearest").reshape(PATCH, PATCH)
            Hh.append(ego[k - HIST:k] / s)
            Pp.append(np.stack([ch_pav, ch_obs, ch_pav2]).astype(np.float16))
            Sg.append([math.cos(gb), math.sin(gb), rem, crowd])
            Y.append(ego[k] / s)
            Yf.append([fxf / s, fyf / s])
            Pos.append([2 * pos[0] / W - 1, 2 * pos[1] / H - 1])
            Frm.append(int(f[k]))

    return dict(
        hist=np.asarray(Hh, np.float32),
        patch=np.asarray(Pp, np.float16),
        stat_geom=np.asarray(Sg, np.float32),
        y=np.asarray(Y, np.float32),
        y_fut=np.asarray(Yf, np.float32),
        pos_norm=np.asarray(Pos, np.float32),
        frame=np.asarray(Frm, np.int64),
    )


def build_scene(scene, cache=True):
    """Aggregate samples over all videos of a scene."""
    cache_fp = SDD / scene / "samples_v2.npz"
    if cache and cache_fp.exists():
        c = np.load(cache_fp)
        return {k: c[k] for k in c.files}
    vids = sorted((SDD / scene).glob("video*"))
    parts = [_build_video(d) for d in vids if (d / "peds.npz").exists()]
    parts = [p for p in parts if len(p["y"])]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if cache:
        np.savez_compressed(cache_fp, **out)
    return out


if __name__ == "__main__":
    for sc in SCENES:
        o = build_scene(sc, cache=False)
        print(f"{sc:8s} samples {len(o['y']):6d}  patch {o['patch'].shape}  "
              f"y_std {o['y'].std(0).round(2)}  crowd_max {o['stat_geom'][:,3].max():.1f}")
