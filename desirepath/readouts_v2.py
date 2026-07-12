"""DesirePath-NF v2 read-outs: the descriptive desire-path deliverable.

These do NOT depend on the (negative) predictive transfer result.  From real
trajectories x real walkable-surface masks we produce:

  (1) desire-path map : where pedestrians systematically leave the pavement
                        (off-pavement visitation hotspots) = 芝を突っ切る道.
  (2) shortcut pressure: paved-only shortest path length vs actual path length
                        -> peds who beat the paved network by cutting grass.
  (3) paving proposal  : greedily pave the grass cells carrying the most
                        off-pavement traffic; utility = traffic captured vs a
                        naive "pave the densest visited cells" baseline.

Run:  uv run python -m desirepath.readouts_v2
Out:  reports/figures/desirepath.png, results/readouts_v2.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

HERE = Path(__file__).resolve().parent.parent
SDD = HERE / "data" / "sdd"
FIG = HERE / "reports" / "figures"
RES = HERE / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

DS = 8               # mask downsample factor for the geodesic grid
FPS_DS = 12
HIST = 8


def _load(scene, video):
    import matplotlib.image as mpimg
    from desirepath.prep_sdd import SDD_RAW
    video_dir = SDD / scene / video
    def mask(name):
        a = mpimg.imread(video_dir / name)
        if a.ndim == 3:
            a = a[..., 0]
        return a > 0.5
    paved = mask("paved_mask.png")
    walk = mask("walkable_mask.png")
    z = np.load(video_dir / "peds.npz")
    ref = mpimg.imread(SDD_RAW / scene / video / "reference.jpg")
    return paved, walk, z, ref


def _tracks(z):
    ids, frame, cx, cy = z["ids"], z["frame"], z["cx"], z["cy"]
    out = []
    for pid in np.unique(ids):
        m = ids == pid
        f, X, Y = frame[m], cx[m], cy[m]
        o = np.argsort(f)
        f, X, Y = f[o], X[o], Y[o]
        sel = np.unique(f // FPS_DS, return_index=True)[1]
        X, Y = X[sel], Y[sel]
        if len(X) >= HIST + 3:
            out.append(np.stack([X, Y], 1))
    return out


def _grid_graph(passable):
    """8-connected grid over passable cells -> (csr adjacency, node_id grid)."""
    h, w = passable.shape
    idx = -np.ones((h, w), int)
    nodes = np.argwhere(passable)
    idx[passable] = np.arange(len(nodes))
    rows, cols, data = [], [], []
    for dy, dx, cost in [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),
                         (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]:
        ys, xs = nodes[:, 0], nodes[:, 1]
        ny, nx = ys + dy, xs + dx
        ok = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        ok &= passable[np.clip(ny, 0, h - 1), np.clip(nx, 0, w - 1)]
        rows.append(idx[ys[ok], xs[ok]]); cols.append(idx[ny[ok], nx[ok]])
        data.append(np.full(ok.sum(), cost))
    A = csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(nodes), len(nodes)))
    return A, idx


def _geo_from(A, idx, cell):
    src = idx[cell[0], cell[1]]
    if src < 0:
        return None
    d = dijkstra(A, indices=src, directed=False)
    return d


def analyse_video(scene, video):
    paved, walk, z, ref = _load(scene, video)
    H, W = paved.shape
    # downsampled passability
    pav_d = paved[::DS, ::DS]
    walk_d = walk[::DS, ::DS]
    hh, ww = walk_d.shape
    A_pav, idx_pav = _grid_graph(pav_d)
    A_walk, idx_walk = _grid_graph(walk_d)

    tracks = _tracks(z)
    # (1) off-pavement visitation
    on_grass_pts = []
    total_pts = 0
    grass_traffic = np.zeros((hh, ww))          # off-pavement traffic per cell
    all_traffic = np.zeros((hh, ww))
    detours = []                                # actual / paved-shortest
    beat_paved = 0
    for tr in tracks:
        xi = np.clip((tr[:, 0] / DS).astype(int), 0, ww - 1)
        yi = np.clip((tr[:, 1] / DS).astype(int), 0, hh - 1)
        onp = paved[np.clip(tr[:, 1].astype(int), 0, H - 1),
                    np.clip(tr[:, 0].astype(int), 0, W - 1)]
        onw = walk[np.clip(tr[:, 1].astype(int), 0, H - 1),
                   np.clip(tr[:, 0].astype(int), 0, W - 1)]
        off = onw & ~onp                        # on grass (walkable, not paved)
        total_pts += len(tr)
        for j in range(len(tr)):
            all_traffic[yi[j], xi[j]] += 1
            if off[j]:
                on_grass_pts.append(tr[j])
                grass_traffic[yi[j], xi[j]] += 1
        # shortcut: actual length vs paved-only shortest path between endpoints
        actual = np.hypot(*np.diff(tr, axis=0).T).sum() / DS
        start, goal = (yi[0], xi[0]), (yi[-1], xi[-1])
        dgoal = _geo_from(A_pav, idx_pav, goal)
        if dgoal is not None and idx_pav[start] >= 0:
            pav_sp = dgoal[idx_pav[start]]
            if np.isfinite(pav_sp) and pav_sp > 1:
                detours.append(actual / pav_sp)
                if actual < 0.98 * pav_sp:
                    beat_paved += 1
    on_grass_pts = np.array(on_grass_pts) if on_grass_pts else np.zeros((0, 2))

    # (3) paving proposal: greedy grass cells by off-pavement traffic
    grass_only = grass_traffic * (walk_d & ~pav_d)
    K = 40
    flat = np.argsort(grass_only.ravel())[::-1][:K]
    prop = np.zeros((hh, ww), bool)
    prop.ravel()[flat] = grass_only.ravel()[flat] > 0
    # utility: off-pavement traffic within radius of proposal vs naive densest-visited cells
    def captured(cells):
        yy, xx = np.where(cells)
        cap = 0.0
        gy, gx = np.where(grass_traffic > 0)
        gt = grass_traffic[gy, gx]
        for k in range(len(gy)):
            if ((yy - gy[k]) ** 2 + (xx - gx[k]) ** 2 <= 4).any():
                cap += gt[k]
        return cap / max(grass_traffic.sum(), 1)
    # naive baseline: pave K random grass cells (mean over seeds) — a fair
    # "place pavement without using desire-path traffic" control.
    grass_cells = np.argwhere(walk_d & ~pav_d)
    rng = np.random.RandomState(0)
    naive_caps = []
    for _ in range(5):
        sel = grass_cells[rng.choice(len(grass_cells), min(K, len(grass_cells)), replace=False)]
        nb = np.zeros((hh, ww), bool)
        nb[sel[:, 0], sel[:, 1]] = True
        naive_caps.append(captured(nb))
    util_prop = captured(prop)
    util_naive = float(np.mean(naive_caps))

    metrics = dict(
        scene=scene, video=video, n_tracks=len(tracks),
        off_pavement_rate=float(len(on_grass_pts) / max(total_pts, 1)),
        median_detour_ratio=float(np.median(detours)) if detours else None,
        frac_beat_paved=float(beat_paved / max(len(detours), 1)) if detours else None,
        paving_capture_proposed=float(util_prop),
        paving_capture_naive=float(util_naive),
    )
    return metrics, dict(ref=ref, paved=paved, walk=walk, H=H, W=W, DS=DS,
                         on_grass=on_grass_pts, grass_traffic=grass_traffic,
                         prop=prop, pav_d=pav_d, walk_d=walk_d)


def figure(scene, video, m, R):
    H, W, DS = R["H"], R["W"], R["DS"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.4))
    # (1) desire-path map: off-pavement hotspots on the reference
    ax[0].imshow(R["ref"])
    if len(R["on_grass"]):
        ax[0].hexbin(R["on_grass"][:, 0], R["on_grass"][:, 1], gridsize=40,
                     cmap="autumn", mincnt=1, alpha=0.75)
    ax[0].set_title(f"(1) Desire-path map: off-pavement visits\n{scene} — off-pave rate {m['off_pavement_rate']*100:.1f}%")
    # (2) walkable vs paved surface + trajectories
    surf = np.zeros((*R["paved"].shape, 3))
    surf[R["walk"] & ~R["paved"]] = (0.35, 0.7, 0.35)   # grass
    surf[R["paved"]] = (0.6, 0.6, 0.6)                   # paved
    ax[1].imshow(surf)
    if len(R["on_grass"]):
        ax[1].scatter(R["on_grass"][:, 0], R["on_grass"][:, 1], s=2, c="#c0142c", alpha=.5)
    dr = m["median_detour_ratio"]; fb = m["frac_beat_paved"]
    ax[1].set_title(f"(2) Surface + grass-cutting steps (red)\nmedian actual/paved-shortest {dr:.2f}, {fb*100:.0f}% beat paved net")
    # (3) paving proposal
    ax[2].imshow(R["ref"])
    yy, xx = np.where(R["prop"])
    ax[2].scatter(xx * DS, yy * DS, s=60, marker="s", c="#1f6feb",
                  edgecolors="white", linewidths=0.5, label="proposed paving")
    ax[2].set_title(f"(3) Paving proposal (greedy)\ncaptures {m['paving_capture_proposed']*100:.0f}% off-pave traffic vs naive {m['paving_capture_naive']*100:.0f}%")
    ax[2].legend(loc="lower right", fontsize=8)
    for a in ax:
        a.set_xlim(0, W); a.set_ylim(H, 0); a.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "desirepath.png", dpi=105, bbox_inches="tight")
    plt.close(fig)


def main():
    scene, video = "hyang", "video0"
    print(f"[desire-path read-outs] {scene}/{video}")
    m, R = analyse_video(scene, video)
    figure(scene, video, m, R)
    # aggregate off-pavement rate across all scenes/videos (metric only)
    agg = {}
    for sc in ["hyang", "little", "quad", "gates"]:
        rates = []
        for vd in sorted((SDD / sc).glob("video*")):
            mm, _ = analyse_video(sc, vd.name)
            rates.append(mm["off_pavement_rate"])
        agg[sc] = float(np.mean(rates))
    out = dict(headline=m, off_pavement_rate_by_scene=agg)
    (RES / "readouts_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("saved reports/figures/desirepath.png")


if __name__ == "__main__":
    main()
