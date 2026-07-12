"""Intuitive picture of WHY the flow beats a Gaussian at decision points.

At a decision point the 8-step future branches: people go LEFT or RIGHT around
the obstacle/corner, rarely straight through the middle.  So the deviation
(residual = actual endpoint - goal-directed endpoint) is BIMODAL.  A Gaussian
can only place one blob (on the empty middle); the conditional flow places mass
on both branches.  Three panels:

  (A) where decisions happen  : decision-point locations on the reference.
  (B) actual futures branch   : real residual endpoints (egocentric, forward=up)
                                -> two lobes; the best-fit Gaussian ellipse sits
                                   on the empty middle.
  (C) flow captures both      : residuals SAMPLED from the terrain flow (trained
                                   on the other 3 scenes, held-out on hyang)
                                   -> reproduces both lobes.

Run:  uv run python -m desirepath.decision_viz
Out:  reports/figures/decision_viz.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from desirepath import data_v2 as D
from desirepath import pipeline_v2 as P

FIG = D.HERE / "reports" / "figures"
SCENE = "hyang"
OTHERS = ["little", "quad", "gates"]
DEC_Q = 75


def _residual(d, sp8):
    return (d["y_fut"] - sp8 * d["stat_geom"][:, 0:2]).astype(np.float32)


def _gauss_ellipses(ax, mu, cov, color="#333"):
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    for k, a in [(1, 0.9), (2, 0.5)]:
        w, h = 2 * k * np.sqrt(vals)
        ax.add_patch(Ellipse(mu, w, h, angle=ang, fill=False, ec=color, lw=2, alpha=a, ls="--"))


def main():
    # --- training scenes -> sp8 and a terrain flow on the residual target ---
    train = {k: np.concatenate([D.build_scene(s)[k] for s in OTHERS])
             for k in ("hist", "patch", "stat_geom", "y", "y_fut", "pos_norm")}
    train["scene_idx"] = np.zeros(len(train["y"]), np.int64)
    sp8 = float(np.linalg.norm(train["y_fut"], axis=1).mean())
    train["y_resid"] = _residual(train, sp8)
    # cap for speed
    rng = np.random.RandomState(0)
    idx = rng.choice(len(train["y_resid"]), min(24000, len(train["y_resid"])), replace=False)
    tr = {k: v[idx] for k, v in train.items()}
    print(f"training terrain flow on residual ({len(tr['y_resid'])} samples)...")
    m, mu, sd = P.train_flow_v2(tr, mode="terrain", n_scene=4, epochs=20, target_key="y_resid")
    m.eval()

    # --- held-out hyang: residual + decision mask ---
    H = D.build_scene(SCENE)
    resid = _residual(H, sp8)
    dev = np.linalg.norm(resid, axis=1)
    dec = dev >= np.percentile(dev, DEC_Q)
    act = resid[dec]                                   # actual decision-point residuals

    # --- sample the flow at held-out decision contexts ---
    di = np.where(dec)[0]
    di = rng.choice(di, min(4000, len(di)), replace=False)
    with torch.no_grad():
        hist = torch.as_tensor(H["hist"][di], dtype=torch.float32)
        patch = torch.as_tensor(np.asarray(H["patch"][di], np.float32))
        stat = torch.as_tensor(H["stat_geom"][di], dtype=torch.float32)
        pos = torch.as_tensor(H["pos_norm"][di], dtype=torch.float32)
        sc = torch.zeros(len(di), dtype=torch.long)
        s = m.sample(hist, patch, stat, pos, sc, n=4)      # (4, B, 2) normalized
    samp = (s.reshape(-1, 2).numpy() * sd + mu)            # flow residual samples

    # Gaussian best-fit to the actual decision residuals (the fair single-blob model)
    gmu, gcov = act.mean(0), np.cov(act.T)

    # --- panel A: where decisions happen (single video for a clean overlay) ---
    import matplotlib.image as mpimg
    from desirepath.prep_sdd import SDD_RAW
    V = D._build_video(D.SDD / SCENE / "video0")
    vr = _residual(V, sp8); vdev = np.linalg.norm(vr, axis=1)
    vdec = vdev >= np.percentile(vdev, DEC_Q)
    ref = mpimg.imread(SDD_RAW / SCENE / "video0" / "reference.jpg")
    Hh, Ww = ref.shape[:2]
    px = (V["pos_norm"][vdec, 0] + 1) / 2 * Ww
    py = (V["pos_norm"][vdec, 1] + 1) / 2 * Hh

    lim = 12
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
    ax[0].imshow(ref)
    ax[0].hexbin(px, py, gridsize=34, cmap="plasma", mincnt=1, alpha=0.75)
    ax[0].set_xlim(0, Ww); ax[0].set_ylim(Hh, 0); ax[0].axis("off")
    ax[0].set_title("(A) Where decisions happen\n(steps that deviate most from the goal direction)")

    for a, data, ttl in [(ax[1], act, "(B) Actual futures BRANCH (bimodal)\nreal residual endpoints — two lobes, empty middle"),
                         (ax[2], samp, "(C) The flow captures BOTH branches\nsamples from terrain flow (held-out on hyang)")]:
        a.hexbin(data[:, 1], data[:, 0], gridsize=40, cmap="magma", mincnt=1, extent=(-lim, lim, -lim, lim))
        _gauss_ellipses(a, [gmu[1], gmu[0]], np.array([[gcov[1, 1], gcov[1, 0]], [gcov[0, 1], gcov[0, 0]]]), "#39d")
        a.scatter([0], [0], c="w", s=40, marker="o", edgecolors="k", zorder=5)
        a.annotate("now", (0, 0), (0.6, -1.6), color="w", fontsize=8)
        a.arrow(0, 0, 0, 3, head_width=0.6, color="w", alpha=0.7)
        a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_aspect("equal")
        a.set_xlabel("lateral deviation  (left  −   +  right)"); a.set_ylabel("forward")
        a.set_title(ttl, fontsize=9.5)
    ax[1].text(-lim + 0.5, lim - 1.5, "dashed = best-fit Gaussian\n(sits on the empty middle)", color="#39d", fontsize=8)
    fig.suptitle("Why the normalizing flow wins at decision points: the future is two branches, not one blob", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "decision_viz.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    lat = act[:, 1]
    print(f"decision residual lateral: left {np.mean(lat < -1)*100:.0f}% / middle {np.mean(np.abs(lat) <= 1)*100:.0f}% / right {np.mean(lat > 1)*100:.0f}%")
    print("saved reports/figures/decision_viz.png")


if __name__ == "__main__":
    main()
