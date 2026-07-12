"""DesirePath-NF: build everything (LOSO evaluation, full-model read-outs,
figures, metrics.json, HTML report). Run:  uv run python run_all.py"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from desirepath import pipeline as P

FIG = P.FIG
P.RESULTS.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def load_all():
    scenes, order = {}, list(P.SCENES)
    for si, (name, fn) in enumerate(P.SCENES.items()):
        peds, frames = P.load_scene(P.DATA / fn)
        bbox = P.scene_bbox(peds)
        Xh, Xs, Y = P.build_samples(peds, frames, si, bbox)
        scenes[name] = dict(peds=peds, bbox=bbox, Xh=Xh, Xs=Xs, Y=Y, idx=si)
        print(f"  {name}: {len(Y)} steps, {len(peds)} peds")
    return scenes, order


def pack(scenes, names):
    Xh = np.concatenate([scenes[n]["Xh"] for n in names])
    Xs = np.concatenate([scenes[n]["Xs"] for n in names])
    Y = np.concatenate([scenes[n]["Y"] for n in names])
    return Xh, Xs, Y


def loso(scenes, order):
    rows = []
    for test in order:
        train_names = [n for n in order if n != test]
        Xh, Xs, Y = pack(scenes, train_names)
        mu, sd = P.normalize_fit(Y)
        tr = (Xh, Xs, Y, mu, sd)
        ev = (scenes[test]["Xh"], scenes[test]["Xs"], scenes[test]["Y"])
        # models
        m_full = P.train_flow(tr, use_pos=True, n_scene=len(order))
        m_blind = P.train_flow(tr, use_pos=False, n_scene=len(order))
        Yn_tr = (Y - mu) / sd
        Yn_ev = (ev[2] - mu) / sd
        row = dict(
            scene=test,
            flow_pos=P.nll_flow(m_full, ev, mu, sd),
            flow_nopos=P.nll_flow(m_blind, ev, mu, sd),
            gauss=P.nll_gauss(Yn_tr, Yn_ev),
            geodesic=P.nll_geodesic((Xh, Xs, Y), ev, mu, sd),
            n_test=int(len(ev[2])),
        )
        print("  LOSO", row)
        rows.append(row)
    return rows


def train_full(scenes, order):
    Xh, Xs, Y = pack(scenes, order)
    mu, sd = P.normalize_fit(Y)
    m = P.train_flow((Xh, Xs, Y, mu, sd), use_pos=True, epochs=40, n_scene=len(order))
    return m, mu, sd


def readout_figs(m, mu, sd, scenes, order):
    """Flow field + surprise map + desire-vs-geodesic deviation for one dense scene."""
    name = "students03"
    S = scenes[name]
    peds, bbox = S["peds"], S["bbox"]
    xmin, xmax, ymin, ymax = bbox

    # --- surprise map (from actual steps) ---
    with torch.no_grad():
        ll = m.log_prob(P.to_t(S["Xh"]), P.to_t(S["Xs"]),
                        P.to_t((S["Y"] - mu) / sd)).numpy()
    surprise = -ll
    pos = np.stack([S["Xs"][:, 0], S["Xs"][:, 1]], 1)  # normalized [-1,1]
    px = (pos[:, 0] + 1) / 2 * (xmax - xmin) + xmin
    py = (pos[:, 1] + 1) / 2 * (ymax - ymin) + ymin

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    # (1) trajectories = observed flow
    for xy in peds.values():
        ax[0].plot(xy[:, 0], xy[:, 1], lw=0.6, alpha=0.35, color="#3b6ea5")
    ax[0].set_title("Observed pedestrian flow (students03)")
    # (2) surprise heatmap
    hb = ax[1].hexbin(px, py, C=surprise, gridsize=28, cmap="magma", reduce_C_function=np.mean)
    fig.colorbar(hb, ax=ax[1], label="surprise  -log p(step)")
    ax[1].set_title("Surprise map (decision / off-corridor)")
    # (3) desire-vs-geodesic deviation field
    #     compare model's expected step direction vs straight-to-goal bearing
    with torch.no_grad():
        smp = m.sample(P.to_t(S["Xh"]), P.to_t(S["Xs"]), n=8).numpy()  # (8,N,2)
    exp_step = smp.mean(0) * sd + mu                                    # (N,2) egocentric
    # geodesic egocentric direction is stat[:,2:4] = (cos,sin) of goal bearing
    geo = S["Xs"][:, 2:4]
    # deviation angle between expected step and geodesic (egocentric frame)
    a_exp = np.arctan2(exp_step[:, 1], exp_step[:, 0])
    a_geo = np.arctan2(geo[:, 1], geo[:, 0])
    dev = np.arctan2(np.sin(a_exp - a_geo), np.cos(a_exp - a_geo))
    hb2 = ax[2].hexbin(px, py, C=np.abs(dev), gridsize=28, cmap="viridis", reduce_C_function=np.mean)
    fig.colorbar(hb2, ax=ax[2], label="|desire - geodesic|  (rad)")
    ax[2].set_title("Desire vs straight-to-goal deviation")
    for a in ax:
        a.set_aspect("equal"); a.set_xlabel("x [m]"); a.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(FIG / "readouts.png", dpi=110)
    plt.close(fig)
    return float(np.mean(np.abs(dev)))


def metrics_fig(rows):
    labels = ["flow_pos", "flow_nopos", "geodesic", "gauss"]
    means = {k: np.mean([r[k] for r in rows]) for k in labels}
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2a7", "#7a2", "#a72", "#999"]
    ax.bar(labels, [means[k] for k in labels], color=colors)
    ax.set_ylabel("held-out NLL (nats/step, lower=better)")
    ax.set_title("Leave-one-scene-out NLL (mean over 5 scenes)")
    for i, k in enumerate(labels):
        ax.text(i, means[k], f"{means[k]:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "metrics.png", dpi=110); plt.close(fig)
    return means


def main():
    print("[1/4] loading scenes"); scenes, order = load_all()
    print("[2/4] leave-one-scene-out"); rows = loso(scenes, order)
    print("[3/4] full model + read-outs")
    m, mu, sd = train_full(scenes, order)
    mean_dev = readout_figs(m, mu, sd, scenes, order)
    means = metrics_fig(rows)
    print("[4/4] writing metrics.json")
    out = dict(loso=rows, loso_mean=means, mean_desire_deviation_rad=mean_dev,
               n_scenes=len(order), scenes=order)
    (P.RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(means, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
