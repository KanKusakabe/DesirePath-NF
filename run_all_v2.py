"""DesirePath-NF v2: leave-one-scene-out TRANSFER test (the P1 core).

Headline question (v1 weakness #2): on a held-out scene, does the local-terrain
model beat the geometry-blind control while the v1-style absolute-position model
collapses?  Also: does shuffling the terrain patch destroy the terrain model's
edge (confound control)?

Run:  uv run python run_all_v2.py
Out:  results/metrics_v2.json, reports/figures/metrics_v2.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from desirepath import data_v2 as D
from desirepath import pipeline_v2 as P

HERE = Path(__file__).resolve().parent
FIG = HERE / "reports" / "figures"
RES = HERE / "results"
FIG.mkdir(parents=True, exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)

SCENES = D.SCENES                      # ["hyang","little","quad","gates"]
SIDX = {s: i for i, s in enumerate(SCENES)}
CAP = 12000                            # per-scene train cap (balance + speed)
EPOCHS = 25
TARGET = "y_fut"                       # K-step-ahead occupancy target (reformulation B)
RNG = np.random.RandomState(0)


def _cap(o, si, n):
    idx = RNG.choice(len(o[TARGET]), min(n, len(o[TARGET])), replace=False)
    d = {k: o[k][idx] for k in ("hist", "patch", "stat_geom", "y", "y_fut", "pos_norm")}
    d["scene_idx"] = np.full(len(idx), si, np.int64)
    return d


def _pack(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _patch_permuted(d):
    """Break patch<->sample correspondence (train AND eval) -> terrain-content control."""
    q = dict(d)
    q["patch"] = d["patch"][RNG.permutation(len(d["patch"]))]
    return q


def loso(scene_data):
    rows = []
    for test in SCENES:
        train = _pack([_cap(scene_data[s], SIDX[s], CAP) for s in SCENES if s != test])
        ev = _cap(scene_data[test], SIDX[test], 10 ** 9)   # full held-out scene
        row = {"scene": test, "n_test": int(len(ev[TARGET]))}
        for mode in ("terrain", "blind", "abspos"):
            m, mu, sd = P.train_flow_v2(train, mode=mode, n_scene=len(SCENES),
                                        epochs=EPOCHS, target_key=TARGET)
            row[mode] = P.nll_flow_v2(m, ev, mu, sd, target_key=TARGET)
        # confound: destroy terrain content (permute patches at train AND eval);
        # if terrain content matters, real "terrain" beats this.
        tr_sh, ev_sh = _patch_permuted(train), _patch_permuted(ev)
        m, mu, sd = P.train_flow_v2(tr_sh, mode="terrain", n_scene=len(SCENES),
                                    epochs=EPOCHS, target_key=TARGET)
        row["terrain_shuffled"] = P.nll_flow_v2(m, ev_sh, mu, sd, target_key=TARGET)
        row["geodesic"] = P.nll_geodesic(train, ev, target_key=TARGET)
        row["gauss"] = P.nll_gauss(train[TARGET], ev[TARGET])
        print("  LOSO", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
        rows.append(row)
    return rows


def figure(rows):
    keys = ["terrain", "blind", "abspos", "terrain_shuffled", "geodesic", "gauss"]
    labels = ["terrain\n(local geom)", "blind", "abspos\n(v1)", "terrain\nshuffled", "geodesic", "gauss"]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    colors = ["#2a7", "#7a2", "#c33", "#999", "#a72", "#bbb"]
    ax[0].bar(labels, [means[k] for k in keys], color=colors)
    ax[0].set_ylabel("held-out NLL (nats/step, lower=better)")
    ax[0].set_title("Leave-one-scene-out mean NLL")
    for i, k in enumerate(keys):
        ax[0].text(i, means[k], f"{means[k]:.2f}", ha="center", va="bottom", fontsize=8)
    # per-scene terrain vs blind vs abspos (transfer robustness)
    x = np.arange(len(rows)); w = 0.27
    for j, (k, c) in enumerate(zip(["terrain", "blind", "abspos"], ["#2a7", "#7a2", "#c33"])):
        ax[1].bar(x + (j - 1) * w, [r[k] for r in rows], w, label=k, color=c)
    ax[1].set_xticks(x); ax[1].set_xticklabels([r["scene"] for r in rows])
    ax[1].set_ylabel("held-out NLL"); ax[1].set_title("Per held-out scene (transfer)")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "metrics_v2.png", dpi=110); plt.close(fig)
    return means


def main():
    print("[1/3] loading scene samples (cached)")
    scene_data = {s: D.build_scene(s) for s in SCENES}
    for s in SCENES:
        print(f"  {s}: {len(scene_data[s]['y'])} samples")
    print("[2/3] leave-one-scene-out transfer test")
    rows = loso(scene_data)
    print("[3/3] figure + metrics_v2.json")
    means = figure(rows)
    # Go-gate checks
    per = {r["scene"]: r for r in rows}
    terrain_beats_blind = {s: per[s]["terrain"] < per[s]["blind"] for s in SCENES}
    abspos_collapse = {s: per[s]["abspos"] > per[s]["blind"] for s in SCENES}
    out = dict(loso=rows, loso_mean=means, cap=CAP, epochs=EPOCHS,
               scenes=SCENES,
               gate_terrain_beats_blind_all=all(terrain_beats_blind.values()),
               gate_terrain_beats_blind_by_scene=terrain_beats_blind,
               abspos_collapse_by_scene=abspos_collapse)
    (RES / "metrics_v2.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: means[k] for k in means}, indent=2))
    print("terrain<blind all folds:", out["gate_terrain_beats_blind_all"], terrain_beats_blind)
    print("done.")


if __name__ == "__main__":
    main()
