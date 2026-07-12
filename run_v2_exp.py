"""DesirePath-NF v2 — experiments ② (residual target) and ③ (decision-point eval).

Motivation: on the raw K-step target, geometry-blind wins because goal-directed
motion dominates and the distribution is ~unimodal (a flow / terrain add little).
Two changes to make the flow's strengths matter:

  ②  RESIDUAL target = (actual K-step endpoint) - (goal-directed endpoint).
      Subtracts the dominant goal-seeking component; what remains is the
      terrain/social DEVIATION -- smaller, multi-modal (detour left vs right).

  ③  DECISION-POINT evaluation: split held-out steps into
        decision = large deviation from the goal direction (top 25%)  -> a real
                   "which way around" choice, where the future is multi-modal;
        straight = small deviation (bottom 50%)                       -> unimodal.
      A conditional flow with terrain should beat both the geometry-blind flow
      and a Gaussian *at decision points* (exact multi-modal density), even if
      it ties on straight walking.

Run:  uv run python run_v2_exp.py
Out:  results/metrics_exp.json, reports/figures/decision_points.png
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

SCENES = D.SCENES
SIDX = {s: i for i, s in enumerate(SCENES)}
CAP = 12000
EPOCHS = 20
DEC_Q = 75          # decision points = deviation >= this percentile
STR_Q = 50          # straight       = deviation <= this percentile
RNG = np.random.RandomState(0)

KEYS = ("hist", "patch", "stat_geom", "y", "y_fut", "pos_norm")


def _cap(o, si, n):
    idx = RNG.choice(len(o["y_fut"]), min(n, len(o["y_fut"])), replace=False)
    d = {k: o[k][idx] for k in KEYS}
    d["scene_idx"] = np.full(len(idx), si, np.int64)
    return d


def _pack(parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def _add_residual(d, sp8):
    """residual = K-step endpoint - goal-directed endpoint (magnitude sp8)."""
    ghat = d["stat_geom"][:, 0:2]                      # (cos,sin) goal bearing, egocentric
    d["y_resid"] = (d["y_fut"] - sp8 * ghat).astype(np.float32)
    return d


def _slice(d, mask):
    out = {k: v[mask] for k, v in d.items()}
    return out


def _eval_subsets(m, ev, mu, sd, target, masks):
    return {name: P.nll_flow_v2(m, _slice(ev, mk), mu, sd, target_key=target)
            for name, mk in masks.items()}


def _gauss_subsets(train, ev, target, masks):
    return {name: P.nll_gauss(train[target], _slice(ev, mk)[target])
            for name, mk in masks.items()}


def loso(scene_data):
    rows = []
    for test in SCENES:
        train = _pack([_cap(scene_data[s], SIDX[s], CAP) for s in SCENES if s != test])
        ev = _cap(scene_data[test], SIDX[test], 10 ** 9)
        # goal-directed K-step speed fit on TRAIN only
        sp8 = float(np.linalg.norm(train["y_fut"], axis=1).mean())
        _add_residual(train, sp8); _add_residual(ev, sp8)
        # decision-point split from deviation-from-goal (= |residual|) on held-out
        dev = np.linalg.norm(ev["y_resid"], axis=1)
        dth, sth = np.percentile(dev, DEC_Q), np.percentile(dev, STR_Q)
        masks = {"all": np.ones(len(dev), bool),
                 "decision": dev >= dth,
                 "straight": dev <= sth}
        row = {"scene": test, "sp8": sp8,
               "n": {k: int(mk.sum()) for k, mk in masks.items()}}
        for target in ("y_fut", "y_resid"):
            for mode in ("terrain", "blind"):
                m, mu, sd = P.train_flow_v2(train, mode=mode, n_scene=len(SCENES),
                                            epochs=EPOCHS, target_key=target)
                row[f"{target}:{mode}"] = _eval_subsets(m, ev, mu, sd, target, masks)
            row[f"{target}:gauss"] = _gauss_subsets(train, ev, target, masks)
        print("  fold", test, "done; decision n =", row["n"]["decision"])
        rows.append(row)
    return rows


def _mean_over_folds(rows, key, subset):
    """Sample-weighted mean so a tiny fold (e.g. quad) can't dominate."""
    v = np.array([r[key][subset] for r in rows])
    w = np.array([r["n"][subset] for r in rows], float)
    return float((v * w).sum() / w.sum())


def figure(rows):
    subsets = ["straight", "all", "decision"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for pi, (target, title) in enumerate([("y_fut", "Raw K-step target"),
                                          ("y_resid", "② Residual target (desire = actual − goal-directed)")]):
        models = [(f"{target}:terrain", "terrain", "#2a7"),
                  (f"{target}:blind", "blind", "#7a2"),
                  (f"{target}:gauss", "gauss", "#bbb")]
        x = np.arange(len(subsets)); w = 0.26
        for j, (key, lab, c) in enumerate(models):
            vals = [_mean_over_folds(rows, key, s) for s in subsets]
            ax[pi].bar(x + (j - 1) * w, vals, w, label=lab, color=c)
        ax[pi].set_xticks(x); ax[pi].set_xticklabels(subsets)
        ax[pi].set_title(title, fontsize=10)
        ax[pi].set_ylabel("held-out NLL (lower=better)")
        ax[pi].axvline(2 - 0.5, ls=":", c="#999", lw=0.8)
        ax[pi].legend(fontsize=8)
    fig.suptitle("③ Decision points (deviation from goal) — where terrain + flow should earn their keep", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "decision_points.png", dpi=110); plt.close(fig)


def main():
    print("[1/3] load"); scene_data = {s: D.build_scene(s) for s in SCENES}
    print("[2/3] LOSO with residual + decision-point split"); rows = loso(scene_data)
    print("[3/3] figure + metrics_exp.json"); figure(rows)
    summary = {}
    for target in ("y_fut", "y_resid"):
        summary[target] = {}
        for sub in ("all", "decision", "straight"):
            t = _mean_over_folds(rows, f"{target}:terrain", sub)
            b = _mean_over_folds(rows, f"{target}:blind", sub)
            g = _mean_over_folds(rows, f"{target}:gauss", sub)
            summary[target][sub] = dict(terrain=t, blind=b, gauss=g,
                                        terrain_minus_blind=t - b, terrain_minus_gauss=t - g)
    out = dict(folds=rows, summary=summary, cap=CAP, epochs=EPOCHS,
               dec_q=DEC_Q, str_q=STR_Q, scenes=SCENES)
    (RES / "metrics_exp.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(summary, indent=2))
    print("done.")


if __name__ == "__main__":
    main()
