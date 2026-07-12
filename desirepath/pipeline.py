"""End-to-end pipeline for DesirePath-NF.

Data:   ETH/UCY obsmat.txt (cols: frame, id, x, z, y, vx, vz, vy) -> ground (x,y).
Target: egocentric one-step increment (d_forward, d_lateral) in the local heading
        frame (meters per annotation step).
Cond c: GRU over the increment history + [normalized position, goal-bearing
        (cos,sin), local crowd density, scene embedding].
Model:  zuko conditional Neural Spline Flow over the 2-D increment.

Read-outs: flow field, surprise map, and the actual-vs-geodesic deviation field
(a paved-map-free proxy for "desire paths"). Everything is scored against honest
baselines with leave-one-scene-out generalization.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import zuko

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIG = HERE / "reports" / "figures"

SCENES = {
    "eth":       "ETH_seq_eth_obsmat.txt",
    "hotel":     "ETH_seq_hotel_obsmat.txt",
    "zara01":    "UCY_zara01_obsmat.txt",
    "zara02":    "UCY_zara02_obsmat.txt",
    "students03":"UCY_students03_obsmat.txt",
}
HIST = 8          # history length for the GRU
DIM = 2           # (d_forward, d_lateral)
torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------------- data
def load_scene(path: Path):
    """Return {ped_id: Nx2 array of (x,y) sorted by frame}."""
    raw = np.loadtxt(path)
    peds = {}
    for pid in np.unique(raw[:, 1]):
        rows = raw[raw[:, 1] == pid]
        rows = rows[np.argsort(rows[:, 0])]
        xy = rows[:, [2, 4]]
        if len(xy) >= HIST + 3:
            peds[int(pid)] = xy
    frames = raw[:, [0, 1, 2, 4]]  # frame, id, x, y  (for crowd density)
    return peds, frames


def _rot(dx, dy, heading):
    c, s = math.cos(-heading), math.sin(-heading)
    return c * dx - s * dy, s * dx + c * dy


def build_samples(peds, frames, scene_idx, bbox):
    """Yield per-step samples: target increment + conditioning features."""
    (xmin, xmax, ymin, ymax) = bbox
    sx, sy = max(xmax - xmin, 1e-3), max(ymax - ymin, 1e-3)
    # crowd density: peds present per frame
    fr_ids = {}
    for f, i, x, y in frames:
        fr_ids.setdefault(f, []).append((x, y))

    X_hist, X_stat, Y = [], [], []
    for pid, xy in peds.items():
        goal = xy[-1]
        # world increments
        d = np.diff(xy, axis=0)                       # (N-1, 2)
        head = np.arctan2(d[:, 1], d[:, 0])           # heading of each step
        # egocentric increments (rotate each step into the *previous* heading)
        ego = np.zeros_like(d)
        for k in range(len(d)):
            h = head[k - 1] if k > 0 else head[k]
            fx, fy = _rot(d[k, 0], d[k, 1], h)
            ego[k] = (fx, fy)
        for k in range(HIST, len(d)):
            hist = ego[k - HIST:k]                     # (HIST, 2)
            pos = xy[k]                                # current position
            h = head[k - 1]
            # goal bearing in egocentric frame
            gx, gy = _rot(goal[0] - pos[0], goal[1] - pos[1], h)
            gb = math.atan2(gy, gx)
            # crowd density within 2 m at the (approx) current frame
            fr = list(fr_ids.keys())
            # cheap density proxy: neighbours of this ped's position among all rows
            X_hist.append(hist)
            X_stat.append([
                2 * (pos[0] - xmin) / sx - 1,
                2 * (pos[1] - ymin) / sy - 1,
                math.cos(gb), math.sin(gb),
                min(np.hypot(goal[0]-pos[0], goal[1]-pos[1]) / 10.0, 3.0),
                float(scene_idx),
            ])
            Y.append(ego[k])
    return (np.array(X_hist, np.float32),
            np.array(X_stat, np.float32),
            np.array(Y, np.float32))


def scene_bbox(peds):
    allxy = np.concatenate(list(peds.values()))
    return (allxy[:, 0].min(), allxy[:, 0].max(),
            allxy[:, 1].min(), allxy[:, 1].max())


# ----------------------------------------------------------------------------- model
class CondEnc(nn.Module):
    def __init__(self, n_scene, stat_dim=6, gru=32, semb=4, out=48, use_pos=True):
        super().__init__()
        self.use_pos = use_pos
        self.gru = nn.GRU(DIM, gru, batch_first=True)
        self.scene = nn.Embedding(n_scene, semb)
        cont = (stat_dim - 1) + semb  # replace scene id with its embedding
        if not use_pos:
            cont -= 2                  # drop position -> confound control
        self.mlp = nn.Sequential(nn.Linear(gru + cont, out), nn.ReLU(),
                                  nn.Linear(out, out), nn.ReLU())
        self.out = out

    def forward(self, hist, stat):
        _, h = self.gru(hist)
        h = h.squeeze(0)
        cont = stat[:, :-1]
        if not self.use_pos:
            cont = cont[:, 2:]
        semb = self.scene(stat[:, -1].long())
        return self.mlp(torch.cat([h, cont, semb], -1))


class FlowModel(nn.Module):
    def __init__(self, n_scene, use_pos=True):
        super().__init__()
        self.enc = CondEnc(n_scene, use_pos=use_pos)
        self.flow = zuko.flows.NSF(features=DIM, context=self.enc.out,
                                   transforms=3, hidden_features=(64, 64))

    def log_prob(self, hist, stat, y):
        c = self.enc(hist, stat)
        return self.flow(c).log_prob(y)

    def sample(self, hist, stat, n=1):
        c = self.enc(hist, stat)
        return self.flow(c).sample((n,))


# ----------------------------------------------------------------------------- train/eval
def normalize_fit(Y):
    mu, sd = Y.mean(0), Y.std(0) + 1e-6
    return mu, sd


def to_t(a):
    return torch.as_tensor(a, dtype=torch.float32)


def train_flow(tr, use_pos=True, epochs=30, bs=256, lr=3e-3, n_scene=5):
    Xh, Xs, Y, mu, sd = tr
    Yn = (Y - mu) / sd
    m = FlowModel(n_scene, use_pos=use_pos)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xh_t, Xs_t, Y_t = to_t(Xh), to_t(Xs), to_t(Yn)
    n = len(Y)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            ll = m.log_prob(Xh_t[idx], Xs_t[idx], Y_t[idx])
            loss = -ll.mean()
            loss.backward()
            opt.step()
    return m


def nll_flow(m, ev, mu, sd):
    Xh, Xs, Y = ev
    Yn = (Y - mu) / sd
    with torch.no_grad():
        ll = m.log_prob(to_t(Xh), to_t(Xs), to_t(Yn))
    return float(-ll.mean())


def nll_gauss(train_Yn, ev_Yn):
    """Position-blind diagonal Gaussian baseline in normalized space."""
    mu, sd = train_Yn.mean(0), train_Yn.std(0) + 1e-6
    var = sd ** 2
    d = ev_Yn - mu
    ll = -0.5 * (np.log(2*math.pi*var) + d**2/var).sum(1)
    return float(-ll.mean())


def nll_geodesic(train, ev, mu, sd):
    """Straight-to-goal baseline: predict the increment that points at the goal
    with the empirical mean speed; Gaussian residual. This is the 'provided /
    geodesic' reference the desire-path deviation is measured against."""
    Xh_tr, Xs_tr, Y_tr = train[0], train[1], train[2]
    # goal direction in egocentric frame = (cos,sin) already in stat[:,2:4]
    sp_tr = np.linalg.norm(Y_tr, axis=1, keepdims=True)
    pred_tr = Xs_tr[:, 2:4] * sp_tr.mean()
    res_tr = (Y_tr - pred_tr - mu) / sd
    g_mu, g_sd = res_tr.mean(0), res_tr.std(0) + 1e-6
    Xh_ev, Xs_ev, Y_ev = ev
    pred_ev = Xs_ev[:, 2:4] * sp_tr.mean()
    d = ((Y_ev - pred_ev - mu) / sd - g_mu)
    var = g_sd ** 2
    ll = -0.5 * (np.log(2*math.pi*var) + d**2/var).sum(1)
    return float(-ll.mean())
