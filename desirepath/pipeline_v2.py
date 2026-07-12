"""DesirePath-NF v2 model: p(next step | history, LOCAL TERRAIN, goal/social).

Three conditioning modes share one flow head so held-out NLL is comparable:
  - "terrain": GRU(history) + TerrainCNN(heading-aligned 3ch patch) + MLP(goal/social)
               -> the transfer-safe model; grammar binds to local geometry.
  - "blind"  : GRU(history) + MLP(goal/social)         [geometry-blind control]
  - "abspos" : GRU + MLP(goal/social + abs position + scene embedding)  [v1-style]
               -> reproduces v1's memorise-absolute-coords failure mode.

The scientific question (v1 weakness #2): does "terrain" beat "blind" on a
held-out scene where "abspos" collapses?  Port of Layout-NF's occupancy CNN +
motionsim/MERL-NF flow skeleton.  torch.set_num_threads(1) avoids the v1 segfault.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import zuko

torch.set_num_threads(1)
torch.manual_seed(0)
np.random.seed(0)

DIM = 2            # (d_forward, d_lateral)
PATCH_CH = 3       # [is_paved, dist_to_obstacle, dist_to_paved]


class TerrainCNN(nn.Module):
    """3 x 32 x 32 heading-aligned terrain crop -> feature vector.

    Deliberately SMALL + dropout: an over-parameterized CNN memorizes
    training-scene terrain *appearance* that does not transfer (verified: the
    32-wide no-dropout version was beaten by its own shuffled-patch control on
    held-out scenes).  This capacity-limited, regularized version makes the
    local-geometry signal transfer instead of overfitting.
    """

    def __init__(self, out_dim=24, drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(PATCH_CH, 12, 3, stride=2, padding=1), nn.ReLU(),  # 16x16
            nn.Conv2d(12, 16, 3, stride=2, padding=1), nn.ReLU(),        # 8x8
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),        # 4x4
            nn.Flatten(), nn.Dropout(drop), nn.Linear(16 * 16, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, patch):           # [B,3,32,32]
        return self.net(patch)


class CondEncV2(nn.Module):
    def __init__(self, mode="terrain", n_scene=4, gru=32, cnn=24, out=64,
                 stat_dim=4):
        super().__init__()
        assert mode in ("terrain", "blind", "abspos")
        self.mode = mode
        self.gru = nn.GRU(DIM, gru, batch_first=True)
        self.cnn = TerrainCNN(cnn) if mode == "terrain" else None
        self.scene = nn.Embedding(n_scene, 4) if mode == "abspos" else None
        in_dim = gru + stat_dim
        if mode == "terrain":
            in_dim += cnn
        if mode == "abspos":
            in_dim += 2 + 4                      # abs pos + scene embedding
        self.mlp = nn.Sequential(nn.Linear(in_dim, out), nn.ReLU(),
                                 nn.Linear(out, out), nn.ReLU())
        self.out = out

    def forward(self, hist, patch, stat, pos, scene_idx):
        _, h = self.gru(hist)
        parts = [h.squeeze(0), stat]
        if self.mode == "terrain":
            parts.append(self.cnn(patch))
        if self.mode == "abspos":
            parts.append(pos)
            parts.append(self.scene(scene_idx))
        return self.mlp(torch.cat(parts, -1))


class FlowModelV2(nn.Module):
    def __init__(self, mode="terrain", n_scene=4):
        super().__init__()
        self.enc = CondEncV2(mode=mode, n_scene=n_scene)
        self.flow = zuko.flows.NSF(features=DIM, context=self.enc.out,
                                   transforms=3, hidden_features=(64, 64))

    def log_prob(self, hist, patch, stat, pos, scene_idx, y):
        c = self.enc(hist, patch, stat, pos, scene_idx)
        return self.flow(c).log_prob(y)

    def sample(self, hist, patch, stat, pos, scene_idx, n=1):
        c = self.enc(hist, patch, stat, pos, scene_idx)
        return self.flow(c).sample((n,))


# ----------------------------------------------------------------------------- data plumbing
def normalize_fit(Y):
    return Y.mean(0), Y.std(0) + 1e-6


def _batch_tensors(D, idx, mu, sd, target_key="y", patch_shuffle=False):
    """Slice a batch of numpy arrays -> tensors (patch float16 -> float32)."""
    hist = torch.as_tensor(D["hist"][idx], dtype=torch.float32)
    p = D["patch"][idx]
    if patch_shuffle:                                   # confound control
        p = p[np.random.permutation(len(p))]
    patch = torch.as_tensor(np.asarray(p, np.float32))
    stat = torch.as_tensor(D["stat_geom"][idx], dtype=torch.float32)
    pos = torch.as_tensor(D["pos_norm"][idx], dtype=torch.float32)
    sc = torch.as_tensor(D["scene_idx"][idx], dtype=torch.long)
    y = torch.as_tensor((D[target_key][idx] - mu) / sd, dtype=torch.float32)
    return hist, patch, stat, pos, sc, y


def train_flow_v2(D, mode="terrain", n_scene=4, epochs=20, bs=512, lr=3e-3,
                  weight_decay=1e-4, patch_shuffle=False, target_key="y"):
    mu, sd = normalize_fit(D[target_key])
    m = FlowModelV2(mode=mode, n_scene=n_scene)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(D[target_key])
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            hist, patch, stat, pos, sc, y = _batch_tensors(D, idx, mu, sd, target_key, patch_shuffle)
            opt.zero_grad()
            loss = -m.log_prob(hist, patch, stat, pos, sc, y).mean()
            loss.backward()
            opt.step()
    return m, mu, sd


def nll_flow_v2(m, D, mu, sd, target_key="y", bs=2048):
    m.eval()                              # disable dropout for evaluation
    n = len(D[target_key]); tot = 0.0
    with torch.no_grad():
        for i in range(0, n, bs):
            idx = np.arange(i, min(i + bs, n))
            hist, patch, stat, pos, sc, y = _batch_tensors(D, idx, mu, sd, target_key)
            tot += float(-m.log_prob(hist, patch, stat, pos, sc, y).sum())
    return tot / n


# ----------------------------------------------------------------------------- baselines
def nll_gauss(train_y, ev_y):
    mu, sd = train_y.mean(0), train_y.std(0) + 1e-6
    var = sd ** 2
    d = ev_y - mu
    ll = -0.5 * (np.log(2 * math.pi * var) + d ** 2 / var).sum(1)
    return float(-ll.mean())


def nll_geodesic(train_D, ev_D, target_key="y"):
    """Straight-to-goal reference: predict displacement pointing at the goal
    (goal bearing = stat_geom[:,0:2]) at mean speed; Gaussian residual."""
    Ytr = train_D[target_key]; sp = np.linalg.norm(Ytr, axis=1, keepdims=True).mean()
    pred_tr = train_D["stat_geom"][:, 0:2] * sp
    res = Ytr - pred_tr
    g_mu, g_sd = res.mean(0), res.std(0) + 1e-6
    var = g_sd ** 2
    d = (ev_D[target_key] - ev_D["stat_geom"][:, 0:2] * sp) - g_mu
    ll = -0.5 * (np.log(2 * math.pi * var) + d ** 2 / var).sum(1)
    return float(-ll.mean())
