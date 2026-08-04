from __future__ import annotations

import random
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    input_dim: int
    n_classes: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_synthetic_ring(n: int = 6000, seed: int = 271828) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 5, 4)).astype(np.float32)
    s = x.copy()
    mix = np.array(
        [[.55, -.20, .15, .10], [.10, .60, -.25, .15],
         [-.20, .15, .55, .20], [.15, -.10, .20, .60]], dtype=np.float32
    )
    for _ in range(4):
        s = np.tanh((.50 * s + .85 * np.roll(s, 1, 1) - .70 * np.roll(s, -1, 1)) @ mix)
    scores = np.stack([
        s[:, :, 0].mean(1) + .30 * s[:, :, 1].max(1),
        s[:, :, 1].mean(1) - .25 * s[:, :, 2].min(1),
        s[:, :, 2].mean(1) + .25 * s[:, :, 3].std(1),
        s[:, :, 3].mean(1) - .20 * s[:, :, 0].std(1),
    ], axis=1)
    return x.reshape(n, -1), scores.argmax(1).astype(np.int64)


def load_bundle(name: str, split_seed: int = 314159) -> DatasetBundle:
    if name == "digits":
        d = load_digits(); x = d.data.astype(np.float32) / 16.; y = d.target.astype(np.int64)
    elif name == "breast_cancer":
        d = load_breast_cancer(); x = d.data.astype(np.float32); y = d.target.astype(np.int64)
    elif name == "wine":
        d = load_wine(); x = d.data.astype(np.float32); y = d.target.astype(np.int64)
    elif name == "synthetic_ring":
        x, y = make_synthetic_ring()
    else:
        raise ValueError(name)
    xf, xt, yf, yt = train_test_split(x, y, test_size=.20, random_state=split_seed, stratify=y)
    xr, xv, yr, yv = train_test_split(xf, yf, test_size=.20, random_state=split_seed + 1, stratify=yf)
    if name != "digits":
        sc = StandardScaler(); xr = sc.fit_transform(xr).astype(np.float32)
        xv = sc.transform(xv).astype(np.float32); xt = sc.transform(xt).astype(np.float32)
    return DatasetBundle(name, xr, yr, xv, yv, xt, yt, x.shape[1], int(y.max() + 1))


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, classes: int):
        super().__init__(); self.a = nn.GELU()
        self.fc1 = nn.Linear(input_dim, hidden); self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        h = self.a(self.fc1(x)); h = self.a(self.fc2(h)); z = self.out(h)
        return (z, h) if return_features else z


def topology_matrix(k: int, topology: str, seed: int = 0) -> torch.Tensor:
    a = np.zeros((k, k), np.float32)
    if topology == "none": return torch.from_numpy(a)
    if topology == "ring":
        for i in range(k): a[i, (i - 1) % k] = a[i, (i + 1) % k] = 1
    elif topology == "chain":
        for i in range(k):
            if i: a[i, i - 1] = 1
            if i + 1 < k: a[i, i + 1] = 1
    elif topology == "star":
        for i in range(1, k): a[0, i] = a[i, 0] = 1
    elif topology == "complete":
        a[:] = 1; np.fill_diagonal(a, 0)
    elif topology == "random":
        rng = np.random.default_rng(seed); edges = set(); p = rng.permutation(k)
        for i in range(k): edges.add(tuple(sorted((int(p[i]), int(p[(i + 1) % k])))))
        while len(edges) < k: edges.add(tuple(sorted(rng.choice(k, 2, replace=False).tolist())))
        for u, v in edges: a[u, v] = a[v, u] = 1
    else: raise ValueError(topology)
    degree = a.sum(1, keepdims=True)
    return torch.from_numpy(np.divide(a, degree, out=np.zeros_like(a), where=degree > 0))


class MicrocircuitLayer(nn.Module):
    def __init__(self, input_dim: int, macro: int, k: int, steps: int, topology: str, seed: int):
        super().__init__(); self.macro = macro; self.k = k; self.steps = steps
        self.input_proj = nn.Linear(input_dim, macro * k)
        self.raw_self = nn.Parameter(torch.zeros(macro, k))
        self.raw_neighbor = nn.Parameter(torch.zeros(macro, k))
        self.raw_drive = nn.Parameter(torch.ones(macro, k))
        self.readout_logits = nn.Parameter(torch.zeros(macro, k))
        self.output_bias = nn.Parameter(torch.zeros(macro))
        self.register_buffer("adjacency", topology_matrix(k, topology, seed))
        nn.init.xavier_uniform_(self.input_proj.weight); nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.raw_self, .15, .08); nn.init.normal_(self.raw_neighbor, 0., .08)
        nn.init.normal_(self.readout_logits, 0., .02)

    def forward(self, x: torch.Tensor, return_internal: bool = False):
        drive = torch.tanh(self.input_proj(x).reshape(x.shape[0], self.macro, self.k)); state = drive
        if self.steps:
            gs = .85 * torch.tanh(self.raw_self); gn = .85 * torch.tanh(self.raw_neighbor)
            gd = 1.25 * torch.sigmoid(self.raw_drive); a = self.adjacency.to(state.dtype)
            for _ in range(self.steps):
                neighbors = torch.einsum("bmk,lk->bml", state, a)
                state = torch.tanh(gs * state + gn * neighbors + gd * drive)
        out = (torch.softmax(self.readout_logits, -1) * state).sum(-1) + self.output_bias
        return (out, state) if return_internal else out


class MicrocircuitNet(nn.Module):
    def __init__(self, input_dim: int, macro: int, classes: int, k: int, steps: int, topology: str, seed: int = 0):
        super().__init__(); self.a = nn.GELU()
        self.l1 = MicrocircuitLayer(input_dim, macro, k, steps, topology, seed)
        self.l2 = MicrocircuitLayer(macro, macro, k, steps, topology, seed + 1)
        self.out = nn.Linear(macro, classes)

    def forward(self, x: torch.Tensor, return_features: bool = False, return_internal: bool = False):
        h1, s1 = self.l1(x, True); h1 = self.a(h1); h2, s2 = self.l2(h1, True); h2 = self.a(h2)
        z = self.out(h2)
        if return_internal: return z, h2, (s1, s2)
        return (z, h2) if return_features else z


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def choose_matched_widths(input_dim: int, classes: int, target: int, k: int) -> Tuple[int, int, int, int]:
    best = None
    for h in range(12, 257):
        pm = count_params(MLP(input_dim, h, classes))
        if not target * .55 <= pm <= target * 1.45: continue
        for r in range(6, 160):
            pr = count_params(MicrocircuitNet(input_dim, r, classes, k, 3, "ring"))
            score = 2000 * abs(pm - pr) / max(pm, pr) + abs((pm + pr) / 2 - target) / target
            if best is None or score < best[0]: best = (score, h, r, pm, pr)
    if best is None: raise RuntimeError("matching failed")
    return best[1], best[2], best[3], best[4]


def instantiate(kind: str, data: DatasetBundle, h: int, r: int, k: int, steps: int, topology: str, seed: int) -> nn.Module:
    set_seed(seed)
    return MLP(data.input_dim, h, data.n_classes) if kind == "MLP" else MicrocircuitNet(data.input_dim, r, data.n_classes, k, steps, topology, seed + 1000)


def train_one(model: nn.Module, data: DatasetBundle, seed: int, epochs: int, patience: int, batch: int) -> Tuple[nn.Module, Dict[str, float]]:
    set_seed(seed); gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(data.x_train), torch.from_numpy(data.y_train)), batch_size=batch, shuffle=True, generator=gen)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4); loss_fn = nn.CrossEntropyLoss()
    best = deepcopy(model.state_dict()); best_loss = float("inf"); best_epoch = 0; wait = 0; start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); opt.step()
        model.eval()
        with torch.no_grad(): val = float(loss_fn(model(torch.from_numpy(data.x_val)), torch.from_numpy(data.y_val)))
        if val < best_loss - 1e-5: best_loss = val; best = deepcopy(model.state_dict()); best_epoch = epoch + 1; wait = 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best)
    return model, {"train_seconds": time.perf_counter() - start, "best_epoch": best_epoch, "best_val_loss": best_loss}


@torch.no_grad()
def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray):
    model.eval(); logits, features = model(torch.from_numpy(x), return_features=True)
    probs = torch.softmax(logits, 1).cpu().numpy(); pred = probs.argmax(1)
    return float(accuracy_score(y, pred)), float(log_loss(y, probs, labels=np.arange(probs.shape[1]))), features.cpu().numpy()


def effective_rank(features: np.ndarray):
    z = features.astype(np.float64) - features.mean(0, keepdims=True); s = np.linalg.svd(z, compute_uv=False); e = s * s
    if not e.sum(): return 0., 0.
    p = e / e.sum(); p = p[p > 1e-15]; er = float(np.exp(-(p * np.log(p)).sum()))
    return er, er / max(1, min(z.shape[0] - 1, z.shape[1]))


@torch.no_grad()
def subnode_diversity(model: MicrocircuitNet, x: np.ndarray) -> float:
    _, _, states = model(torch.from_numpy(x[:min(512, len(x))]), return_internal=True); vals = []
    for state in states:
        for node in state.cpu().numpy().transpose(1, 2, 0):
            u = node / (np.linalg.norm(node, axis=1, keepdims=True) + 1e-12); sim = u @ u.T
            vals.extend((1 - sim[np.triu_indices(sim.shape[0], 1)]).tolist())
    return float(np.mean(vals))


@torch.no_grad()
def latency_us(model: nn.Module, input_dim: int, batch: int = 256, repeats: int = 10) -> float:
    model.eval(); x = torch.randn(batch, input_dim)
    for _ in range(5): model(x)
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns(); model(x); vals.append(time.perf_counter_ns() - t0)
    return float(np.median(vals) / 1000 / batch)
