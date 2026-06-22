#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClosureGraphRealBenchmark v3 TRIANGLE

Цель:
  Проверить closure-layer на реальных графах честнее, чем v1:
    - deep_big baseline с числом параметров ближе к closure
    - GCN/message-passing baseline
    - несколько seed
    - IN/OOD corruption
    - optional inductive mode: train on some graphs, test on unseen graphs
    - eval iters ablation for closure: 1/2/3/5/6...
    - triangle update: sum_k h[i,k]*h[k,j], not only row/col mean-field
    - raw-only ablation (--rel-mode raw)
    - optional directed oriented graphs (--directed)

Задачи:
  edge  : восстановить настоящие рёбра clean graph
  path2 : восстановить 2-hop/path-closure структуру clean graph
  comm  : восстановить community/basin relation clean graph

Запуск быстрый:
  python -u closure_graph_real_benchmark_v3_triangle.py --device cuda --amp fp16 --experiment per_graph --seeds 0 --steps 120

Запуск сильнее:
  python -u closure_graph_real_benchmark_v3_triangle.py --device cuda --amp fp16 --experiment per_graph --seeds 0,1,2 --steps 150

Inductive:
  python -u closure_graph_real_benchmark_v3_triangle.py --device cuda --amp fp16 --experiment inductive --train-graphs karate,lesmis --test-graphs florentine,davis --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import networkx as nx
except Exception as e:
    raise SystemExit("networkx is required: pip install networkx") from e

try:
    from sklearn.metrics import roc_auc_score
except Exception as e:
    raise SystemExit("scikit-learn is required: pip install scikit-learn") from e


# ------------------------- utils -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
    y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
    ok = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[ok]
    y_score = y_score[ok]
    if y_true.size < 4 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def offdiag_active_mask(active: torch.Tensor) -> torch.Tensor:
    # active [B,N] bool -> [B,N,N]
    b, n = active.shape
    m = active[:, :, None] & active[:, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=active.device)[None]
    return m & (~eye)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # logits/target [B,N,N,3], mask [B,N,N]
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return loss[mask].mean()


def sym_bin(a: np.ndarray) -> np.ndarray:
    a = (a > 0).astype(np.float32)
    a = np.maximum(a, a.T)
    np.fill_diagonal(a, 0.0)
    return a


# ------------------------- graph loading/sampling -------------------------

def load_builtin_graphs() -> Dict[str, nx.Graph]:
    graphs: Dict[str, nx.Graph] = {}
    graphs["karate"] = nx.karate_club_graph()
    graphs["lesmis"] = nx.les_miserables_graph()
    graphs["florentine"] = nx.florentine_families_graph()
    try:
        graphs["davis"] = nx.davis_southern_women_graph()
    except Exception:
        pass
    # make all simple undirected unweighted
    out = {}
    for name, g in graphs.items():
        h = nx.Graph()
        h.add_nodes_from(g.nodes())
        h.add_edges_from(g.edges())
        h.remove_edges_from(nx.selfloop_edges(h))
        # keep largest component if disconnected but preserve small graphs if mostly connected
        if not nx.is_connected(h):
            comp = max(nx.connected_components(h), key=len)
            h = h.subgraph(comp).copy()
        out[name] = nx.convert_node_labels_to_integers(h)
    return out


def sample_nodes_bfs(g: nx.Graph, n: int, rng: np.random.Generator) -> List[int]:
    nodes = list(g.nodes())
    if len(nodes) <= n:
        return nodes
    start = int(rng.choice(nodes))
    seen = [start]
    q = [start]
    seen_set = {start}
    while q and len(seen) < n:
        u = q.pop(0)
        neigh = list(g.neighbors(u))
        rng.shuffle(neigh)
        for v in neigh:
            if v not in seen_set:
                seen_set.add(v)
                seen.append(v)
                q.append(v)
                if len(seen) >= n:
                    break
    if len(seen) < n:
        rest = [x for x in nodes if x not in seen_set]
        rng.shuffle(rest)
        seen.extend(rest[: n - len(seen)])
    return seen[:n]


def graph_to_padded_adj(g: nx.Graph, n: int, rng: np.random.Generator, directed: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    chosen = sample_nodes_bfs(g, n, rng)
    sg = g.subgraph(chosen).copy()
    sg = nx.convert_node_labels_to_integers(sg)
    k = sg.number_of_nodes()
    a_small = nx.to_numpy_array(sg, dtype=np.float32, weight=None)
    a_small = (a_small > 0).astype(np.float32)
    np.fill_diagonal(a_small, 0.0)
    if directed:
        # Builtin graphs are undirected. For directed stress-test, orient each undirected edge randomly.
        # This keeps exactly one direction u->v or v->u and creates real directionality.
        d = np.zeros_like(a_small, dtype=np.float32)
        iu, ju = np.triu_indices(k, 1)
        edges = a_small[iu, ju] > 0.5
        flips = rng.random(edges.shape) < 0.5
        u = iu[edges]; v = ju[edges]; f = flips[edges]
        d[u[f], v[f]] = 1.0
        d[v[~f], u[~f]] = 1.0
        a_small = d
    else:
        a_small = np.maximum(a_small, a_small.T)
    a = np.zeros((n, n), dtype=np.float32)
    a[:k, :k] = a_small
    active = np.zeros((n,), dtype=np.bool_)
    active[:k] = True
    return a, active


def communities_matrix(a: np.ndarray, active: np.ndarray) -> np.ndarray:
    n = a.shape[0]
    k = int(active.sum())
    out = np.zeros((n, n), dtype=np.float32)
    if k <= 1:
        return out
    g = nx.from_numpy_array(a[:k, :k])
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(g))
    except Exception:
        comms = [set(c) for c in nx.connected_components(g)]
    label = np.zeros((k,), dtype=np.int64)
    for ci, c in enumerate(comms):
        for u in c:
            label[int(u)] = ci
    out[:k, :k] = (label[:, None] == label[None, :]).astype(np.float32)
    np.fill_diagonal(out, 0.0)
    return out


def corrupt_adj(a: np.ndarray, active: np.ndarray, drop_p: float, add_p: float, rng: np.random.Generator, directed: bool = False) -> np.ndarray:
    n = a.shape[0]
    c = a.copy()
    k = int(active.sum())
    if k <= 1:
        return c
    if directed:
        iu, ju = np.where(~np.eye(k, dtype=bool))
        edge = a[iu, ju] > 0.5
        drop = edge & (rng.random(edge.shape) < drop_p)
        c[iu[drop], ju[drop]] = 0.0
        non = (~edge)
        add = non & (rng.random(edge.shape) < add_p)
        c[iu[add], ju[add]] = 1.0
    else:
        iu, ju = np.triu_indices(k, 1)
        edge = a[iu, ju] > 0.5
        drop = edge & (rng.random(edge.shape) < drop_p)
        c[iu[drop], ju[drop]] = 0.0
        c[ju[drop], iu[drop]] = 0.0
        non = (~edge)
        add = non & (rng.random(edge.shape) < add_p)
        c[iu[add], ju[add]] = 1.0
        c[ju[add], iu[add]] = 1.0
    np.fill_diagonal(c, 0.0)
    return c


def common_neighbors(a: np.ndarray) -> np.ndarray:
    cn = a @ a
    np.fill_diagonal(cn, 0.0)
    mx = cn.max()
    if mx > 0:
        cn = cn / mx
    return cn.astype(np.float32)


def jaccard_scores(a: np.ndarray) -> np.ndarray:
    deg = a.sum(axis=1)
    inter = a @ a
    union = deg[:, None] + deg[None, :] - inter
    j = inter / (union + 1e-6)
    np.fill_diagonal(j, 0.0)
    return j.astype(np.float32)


def spectral_scores(a: np.ndarray, dim: int = 8) -> np.ndarray:
    n = a.shape[0]
    try:
        aa = a.astype(np.float64)
        if np.allclose(aa, aa.T, atol=1e-6):
            w, v = np.linalg.eigh(aa)
            idx = np.argsort(np.abs(w))[::-1][: min(dim, n)]
            emb = v[:, idx] * np.sqrt(np.abs(w[idx]) + 1e-6)
            s = emb @ emb.T
        else:
            u, sv, vt = np.linalg.svd(aa, full_matrices=False)
            r = min(dim, len(sv))
            left = u[:, :r] * np.sqrt(sv[:r] + 1e-6)
            right = vt[:r, :].T * np.sqrt(sv[:r] + 1e-6)
            s = left @ right.T
        s = (s - s.min()) / (s.max() - s.min() + 1e-6)
        np.fill_diagonal(s, 0.0)
        return s.astype(np.float32)
    except Exception:
        return np.zeros_like(a, dtype=np.float32)


def build_relation_features(corrupt: np.ndarray, active: np.ndarray, rel_mode: str = "full") -> np.ndarray:
    # full: [N,N,8], raw: [N,N,1]
    a = corrupt.astype(np.float32)
    n = a.shape[0]
    if rel_mode == "raw":
        rel = a[..., None].astype(np.float32)
        np.fill_diagonal(rel[..., 0], 0.0)
        return rel
    out_deg = a.sum(axis=1, keepdims=True)
    in_deg = a.sum(axis=0, keepdims=True).T
    k = max(float(active.sum() - 1), 1.0)
    outn = out_deg / k
    inn = in_deg / k
    cn = common_neighbors(a)
    jac = jaccard_scores(a)
    p2 = ((a @ a) > 0).astype(np.float32)
    np.fill_diagonal(p2, 0.0)
    rel = np.zeros((n, n, 8), dtype=np.float32)
    rel[..., 0] = a
    rel[..., 1] = cn
    rel[..., 2] = jac
    rel[..., 3] = p2
    rel[..., 4] = outn.repeat(n, axis=1)
    rel[..., 5] = inn.T.repeat(n, axis=0)
    # directional degree mismatch: out_i vs in_j
    rel[..., 6] = np.abs(outn - inn.T)
    rel[..., 7] = (active[:, None] & active[None, :]).astype(np.float32)
    for c in range(rel.shape[-1]):
        np.fill_diagonal(rel[..., c], 0.0)
    return rel


def targets_from_clean(a: np.ndarray, active: np.ndarray) -> np.ndarray:
    edge = a.astype(np.float32)
    path2 = ((a @ a) > 0).astype(np.float32)
    np.fill_diagonal(path2, 0.0)
    comm = communities_matrix(a, active)
    return np.stack([edge, path2, comm], axis=-1).astype(np.float32)


@dataclass
class Batch:
    rel: torch.Tensor       # [B,N,N,C]
    target: torch.Tensor    # [B,N,N,3]
    active: torch.Tensor    # [B,N]
    baselines: Dict[str, torch.Tensor]  # [B,N,N]


def make_batch(graphs: List[nx.Graph], args, rng: np.random.Generator, device: torch.device, hard: bool = False) -> Batch:
    rels, targs, actives = [], [], []
    raws, cns, jacs, specs = [], [], [], []
    drop = args.hard_drop if hard else args.drop
    add = args.hard_add if hard else args.add
    for _ in range(args.batch_current):
        g = graphs[int(rng.integers(0, len(graphs)))]
        clean, active = graph_to_padded_adj(g, args.n, rng, directed=args.directed)
        corr = corrupt_adj(clean, active, drop, add, rng, directed=args.directed)
        rel = build_relation_features(corr, active, rel_mode=args.rel_mode)
        targ = targets_from_clean(clean, active)
        rels.append(rel); targs.append(targ); actives.append(active)
        raws.append(corr.astype(np.float32))
        cns.append(common_neighbors(corr))
        jacs.append(jaccard_scores(corr))
        specs.append(spectral_scores(corr))
    def T(x, dtype=torch.float32):
        return torch.tensor(np.stack(x), dtype=dtype, device=device)
    return Batch(
        rel=T(rels),
        target=T(targs),
        active=T(actives, dtype=torch.bool),
        baselines={
            "raw": T(raws),
            "cn": T(cns),
            "jaccard": T(jacs),
            "spectral": T(specs),
        }
    )


# ------------------------- models -------------------------

class PairMLP(nn.Module):
    def __init__(self, rel_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rel_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        return self.net(rel)


class ContextMLP(nn.Module):
    def __init__(self, rel_dim: int, hidden: int, depth: int = 3):
        super().__init__()
        # rel,row,col,global,row_std,col_std + 6 scalar graph features
        in_dim = rel_dim * 6 + 6
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)
    def _ch(self, rel: torch.Tensor, idx: int) -> torch.Tensor:
        if rel.shape[-1] > idx:
            return rel[..., idx:idx+1]
        return torch.zeros_like(rel[..., :1])
    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        row = rel.mean(dim=2, keepdim=True).expand_as(rel)
        col = rel.mean(dim=1, keepdim=True).expand_as(rel)
        glob = rel.mean(dim=(1,2), keepdim=True).expand_as(rel)
        row_std = rel.std(dim=2, keepdim=True).expand_as(rel)
        col_std = rel.std(dim=1, keepdim=True).expand_as(rel)
        a = self._ch(rel, 0)
        deg_i = a.sum(dim=2, keepdim=True).expand_as(a) / max(rel.shape[1]-1, 1)
        deg_j = a.sum(dim=1, keepdim=True).expand_as(a) / max(rel.shape[1]-1, 1)
        cn = self._ch(rel, 1)
        jac = self._ch(rel, 2)
        p2 = self._ch(rel, 3)
        diff_deg = torch.abs(deg_i - deg_j)
        extra = torch.cat([deg_i, deg_j, diff_deg, cn, jac, p2], dim=-1)
        x = torch.cat([rel, row, col, glob, row_std, col_std, extra], dim=-1)
        return self.net(x)


class GCNBaseline(nn.Module):
    def __init__(self, rel_dim: int, hidden: int):
        super().__init__()
        node_in = rel_dim * 4
        self.node_proj = nn.Linear(node_in, hidden)
        self.g1 = nn.Linear(hidden, hidden)
        self.g2 = nn.Linear(hidden, hidden)
        self.pair = nn.Sequential(
            nn.Linear(hidden * 4 + rel_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
    def norm_adj(self, a: torch.Tensor) -> torch.Tensor:
        b, n, _ = a.shape
        eye = torch.eye(n, device=a.device, dtype=a.dtype)[None]
        ah = a + eye
        deg = ah.sum(dim=-1).clamp_min(1e-6)
        dinv = deg.pow(-0.5)
        return ah * dinv[:, :, None] * dinv[:, None, :]
    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        a = rel[..., 0]
        row_mean = rel.mean(dim=2)
        row_std = rel.std(dim=2)
        row_max = rel.max(dim=2).values
        deg = a.sum(dim=2, keepdim=True).expand(-1, -1, rel.shape[-1]) / max(rel.shape[1]-1, 1)
        node = torch.cat([row_mean, row_std, row_max, deg], dim=-1)
        h = F.gelu(self.node_proj(node))
        an = self.norm_adj(a)
        h = F.gelu(self.g1(torch.bmm(an, h)))
        h = F.gelu(self.g2(torch.bmm(an, h)))
        hi = h[:, :, None, :].expand(-1, -1, rel.shape[2], -1)
        hj = h[:, None, :, :].expand(-1, rel.shape[1], -1, -1)
        x = torch.cat([hi, hj, torch.abs(hi - hj), hi * hj, rel], dim=-1)
        return self.pair(x)


class ClosureLayer(nn.Module):
    def __init__(self, rel_dim: int, dim: int, hidden: int, use_triangle: bool = True):
        super().__init__()
        self.use_triangle = use_triangle
        self.enc = nn.Sequential(nn.Linear(rel_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        # h,row,col,global,row_std,col_std,(triangle if enabled) + 6 scalar graph features
        feat_dim = dim * (7 if use_triangle else 6) + 6
        self.step = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.gate = nn.Sequential(nn.Linear(feat_dim, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Sigmoid())
        self.read = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.ln = nn.LayerNorm(dim)
    def _ch(self, rel: torch.Tensor, idx: int) -> torch.Tensor:
        if rel.shape[-1] > idx:
            return rel[..., idx:idx+1]
        return torch.zeros_like(rel[..., :1])
    def triangle_update(self, h: torch.Tensor) -> torch.Tensor:
        # true path composition: tri[i,j,c] = sum_k h[i,k,c] * h[k,j,c]
        # This is the important v3 change versus mean-field row*col agreement.
        b, n, _, d = h.shape
        hc = h.permute(0, 3, 1, 2).contiguous().reshape(b * d, n, n)
        tri = torch.bmm(hc, hc) / max(float(n), 1.0)
        tri = tri.reshape(b, d, n, n).permute(0, 2, 3, 1).contiguous()
        return tri
    def step_once(self, h: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        row = h.mean(dim=2, keepdim=True).expand_as(h)
        col = h.mean(dim=1, keepdim=True).expand_as(h)
        glob = h.mean(dim=(1,2), keepdim=True).expand_as(h)
        row_std = h.std(dim=2, keepdim=True).expand_as(h)
        col_std = h.std(dim=1, keepdim=True).expand_as(h)
        pieces = [h, row, col, glob, row_std, col_std]
        if self.use_triangle:
            pieces.append(self.triangle_update(h))
        a = self._ch(rel, 0)
        deg_i = a.sum(dim=2, keepdim=True).expand_as(a) / max(rel.shape[1]-1, 1)
        deg_j = a.sum(dim=1, keepdim=True).expand_as(a) / max(rel.shape[1]-1, 1)
        cn = self._ch(rel, 1)
        jac = self._ch(rel, 2)
        p2 = self._ch(rel, 3)
        diff_deg = torch.abs(deg_i - deg_j)
        extra = torch.cat([deg_i, deg_j, diff_deg, cn, jac, p2], dim=-1)
        x = torch.cat(pieces + [extra], dim=-1)
        delta = self.step(x)
        gate = self.gate(x)
        return self.ln(h + gate * delta)
    def forward(self, rel: torch.Tensor, iters: int = 5, return_curve: bool = False, return_h: bool = False):
        h = self.enc(rel)
        curve = []
        prev = h
        for _ in range(iters):
            h = self.step_once(h, rel)
            if return_curve:
                curve.append(float((h - prev).pow(2).mean().detach().sqrt().cpu()))
            prev = h
        out = self.read(h)
        if return_curve and return_h:
            return out, curve, h
        if return_curve:
            return out, curve
        if return_h:
            return out, h
        return out


# ------------------------- train/eval -------------------------

def model_forward(model: nn.Module, rel: torch.Tensor, iters: int) -> torch.Tensor:
    if isinstance(model, ClosureLayer):
        return model(rel, iters=iters)
    return model(rel)


def train_one_step(models: Dict[str, nn.Module], opt: torch.optim.Optimizer, batch: Batch, amp: str, iters: int, fixed_w: float = 0.02) -> Tuple[float, List[float], float, float]:
    opt.zero_grad(set_to_none=True)
    mask = offdiag_active_mask(batch.active)
    use_amp = batch.rel.is_cuda and amp in ("fp16", "bf16")
    dtype = torch.float16 if amp == "fp16" else torch.bfloat16
    main_curve: List[float] = []
    fixed_loss = torch.tensor(0.0, device=batch.rel.device)
    ratio_val = 0.0
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
        total = torch.tensor(0.0, device=batch.rel.device)
        n_terms = 0
        for name, model in models.items():
            if isinstance(model, ClosureLayer):
                logits, curve, h = model(batch.rel, iters=iters, return_curve=True, return_h=True)
                if name == "closure":
                    main_curve = curve
                    if len(curve) >= 2 and curve[0] > 1e-9:
                        ratio_val = float(curve[-1] / (curve[0] + 1e-9))
                # Optional true fixed-point regularization: one more learned step should change little.
                # Default is off for speed; curve ratio is still logged.
                if fixed_w > 0.0 and name == "closure":
                    h_next = model.step_once(h, batch.rel)
                    fixed_loss = fixed_loss + (h_next - h).pow(2).mean()
            else:
                logits = model(batch.rel)
            total = total + masked_bce(logits, batch.target, mask)
            n_terms += 1
        loss = total / max(n_terms, 1) + fixed_w * fixed_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for m in models.values() for p in m.parameters()], 1.0)
    opt.step()
    return float(loss.detach().cpu()), main_curve, float(fixed_loss.detach().cpu()), ratio_val


def collect_scores_for_batch(models: Dict[str, nn.Module], batch: Batch, iters: int, eval_iters: Optional[List[int]] = None) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    # returns model -> task -> (y, score)
    mask = offdiag_active_mask(batch.active).detach().cpu().numpy().reshape(-1)
    target = batch.target.detach().cpu().numpy().reshape(-1, 3)
    out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    task_names = ["edge", "path2", "comm"]

    def add(name: str, scores_3: np.ndarray):
        flat = scores_3.reshape(-1, 3)
        out[name] = {}
        for ti, tn in enumerate(task_names):
            out[name][tn] = (target[:, ti][mask], flat[:, ti][mask])

    # baselines mapped to three tasks
    raw = batch.baselines["raw"].detach().cpu().numpy()
    cn = batch.baselines["cn"].detach().cpu().numpy()
    jac = batch.baselines["jaccard"].detach().cpu().numpy()
    spec = batch.baselines["spectral"].detach().cpu().numpy()
    path2_raw = ((raw @ raw) > 0).astype(np.float32)
    for arr in (path2_raw,):
        for b in range(arr.shape[0]):
            np.fill_diagonal(arr[b], 0.0)
    add("raw", np.stack([raw, path2_raw, spec], axis=-1))
    add("cn", np.stack([cn, cn, cn], axis=-1))
    add("jaccard", np.stack([jac, jac, jac], axis=-1))
    add("spectral", np.stack([spec, spec, spec], axis=-1))

    with torch.no_grad():
        for name, model in models.items():
            logits = model_forward(model, batch.rel, iters)
            add(name, torch.sigmoid(logits).detach().cpu().numpy())
        if eval_iters:
            for k in eval_iters:
                logits = models["closure"](batch.rel, iters=k)
                add(f"closure_i{k}", torch.sigmoid(logits).detach().cpu().numpy())
    return out


def merge_auc(accum: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    for m, bytask in accum.items():
        res[m] = {}
        for t, pairs in bytask.items():
            ys = np.concatenate([p[0] for p in pairs])
            ss = np.concatenate([p[1] for p in pairs])
            res[m][t] = safe_auc(ys, ss)
    return res


def evaluate(models: Dict[str, nn.Module], graphs: List[nx.Graph], args, rng: np.random.Generator, device: torch.device, hard: bool, eval_iters: Optional[List[int]]) -> Dict[str, Dict[str, float]]:
    for m in models.values():
        m.eval()
    accum: Dict[str, Dict[str, List[Tuple[np.ndarray, np.ndarray]]]] = {}
    old_batch = args.batch_current
    args.batch_current = args.eval_batch
    for _ in range(args.eval_batches):
        batch = make_batch(graphs, args, rng, device, hard=hard)
        scores = collect_scores_for_batch(models, batch, args.iters, eval_iters=eval_iters)
        for m, bytask in scores.items():
            accum.setdefault(m, {})
            for t, pair in bytask.items():
                accum[m].setdefault(t, []).append(pair)
    args.batch_current = old_batch
    for m in models.values():
        m.train()
    return merge_auc(accum)


def best_other_delta(aucs: Dict[str, Dict[str, float]], task: str, closure_name: str = "closure") -> Tuple[float, float, str]:
    cl = aucs.get(closure_name, {}).get(task, float("nan"))
    best_val = -1.0
    best_name = "none"
    for m, vals in aucs.items():
        if m.startswith("closure"):
            continue
        v = vals.get(task, float("nan"))
        if np.isfinite(v) and v > best_val:
            best_val = v; best_name = m
    return cl, cl - best_val, best_name


def make_models(args, device: torch.device) -> Dict[str, nn.Module]:
    rel_dim = 1 if args.rel_mode == "raw" else 8
    models: Dict[str, nn.Module] = {
        "pair": PairMLP(rel_dim, args.hidden).to(device),
        "deep": ContextMLP(rel_dim, args.hidden, depth=3).to(device),
        "deep_big": ContextMLP(rel_dim, args.big_hidden, depth=args.big_depth).to(device),
        "gcn": GCNBaseline(rel_dim, args.hidden).to(device),
        "closure": ClosureLayer(rel_dim, args.dim, args.hidden, use_triangle=True).to(device),
    }
    if args.ablate_tri:
        models["closure_no_tri"] = ClosureLayer(rel_dim, args.dim, args.hidden, use_triangle=False).to(device)
    return models


def run_train_eval(graphs_train: List[nx.Graph], graphs_eval: Dict[str, List[nx.Graph]], args, seed: int, label: str, writer: Optional[csv.DictWriter]) -> None:
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    rng = np.random.default_rng(seed)
    models = make_models(args, device)
    params = {k: count_params(v) for k, v in models.items()}
    print(f"\nRUN {label} seed={seed} params: " + " ".join([f"{k}={v:,}" for k,v in params.items()]), flush=True)
    opt = torch.optim.AdamW([p for m in models.values() for p in m.parameters()], lr=args.lr, weight_decay=args.wd)
    args.batch_current = args.batch
    eval_iters = [int(x) for x in args.eval_iters.split(",") if x.strip()] if args.eval_iters else []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = make_batch(graphs_train, args, rng, device, hard=False)
        loss, curve, fixed_loss, curve_ratio = train_one_step(models, opt, batch, args.amp, args.iters, fixed_w=args.fixed_w)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            print(f"step {step:05d}/{args.steps} loss={loss:.4f} fixed={fixed_loss:.5f} curve={curve[0] if curve else 0:.4f}->{curve[-1] if curve else 0:.4f} ratio={curve_ratio:.3f} t={time.time()-t0:.1f}s", flush=True)
            for split_name, gs in graphs_eval.items():
                for hard in (False, True):
                    mode = "OOD" if hard else "IN"
                    aucs = evaluate(models, gs, args, rng, device, hard=hard, eval_iters=eval_iters)
                    parts = []
                    for task in ["edge", "path2", "comm"]:
                        cl, delta, best = best_other_delta(aucs, task, closure_name="closure")
                        parts.append(f"{task}: cl={cl:.3f} Δ={delta:+.3f} best={best}")
                    # iters ablation on edge/path2/comm if present
                    if eval_iters:
                        iter_bits = []
                        for k in eval_iters:
                            name = f"closure_i{k}"
                            if name in aucs:
                                iter_bits.append(f"i{k} e/p/c={aucs[name]['edge']:.2f}/{aucs[name]['path2']:.2f}/{aucs[name]['comm']:.2f}")
                        iter_s = " | " + " ; ".join(iter_bits[:4]) if iter_bits else ""
                    else:
                        iter_s = ""
                    print(f"  {split_name}:{mode} " + " | ".join(parts) + iter_s, flush=True)
                    if writer is not None and step == args.steps:
                        for model_name, bytask in aucs.items():
                            for task, val in bytask.items():
                                cl, delta, best = best_other_delta(aucs, task, closure_name="closure")
                                writer.writerow({
                                    "experiment": args.experiment,
                                    "run": label,
                                    "seed": seed,
                                    "split": split_name,
                                    "corruption": mode,
                                    "task": task,
                                    "model": model_name,
                                    "auc": val,
                                    "closure_auc": cl,
                                    "closure_delta_vs_best_nonclosure": delta,
                                    "best_nonclosure": best,
                                    "steps": args.steps,
                                    "n": args.n,
                                    "dim": args.dim,
                                    "hidden": args.hidden,
                                    "iters": args.iters,
                                    "params": params.get(model_name, -1),
                                    "rel_mode": args.rel_mode,
                                    "directed": int(args.directed),
                                })


def parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default="fp16", choices=["off", "fp16", "bf16"])
    p.add_argument("--experiment", default="per_graph", choices=["per_graph", "inductive"])
    p.add_argument("--datasets", default="builtin", choices=["builtin"])
    p.add_argument("--train-graphs", default="karate,lesmis")
    p.add_argument("--test-graphs", default="florentine,davis")
    p.add_argument("--seeds", default="0")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--big-hidden", type=int, default=128)
    p.add_argument("--big-depth", type=int, default=4)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--eval-iters", default="1,2,3,5")
    p.add_argument("--rel-mode", default="full", choices=["full", "raw"], help="full uses handcrafted graph channels; raw uses only corrupted adjacency")
    p.add_argument("--directed", action="store_true", help="randomly orient builtin undirected edges and corrupt as directed graph")
    p.add_argument("--ablate-tri", action="store_true", help="also train closure_no_tri mean-field ablation")
    p.add_argument("--fixed-w", type=float, default=0.0, help="optional fixed point loss weight for main closure layer; off by default for speed")
    p.add_argument("--drop", type=float, default=0.25)
    p.add_argument("--add", type=float, default=0.06)
    p.add_argument("--hard-drop", type=float, default=0.45)
    p.add_argument("--hard-add", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--results-csv", default="closure_graph_real_v3_summary.csv")
    args = p.parse_args()

    print("ClosureGraphRealBenchmark v3 TRIANGLE — real graph closure layer", flush=True)
    print(f"device={args.device} amp={args.amp} experiment={args.experiment} n={args.n} dim={args.dim} hidden={args.hidden} iters={args.iters} rel_mode={args.rel_mode} directed={args.directed}", flush=True)
    print(f"corruption train drop/add={args.drop}/{args.add}; OOD={args.hard_drop}/{args.hard_add}", flush=True)
    print("baselines: raw, common-neighbors, jaccard, spectral, pair-MLP, deep-MLP, deep_big, GCN; closure uses triangle update", flush=True)

    graphs = load_builtin_graphs()
    for name, g in graphs.items():
        print(f"graph {name}: nodes={g.number_of_nodes()} edges={g.number_of_edges()}", flush=True)

    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    file_exists = os.path.exists(args.results_csv)
    f = open(args.results_csv, "a", newline="", encoding="utf-8")
    fieldnames = ["experiment","run","seed","split","corruption","task","model","auc","closure_auc","closure_delta_vs_best_nonclosure","best_nonclosure","steps","n","dim","hidden","iters","params","rel_mode","directed"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    seeds = [int(x) for x in parse_list(args.seeds)]
    try:
        if args.experiment == "per_graph":
            for seed in seeds:
                for name, g in graphs.items():
                    run_train_eval([g], {name: [g]}, args, seed, label=name, writer=writer)
        else:
            train_names = parse_list(args.train_graphs)
            test_names = parse_list(args.test_graphs)
            train_g = [graphs[n] for n in train_names if n in graphs]
            test_g = [graphs[n] for n in test_names if n in graphs]
            if not train_g or not test_g:
                raise SystemExit(f"Bad train/test graph names. Available: {list(graphs)}")
            for seed in seeds:
                run_train_eval(train_g, {"train_seen": train_g, "test_unseen": test_g}, args, seed, label=f"inductive_{'-'.join(train_names)}__to__{'-'.join(test_names)}", writer=writer)
    finally:
        f.close()
    print(f"\nDone. Results appended: {args.results_csv}", flush=True)


if __name__ == "__main__":
    main()
