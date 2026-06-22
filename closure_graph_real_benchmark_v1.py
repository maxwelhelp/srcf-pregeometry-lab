#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClosureGraphRealBenchmark v1.1

Цель: проверить closure-layer не на anomaly/DNA, а на реальных графах:
  1) edge repair: восстановить настоящие рёбра после удаления/добавления шума
  2) path2 closure: восстановить 2-hop/транзитивную структуру
  3) community/basin: восстановить same-community relation
  4) OOD corruption: проверить перенос на более сильное повреждение

Сравнение:
  raw_corrupt
  common_neighbors
  jaccard
  spectral
  local_mlp
  deep_context_mlp
  closure_layer

Данные:
  встроенные реальные графы NetworkX: karate, lesmis, florentine, davis
  опционально Cora citation graph с авто-скачиванием (--datasets cora/all --allow-download)

Важно: встроенные NetworkX графы — маленькие реальные исторические сети, не синтетика.
Cora — внешний реальный citation graph, качается отдельно.
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
import sys
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import networkx as nx
except Exception as e:
    raise SystemExit("[ERR] Нужен networkx: pip install networkx") from e

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


# ------------------------- utils -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    mask = np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if roc_auc_score is not None:
        return float(roc_auc_score(y_true, y_score))
    # simple rank AUC fallback
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score)) + 1
    pos = y_true == 1
    n_pos = pos.sum(); n_neg = len(y_true) - n_pos
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-12))


def ap_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if average_precision_score is None:
        return float("nan")
    y_true = y_true.astype(np.int32).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def upper_mask(n: int, device=None) -> torch.Tensor:
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return torch.maximum(x, x.transpose(-1, -2))


def remove_diag(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    return x * (1.0 - eye)


# ------------------------- real graph loaders -------------------------

def _to_simple_connected(G: nx.Graph) -> nx.Graph:
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        raise ValueError("empty graph")
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(comps[0]).copy()
    return nx.convert_node_labels_to_integers(G)


def load_builtin_graph(name: str) -> nx.Graph:
    name = name.lower()
    if name == "karate":
        return _to_simple_connected(nx.karate_club_graph())
    if name == "lesmis":
        return _to_simple_connected(nx.les_miserables_graph())
    if name == "florentine":
        return _to_simple_connected(nx.florentine_families_graph())
    if name == "davis":
        return _to_simple_connected(nx.davis_southern_women_graph())
    raise ValueError(f"unknown builtin graph: {name}")


def download_cora(cache_dir: str = "data/cora") -> Path:
    """Download small Cora files from pygcn mirror."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cites = cache / "cora.cites"
    content = cache / "cora.content"
    if cites.exists() and content.exists():
        return cache
    base = "https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/"
    for fn in ["cora.cites", "cora.content"]:
        url = base + fn
        out = cache / fn
        print(f"downloading Cora: {url}", flush=True)
        urllib.request.urlretrieve(url, out)
    return cache


def load_cora(allow_download: bool, cache_dir: str = "data/cora") -> nx.Graph:
    cache = Path(cache_dir)
    if not ((cache / "cora.cites").exists() and (cache / "cora.content").exists()):
        if not allow_download:
            raise FileNotFoundError("Cora не найден. Запусти с --allow-download или используй builtin graphs.")
        download_cora(cache_dir)
    cites = cache / "cora.cites"
    # Cora ids are strings
    G = nx.Graph()
    with open(cites, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split()
            G.add_edge(a, b)
    return _to_simple_connected(G)


def sample_bfs_subgraph(G: nx.Graph, n: int, rng: random.Random) -> nx.Graph:
    G = _to_simple_connected(G)
    if G.number_of_nodes() <= n:
        return nx.convert_node_labels_to_integers(G)
    nodes = list(G.nodes())
    for _ in range(50):
        start = rng.choice(nodes)
        seen = [start]
        seen_set = {start}
        q = [start]
        while q and len(seen) < n:
            u = q.pop(0)
            nbrs = list(G.neighbors(u))
            rng.shuffle(nbrs)
            for v in nbrs:
                if v not in seen_set:
                    seen.append(v); seen_set.add(v); q.append(v)
                    if len(seen) >= n:
                        break
        if len(seen) >= max(8, min(n, G.number_of_nodes())):
            H = G.subgraph(seen[:n]).copy()
            if H.number_of_edges() > 0:
                return nx.convert_node_labels_to_integers(_to_simple_connected(H))
    chosen = rng.sample(nodes, n)
    return nx.convert_node_labels_to_integers(_to_simple_connected(G.subgraph(chosen).copy()))


def graph_to_adj(G: nx.Graph, n: int, rng: random.Random) -> torch.Tensor:
    H = sample_bfs_subgraph(G, n, rng)
    # if sampled connected comp got smaller, pad with isolated placeholders? Better resample / pad.
    N = H.number_of_nodes()
    A = nx.to_numpy_array(H, dtype=np.float32)
    if N < n:
        P = np.zeros((n, n), dtype=np.float32)
        P[:N, :N] = A
        A = P
    elif N > n:
        A = A[:n, :n]
    A = np.maximum(A, A.T)
    # В NetworkX некоторые реальные графы (например karate) имеют edge weights.
    # Для repair/link задач нужен бинарный факт связи, иначе target_edge становится 0..N.
    A = (A > 0).astype(np.float32)
    np.fill_diagonal(A, 0.0)
    return torch.from_numpy(A)


# ------------------------- data generation -------------------------

@dataclass
class Batch:
    r: torch.Tensor          # [B,N,N,C]
    target_edge: torch.Tensor
    target_path2: torch.Tensor
    target_comm: torch.Tensor
    corrupt_adj: torch.Tensor
    clean_adj: torch.Tensor


def clean_targets(A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A [N,N] -> edge, path2, community-like target."""
    n = A.shape[0]
    A = remove_diag((symmetrize(A.float()) > 0).float())
    path2 = ((A @ A) > 0).float()
    path2 = remove_diag(torch.maximum(A, path2))
    # community labels by greedy modularity on clean graph; fallback connected components
    G = nx.from_numpy_array(A.cpu().numpy())
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G))
    except Exception:
        comms = list(nx.connected_components(G))
    labels = torch.zeros(n, dtype=torch.long, device=A.device)
    for k, c in enumerate(comms):
        for node in c:
            if node < n:
                labels[node] = k
    comm = (labels[:, None] == labels[None, :]).float()
    comm = remove_diag(comm)
    return A, path2, comm


def corrupt_adj(A: torch.Tensor, drop_p: float, add_p: float) -> torch.Tensor:
    device = A.device
    n = A.shape[-1]
    mask = upper_mask(n, device)
    Aup = A[mask]
    keep = torch.ones_like(Aup)
    drop = (torch.rand_like(Aup) < drop_p) & (Aup > 0.5)
    add = (torch.rand_like(Aup) < add_p) & (Aup < 0.5)
    Cup = Aup.clone()
    Cup[drop] = 0.0
    Cup[add] = 1.0
    C = torch.zeros_like(A)
    C[mask] = Cup
    C = C + C.t()
    return C


def relation_features(C: torch.Tensor) -> torch.Tensor:
    """C [B,N,N] -> R [B,N,N,8]."""
    B, N, _ = C.shape
    C = remove_diag(symmetrize(C.float()))
    deg = C.sum(-1) / max(1, N - 1)                 # [B,N]
    common = torch.bmm(C, C) / max(1, N)            # [B,N,N]
    twohop = (common > 0).float()
    jacc = common / (deg[:, :, None] + deg[:, None, :] - common + 1e-6)
    deg_diff = (deg[:, :, None] - deg[:, None, :]).abs()
    deg_prod = deg[:, :, None] * deg[:, None, :]
    row_mean = C.mean(-1)[:, :, None].expand(B, N, N)
    col_mean = C.mean(-2)[:, None, :].expand(B, N, N)
    # simple closure pressure: if two nodes share neighbors but no edge
    missing_pressure = (1.0 - C) * common
    R = torch.stack([C, common, twohop, jacc, deg_diff, deg_prod, row_mean, missing_pressure], dim=-1)
    return R


def make_batch(A_clean: torch.Tensor, batch: int, drop_p: float, add_p: float, device: str) -> Batch:
    A_clean = A_clean.to(device)
    edge, path2, comm = clean_targets(A_clean)
    C_list = []
    for _ in range(batch):
        C_list.append(corrupt_adj(edge, drop_p, add_p))
    C = torch.stack(C_list, dim=0)
    R = relation_features(C)
    edge_b = edge[None].expand(batch, -1, -1)
    path2_b = path2[None].expand(batch, -1, -1)
    comm_b = comm[None].expand(batch, -1, -1)
    clean_b = edge[None].expand(batch, -1, -1)
    return Batch(R, edge_b, path2_b, comm_b, C, clean_b)


# ------------------------- models -------------------------

class PairMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.net(r)  # [B,N,N,3]


class DeepContextMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        ctx_dim = in_dim * 4
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
    def forward(self, r: torch.Tensor) -> torch.Tensor:
        row = r.mean(dim=2, keepdim=True).expand_as(r)
        col = r.mean(dim=1, keepdim=True).expand_as(r)
        glob = r.mean(dim=(1,2), keepdim=True).expand_as(r)
        x = torch.cat([r, row, col, glob], dim=-1)
        return self.net(x)


class ClosureLayer(nn.Module):
    def __init__(self, in_dim: int, dim: int, hidden: int, iters: int):
        super().__init__()
        self.iters = iters
        self.enc = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.step = nn.Sequential(
            nn.Linear(dim * 5, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.gate = nn.Sequential(nn.Linear(dim * 4, hidden), nn.GELU(), nn.Linear(hidden, dim), nn.Sigmoid())
        self.dec = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.norm = nn.LayerNorm(dim)

    def step_once(self, h: torch.Tensor) -> torch.Tensor:
        row = h.mean(dim=2, keepdim=True).expand_as(h)
        col = h.mean(dim=1, keepdim=True).expand_as(h)
        glob = h.mean(dim=(1,2), keepdim=True).expand_as(h)
        # second-order pressure: row/col agreement proxy
        agree = row * col
        x = torch.cat([h, row, col, glob, agree], dim=-1)
        delta = self.step(x)
        g = self.gate(torch.cat([h, row, col, glob], dim=-1))
        return self.norm(h + g * delta)

    def forward(self, r: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        h = self.enc(r)
        curve = []
        prev = h
        for _ in range(self.iters):
            h = self.step_once(h)
            curve.append(float((h - prev).pow(2).mean().detach().sqrt().cpu()))
            prev = h
        return self.dec(h), curve


def masked_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = target.shape[-1]
    mask = upper_mask(n, target.device)[None].expand(target.shape[0], -1, -1)
    return F.binary_cross_entropy_with_logits(logits[mask], target[mask])


def train_step(models, opt, batch: Batch, amp: str, weights=(1.0, 1.0, 0.75)):
    pair, deep, closure = models
    opt.zero_grad(set_to_none=True)
    targets = torch.stack([batch.target_edge, batch.target_path2, batch.target_comm], dim=-1)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp == "fp16" and batch.r.is_cuda)):
        logits_pair = pair(batch.r)
        logits_deep = deep(batch.r)
        logits_cl, curve = closure(batch.r)
        loss_pair = 0.0
        loss_deep = 0.0
        loss_cl = 0.0
        for k, w in enumerate(weights):
            loss_pair = loss_pair + w * masked_bce(logits_pair[..., k], targets[..., k])
            loss_deep = loss_deep + w * masked_bce(logits_deep[..., k], targets[..., k])
            loss_cl = loss_cl + w * masked_bce(logits_cl[..., k], targets[..., k])
        fixed = torch.tensor(0.0, device=batch.r.device)
        if len(curve) >= 2:
            # small penalty if last movement not below early movement (diagnostic, weak)
            fixed = torch.tensor(max(0.0, curve[-1] - curve[0]), device=batch.r.device)
        loss = loss_pair + loss_deep + loss_cl + 0.01 * fixed
    loss.backward()
    torch.nn.utils.clip_grad_norm_(list(pair.parameters()) + list(deep.parameters()) + list(closure.parameters()), 1.0)
    opt.step()
    return float(loss.detach().cpu()), curve


# ------------------------- evaluation -------------------------

@torch.no_grad()
def spectral_score(C: torch.Tensor, rank: int = 8) -> torch.Tensor:
    """C [B,N,N] -> score [B,N,N] by spectral embedding dot product."""
    B, N, _ = C.shape
    outs = []
    for b in range(B):
        A = remove_diag(symmetrize(C[b].float()))
        # normalized adjacency-like
        deg = A.sum(-1)
        D = torch.diag(1.0 / torch.sqrt(deg + 1e-6))
        M = D @ A @ D
        try:
            vals, vecs = torch.linalg.eigh(M)
            idx = torch.argsort(vals, descending=True)[:min(rank, N)]
            X = vecs[:, idx] * vals[idx].clamp(min=0).sqrt()[None, :]
            S = X @ X.t()
        except Exception:
            S = A
        outs.append(S)
    return torch.stack(outs, dim=0)


def collect_auc(scores: torch.Tensor, target: torch.Tensor) -> float:
    n = target.shape[-1]
    mask = upper_mask(n, target.device)[None].expand(target.shape[0], -1, -1)
    y = target[mask].detach().cpu().numpy()
    s = scores[mask].detach().cpu().numpy()
    return auc_score(y, s)


@torch.no_grad()
def eval_models(models, A_clean: torch.Tensor, batch_size: int, batches: int, drop_p: float, add_p: float, device: str, amp: str) -> Dict[str, float]:
    pair, deep, closure = models
    pair.eval(); deep.eval(); closure.eval()
    metrics: Dict[str, List[float]] = {}
    curves = []
    for _ in range(batches):
        b = make_batch(A_clean, batch_size, drop_p, add_p, device)
        targets = {
            "edge": b.target_edge,
            "path2": b.target_path2,
            "comm": b.target_comm,
        }
        raw = b.corrupt_adj
        common = b.r[..., 1]
        jacc = b.r[..., 3]
        spec = spectral_score(b.corrupt_adj)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(amp == "fp16" and b.r.is_cuda)):
            lp = pair(b.r)
            ld = deep(b.r)
            lc, curve = closure(b.r)
        curves.append(curve[-1] - curve[0] if curve else 0.0)
        model_scores = {
            "raw": raw,
            "common": common,
            "jaccard": jacc,
            "spectral": spec,
            "pair": torch.sigmoid(lp.float()),
            "deep": torch.sigmoid(ld.float()),
            "closure": torch.sigmoid(lc.float()),
        }
        for task_idx, task_name in enumerate(["edge", "path2", "comm"]):
            tgt = targets[task_name]
            for name, sc in model_scores.items():
                if sc.dim() == 4:
                    s = sc[..., task_idx]
                else:
                    s = sc
                key = f"{task_name}_{name}"
                metrics.setdefault(key, []).append(collect_auc(s.float(), tgt.float()))
    out = {k: float(np.nanmean(v)) for k, v in metrics.items()}
    out["curve_drop"] = float(-np.nanmean(curves))  # positive means last < first
    return out


def print_eval(step: int, res: Dict[str, float]) -> None:
    # focus on strongest tests
    def g(task, base): return res.get(f"{task}_{base}", float("nan"))
    line = (
        f"step {step:05d} | "
        f"edge: raw={g('edge','raw'):.3f} cn={g('edge','common'):.3f} spec={g('edge','spectral'):.3f} deep={g('edge','deep'):.3f} cl={g('edge','closure'):.3f} Δ={g('edge','closure')-max(g('edge','deep'),g('edge','spectral'),g('edge','common')):+.3f} | "
        f"path2: raw={g('path2','raw'):.3f} cn={g('path2','common'):.3f} spec={g('path2','spectral'):.3f} deep={g('path2','deep'):.3f} cl={g('path2','closure'):.3f} Δ={g('path2','closure')-max(g('path2','deep'),g('path2','spectral'),g('path2','common')):+.3f} | "
        f"comm: raw={g('comm','raw'):.3f} cn={g('comm','common'):.3f} spec={g('comm','spectral'):.3f} deep={g('comm','deep'):.3f} cl={g('comm','closure'):.3f} Δ={g('comm','closure')-max(g('comm','deep'),g('comm','spectral'),g('comm','common')):+.3f} | "
        f"curve_drop={res.get('curve_drop',0):.4f}"
    )
    print(line, flush=True)


def append_summary(path: str, dataset: str, graph_name: str, mode: str, res: Dict[str, float], args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["dataset","graph","mode","metric","value","n","dim","hidden","iters","steps","drop","add"])
        for k, v in sorted(res.items()):
            w.writerow([dataset, graph_name, mode, k, f"{v:.6f}", args.n, args.dim, args.hidden, args.iters, args.steps, args.drop_p, args.add_p])


# ------------------------- main -------------------------

def run_graph(name: str, G: nx.Graph, args) -> None:
    print("\n" + "="*80, flush=True)
    print(f"REAL GRAPH: {name} nodes={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)
    rng = random.Random(args.seed)
    device = args.device
    A = graph_to_adj(G, args.n, rng).to(device)
    N_eff = int((A.sum(0) + A.sum(1) > 0).sum().item())
    print(f"sample/crop n={args.n}, active_nodes≈{N_eff}, clean_edges={int(A.sum().item()/2)}", flush=True)
    in_dim = 8
    pair = PairMLP(in_dim, args.hidden).to(device)
    deep = DeepContextMLP(in_dim, args.hidden).to(device)
    closure = ClosureLayer(in_dim, args.dim, args.hidden, args.iters).to(device)
    print(f"params: pair={count_params(pair):,} deep={count_params(deep):,} closure={count_params(closure):,}", flush=True)
    opt = torch.optim.AdamW(list(pair.parameters()) + list(deep.parameters()) + list(closure.parameters()), lr=args.lr, weight_decay=1e-4)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        b = make_batch(A, args.batch, args.drop_p, args.add_p, device)
        loss, curve = train_step((pair, deep, closure), opt, b, args.amp)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            res_in = eval_models((pair, deep, closure), A, args.eval_batch, args.eval_batches, args.drop_p, args.add_p, device, args.amp)
            res_ood = eval_models((pair, deep, closure), A, args.eval_batch, args.eval_batches, args.hard_drop_p, args.hard_add_p, device, args.amp)
            print(f"train step {step:05d} loss={loss:.4f} t={time.time()-t0:.1f}s", flush=True)
            print("  IN :", end=" ", flush=True); print_eval(step, res_in)
            print("  OOD:", end=" ", flush=True); print_eval(step, res_ood)
            if args.results_csv:
                append_summary(args.results_csv, "real_graph", name, f"in_step{step}", res_in, args)
                append_summary(args.results_csv, "real_graph", name, f"ood_step{step}", res_ood, args)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({"pair": pair.state_dict(), "deep": deep.state_dict(), "closure": closure.state_dict(), "args": vars(args)}, os.path.join(args.save_dir, f"closure_graph_{name}.pt"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", default="fp16", choices=["none", "fp16"])
    p.add_argument("--datasets", default="builtin", help="builtin,karate,lesmis,florentine,davis,cora,all or comma-list")
    p.add_argument("--allow-download", action="store_true", help="allow Cora download")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--eval-batch", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=3)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--drop-p", type=float, default=0.25)
    p.add_argument("--add-p", type=float, default=0.06)
    p.add_argument("--hard-drop-p", type=float, default=0.45)
    p.add_argument("--hard-add-p", type=float, default=0.15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--results-csv", default="results/closure_graph_real_v1_summary.csv")
    p.add_argument("--save-dir", default="")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print("ClosureGraphRealBenchmark v1.1 — real graph repair/path/community", flush=True)
    print(f"device={args.device} amp={args.amp} datasets={args.datasets} n={args.n} dim={args.dim} hidden={args.hidden} iters={args.iters}", flush=True)
    print(f"train corruption drop/add={args.drop_p}/{args.add_p}; OOD={args.hard_drop_p}/{args.hard_add_p}", flush=True)
    names = []
    if args.datasets == "builtin":
        names = ["karate", "lesmis", "florentine", "davis"]
    elif args.datasets == "all":
        names = ["karate", "lesmis", "florentine", "davis", "cora"]
    else:
        names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for name in names:
        try:
            if name == "cora":
                G = load_cora(args.allow_download)
            else:
                G = load_builtin_graph(name)
        except Exception as e:
            print(f"[SKIP] {name}: {e}", flush=True)
            continue
        run_graph(name, G, args)
    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
