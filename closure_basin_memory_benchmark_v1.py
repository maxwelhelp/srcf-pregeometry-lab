#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closure Basin Memory Benchmark v1

Цель: проверить именно память/basin retrieval, а не обычный graph repair.
Есть K clean-графов памяти. Query = damaged(A) + random false edges + foreign edges from B.
Модель должна:
  1) определить basin A,
  2) восстановить clean A,
  3) откинуть чужие foreign edges,
  4) показать, помогает ли AttractorBank относительно closure без памяти.

Без интернета. Использует networkx builtin-графы + sampled subgraphs.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import networkx as nx
except Exception as e:
    raise SystemExit("networkx нужен для builtin-графов: pip install networkx") from e

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


# --------------------------- utils ---------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auroc(y_true, y_score) -> float:
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(y_score).reshape(-1)
    if roc_auc_score is None:
        return float('nan')
    if len(np.unique(y)) < 2:
        return float('nan')
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float('nan')


def ap_score(y_true, y_score) -> float:
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(y_score).reshape(-1)
    if average_precision_score is None:
        return float('nan')
    if len(np.unique(y)) < 2:
        return float('nan')
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float('nan')


def sym_bin(A: np.ndarray) -> np.ndarray:
    A = (A > 0).astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    return A


def graph_to_adj(G: nx.Graph, n: int, seed: int = 0) -> np.ndarray:
    """Sample/pad graph to fixed n."""
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    if len(nodes) > n:
        # random BFS-ish crop: choose start then expand neighbors
        start = rng.choice(nodes)
        chosen = [start]
        frontier = [start]
        seen = {start}
        while len(chosen) < n and frontier:
            u = frontier.pop(0)
            neigh = list(G.neighbors(u))
            rng.shuffle(neigh)
            for v in neigh:
                if v not in seen:
                    seen.add(v); chosen.append(v); frontier.append(v)
                    if len(chosen) >= n:
                        break
        if len(chosen) < n:
            rest = [x for x in nodes if x not in seen]
            rng.shuffle(rest)
            chosen += rest[: n - len(chosen)]
        nodes = chosen[:n]
    H = G.subgraph(nodes).copy()
    A0 = nx.to_numpy_array(H, nodelist=nodes, weight=None, dtype=np.float32)
    A = np.zeros((n, n), dtype=np.float32)
    m = min(n, A0.shape[0])
    A[:m, :m] = A0[:m, :m]
    return sym_bin(A)


def builtin_graphs() -> Dict[str, nx.Graph]:
    graphs = {
        "karate": nx.karate_club_graph(),
        "lesmis": nx.les_miserables_graph(),
        "florentine": nx.florentine_families_graph(),
        "davis": nx.davis_southern_women_graph(),
    }
    # Convert bipartite/string labels ok; all undirected builtin.
    return graphs


def make_basins(k: int, n: int, seed: int) -> Tuple[np.ndarray, List[str]]:
    """Create K clean basin adjacency matrices [K,N,N]."""
    rng = np.random.default_rng(seed)
    bases = builtin_graphs()
    basins: List[np.ndarray] = []
    names: List[str] = []

    # first use the four canonical builtin graphs
    for idx, (name, G) in enumerate(bases.items()):
        if len(basins) >= k:
            break
        basins.append(graph_to_adj(G, n=n, seed=seed + idx * 11))
        names.append(name)

    # add sampled ego/subgraphs from larger graphs to increase capacity
    source_items = list(bases.items())
    attempt = 0
    while len(basins) < k and attempt < 200:
        name, G = source_items[attempt % len(source_items)]
        A = graph_to_adj(G, n=n, seed=seed + 1000 + attempt)
        # small random rewiring/drop for distinct basins but still real-derived
        # keep it light so it remains a subgraph-like memory, not pure synthetic
        mask = rng.random((n, n))
        mask = np.triu(mask, 1)
        drop = (mask < 0.03).astype(np.float32)
        A2 = A.copy()
        A2[drop > 0] = 0
        A2 = sym_bin(A2)
        # reject near-duplicates by cosine similarity
        flat = A2.reshape(-1)
        ok = True
        for B in basins:
            sim = (flat @ B.reshape(-1)) / (np.linalg.norm(flat) * np.linalg.norm(B.reshape(-1)) + 1e-8)
            if sim > 0.98:
                ok = False
                break
        if ok:
            basins.append(A2)
            names.append(f"{name}_sub{attempt}")
        attempt += 1
    if len(basins) < k:
        raise RuntimeError(f"Could not create K={k} basins")
    return np.stack(basins, axis=0).astype(np.float32), names


def upper_no_diag_mask(n: int, device=None) -> torch.Tensor:
    m = torch.ones(n, n, dtype=torch.bool, device=device)
    m.fill_diagonal_(False)
    return m


@dataclass
class Batch:
    query: torch.Tensor       # [B,N,N]
    clean: torch.Tensor       # [B,N,N]
    foreign: torch.Tensor     # [B,N,N]
    basin: torch.Tensor       # [B]
    foreign_basin: torch.Tensor  # [B]


class BasinGenerator:
    def __init__(self, basins_np: np.ndarray, device: str, drop_p: float, add_p: float, foreign_p: float, seed: int):
        self.basins = torch.tensor(basins_np, dtype=torch.float32, device=device)
        self.K, self.N, _ = self.basins.shape
        self.drop_p = drop_p
        self.add_p = add_p
        self.foreign_p = foreign_p
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.device = device
        self.mask = upper_no_diag_mask(self.N, device=device).float()

    def sample(self, batch: int) -> Batch:
        K, N = self.K, self.N
        device = self.device
        basin = torch.randint(0, K, (batch,), device=device, generator=self.rng)
        foreign_basin = torch.randint(0, K - 1, (batch,), device=device, generator=self.rng)
        foreign_basin = foreign_basin + (foreign_basin >= basin).long()
        clean = self.basins[basin].clone()
        foreign_clean = self.basins[foreign_basin]

        # Work upper triangular then symmetrize.
        rand = torch.rand((batch, N, N), device=device, generator=self.rng)
        keep = ((rand > self.drop_p).float() * clean)

        rand_add = torch.rand((batch, N, N), device=device, generator=self.rng)
        non_edges = (1.0 - clean) * self.mask
        add = ((rand_add < self.add_p).float() * non_edges)

        rand_f = torch.rand((batch, N, N), device=device, generator=self.rng)
        foreign_candidates = foreign_clean * (1.0 - clean) * self.mask
        foreign_edges = ((rand_f < self.foreign_p).float() * foreign_candidates)

        q = torch.maximum(keep, torch.maximum(add, foreign_edges))
        q = torch.triu(q, diagonal=1)
        q = q + q.transpose(1, 2)

        f = torch.triu(foreign_edges, diagonal=1)
        f = f + f.transpose(1, 2)
        return Batch(query=q, clean=clean, foreign=f, basin=basin, foreign_basin=foreign_basin)


# --------------------------- baselines ---------------------------

def raw_nearest_scores(query: torch.Tensor, basins: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predicted basin by raw cosine and selected clean adjacency/foreign score."""
    B, N, _ = query.shape
    K = basins.shape[0]
    qf = query.reshape(B, -1)
    bf = basins.reshape(K, -1)
    sim = F.cosine_similarity(qf[:, None, :], bf[None, :, :], dim=-1)  # [B,K]
    pred = sim.argmax(dim=-1)
    selected = basins[pred]
    # score for foreign/noisy edge: edge present in query but absent in selected memory
    foreign_score = query * (1.0 - selected)
    return pred, selected, foreign_score


# --------------------------- models ---------------------------

class DeepBigMemory(nn.Module):
    def __init__(self, n: int, k: int, hidden: int = 256, depth: int = 4):
        super().__init__()
        self.n = n
        layers = []
        inp = n * n
        for _ in range(depth):
            layers += [nn.Linear(inp, hidden), nn.GELU(), nn.LayerNorm(hidden)]
            inp = hidden
        self.net = nn.Sequential(*layers)
        self.cls = nn.Linear(hidden, k)
        self.edge = nn.Linear(hidden, n * n)
        self.foreign = nn.Linear(hidden, n * n)

    def forward(self, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, N, _ = q.shape
        z = self.net(q.reshape(B, -1))
        edge = self.edge(z).reshape(B, N, N)
        foreign = self.foreign(z).reshape(B, N, N)
        return {"basin_logits": self.cls(z), "edge_logits": edge, "foreign_logits": foreign}


class AttractorBank(nn.Module):
    def __init__(self, dim: int, n_slots: int = 16, beta: float = 1.0):
        super().__init__()
        self.bank = nn.Parameter(torch.randn(n_slots, dim) * 0.02)
        self.beta = beta

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # h [B,N,N,D]
        sim = torch.einsum('bijd,kd->bijk', h, F.normalize(self.bank, dim=-1)) * self.beta
        attn = sim.softmax(dim=-1)
        mem = torch.einsum('bijk,kd->bijd', attn, self.bank)
        pmax = attn.max(dim=-1).values.mean()
        ent = (-(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1)).mean()
        h_norm = F.normalize(h, dim=-1)
        b_norm = F.normalize(self.bank, dim=-1)
        cos = torch.einsum('bijd,kd->bijk', h_norm, b_norm).max(dim=-1).values.mean()
        return mem, {"slot_pmax": pmax, "slot_entropy": ent, "slot_cos": cos}


class ClosureMemoryNet(nn.Module):
    def __init__(self, n: int, k: int, dim: int = 48, hidden: int = 96, iters: int = 5,
                 use_memory: bool = False, memory_slots: int = 16, memory_beta: float = 2.0,
                 memory_strength: float = 0.15, late_memory: bool = False):
        super().__init__()
        self.n = n
        self.k = k
        self.dim = dim
        self.iters = iters
        self.use_memory = use_memory
        self.memory_strength = memory_strength
        self.late_memory = late_memory
        self.enc = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.LayerNorm(dim))
        # h,row,col,glob,tri,raw_enc
        self.step = nn.Sequential(
            nn.Linear(dim * 6, hidden), nn.GELU(), nn.LayerNorm(hidden),
            nn.Linear(hidden, dim),
        )
        self.mem = AttractorBank(dim, memory_slots, memory_beta) if use_memory else None
        self.desc = nn.Sequential(nn.Linear(dim * 3, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.cls = nn.Linear(hidden, k)
        self.edge = nn.Sequential(nn.Linear(dim + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.foreign = nn.Sequential(nn.Linear(dim + 1, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def step_once(self, h: torch.Tensor, raw_enc: torch.Tensor, raw_q: torch.Tensor, t: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, _, D = h.shape
        row = h.mean(dim=2, keepdim=True).expand(-1, -1, N, -1)
        col = h.mean(dim=1, keepdim=True).expand(-1, N, -1, -1)
        glob = h.mean(dim=(1, 2), keepdim=True).expand(-1, N, N, -1)
        hc = h.permute(0, 3, 1, 2).contiguous()  # [B,D,N,N]
        tri = torch.matmul(hc, hc) / max(1, N)
        tri = tri.permute(0, 2, 3, 1).contiguous()
        x = torch.cat([h, row, col, glob, tri, raw_enc], dim=-1)
        dh = self.step(x)
        h2 = h + 0.35 * torch.tanh(dh)
        info: Dict[str, torch.Tensor] = {}
        if self.mem is not None:
            mem, mi = self.mem(h2)
            if self.late_memory and self.iters > 1:
                s = self.memory_strength * (float(t) / float(self.iters - 1)) ** 2
            else:
                s = self.memory_strength
            h2 = h2 + s * (mem - h2)
            info.update({k: v.detach() for k, v in mi.items()})
            info["mem_strength_t"] = torch.tensor(s, device=h.device)
        delta = (h2 - h).pow(2).mean().sqrt()
        info["delta"] = delta.detach()
        return h2, info

    def forward(self, q: torch.Tensor, iters: int | None = None) -> Dict[str, torch.Tensor]:
        if iters is None:
            iters = self.iters
        B, N, _ = q.shape
        raw = q.unsqueeze(-1)
        raw_enc = self.enc(raw)
        h = raw_enc
        curves = []
        mem_logs = []
        for t in range(iters):
            h, info = self.step_once(h, raw_enc, q, t)
            curves.append(info.get("delta", torch.tensor(0., device=q.device)))
            if "slot_cos" in info:
                mem_logs.append(info)
        mean = h.mean(dim=(1, 2))
        std = h.std(dim=(1, 2))
        mx = h.amax(dim=(1, 2))
        z = self.desc(torch.cat([mean, std, mx], dim=-1))
        basin_logits = self.cls(z)
        pair_in = torch.cat([h, raw], dim=-1)
        edge_logits = self.edge(pair_in).squeeze(-1)
        foreign_logits = self.foreign(pair_in).squeeze(-1)
        out = {"basin_logits": basin_logits, "edge_logits": edge_logits, "foreign_logits": foreign_logits,
               "curve": torch.stack(curves) if curves else torch.empty(0, device=q.device)}
        if mem_logs:
            for key in ["slot_cos", "slot_entropy", "slot_pmax", "mem_strength_t"]:
                vals = [m[key] for m in mem_logs if key in m]
                if vals:
                    out[key] = torch.stack(vals)
        # closure_error map: last step delta cannot be exact without storing prev, use residual to edge probability as proxy optional
        return out


# --------------------------- train/eval ---------------------------

def loss_for(out: Dict[str, torch.Tensor], batch: Batch, edge_w: float = 0.5, foreign_w: float = 0.25) -> torch.Tensor:
    loss_cls = F.cross_entropy(out["basin_logits"], batch.basin)
    loss_edge = F.binary_cross_entropy_with_logits(out["edge_logits"], batch.clean)
    loss_foreign = F.binary_cross_entropy_with_logits(out["foreign_logits"], batch.foreign)
    return loss_cls + edge_w * loss_edge + foreign_w * loss_foreign


@torch.no_grad()
def eval_model(name: str, model, gen: BasinGenerator, basins: torch.Tensor, eval_batch: int, eval_batches: int, amp: str) -> Dict[str, float]:
    device_type = 'cuda' if basins.is_cuda else 'cpu'
    y_basin = []
    pred_basin = []
    basin_conf = []
    clean_all = []
    edge_score_all = []
    foreign_all = []
    foreign_score_all = []
    mem_diag = {"slot_cos_last": [], "slot_pmax_last": [], "slot_entropy_last": [], "curve_ratio": []}

    for _ in range(eval_batches):
        b = gen.sample(eval_batch)
        if name == "raw_nearest":
            pred, selected, fscore = raw_nearest_scores(b.query, basins)
            y_basin.append(b.basin.detach().cpu().numpy())
            pred_basin.append(pred.detach().cpu().numpy())
            basin_conf.append(torch.ones_like(pred, dtype=torch.float32).detach().cpu().numpy())
            clean_all.append(b.clean.detach().cpu().numpy())
            edge_score_all.append(selected.detach().cpu().numpy())
            foreign_all.append(b.foreign.detach().cpu().numpy())
            foreign_score_all.append(fscore.detach().cpu().numpy())
        else:
            model.eval()
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(amp == 'fp16' and device_type == 'cuda')):
                out = model(b.query)
            prob = out["basin_logits"].softmax(dim=-1)
            pred = prob.argmax(dim=-1)
            y_basin.append(b.basin.detach().cpu().numpy())
            pred_basin.append(pred.detach().cpu().numpy())
            basin_conf.append(prob.max(dim=-1).values.detach().cpu().numpy())
            clean_all.append(b.clean.detach().cpu().numpy())
            edge_score_all.append(out["edge_logits"].sigmoid().detach().float().cpu().numpy())
            foreign_all.append(b.foreign.detach().cpu().numpy())
            foreign_score_all.append(out["foreign_logits"].sigmoid().detach().float().cpu().numpy())
            if "slot_cos" in out:
                mem_diag["slot_cos_last"].append(float(out["slot_cos"][-1].detach().float().cpu()))
                mem_diag["slot_pmax_last"].append(float(out["slot_pmax"][-1].detach().float().cpu()))
                mem_diag["slot_entropy_last"].append(float(out["slot_entropy"][-1].detach().float().cpu()))
            if "curve" in out and len(out["curve"]) > 1:
                c = out["curve"].detach().float().cpu().numpy()
                mem_diag["curve_ratio"].append(float(c[-1] / (c[0] + 1e-8)))

    yb = np.concatenate(y_basin)
    pb = np.concatenate(pred_basin)
    acc = float((yb == pb).mean())
    edge_auc = auroc(np.concatenate(clean_all), np.concatenate(edge_score_all))
    edge_ap = ap_score(np.concatenate(clean_all), np.concatenate(edge_score_all))
    foreign_auc = auroc(np.concatenate(foreign_all), np.concatenate(foreign_score_all))
    foreign_ap = ap_score(np.concatenate(foreign_all), np.concatenate(foreign_score_all))
    res = {
        "retrieval_acc": acc,
        "edge_auc": edge_auc,
        "edge_ap": edge_ap,
        "foreign_auc": foreign_auc,
        "foreign_ap": foreign_ap,
    }
    for k, vals in mem_diag.items():
        if vals:
            res[k] = float(np.mean(vals))
    return res


def format_metrics(prefix: str, m: Dict[str, float]) -> str:
    parts = [f"{prefix}: acc={m['retrieval_acc']:.3f}", f"edge={m['edge_auc']:.3f}", f"foreign={m['foreign_auc']:.3f}"]
    if "slot_cos_last" in m:
        parts.append(f"slot_cos={m['slot_cos_last']:.3f}")
        parts.append(f"slot_pmax={m['slot_pmax_last']:.3f}")
        parts.append(f"slot_H={m['slot_entropy_last']:.2f}")
    if "curve_ratio" in m:
        parts.append(f"curve_ratio={m['curve_ratio']:.3f}")
    return " ".join(parts)


def train_one(args, K: int, seed: int, writer=None) -> Dict[str, Dict[str, float]]:
    seed_all(seed)
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    basins_np, names = make_basins(K, args.n, seed)
    basins = torch.tensor(basins_np, dtype=torch.float32, device=device)
    train_gen = BasinGenerator(basins_np, device, args.drop_p, args.add_p, args.foreign_p, seed + 10)
    hard_gen = BasinGenerator(basins_np, device, args.hard_drop_p, args.hard_add_p, args.hard_foreign_p, seed + 20)

    deep = DeepBigMemory(args.n, K, hidden=args.deep_hidden, depth=args.deep_depth).to(device)
    cl_nomem = ClosureMemoryNet(args.n, K, dim=args.dim, hidden=args.hidden, iters=args.iters,
                                use_memory=False).to(device)
    cl_mem = ClosureMemoryNet(args.n, K, dim=args.dim, hidden=args.hidden, iters=args.iters,
                              use_memory=True, memory_slots=args.memory_slots, memory_beta=args.memory_beta,
                              memory_strength=args.memory_strength, late_memory=args.late_memory).to(device)
    params = {
        "deep_big": sum(p.numel() for p in deep.parameters()),
        "closure_no_mem": sum(p.numel() for p in cl_nomem.parameters()),
        "closure_mem": sum(p.numel() for p in cl_mem.parameters()),
    }
    print(f"\n=== K={K} seed={seed} basins={names[:min(6,len(names))]}{'...' if len(names)>6 else ''}")
    print(f"params: deep_big={params['deep_big']:,} closure_no_mem={params['closure_no_mem']:,} closure_mem={params['closure_mem']:,}")

    opt = torch.optim.AdamW(list(deep.parameters()) + list(cl_nomem.parameters()) + list(cl_mem.parameters()), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == 'fp16' and device == 'cuda'))
    device_type = 'cuda' if device == 'cuda' else 'cpu'
    t0 = time.time()

    for step in range(1, args.steps + 1):
        b = train_gen.sample(args.batch)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(args.amp == 'fp16' and device == 'cuda')):
            out_d = deep(b.query)
            out_n = cl_nomem(b.query)
            out_m = cl_mem(b.query)
            loss = loss_for(out_d, b) + loss_for(out_n, b) + loss_for(out_m, b)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(deep.parameters()) + list(cl_nomem.parameters()) + list(cl_mem.parameters()), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            raw_in = eval_model("raw_nearest", None, train_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            deep_in = eval_model("deep", deep, train_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            nomem_in = eval_model("nomem", cl_nomem, train_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            mem_in = eval_model("mem", cl_mem, train_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            raw_h = eval_model("raw_nearest", None, hard_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            deep_h = eval_model("deep", deep, hard_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            nomem_h = eval_model("nomem", cl_nomem, hard_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            mem_h = eval_model("mem", cl_mem, hard_gen, basins, args.eval_batch, args.eval_batches, args.amp)
            print(f"step {step:05d}/{args.steps} loss={float(loss.detach().cpu()):.4f} t={time.time()-t0:.1f}s")
            print("  IN   " + format_metrics("raw", raw_in))
            print("       " + format_metrics("deep", deep_in))
            print("       " + format_metrics("cl_no_mem", nomem_in))
            print("       " + format_metrics("cl_mem", mem_in) + f" | memΔ_acc={mem_in['retrieval_acc']-nomem_in['retrieval_acc']:+.3f} memΔ_foreign={mem_in['foreign_auc']-nomem_in['foreign_auc']:+.3f}")
            print("  HARD " + format_metrics("raw", raw_h))
            print("       " + format_metrics("deep", deep_h))
            print("       " + format_metrics("cl_no_mem", nomem_h))
            print("       " + format_metrics("cl_mem", mem_h) + f" | memΔ_acc={mem_h['retrieval_acc']-nomem_h['retrieval_acc']:+.3f} memΔ_foreign={mem_h['foreign_auc']-nomem_h['foreign_auc']:+.3f}")
            if writer is not None:
                for split, metrics_by_name in [("in", {"raw": raw_in, "deep": deep_in, "closure_no_mem": nomem_in, "closure_mem": mem_in}),
                                               ("hard", {"raw": raw_h, "deep": deep_h, "closure_no_mem": nomem_h, "closure_mem": mem_h})]:
                    for model_name, mm in metrics_by_name.items():
                        row = {"K": K, "seed": seed, "step": step, "split": split, "model": model_name}
                        row.update(mm)
                        writer.writerow(row)
    return {}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', default='fp16', choices=['none','fp16'])
    p.add_argument('--K-list', default='2,4,8')
    p.add_argument('--seeds', default='0')
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--eval-batch', type=int, default=64)
    p.add_argument('--eval-batches', type=int, default=3)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--dim', type=int, default=48)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--iters', type=int, default=5)
    p.add_argument('--deep-hidden', type=int, default=256)
    p.add_argument('--deep-depth', type=int, default=4)
    p.add_argument('--memory-slots', type=int, default=16)
    p.add_argument('--memory-beta', type=float, default=1.5)
    p.add_argument('--memory-strength', type=float, default=0.12)
    p.add_argument('--late-memory', action='store_true')
    p.add_argument('--drop-p', type=float, default=0.40)
    p.add_argument('--add-p', type=float, default=0.08)
    p.add_argument('--foreign-p', type=float, default=0.25)
    p.add_argument('--hard-drop-p', type=float, default=0.60)
    p.add_argument('--hard-add-p', type=float, default=0.15)
    p.add_argument('--hard-foreign-p', type=float, default=0.40)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--eval-every', type=int, default=30)
    p.add_argument('--results-csv', default='results/closure_basin_memory_v1.csv')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.results_csv) or '.', exist_ok=True)
    Ks = [int(x) for x in args.K_list.split(',') if x.strip()]
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    print("Closure Basin Memory Benchmark v1")
    print(f"device={args.device} amp={args.amp} K={Ks} seeds={seeds} n={args.n} dim={args.dim} iters={args.iters}")
    print(f"query: drop={args.drop_p} add={args.add_p} foreign={args.foreign_p}; HARD drop={args.hard_drop_p} add={args.hard_add_p} foreign={args.hard_foreign_p}")
    print("metrics: retrieval_acc, edge_auc, foreign_auc. raw_nearest is the must-beat memory baseline.")

    fieldnames = ["K","seed","step","split","model","retrieval_acc","edge_auc","edge_ap","foreign_auc","foreign_ap",
                  "slot_cos_last","slot_pmax_last","slot_entropy_last","curve_ratio"]
    with open(args.results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if f.tell() == 0:
            writer.writeheader()
        for K in Ks:
            for seed in seeds:
                train_one(args, K, seed, writer=writer)
    print(f"\nDone. Results appended: {args.results_csv}")


if __name__ == '__main__':
    main()
