#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Closure Relation Diagnostics v1

Один файл, но три проверки:

1) mode=other
   Проверяет наше сильное место: foreign/other relation detection.
   Важно: это НЕ память. Это диагностика closure как self/other фильтра.

2) mode=memory
   Проверяет настоящую graph-level basin memory:
   query = damaged(A) + foreign edges(B)
   надо выбрать basin A, восстановить A, откинуть чужие связи.

3) mode=nongraph
   Проверяет тезис "не граф, а матрица отношений" на co-occurrence матрицах
   из реальных локальных текстов/кода репозитория.

Запуск быстрый:
  python -u closure_relation_diagnostics_v1.py --mode all --device cuda --amp fp16 \
    --K-list 2,4 --steps 120 --batch 24 --eval-batch 48 --eval-every 30
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import re
import time
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
    nx = None

# ------------------------- utils -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auc_np(y_true, y_score) -> float:
    y = np.asarray(y_true).astype(np.int64).reshape(-1)
    s = np.asarray(y_score).astype(np.float64).reshape(-1)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if y.size == 0 or len(np.unique(y)) < 2:
        return float('nan')
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    # average ranks for ties, 1-indexed
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i + 1
        while j < len(sorted_s) and sorted_s[j] == sorted_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[y == 1].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def best_auc_np(y_true, y_score) -> Tuple[float, float, float]:
    a = auc_np(y_true, y_score)
    if not np.isfinite(a):
        return float('nan'), float('nan'), float('nan')
    return a, 1.0 - a, max(a, 1.0 - a)


def offdiag_mask(n: int, device=None) -> torch.Tensor:
    return (~torch.eye(n, dtype=torch.bool, device=device))


def sym_bin(a: torch.Tensor) -> torch.Tensor:
    a = torch.triu(a, diagonal=1)
    a = a + a.transpose(-1, -2)
    return (a > 0).float()


def to_device_batch(batch, device: torch.device):
    return Batch(
        query=batch.query.to(device),
        clean=batch.clean.to(device),
        foreign=batch.foreign.to(device),
        label=batch.label.to(device),
        candidate=batch.candidate.to(device),
    )

# ------------------------- basins -------------------------

def nx_to_adj(G, n: int, seed: int = 0) -> np.ndarray:
    # connected largest component, sample/crop/pad to n
    if G.is_directed():
        H = G.copy()
    else:
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        H = G.subgraph(list(comps[0])).copy()
    nodes = list(H.nodes())
    rng = random.Random(seed)
    if len(nodes) > n:
        # BFS-ish sample for coherent real subgraph
        start = rng.choice(nodes)
        seen = [start]
        q = [start]
        while q and len(seen) < n:
            u = q.pop(0)
            neigh = list(H.neighbors(u))
            rng.shuffle(neigh)
            for v in neigh:
                if v not in seen:
                    seen.append(v)
                    q.append(v)
                    if len(seen) >= n:
                        break
        if len(seen) < n:
            rest = [x for x in nodes if x not in seen]
            rng.shuffle(rest)
            seen.extend(rest[: n - len(seen)])
        nodes = seen[:n]
        H = H.subgraph(nodes).copy()
    else:
        nodes = nodes[:]
    idx = {node: i for i, node in enumerate(nodes)}
    a = np.zeros((n, n), dtype=np.float32)
    for u, v in H.edges():
        if u in idx and v in idx:
            i, j = idx[u], idx[v]
            if i != j:
                a[i, j] = 1.0
                if not H.is_directed():
                    a[j, i] = 1.0
    np.fill_diagonal(a, 0.0)
    return a


def builtin_graphs(n: int, K: int, seed: int = 0) -> Tuple[List[str], torch.Tensor]:
    if nx is None:
        raise RuntimeError("networkx не установлен")
    base = []
    base.append(("karate", nx.karate_club_graph()))
    try:
        base.append(("lesmis", nx.les_miserables_graph()))
    except Exception:
        pass
    try:
        base.append(("florentine", nx.florentine_families_graph()))
    except Exception:
        pass
    try:
        base.append(("davis", nx.davis_southern_women_graph()))
    except Exception:
        pass
    # expand by real subgraphs from available larger graphs
    out_names, out_adj = [], []
    s = seed * 1000
    for name, G in base:
        out_names.append(name)
        out_adj.append(nx_to_adj(G, n, s))
        s += 1
        if len(out_names) >= K:
            break
    gi = 0
    while len(out_names) < K:
        name, G = base[gi % len(base)]
        out_names.append(f"{name}_sub{gi}")
        out_adj.append(nx_to_adj(G, n, s + gi * 17))
        gi += 1
    return out_names[:K], torch.tensor(np.stack(out_adj[:K]), dtype=torch.float32)


def load_text_relation_basins(root: str, n: int, K: int, seed: int = 0) -> Tuple[List[str], torch.Tensor]:
    """Build non-graph token co-occurrence relation matrices from local repo text/code files."""
    rng = random.Random(seed)
    rootp = Path(root)
    exts = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}
    files = []
    for p in rootp.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_size < 500_000:
            if any(part.startswith(".git") for part in p.parts):
                continue
            files.append(p)
    rng.shuffle(files)
    docs = []
    for p in files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        toks = re.findall(r"[A-Za-z_А-Яа-я0-9]{3,}", txt.lower())
        if len(toks) >= 200:
            docs.append((str(p.relative_to(rootp))[:40], toks[:6000]))
        if len(docs) >= max(K * 3, K):
            break
    if len(docs) < K:
        raise RuntimeError(f"Недостаточно локальных текстов/кода для nongraph: found={len(docs)}, need={K}")
    # shared vocab by frequency over selected docs
    from collections import Counter
    cnt = Counter()
    for _, toks in docs[: max(K * 2, K)]:
        cnt.update(toks)
    vocab = [w for w, _ in cnt.most_common(n)]
    vid = {w: i for i, w in enumerate(vocab)}
    mats, names = [], []
    for name, toks in docs[:K]:
        mat = np.zeros((n, n), dtype=np.float32)
        ids = [vid[t] for t in toks if t in vid]
        window = 4
        for i, a in enumerate(ids):
            for b in ids[i + 1: i + 1 + window]:
                if a != b:
                    mat[a, b] += 1.0
                    mat[b, a] += 1.0
        # binarize top relation edges, keep roughly 10-20% density
        if mat.max() > 0:
            vals = mat[mat > 0]
            thr = np.quantile(vals, 0.75) if len(vals) > 10 else 0.0
            mat = (mat >= thr).astype(np.float32)
        np.fill_diagonal(mat, 0.0)
        mats.append(mat)
        names.append("text:" + name)
    return names, torch.tensor(np.stack(mats), dtype=torch.float32)

# ------------------------- data generation -------------------------

@dataclass
class Batch:
    query: torch.Tensor      # [B,N,N]
    clean: torch.Tensor      # [B,N,N]
    foreign: torch.Tensor    # [B,N,N]
    label: torch.Tensor      # [B]
    candidate: torch.Tensor  # [B,K,N,N]


def sample_query(clean_bank: torch.Tensor, batch: int, drop_p: float, add_p: float, foreign_p: float, device=None) -> Batch:
    K, N, _ = clean_bank.shape
    clean_bank = clean_bank.to(device) if device is not None else clean_bank
    q_list, c_list, f_list, labels, cand_list = [], [], [], [], []
    eye = torch.eye(N, device=clean_bank.device).bool()
    upper = torch.triu(torch.ones(N, N, device=clean_bank.device).bool(), diagonal=1)
    for _ in range(batch):
        s = torch.randint(0, K, (1,), device=clean_bank.device).item()
        # choose different contaminant if possible
        if K > 1:
            b = torch.randint(0, K - 1, (1,), device=clean_bank.device).item()
            if b >= s:
                b += 1
        else:
            b = s
        A = clean_bank[s]
        B = clean_bank[b]
        # undirected symmetric query corruption
        A_up = (A > 0.5) & upper
        keep = (torch.rand(N, N, device=A.device) > drop_p) & upper
        q_up = A_up & keep
        # random add from non-A
        nonA = (~A_up) & upper
        add = (torch.rand(N, N, device=A.device) < add_p) & nonA
        # foreign edges from B that are not A
        B_up = (B > 0.5) & upper
        foreign_candidates = B_up & (~A_up)
        foreign = (torch.rand(N, N, device=A.device) < foreign_p) & foreign_candidates
        q_up = q_up | add | foreign
        q = q_up.float() + q_up.float().T
        fm = foreign.float() + foreign.float().T
        q = q.masked_fill(eye, 0.0)
        fm = fm.masked_fill(eye, 0.0)
        q_list.append(q)
        c_list.append(A.float())
        f_list.append(fm)
        labels.append(s)
        cand_list.append(clean_bank)
    return Batch(
        query=torch.stack(q_list),
        clean=torch.stack(c_list),
        foreign=torch.stack(f_list),
        label=torch.tensor(labels, device=clean_bank.device, dtype=torch.long),
        candidate=torch.stack(cand_list),
    )


def relation_features(adj: torch.Tensor, rel_mode: str = "full") -> torch.Tensor:
    # adj [B,N,N]
    B, N, _ = adj.shape
    if rel_mode == "raw":
        return adj.unsqueeze(-1)
    deg = adj.sum(-1) / max(1, N - 1)
    di = deg[:, :, None].expand(B, N, N)
    dj = deg[:, None, :].expand(B, N, N)
    deg_diff = (di - dj).abs()
    deg_prod = di * dj
    # common neighbors / path2 normalized
    cn = torch.bmm(adj, adj) / max(1, N)
    union = (di + dj - cn).clamp_min(1e-6)
    jacc = cn / union
    rev = adj.transpose(1, 2)
    return torch.stack([adj, rev, di, dj, deg_diff, deg_prod, cn, jacc], dim=-1)

# ------------------------- models -------------------------

class DeepBig(nn.Module):
    def __init__(self, in_dim: int, Kmax: int, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        self.pair = nn.Sequential(*layers)
        self.edge = nn.Linear(hidden, 1)
        self.foreign = nn.Linear(hidden, 1)
        self.cls = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, Kmax))
    def forward(self, x: torch.Tensor, K: int):
        h = self.pair(x)
        edge = self.edge(h).squeeze(-1)
        foreign = self.foreign(h).squeeze(-1)
        mean = h.mean(dim=(1, 2))
        std = h.std(dim=(1, 2), unbiased=False)
        logits = self.cls(torch.cat([mean, std], dim=-1))[:, :K]
        return edge, foreign, logits, {}

class ClosureCore(nn.Module):
    def __init__(self, in_dim: int, Kmax: int, dim: int = 48, hidden: int = 96, iters: int = 5, graph_memory: bool = False):
        super().__init__()
        self.dim = dim
        self.iters = iters
        self.graph_memory = graph_memory
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # h, rev, row, col, glob, tri
        self.update = nn.Sequential(nn.Linear(dim * 6, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.norm = nn.LayerNorm(dim)
        self.cls_plain = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, Kmax))
        self.prototypes = nn.Parameter(torch.randn(Kmax, dim) * 0.02)
        self.proto_to_ctx = nn.Linear(dim, dim)
        self.edge = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.foreign = nn.Sequential(nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def step_once(self, h: torch.Tensor):
        B, N, _, D = h.shape
        rev = h.transpose(1, 2)
        row = h.mean(2, keepdim=True).expand(B, N, N, D)
        col = h.mean(1, keepdim=True).expand(B, N, N, D)
        glob = h.mean((1, 2), keepdim=True).expand(B, N, N, D)
        hc = h.permute(0, 3, 1, 2).contiguous().view(B * D, N, N)
        tri = torch.bmm(hc, hc) / math.sqrt(max(1, N))
        tri = tri.view(B, D, N, N).permute(0, 2, 3, 1).contiguous()
        u = torch.cat([h, rev, row, col, glob, tri], dim=-1)
        dh = torch.tanh(self.update(u))
        h2 = self.norm(h + 0.25 * dh)
        return h2
    def forward(self, x: torch.Tensor, K: int):
        h = self.embed(x)
        curves = []
        for _ in range(self.iters):
            hp = h
            h = self.step_once(h)
            curves.append((h - hp).pow(2).mean().detach())
        mean = h.mean(dim=(1, 2))
        std = h.std(dim=(1, 2), unbiased=False)
        z = torch.cat([mean, std], dim=-1)
        if self.graph_memory:
            proto = self.prototypes[:K]
            sim = mean @ proto.t() / math.sqrt(self.dim)
            w = F.softmax(sim, dim=-1)
            mem = w @ proto
            ctx = self.proto_to_ctx(mem).view(mem.shape[0], 1, 1, self.dim).expand_as(h)
            logits = sim
            slot_pmax = w.max(dim=-1).values.mean()
            slot_H = (-(w.clamp_min(1e-8) * w.clamp_min(1e-8).log()).sum(-1)).mean()
            slot_cos = F.cosine_similarity(mean, mem, dim=-1).mean()
        else:
            ctx = torch.zeros_like(h)
            logits = self.cls_plain(z)[:, :K]
            slot_pmax = torch.tensor(float('nan'), device=h.device)
            slot_H = torch.tensor(float('nan'), device=h.device)
            slot_cos = torch.tensor(float('nan'), device=h.device)
        dec_in = torch.cat([h, ctx], dim=-1)
        edge = self.edge(dec_in).squeeze(-1)
        foreign = self.foreign(dec_in).squeeze(-1)
        curve_ratio = (curves[-1] / curves[0].clamp_min(1e-8)) if curves else torch.tensor(1.0, device=h.device)
        info = dict(curve_ratio=curve_ratio, slot_pmax=slot_pmax.detach(), slot_H=slot_H.detach(), slot_cos=slot_cos.detach())
        return edge, foreign, logits, info

# ------------------------- baselines/eval -------------------------

def raw_nearest_scores(batch: Batch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns pred labels [B], edge score [B,N,N], foreign score [B,N,N]
    q = batch.query.detach().cpu().numpy()
    cand = batch.candidate.detach().cpu().numpy()
    B, K, N, _ = cand.shape
    pred = []
    edge_scores = []
    foreign_scores = []
    for b in range(B):
        sims = []
        qb = q[b] > 0.5
        for k in range(K):
            ck = cand[b, k] > 0.5
            inter = np.logical_and(qb, ck).sum()
            union = np.logical_or(qb, ck).sum() + 1e-9
            sims.append(inter / union)
        p = int(np.argmax(sims))
        pred.append(p)
        ep = cand[b, p]
        edge_scores.append(ep)
        # observed edge absent from nearest clean => likely foreign
        foreign_scores.append(q[b] * (1.0 - ep))
    return np.asarray(pred), np.stack(edge_scores), np.stack(foreign_scores)


def local_foreign_baselines(batch: Batch) -> Dict[str, np.ndarray]:
    q = batch.query.detach().cpu().numpy()
    B, N, _ = q.shape
    out = {}
    deg = q.sum(-1) / max(1, N - 1)
    deg_diff = np.abs(deg[:, :, None] - deg[:, None, :])
    deg_prod = deg[:, :, None] * deg[:, None, :]
    cn = np.matmul(q, q) / max(1, N)
    union = deg[:, :, None] + deg[:, None, :] - cn + 1e-6
    jacc = cn / union
    out["deg_diff"] = deg_diff
    out["deg_prod"] = deg_prod
    out["low_cn"] = -cn
    out["low_jacc"] = -jacc
    out["edge_absent"] = 1.0 - q
    return out


def eval_outputs(batch: Batch, edge_logits, foreign_logits, cls_logits, prefix: str) -> Dict[str, float]:
    off = offdiag_mask(batch.query.shape[-1], device=batch.query.device)
    label = batch.label.detach().cpu().numpy()
    cls = cls_logits.detach().float().cpu().numpy().argmax(axis=-1)
    acc = float((cls == label).mean())
    edge_score = torch.sigmoid(edge_logits).detach().float().cpu().numpy()
    foreign_score = torch.sigmoid(foreign_logits).detach().float().cpu().numpy()
    clean = batch.clean.detach().cpu().numpy()
    foreign = batch.foreign.detach().cpu().numpy()
    query = batch.query.detach().cpu().numpy()
    off_np = off.detach().cpu().numpy()
    edge_auc = auc_np(clean[:, off_np], edge_score[:, off_np])
    # foreign among observed query edges only
    y_f, s_f = [], []
    for b in range(query.shape[0]):
        m = (query[b] > 0.5) & off_np
        if m.sum() > 0:
            y_f.append(foreign[b][m])
            s_f.append(foreign_score[b][m])
    foreign_auc = auc_np(np.concatenate(y_f), np.concatenate(s_f)) if y_f else float('nan')
    return {f"{prefix}_acc": acc, f"{prefix}_edge": edge_auc, f"{prefix}_foreign": foreign_auc}


def eval_raw(batch: Batch, prefix: str) -> Dict[str, float]:
    pred, edge_s, foreign_s = raw_nearest_scores(batch)
    label = batch.label.detach().cpu().numpy()
    acc = float((pred == label).mean())
    N = batch.query.shape[-1]
    off_np = (~np.eye(N, dtype=bool))
    clean = batch.clean.detach().cpu().numpy()
    foreign = batch.foreign.detach().cpu().numpy()
    query = batch.query.detach().cpu().numpy()
    edge_auc = auc_np(clean[:, off_np], edge_s[:, off_np])
    y_f, s_f = [], []
    for b in range(query.shape[0]):
        m = (query[b] > 0.5) & off_np
        if m.sum() > 0:
            y_f.append(foreign[b][m])
            s_f.append(foreign_s[b][m])
    foreign_auc = auc_np(np.concatenate(y_f), np.concatenate(s_f)) if y_f else float('nan')
    return {f"{prefix}_acc": acc, f"{prefix}_edge": edge_auc, f"{prefix}_foreign": foreign_auc}


def eval_local_baselines(batch: Batch) -> Dict[str, float]:
    res = {}
    loc = local_foreign_baselines(batch)
    foreign = batch.foreign.detach().cpu().numpy()
    query = batch.query.detach().cpu().numpy()
    N = query.shape[-1]
    off_np = (~np.eye(N, dtype=bool))
    for name, score in loc.items():
        y, s = [], []
        for b in range(query.shape[0]):
            m = (query[b] > 0.5) & off_np
            if m.sum() > 0:
                y.append(foreign[b][m])
                s.append(score[b][m])
        hi, lo, best = best_auc_np(np.concatenate(y), np.concatenate(s)) if y else (float('nan'),)*3
        res[f"local_{name}_best_foreign"] = best
        res[f"local_{name}_high_foreign"] = hi
    return res

# ------------------------- train/eval loop -------------------------

def bce_masked(logits, target):
    mask = offdiag_mask(target.shape[-1], device=target.device)
    return F.binary_cross_entropy_with_logits(logits[:, mask], target[:, mask])


def train_one(K: int, seed: int, args, domain: str) -> None:
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    if domain == "nongraph":
        names, bank_cpu = load_text_relation_basins(args.text_root, args.n, K, seed)
    else:
        names, bank_cpu = builtin_graphs(args.n, K, seed)
    bank = bank_cpu.to(device)
    in_dim = 1 if args.rel_mode == 'raw' else 8
    deep = DeepBig(in_dim, Kmax=max(K, 16), hidden=args.deep_hidden, depth=args.deep_depth).to(device)
    cl_no = ClosureCore(in_dim, Kmax=max(K, 16), dim=args.dim, hidden=args.hidden, iters=args.iters, graph_memory=False).to(device)
    cl_mem = ClosureCore(in_dim, Kmax=max(K, 16), dim=args.dim, hidden=args.hidden, iters=args.iters, graph_memory=True).to(device)
    models = [deep, cl_no, cl_mem]
    opt = torch.optim.AdamW([p for m in models for p in m.parameters()], lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == 'fp16' and device.type == 'cuda'))
    print(f"\n=== domain={domain} K={K} seed={seed} basins={names[:K]}")
    print(f"params: deep={sum(p.numel() for p in deep.parameters()):,} cl_no_mem={sum(p.numel() for p in cl_no.parameters()):,} cl_mem={sum(p.numel() for p in cl_mem.parameters()):,}")
    t0 = time.time()
    rows = []
    for step in range(1, args.steps + 1):
        batch = sample_query(bank, args.batch, args.drop_p, args.add_p, args.foreign_p, device=device)
        x = relation_features(batch.query, args.rel_mode)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(args.amp == 'fp16' and device.type == 'cuda')):
            losses = []
            for model in models:
                e, f, c, _ = model(x, K)
                loss = bce_masked(e, batch.clean) + args.foreign_w * bce_masked(f, batch.foreign) + args.cls_w * F.cross_entropy(c, batch.label)
                losses.append(loss)
            loss = sum(losses)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for m in models for p in m.parameters()], args.grad_clip)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            print(f"step {step:05d}/{args.steps} loss={float(loss.detach()):.4f} t={time.time()-t0:.1f}s")
            for split, dp, ap, fp in [("IN", args.drop_p, args.add_p, args.foreign_p), ("HARD", args.hard_drop_p, args.hard_add_p, args.hard_foreign_p)]:
                # aggregate eval batches
                agg = []
                for _ in range(args.eval_batches):
                    eb = sample_query(bank, args.eval_batch, dp, ap, fp, device=device)
                    ex = relation_features(eb.query, args.rel_mode)
                    with torch.no_grad():
                        de = deep(ex, K)
                        no = cl_no(ex, K)
                        me = cl_mem(ex, K)
                    d = {}
                    d.update(eval_raw(eb, "raw"))
                    d.update(eval_outputs(eb, *de[:3], prefix="deep"))
                    d.update(eval_outputs(eb, *no[:3], prefix="cl_no"))
                    d.update(eval_outputs(eb, *me[:3], prefix="cl_mem"))
                    d.update(eval_local_baselines(eb))
                    d["cl_no_curve"] = float(no[3]["curve_ratio"].cpu())
                    d["cl_mem_curve"] = float(me[3]["curve_ratio"].cpu())
                    d["slot_cos"] = float(me[3]["slot_cos"].cpu())
                    d["slot_pmax"] = float(me[3]["slot_pmax"].cpu())
                    d["slot_H"] = float(me[3]["slot_H"].cpu())
                    agg.append(d)
                mean = {k: float(np.nanmean([a[k] for a in agg])) for k in agg[0]}
                print(
                    f"  {split:<4} raw: acc={mean['raw_acc']:.3f} edge={mean['raw_edge']:.3f} foreign={mean['raw_foreign']:.3f} | "
                    f"deep: acc={mean['deep_acc']:.3f} edge={mean['deep_edge']:.3f} foreign={mean['deep_foreign']:.3f} | "
                    f"cl_no: acc={mean['cl_no_acc']:.3f} edge={mean['cl_no_edge']:.3f} foreign={mean['cl_no_foreign']:.3f} curve={mean['cl_no_curve']:.3f} | "
                    f"cl_mem: acc={mean['cl_mem_acc']:.3f} edge={mean['cl_mem_edge']:.3f} foreign={mean['cl_mem_foreign']:.3f} "
                    f"slot_p={mean['slot_pmax']:.2f} H={mean['slot_H']:.2f} curve={mean['cl_mem_curve']:.3f} | "
                    f"deg_best={mean['local_deg_diff_best_foreign']:.3f} cn_best={mean['local_low_cn_best_foreign']:.3f}"
                )
                row = dict(domain=domain, K=K, seed=seed, step=step, split=split)
                row.update(mean)
                rows.append(row)
    if args.results_csv:
        path = Path(args.results_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"appended csv: {path}")

# ------------------------- main -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['other','memory','nongraph','all'], default='all', help='other/memory currently run same diagnostic; nongraph uses text cooccurrence matrices')
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', choices=['no','fp16'], default='fp16')
    p.add_argument('--K-list', default='2,4,8')
    p.add_argument('--seeds', default='0')
    p.add_argument('--steps', type=int, default=120)
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--eval-batch', type=int, default=48)
    p.add_argument('--eval-batches', type=int, default=2)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--dim', type=int, default=48)
    p.add_argument('--hidden', type=int, default=96)
    p.add_argument('--deep-hidden', type=int, default=256)
    p.add_argument('--deep-depth', type=int, default=4)
    p.add_argument('--iters', type=int, default=5)
    p.add_argument('--rel-mode', choices=['raw','full'], default='raw')
    p.add_argument('--drop-p', type=float, default=0.40)
    p.add_argument('--add-p', type=float, default=0.08)
    p.add_argument('--foreign-p', type=float, default=0.25)
    p.add_argument('--hard-drop-p', type=float, default=0.60)
    p.add_argument('--hard-add-p', type=float, default=0.15)
    p.add_argument('--hard-foreign-p', type=float, default=0.40)
    p.add_argument('--foreign-w', type=float, default=1.0)
    p.add_argument('--cls-w', type=float, default=1.0)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--eval-every', type=int, default=30)
    p.add_argument('--results-csv', default='results/closure_relation_diagnostics_v1.csv')
    p.add_argument('--text-root', default='.')
    return p.parse_args()


def main():
    args = parse_args()
    Ks = [int(x) for x in args.K_list.split(',') if x.strip()]
    seeds = [int(x) for x in args.seeds.split(',') if x.strip()]
    print("Closure Relation Diagnostics v1")
    print(f"mode={args.mode} device={args.device} amp={args.amp} K={Ks} seeds={seeds} n={args.n} rel_mode={args.rel_mode}")
    print("checks: raw_nearest, deep_big, closure_no_mem, closure_graph_memory, degree/CN local foreign baselines")
    domains = []
    if args.mode in ('other','memory'):
        domains = ['graph']
    elif args.mode == 'nongraph':
        domains = ['nongraph']
    else:
        domains = ['graph', 'nongraph']
    for domain in domains:
        for K in Ks:
            for seed in seeds:
                try:
                    train_one(K, seed, args, domain)
                except Exception as e:
                    print(f"[SKIP/ERROR] domain={domain} K={K} seed={seed}: {e}")
                    if domain == 'graph':
                        raise
    print("Done.")

if __name__ == '__main__':
    main()
