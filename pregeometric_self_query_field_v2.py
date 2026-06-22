#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-Geometric Self-Query Field v2
---------------------------------
Harder version with operator diversity pressure and stronger diagnostics.

Core rule:
  coordinates are NOT input;
  nodes are anonymous;
  input is only relation tensor R[i,j,c];
  geometry is a verdict/output, not the substrate.

What v1 fixes:
  - harder synthetic tasks with fewer obvious channel/stat leaks;
  - optional per-sample normalization and fixed/random channel mixing;
  - explicit permutation-equivariance test;
  - OOD eval on different N;
  - operator-collapse diagnostics: entropy, effective ops, task-op cosine;
  - optional losses to encourage sparse per-sample ops while keeping global op usage alive.

Tasks:
  0 self/other:     one closure basin vs two independent closure basins
  1 direction:      no consistent axis vs globally consistent directed order
  2 geometry:       metric-like radial relations vs same marginal values with broken metric consistency
  3 boundary mask:  partial closure/fringe nodes, no coordinates

Examples:
  smoke CPU:
    python -u pregeometric_self_query_field_v1.py --device cpu --amp none --train-steps 30 --batch-size 8 --eval-batch-size 8 --eval-batches 1 --n 16 --dim 24 --ops 4 --iters 2 --eval-every 10

  P40 starter:
    python -u pregeometric_self_query_field_v1.py --device cuda --amp fp16 --train-steps 1200 --batch-size 64 --eval-batch-size 64 --eval-batches 2 --n 32 --dim 48 --ops 8 --iters 5 --eval-every 100 --fixed-channel-mix --normalize-rel --ood-n 48 --save-path ./pg_sqf_v1.pt | tee pg_sqf_v1.log
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# utilities
# -----------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: str, amp: str):
    if device.startswith("cuda") and amp != "none":
        dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def random_orthogonal(dim: int, device: str) -> torch.Tensor:
    a = torch.randn(dim, dim, device=device)
    q, r = torch.linalg.qr(a)
    # stabilize sign so seed is deterministic-ish
    q = q * torch.sign(torch.diag(r)).clamp(min=-1, max=1).view(1, -1)
    return q


def _sym(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x + x.transpose(0, 1))


def _normalize_matrix(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mn = x.amin()
    mx = x.amax()
    return (x - mn) / (mx - mn + eps)


def _same_marginal_symmetric(k: torch.Tensor) -> torch.Tensor:
    """Permute off-diagonal values of a symmetric matrix to keep marginal distribution
    but destroy most metric/geometric consistency.
    """
    n = k.shape[0]
    tri = torch.triu_indices(n, n, offset=1, device=k.device)
    vals = k[tri[0], tri[1]]
    vals = vals[torch.randperm(vals.numel(), device=k.device)]
    out = torch.zeros_like(k)
    out[tri[0], tri[1]] = vals
    out = out + out.t()
    out.fill_diagonal_(1.0)
    return out


def inverse_perm(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device)
    return inv


@dataclass
class Batch:
    rel: torch.Tensor       # [B, N, N, rel_dim]
    task: torch.Tensor      # [B]
    cls: torch.Tensor       # [B], valid for task 0/1/2
    node_mask: torch.Tensor # [B, N], valid for task 3


# -----------------------------
# harder coordinate-free generator
# -----------------------------


def _pack_channels(a: torch.Tensor, anti: torch.Tensor, k: torch.Tensor, extra_noise: float = 0.035) -> torch.Tensor:
    """Pack multiple relation views. Channels are later normalized/mixed.
    The semantic meaning of a channel should not be relied on after mixing.
    """
    n = a.shape[0]
    device = a.device
    rel_dim = 8
    rel = torch.zeros(n, n, rel_dim, device=device)
    eye = torch.eye(n, device=device)

    # All tasks fill all channels; we do not leave easy blank channels.
    closure = (a @ a) / max(n, 1)
    closure = _normalize_matrix(closure)
    degree_sum = 0.5 * (a.mean(dim=1, keepdim=True) + a.mean(dim=0, keepdim=True))
    degree_pair = degree_sum.expand(n, n)

    rel[..., 0] = a
    rel[..., 1] = anti
    rel[..., 2] = k
    rel[..., 3] = closure
    rel[..., 4] = degree_pair
    rel[..., 5] = eye
    rel[..., 6] = _sym(torch.randn(n, n, device=device)) * extra_noise + 0.5 * (a + k)
    rel[..., 7] = (torch.randn(n, n, device=device) * extra_noise + 0.25 * anti + 0.25 * (a - k))
    return rel


@torch.no_grad()
def make_one_sample(
    n: int,
    device: str,
    force_task: Optional[int] = None,
    normalize_rel: bool = True,
    channel_mix: Optional[torch.Tensor] = None,
    random_channel_mix: bool = False,
) -> Tuple[torch.Tensor, int, int, torch.Tensor]:
    task = int(torch.randint(0, 4, (), device=device).item()) if force_task is None else int(force_task)
    node_mask = torch.zeros(n, device=device)
    cls = int(torch.randint(0, 2, (), device=device).item())

    # Shared nuisance base for all tasks.
    z = torch.randn(n, 5, device=device)
    base = torch.sigmoid((z @ z.t()) / math.sqrt(z.shape[-1]))
    base = _sym(base)
    base.fill_diagonal_(1.0)
    noise_sym = _sym(torch.rand(n, n, device=device))
    base = (0.65 * base + 0.35 * noise_sym).clamp(0, 1)
    base.fill_diagonal_(1.0)

    anti_noise = torch.randn(n, n, device=device)
    anti_noise = anti_noise - anti_noise.t()
    anti_noise = anti_noise / anti_noise.abs().amax().clamp_min(1e-6)

    # task 0: one closure basin vs two independent basins.
    if task == 0:
        if cls == 0:
            a = base
        else:
            cut = int(torch.randint(max(4, n // 3), min(n - 4, 2 * n // 3), (), device=device).item())
            a = 0.15 * base
            a[:cut, :cut] = 0.78 + 0.22 * torch.rand(cut, cut, device=device)
            a[cut:, cut:] = 0.78 + 0.22 * torch.rand(n - cut, n - cut, device=device)
            a[:cut, cut:] = 0.01 + 0.08 * torch.rand(cut, n - cut, device=device)
            a[cut:, :cut] = 0.01 + 0.08 * torch.rand(n - cut, cut, device=device)
            a = _sym(a)
            a.fill_diagonal_(1.0)
            # Match global mean roughly to reduce trivial mean leak.
            a = (a - a.mean()) / a.std(unbiased=False).clamp_min(1e-6)
            a = torch.sigmoid(a * 0.65 + 0.25)
            a.fill_diagonal_(1.0)
        anti = 0.25 * anti_noise
        k = 0.70 * a + 0.30 * base

    # task 1: direction defined? consistent transitive axis vs random antisymmetric tournament.
    elif task == 1:
        a = base
        if cls == 0:
            anti = anti_noise
        else:
            order = torch.randperm(n, device=device).float()
            anti = torch.tanh((order[:, None] - order[None, :]) / max(n / 5.0, 1.0))
            anti = anti + 0.20 * anti_noise
            anti = anti - anti.t()
            anti = anti / anti.abs().amax().clamp_min(1e-6)
        k = 0.70 * base + 0.30 * _sym(torch.rand(n, n, device=device))
        k.fill_diagonal_(1.0)

    # task 2: geometry valid? hidden radial kernel vs same marginal but non-metric assignment.
    elif task == 2:
        x = torch.randn(n, 2, device=device)
        d2 = torch.cdist(x, x).pow(2)
        sigma2 = d2[d2 > 0].median().clamp_min(1e-3)
        kgood = torch.exp(-d2 / (2.0 * sigma2))
        kgood.fill_diagonal_(1.0)
        if cls == 1:
            k = kgood
        else:
            k = _same_marginal_symmetric(kgood)
            # Inject a few graph-like contradictions while preserving value range.
            idx = torch.randperm(n, device=device)
            aidx = idx[: max(2, n // 4)]
            bidx = idx[max(2, n // 4): max(4, n // 2)]
            k[aidx[:, None], bidx[None, :]] = k[aidx[:, None], bidx[None, :]].flip(0)
            k[bidx[:, None], aidx[None, :]] = k[aidx[:, None], bidx[None, :]].t()
            k.fill_diagonal_(1.0)
        a = 0.55 * base + 0.45 * k
        a.fill_diagonal_(1.0)
        anti = 0.25 * anti_noise

    # task 3: boundary/fringe mask. cls unused.
    else:
        task = 3
        cls = 0
        core = int(torch.randint(max(4, n // 3), max(5, n // 2), (), device=device).item())
        fringe = int(torch.randint(max(3, n // 6), max(4, n // 4), (), device=device).item())
        outer = n - core - fringe
        if outer < 2:
            outer = 2
            fringe = max(1, n - core - outer)
        a = 0.06 * base
        c0, c1 = 0, core
        f0, f1 = c1, c1 + fringe
        o0, o1 = f1, n
        a[c0:c1, c0:c1] = 0.84 + 0.16 * torch.rand(core, core, device=device)
        a[f0:f1, f0:f1] = 0.35 + 0.25 * torch.rand(fringe, fringe, device=device)
        a[c0:c1, f0:f1] = 0.20 + 0.25 * torch.rand(core, fringe, device=device)
        a[f0:f1, c0:c1] = 0.20 + 0.25 * torch.rand(fringe, core, device=device)
        if outer > 0:
            a[o0:o1, o0:o1] = 0.03 + 0.08 * torch.rand(outer, outer, device=device)
            a[c0:c1, o0:o1] = 0.01 + 0.04 * torch.rand(core, outer, device=device)
            a[o0:o1, c0:c1] = 0.01 + 0.04 * torch.rand(outer, core, device=device)
            a[f0:f1, o0:o1] = 0.06 + 0.10 * torch.rand(fringe, outer, device=device)
            a[o0:o1, f0:f1] = 0.06 + 0.10 * torch.rand(outer, fringe, device=device)
        a = _sym(a)
        a.fill_diagonal_(1.0)
        anti = 0.25 * anti_noise
        k = 0.65 * base + 0.35 * a
        node_mask[f0:f1] = 1.0

    rel = _pack_channels(a, anti, k)

    # normalize each sample/channel to make mean/std leakage less useful
    if normalize_rel:
        mu = rel.mean(dim=(0, 1), keepdim=True)
        sd = rel.std(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(1e-5)
        rel = (rel - mu) / sd

    # mix channels so the human-readable channels are not directly addressable
    if random_channel_mix:
        mix = random_orthogonal(rel.shape[-1], device)
        rel = rel @ mix
    elif channel_mix is not None:
        rel = rel @ channel_mix

    # random permutation: nodes have no stable index/position
    p = torch.randperm(n, device=device)
    rel = rel[p][:, p]
    node_mask = node_mask[p]
    return rel.clamp(-5.0, 5.0), task, cls, node_mask


@torch.no_grad()
def make_batch(
    batch_size: int,
    n: int,
    device: str,
    force_task: Optional[int] = None,
    normalize_rel: bool = True,
    channel_mix: Optional[torch.Tensor] = None,
    random_channel_mix: bool = False,
) -> Batch:
    rels, tasks, cls, masks = [], [], [], []
    for _ in range(batch_size):
        r, t, c, m = make_one_sample(
            n=n,
            device=device,
            force_task=force_task,
            normalize_rel=normalize_rel,
            channel_mix=channel_mix,
            random_channel_mix=random_channel_mix,
        )
        rels.append(r)
        tasks.append(t)
        cls.append(c)
        masks.append(m)
    return Batch(
        rel=torch.stack(rels, dim=0),
        task=torch.tensor(tasks, device=device, dtype=torch.long),
        cls=torch.tensor(cls, device=device, dtype=torch.long),
        node_mask=torch.stack(masks, dim=0),
    )


# -----------------------------
# Model
# -----------------------------


class RelationQuestionOp(nn.Module):
    """Permutation-equivariant self-query operator over pair-relations."""

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.left = nn.Linear(dim, dim, bias=False)
        self.right = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 7, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        b, n, _, d = rel.shape
        rij = rel
        rji = rel.transpose(1, 2)
        row = rel.mean(dim=2, keepdim=True).expand(b, n, n, d)
        col = rel.mean(dim=1, keepdim=True).expand(b, n, n, d)
        glob = rel.mean(dim=(1, 2), keepdim=True).expand(b, n, n, d)
        # O(N^2) closure proxy: left(row) * right(col), not full O(N^3) composition.
        comp = torch.tanh(self.left(row) * self.right(col))
        diff = row - col
        prod = rij * rji
        x = torch.cat([rij, rji, row, col, glob, comp, diff + prod], dim=-1)
        return torch.sigmoid(self.gate) * self.mlp(x)


class PreGeometricSelfQueryFieldV2(nn.Module):
    def __init__(self, rel_dim: int = 8, dim: int = 48, n_tasks: int = 4, n_ops: int = 8, iters: int = 5):
        super().__init__()
        self.rel_dim = rel_dim
        self.dim = dim
        self.n_tasks = n_tasks
        self.n_ops = n_ops
        self.iters = iters

        self.encoder = nn.Sequential(nn.Linear(rel_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.task_emb = nn.Embedding(n_tasks, dim)
        self.ops = nn.ModuleList([RelationQuestionOp(dim) for _ in range(n_ops)])
        self.norm = nn.LayerNorm(dim)

        summary_dim = dim * 5 + dim
        self.controller = nn.Sequential(nn.Linear(summary_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, n_ops))
        self.class_head = nn.Sequential(nn.Linear(summary_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, 2))
        self.node_head = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.Linear(dim, 1))

    def invariant_summary(self, rel: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        q = self.task_emb(task)
        mean = rel.mean(dim=(1, 2))
        std = rel.std(dim=(1, 2), unbiased=False)
        mx = rel.amax(dim=(1, 2))
        mn = rel.amin(dim=(1, 2))
        # relation dispersion: helps density/boundary without coordinates
        row = rel.mean(dim=2)
        row_disp = row.std(dim=1, unbiased=False)
        return torch.cat([mean, std, mx, mn, row_disp, q], dim=-1)

    def forward(self, rel_in: torch.Tensor, task: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel = self.encoder(rel_in)
        weights_per_iter = []
        for _ in range(self.iters):
            summary = self.invariant_summary(rel, task)
            w = F.softmax(self.controller(summary), dim=-1)
            weights_per_iter.append(w)
            deltas = torch.stack([op(rel) for op in self.ops], dim=1)
            delta = (deltas * w[:, :, None, None, None]).sum(dim=1)
            rel = self.norm(rel + delta)
        summary = self.invariant_summary(rel, task)
        class_logits = self.class_head(summary)

        row = rel.mean(dim=2)
        col = rel.mean(dim=1)
        diag = rel.diagonal(dim1=1, dim2=2).transpose(1, 2)
        q = self.task_emb(task)[:, None, :].expand(-1, rel.shape[1], -1)
        node_logits = self.node_head(torch.cat([row, col, diag, q], dim=-1)).squeeze(-1)
        q_weights = torch.stack(weights_per_iter, dim=0).mean(dim=0)
        return class_logits, node_logits, q_weights


# -----------------------------
# Metrics / eval
# -----------------------------


def format_q(q: torch.Tensor, topk: int = 3) -> str:
    vals, idx = torch.topk(q.float().cpu(), k=min(topk, q.numel()))
    return " ".join([f"op{int(i)}:{float(v):.2f}" for v, i in zip(vals, idx)])


def op_stats(q_by_task: Dict[int, torch.Tensor]) -> Dict[str, float]:
    qs = torch.stack([q_by_task[t].float() for t in sorted(q_by_task)])
    q_mean = qs.mean(dim=0)
    entropy = -(q_mean.clamp_min(1e-8) * q_mean.clamp_min(1e-8).log()).sum()
    eff_ops = torch.exp(entropy).item()
    qn = F.normalize(qs, dim=-1)
    cos = qn @ qn.t()
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool)
    return {
        "eff_ops": eff_ops,
        "task_op_cos": cos[mask].mean().item(),
        "task_op_cos_max": cos[mask].max().item(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, args, n_eval: int, channel_mix: Optional[torch.Tensor], random_channel_mix: bool = False) -> Dict[str, object]:
    model.eval()
    out: Dict[str, object] = {}
    q_by_task: Dict[int, torch.Tensor] = {}

    for task in [0, 1, 2]:
        total, good = 0, 0
        q_accum = []
        for _ in range(args.eval_batches):
            batch = make_batch(args.eval_batch_size, n_eval, args.device, force_task=task,
                               normalize_rel=args.normalize_rel, channel_mix=channel_mix,
                               random_channel_mix=random_channel_mix)
            logits, _, q_weights = model(batch.rel, batch.task)
            pred = logits.argmax(dim=-1)
            good += (pred == batch.cls).sum().item()
            total += batch.cls.numel()
            q_accum.append(q_weights.mean(dim=0).detach().float().cpu())
        out[f"task{task}_acc"] = good / max(total, 1)
        q_by_task[task] = torch.stack(q_accum).mean(dim=0)
        out[f"task{task}_q"] = q_by_task[task]

    total, correct = 0, 0
    tp = fp = fn = 0.0
    q_accum = []
    for _ in range(args.eval_batches):
        batch = make_batch(args.eval_batch_size, n_eval, args.device, force_task=3,
                           normalize_rel=args.normalize_rel, channel_mix=channel_mix,
                           random_channel_mix=random_channel_mix)
        _, node_logits, q_weights = model(batch.rel, batch.task)
        prob = torch.sigmoid(node_logits)
        pred = (prob > 0.5).float()
        target = batch.node_mask.float()
        correct += (pred == target).sum().item()
        total += target.numel()
        tp += ((pred == 1) & (target == 1)).sum().item()
        fp += ((pred == 1) & (target == 0)).sum().item()
        fn += ((pred == 0) & (target == 1)).sum().item()
        q_accum.append(q_weights.mean(dim=0).detach().float().cpu())
    prec = tp / max(tp + fp, 1e-9)
    rec = tp / max(tp + fn, 1e-9)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    out["task3_node_acc"] = correct / max(total, 1)
    out["task3_boundary_f1"] = f1
    q_by_task[3] = torch.stack(q_accum).mean(dim=0)
    out["task3_q"] = q_by_task[3]
    out.update(op_stats(q_by_task))
    return out


@torch.no_grad()
def permutation_equivariance_error(model: nn.Module, args, channel_mix: Optional[torch.Tensor]) -> Tuple[float, float]:
    model.eval()
    batch = make_batch(8, args.n, args.device, normalize_rel=args.normalize_rel, channel_mix=channel_mix)
    logits, node_logits, _ = model(batch.rel, batch.task)
    p = torch.randperm(args.n, device=args.device)
    inv = inverse_perm(p)
    rel_p = batch.rel[:, p][:, :, p]
    logits_p, node_logits_p, _ = model(rel_p, batch.task)
    class_err = (logits - logits_p).abs().max().item()
    node_err = (node_logits - node_logits_p[:, inv]).abs().max().item()
    return class_err, node_err


def task_diversity_loss(q_weights: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
    """Penalize different tasks using identical op mixtures, when present in batch."""
    means = []
    for t in range(4):
        m = task == t
        if m.any():
            means.append(q_weights[m].mean(dim=0))
    if len(means) < 2:
        return torch.zeros((), device=q_weights.device)
    q = F.normalize(torch.stack(means), dim=-1)
    cos = q @ q.t()
    mask = ~torch.eye(q.shape[0], dtype=torch.bool, device=q.device)
    return cos[mask].mean()




def op_parameter_cosine(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean/max off-diagonal cosine between operator parameters.

    This is not a proof of semantic difference, but it is a useful diagnostic:
    if it goes near 1.0, several learned question operators became copies.
    """
    vecs = []
    for op in model.ops:
        parts = []
        for p in op.parameters():
            if p.ndim >= 2:
                parts.append(p.flatten())
        v = torch.cat(parts)
        vecs.append(v)
    v = F.normalize(torch.stack(vecs), dim=-1)
    cos = v @ v.t()
    mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)
    return cos[mask].mean(), cos[mask].max()


def op_parameter_diversity_loss(model: nn.Module) -> torch.Tensor:
    mean_cos, max_cos = op_parameter_cosine(model)
    # penalize positive similarity; negative/orthogonal is fine
    return F.relu(mean_cos).pow(2) + 0.25 * F.relu(max_cos - 0.65).pow(2)


# -----------------------------
# train
# -----------------------------


def train(args) -> None:
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    channel_mix = random_orthogonal(args.rel_dim, args.device) if args.fixed_channel_mix else None

    model = PreGeometricSelfQueryFieldV2(
        rel_dim=args.rel_dim,
        dim=args.dim,
        n_tasks=4,
        n_ops=args.ops,
        iters=args.iters,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device.startswith("cuda") and args.amp == "fp16"))

    print("PG-SQF-v2 coordinate-free relation/self-query core")
    print(f"device={args.device} amp={args.amp} n={args.n} dim={args.dim} ops={args.ops} iters={args.iters}")
    print(f"normalize_rel={args.normalize_rel} fixed_channel_mix={args.fixed_channel_mix} train_random_channel_mix={args.train_random_channel_mix}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    for step in range(1, args.train_steps + 1):
        model.train()
        batch = make_batch(args.batch_size, args.n, args.device,
                           normalize_rel=args.normalize_rel,
                           channel_mix=channel_mix,
                           random_channel_mix=args.train_random_channel_mix)
        opt.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            class_logits, node_logits, q_weights = model(batch.rel, batch.task)
            loss = torch.zeros((), device=args.device)
            cls_mask = batch.task != 3
            node_mask = batch.task == 3

            if cls_mask.any():
                loss_cls = F.cross_entropy(class_logits[cls_mask], batch.cls[cls_mask])
                loss = loss + loss_cls
            else:
                loss_cls = torch.zeros((), device=args.device)

            if node_mask.any():
                target = batch.node_mask[node_mask].float()
                logits = node_logits[node_mask]
                pos = target.sum().clamp_min(1.0)
                neg = target.numel() - pos
                pos_weight = (neg / pos).clamp(1.0, 8.0)
                loss_node = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
                loss = loss + args.node_loss_weight * loss_node
            else:
                loss_node = torch.zeros((), device=args.device)

            if args.op_sparse_lambda > 0:
                ent = -(q_weights.clamp_min(1e-8) * q_weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
                loss = loss + args.op_sparse_lambda * ent
            if args.op_usage_lambda > 0:
                q_mean = q_weights.mean(dim=0)
                usage_ent = -(q_mean.clamp_min(1e-8) * q_mean.clamp_min(1e-8).log()).sum()
                loss = loss - args.op_usage_lambda * usage_ent
            if args.task_op_diversity_lambda > 0:
                loss = loss + args.task_op_diversity_lambda * task_diversity_loss(q_weights, batch.task)
            if args.op_param_diversity_lambda > 0:
                loss = loss + args.op_param_diversity_lambda * op_parameter_diversity_loss(model)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        if step == 1 or step % args.eval_every == 0:
            metrics = evaluate(model, args, args.n, channel_mix, random_channel_mix=False)
            class_err, node_err = permutation_equivariance_error(model, args, channel_mix)
            op_p_cos, op_p_max = op_parameter_cosine(model)
            print(
                f"step {step:05d} loss={loss.item():.4f} cls={loss_cls.item():.4f} node={loss_node.item():.4f} | "
                f"self_other={metrics['task0_acc']*100:5.1f}% "
                f"direction={metrics['task1_acc']*100:5.1f}% "
                f"geometry={metrics['task2_acc']*100:5.1f}% "
                f"boundary_acc={metrics['task3_node_acc']*100:5.1f}% "
                f"boundary_f1={metrics['task3_boundary_f1']:.3f} | "
                f"eff_ops={metrics['eff_ops']:.2f} task_op_cos={metrics['task_op_cos']:.2f} "
                f"op_param_cos={op_p_cos.item():.2f}/{op_p_max.item():.2f} "
                f"perm_cls={class_err:.1e} perm_node={node_err:.1e}"
            )
            print(
                "  q top: "
                f"self/other[{format_q(metrics['task0_q'])}] | "
                f"direction[{format_q(metrics['task1_q'])}] | "
                f"geometry[{format_q(metrics['task2_q'])}] | "
                f"boundary[{format_q(metrics['task3_q'])}]"
            )
            if args.ood_n and args.ood_n != args.n:
                ood = evaluate(model, args, args.ood_n, channel_mix, random_channel_mix=False)
                print(
                    f"  OOD-N={args.ood_n}: self_other={ood['task0_acc']*100:5.1f}% "
                    f"direction={ood['task1_acc']*100:5.1f}% geometry={ood['task2_acc']*100:5.1f}% "
                    f"boundary_f1={ood['task3_boundary_f1']:.3f} eff_ops={ood['eff_ops']:.2f}"
                )
            if args.eval_random_channel_mix:
                rnd = evaluate(model, args, args.n, None, random_channel_mix=True)
                print(
                    f"  OOD-random-channel-mix: self_other={rnd['task0_acc']*100:5.1f}% "
                    f"direction={rnd['task1_acc']*100:5.1f}% geometry={rnd['task2_acc']*100:5.1f}% "
                    f"boundary_f1={rnd['task3_boundary_f1']:.3f}"
                )

    if args.save_path:
        torch.save({"model": model.state_dict(), "args": vars(args), "channel_mix": channel_mix}, args.save_path)
        print(f"saved: {args.save_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", type=str, default="fp16", choices=["none", "fp16", "bf16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)

    p.add_argument("--n", type=int, default=32)
    p.add_argument("--ood-n", type=int, default=0)
    p.add_argument("--rel-dim", type=int, default=8)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--ops", type=int, default=8)
    p.add_argument("--iters", type=int, default=5)

    p.add_argument("--train-steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-batches", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=100)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--node-loss-weight", type=float, default=0.8)

    p.add_argument("--normalize-rel", action="store_true", help="per-sample channel normalization to reduce mean/std leaks")
    p.add_argument("--fixed-channel-mix", action="store_true", help="apply one fixed random orthogonal channel mix")
    p.add_argument("--train-random-channel-mix", action="store_true", help="hard mode: random channel mix per sample in training")
    p.add_argument("--eval-random-channel-mix", action="store_true", help="OOD eval with random channel mix per sample")

    p.add_argument("--op-sparse-lambda", type=float, default=0.003, help="positive => lower per-sample op entropy")
    p.add_argument("--op-usage-lambda", type=float, default=0.010, help="positive => avoid global one-op collapse")
    p.add_argument("--task-op-diversity-lambda", type=float, default=0.03, help="positive => different tasks prefer different op mixtures")
    p.add_argument("--op-param-diversity-lambda", type=float, default=0.02, help="positive => discourages question-operator parameter copies")
    p.add_argument("--save-path", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
