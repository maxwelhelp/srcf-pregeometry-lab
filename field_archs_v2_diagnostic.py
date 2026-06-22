#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
field_archs_v2_diagnostic.py

Diagnostic version of the three "field" ideas.
Goal: remove fake signals and make the logical holes visible.

Important changes vs the earlier demo:
  1) no learned per-node identity embedding in Arch 1;
  2) no claim that LayerNorm norm == density;
  3) permutation-equivariance is tested explicitly;
  4) SelfQueryingField is marked as an untrained closed-dynamics demo, not proof of semantic questions;
  5) TopologicalEmergence uses sparse top-k dynamic connectivity + degree normalization to avoid instant full-graph collapse.

Usage:
  python -u field_archs_v2_diagnostic.py
"""
from __future__ import annotations

import math
import sys
from typing import Dict, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_threads(n: int = 4) -> None:
    torch.set_num_threads(max(1, int(n)))


def make_anonymous_relations(n: int, rel_dim: int, device: str = "cpu") -> torch.Tensor:
    """Create a relation tensor with no coordinates and no node identities.

    The generator uses hidden random variables only to create pair-relations.
    The hidden variables are not returned. Nodes are randomly permuted.
    """
    z = torch.randn(n, 4, device=device)
    sim = torch.sigmoid((z @ z.t()) / math.sqrt(z.shape[-1]))
    sim = 0.5 * (sim + sim.t())
    sim.fill_diagonal_(1.0)

    anti_seed = torch.randn(n, device=device)
    anti = torch.tanh(anti_seed[:, None] - anti_seed[None, :])
    anti.fill_diagonal_(0.0)

    noise = torch.randn(n, n, rel_dim, device=device) * 0.05
    rel = noise
    rel[..., 0] = sim
    rel[..., 1] = anti
    if rel_dim > 2:
        rel[..., 2] = sim @ sim / n
    if rel_dim > 3:
        rel[..., 3] = torch.eye(n, device=device)

    # random node permutation: no index means anything
    p = torch.randperm(n, device=device)
    return rel[p][:, p]


def permute_rel(rel: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    return rel[p][:, p]


def inverse_perm(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device)
    return inv


# =============================================================================
# ARCH 1: coordinate-free relational baseline
# =============================================================================

class CoordinateFreeRelationalField(nn.Module):
    """Graph/message-passing baseline with no positional encodings.

    No learned node_init[N,D] is allowed here, because that would secretly give
    every node a persistent identity. Node states are derived only from relation
    rows/columns and global summaries.
    """

    def __init__(self, rel_dim: int = 6, dim: int = 64, layers: int = 3):
        super().__init__()
        self.rel_dim = rel_dim
        self.dim = dim
        self.edge_enc = nn.Sequential(nn.Linear(rel_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.node_enc = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList([RelationMPNNLayer(dim) for _ in range(layers)])
        self.boundary_probe = nn.Linear(dim * 3, 1)

    def forward(self, rel_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        # rel_in [N,N,C]
        e = self.edge_enc(rel_in)
        row = e.mean(dim=1)
        col = e.mean(dim=0)
        glob = e.mean(dim=(0, 1), keepdim=False).unsqueeze(0).expand(row.shape[0], -1)
        h = self.node_enc(torch.cat([row, col, glob], dim=-1))

        for layer in self.layers:
            h = layer(h, e)

        row_h = h
        degree_like = torch.sigmoid(e[..., 0]).mean(dim=1, keepdim=True)
        local_var = h.var(dim=-1, keepdim=True, unbiased=False)
        boundary = torch.sigmoid(self.boundary_probe(torch.cat([row_h, degree_like.expand_as(h), local_var.expand_as(h)], dim=-1))).squeeze(-1)

        # Honest density proxy: outgoing relation variability, not LayerNorm norm.
        density = e.var(dim=(1, 2), unbiased=False)
        dist = emergent_distance(h)
        return {"node_state": h, "edge_state": e, "boundary": boundary, "density": density, "emergent_distance": dist}


class RelationMPNNLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.Sigmoid())
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        n, d = h.shape
        hi = h[:, None, :].expand(n, n, d)
        hj = h[None, :, :].expand(n, n, d)
        x = torch.cat([hi, hj, e], dim=-1)
        m = self.msg(x) * self.gate(x)
        # degree-normalized anonymous aggregation
        agg = m.mean(dim=1)
        return self.norm(h + self.out(agg))


def emergent_distance(h: torch.Tensor) -> torch.Tensor:
    hn = F.normalize(h, dim=-1)
    return (1.0 - hn @ hn.t()).clamp_min(0.0)


@torch.no_grad()
def permutation_test_arch1(model: CoordinateFreeRelationalField, rel: torch.Tensor) -> Tuple[float, float]:
    out = model(rel)
    p = torch.randperm(rel.shape[0], device=rel.device)
    inv = inverse_perm(p)
    out_p = model(permute_rel(rel, p))
    node_err = (out["node_state"] - out_p["node_state"][inv]).abs().max().item()
    dist_err = (out["emergent_distance"] - out_p["emergent_distance"][inv][:, inv]).abs().max().item()
    return node_err, dist_err


# =============================================================================
# ARCH 2: self-querying latent field, honest diagnostic version
# =============================================================================

class SelfQueryingLatentField(nn.Module):
    """Closed self-query dynamics over one latent field vector.

    This is conceptually interesting, but untrained question names are labels for
    us, not proven semantics. The diagnostic prints diversity/stability, not fake
    self-knowledge claims.
    """

    question_names = ["boundary", "density", "other", "orientation", "identity", "scale", "undefined"]

    def __init__(self, dim: int = 96, n_steps: int = 10):
        super().__init__()
        self.dim = dim
        self.n_questions = len(self.question_names)
        self.n_steps = n_steps
        self.h_init = nn.Parameter(torch.randn(1, dim) * 0.05)
        self.templates = nn.Parameter(torch.randn(self.n_questions, dim) * 0.15)
        self.query_from_state = nn.Linear(dim, self.n_questions * dim)
        self.answer = nn.Sequential(nn.Linear(dim * 3, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.mix = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.update_gate = nn.Linear(dim * 2, dim)
        self.update = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def _answers(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        hq = h[:, None, :].expand(-1, self.n_questions, -1)
        return self.answer(torch.cat([hq, q, hq * q], dim=-1))

    @torch.no_grad()
    def diagnostics(self, answers: torch.Tensor, attn: torch.Tensor, history: torch.Tensor) -> Dict[str, float]:
        a = answers[0]
        an = F.normalize(a, dim=-1)
        cos = an @ an.t()
        off = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
        deltas = (history[1:] - history[:-1]).norm(dim=-1)
        # attn [B, Q, Q]
        entropy = -(attn.clamp_min(1e-8) * attn.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return {
            "answer_offdiag_cos_mean": off.mean().item(),
            "answer_offdiag_cos_max": off.max().item(),
            "mean_state_delta": deltas.mean().item(),
            "last_state_delta": deltas[-1].item(),
            "question_attention_entropy": entropy.item(),
        }

    def forward(self) -> Dict[str, torch.Tensor | Dict[str, float]]:
        h = self.h_init
        hist = [h.squeeze(0)]
        last_answers = None
        last_attn = None
        for _ in range(self.n_steps):
            q = self.query_from_state(h).view(1, self.n_questions, self.dim) + self.templates[None, :, :]
            answers = self._answers(h, q)
            mixed, attn = self.mix(answers, answers, answers, need_weights=True)
            u = mixed.mean(dim=1)
            gu = torch.cat([h, u], dim=-1)
            gate = torch.sigmoid(self.update_gate(gu))
            h = self.norm(h + 0.25 * gate * torch.tanh(self.update(gu)))
            hist.append(h.squeeze(0))
            last_answers, last_attn = answers, attn
        history = torch.stack(hist)
        diag = self.diagnostics(last_answers, last_attn, history)
        magnitudes = {name: float(last_answers[0, i].norm().detach()) for i, name in enumerate(self.question_names)}
        return {"final_state": h, "history": history, "answer_magnitudes": magnitudes, "diagnostics": diag}


# =============================================================================
# ARCH 3: dynamic topology without fixed grid, non-collapsing diagnostic
# =============================================================================

class SparseTopologicalEmergence(nn.Module):
    """Dynamic sparse graph. Topology is derived from states, not coordinates.

    The earlier version collapsed into a complete graph. This one uses top-k
    connectivity, degree-normalized messages, and a weak diversity push.
    """

    def __init__(self, n_particles: int = 32, dim: int = 64, steps: int = 10, k: int = 4):
        super().__init__()
        self.n = n_particles
        self.dim = dim
        self.steps = steps
        self.k = k
        self.init = nn.Parameter(torch.randn(n_particles, dim) * 0.05)
        self.msg = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.self_update = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self) -> Dict[str, torch.Tensor | list]:
        h = self.init
        hist = []
        conn = None
        for t in range(self.steps):
            hn = F.normalize(h, dim=-1)
            sim = hn @ hn.t()
            sim = sim.masked_fill(torch.eye(self.n, dtype=torch.bool, device=h.device), -1e9)
            idx = torch.topk(sim, k=min(self.k, self.n - 1), dim=-1).indices
            conn = torch.zeros(self.n, self.n, device=h.device)
            conn.scatter_(1, idx, 1.0)
            # symmetrize but keep sparse-ish graph
            conn = ((conn + conn.t()) > 0).float()
            deg = conn.sum(dim=1, keepdim=True).clamp_min(1.0)

            hi = h[:, None, :].expand(self.n, self.n, self.dim)
            hj = h[None, :, :].expand(self.n, self.n, self.dim)
            m = self.msg(torch.cat([hi, hj], dim=-1))
            agg = (conn[..., None] * m).sum(dim=1) / deg

            # Weak anti-collapse: keep particles distinguishable without coordinates.
            diversity_push = 0.03 * (h - h.mean(dim=0, keepdim=True))
            h = self.norm(h + 0.2 * self.self_update(h) + 0.2 * agg + diversity_push)
            dist = emergent_distance(h)
            hist.append({
                "step": t,
                "edges": int(conn.sum().item()),
                "mean_degree": float(conn.sum(dim=1).mean().item()),
                "distance_mean": float(dist.mean().item()),
                "distance_max": float(dist.max().item()),
            })
        return {"final_state": h, "connectivity": conn, "emergent_distance": emergent_distance(h), "history": hist}


# =============================================================================
# Runner
# =============================================================================


def run_all() -> None:
    set_threads(4)
    torch.manual_seed(42)
    print("=" * 72)
    print("FIELD ARCHITECTURE DIAGNOSTICS v2")
    print("=" * 72)

    print("\n[ARCH 1] Coordinate-free relational baseline")
    rel = make_anonymous_relations(n=24, rel_dim=6)
    arch1 = CoordinateFreeRelationalField(rel_dim=6, dim=64, layers=3)
    with torch.no_grad():
        out1 = arch1(rel)
        node_err, dist_err = permutation_test_arch1(arch1, rel)
    print("  No learned node identity. Node states are derived from pair-relations.")
    print(f"  node_state shape: {tuple(out1['node_state'].shape)}")
    print(f"  boundary first5: {out1['boundary'][:5].cpu().numpy().round(3)}")
    print(f"  density/variance first5: {out1['density'][:5].cpu().numpy().round(3)}")
    print(f"  emergent distance range: [{out1['emergent_distance'].min():.3f}, {out1['emergent_distance'].max():.3f}]")
    print(f"  permutation equivariance max error: node={node_err:.2e}, distance={dist_err:.2e}")

    print("\n[ARCH 2] Self-querying latent field, diagnostic only")
    arch2 = SelfQueryingLatentField(dim=96, n_steps=10)
    with torch.no_grad():
        out2 = arch2()
    print("  No external input and no coordinates. But untrained names are not semantics yet.")
    norms = [round(float(x.norm()), 3) for x in out2["history"]]
    print(f"  state norms over steps: {norms}")
    print("  answer magnitudes:")
    for name, val in out2["answer_magnitudes"].items():
        print(f"    {name:12s}: {val:.4f}")
    print("  diagnostics:")
    for k, v in out2["diagnostics"].items():
        print(f"    {k:30s}: {v:.4f}")

    print("\n[ARCH 3] Sparse topological emergence")
    arch3 = SparseTopologicalEmergence(n_particles=32, dim=64, steps=10, k=4)
    with torch.no_grad():
        out3 = arch3()
    print("  Dynamic top-k graph. Not a grid. Not full graph collapse.")
    for rec in out3["history"][::2]:
        print(f"    step {rec['step']:2d}: edges={rec['edges']:3d}, degree={rec['mean_degree']:.2f}, "
              f"dist_mean={rec['distance_mean']:.3f}, dist_max={rec['distance_max']:.3f}")
    ed = out3["emergent_distance"]
    print(f"  final emergent distance range: [{ed.min():.3f}, {ed.max():.3f}]")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)
    print("""
Arch 1 is the honest baseline: relation-only, no coordinates, no hidden node IDs.
It is not highly novel, but it is the baseline every new idea must beat.

Arch 2 is conceptually interesting but not a proof by itself. Without a task/loss,
'boundary/density/other' are just names for templates. Use it only after training.

Arch 3 shows topology-as-output more honestly than the earlier version: it avoids
instant complete-graph collapse. Still, it is a dynamic graph idea, not the main novelty.

The real research direction should be PG-SQF-v1: relation-only inputs + learned
self-query operators + explicit undefined/geometry-valid/boundary/self-other tasks.
""")


if __name__ == "__main__":
    run_all()
