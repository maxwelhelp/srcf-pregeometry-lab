#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self-Referential Closure Field v5 basin-contractive hard + DNA
============================================

Coordinate-free / object-free / label-free experiment.

This version fixes the main logical holes in v0-v3:
  - prints identity and random/frozen baselines before/while training
  - measures contraction vs input, not just output distances
  - adds non-identity pressure, so a lazy identity operator is penalized
  - prints a per-iteration convergence curve
  - uses a stronger permutation-invariant descriptor
  - supports hard synthetic relation states and optional DNA k-mer relation states
  - DNA mode auto-downloads a small slice from NCBI EFetch, with fallback synthetic DNA
  - v5 adds CSV logging, stricter far-preservation/variance guards, and fixes DNA k-mer indexing

Core idea:
  R0 relation tensor [B,N,N,C]
  H0 = encoder(R0)
  H* = F(F(...F(H0)))       # same learned operator applies to itself

No supervised labels are used. The intrinsic objective is:
  near perturbations should converge together,
  unrelated states should stay distinct,
  final states should be fixed/recoverable,
  dynamics should move then settle,
  descriptors should not collapse.

Smoke:
  python -u self_referential_closure_field_v1_hard_dna.py --device cpu --amp none --steps 20 --batch-size 8 --eval-batch-size 8 --data synthetic --n 16 --dim 24 --ops 4 --iters 5 --eval-every 10

P40 synthetic memsafe:
  python -u self_referential_closure_field_v5_basin_dna.py --device cuda --amp fp16 --steps 1000 --batch-size 32 --eval-batch-size 32 --data synthetic --n 32 --ood-n 48 --dim 48 --ops 8 --iters 6 --eval-every 50 --save-path ./srcf_v2_synth.pt | tee srcf_v2_synth.log

P40 DNA 3-mer relation states memsafe:
  python -u self_referential_closure_field_v5_basin_dna.py --device cuda --amp fp16 --steps 1000 --batch-size 16 --eval-batch-size 16 --data dna --dna-bases 200000 --dna-window 2048 --kmer 3 --dim 48 --ops 8 --iters 6 --eval-every 50 --save-path ./srcf_v2_dna.pt | tee srcf_v2_dna.log
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import csv
import urllib.request
from typing import Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ------------------------- misc -------------------------

DNA_BASES = "ACGT"
BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
RC_BASE = str.maketrans("ACGTacgt", "TGCAtgca")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: str, amp: str):
    if device.startswith("cuda") and amp != "none":
        dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def inverse_perm(p: torch.Tensor) -> torch.Tensor:
    inv = torch.empty_like(p)
    inv[p] = torch.arange(p.numel(), device=p.device)
    return inv


def normalize_rel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=(1, 2), keepdim=True)
    sd = x.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(eps)
    return ((x - mu) / sd).clamp(-6.0, 6.0)


# ------------------------- synthetic data -------------------------

@torch.no_grad()
def random_relation_batch_hard(batch: int, n: int, rel_dim: int, device: str, structure_mix: float = 0.70) -> torch.Tensor:
    """Harder anonymous relation states.

    Mixes latent low-rank affinity, directed transition-like relations, block closure,
    cycles and pure noise. There are no labels; these are initial disturbances.
    """
    z = torch.randn(batch, n, 8, device=device)
    sim = torch.einsum("bik,bjk->bij", z, z) / math.sqrt(z.shape[-1])
    sim = torch.tanh(sim)

    # directed latent flow: z_i A z_j
    A = torch.randn(batch, 8, 8, device=device) / math.sqrt(8)
    flow = torch.einsum("bix,bxy,bjy->bij", z, A, z) / math.sqrt(8)
    flow = torch.tanh(flow)
    anti = flow - flow.transpose(1, 2)

    # anonymous block closure: generate random groups, then permute later
    groups = torch.randint(0, max(2, min(6, n // 4)), (batch, n), device=device)
    block = (groups[:, :, None] == groups[:, None, :]).float() * 2.0 - 1.0

    # cycle-like relation without coordinates: random permutation induces directed successor
    cycle = torch.zeros(batch, n, n, device=device)
    for b in range(batch):
        p = torch.randperm(n, device=device)
        cycle[b, p, torch.roll(p, shifts=-1)] = 1.0
        cycle[b, p, torch.roll(p, shifts=1)] -= 0.5

    noise = torch.randn(batch, n, n, device=device)
    sym_noise = 0.5 * (noise + noise.transpose(1, 2))
    eye = torch.eye(n, device=device)[None].expand(batch, -1, -1)

    bases = [sim, anti, block, cycle, sym_noise, flow]
    rel = []
    for c in range(rel_dim):
        coeffs = torch.randn(len(bases), device=device)
        ch = sum(coeffs[i] * bases[i] for i in range(len(bases))) / math.sqrt(len(bases))
        ch = structure_mix * ch + (1.0 - structure_mix) * torch.randn_like(ch)
        if c % 5 == 0:
            ch = ch + 0.10 * eye
        rel.append(ch)
    r = torch.stack(rel, dim=-1)
    r = normalize_rel(r)

    # enforce anonymity by per-sample permutation
    outs = []
    for b in range(batch):
        p = torch.randperm(n, device=device)
        outs.append(r[b, p][:, p])
    return torch.stack(outs, dim=0)


def perturb_relation(r: torch.Tensor, noise_std: float, renorm: bool = True) -> torch.Tensor:
    x = r + noise_std * torch.randn_like(r)
    return normalize_rel(x) if renorm else x.clamp(-6.0, 6.0)


# ------------------------- DNA data -------------------------

def parse_fasta(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line.upper())
    seq = "".join(lines)
    return "".join(ch for ch in seq if ch in BASE_TO_INT)


def download_dna_sequence(args) -> str:
    cache = args.dna_cache
    if cache and os.path.exists(cache):
        with open(cache, "r", encoding="utf-8") as f:
            seq = parse_fasta(f.read())
        if len(seq) >= min(args.dna_bases // 2, args.dna_window * 2):
            print(f"loaded DNA cache: {cache} bases={len(seq):,}")
            return seq[: args.dna_bases]

    if args.download_dna:
        url = (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=nuccore&id={args.dna_accession}&rettype=fasta&retmode=text"
            f"&seq_start=1&seq_stop={args.dna_bases}"
        )
        try:
            print(f"downloading DNA from NCBI: accession={args.dna_accession} bases={args.dna_bases:,}")
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
            seq = parse_fasta(text)
            if len(seq) < args.dna_window * 2:
                raise RuntimeError(f"downloaded sequence too short: {len(seq)}")
            if cache:
                with open(cache, "w", encoding="utf-8") as f:
                    f.write(f">{args.dna_accession} first {len(seq)} bases\n")
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")
                print(f"saved DNA cache: {cache}")
            return seq[: args.dna_bases]
        except Exception as e:
            print(f"WARNING: DNA download failed: {e}")
            print("fallback: generating synthetic Markov DNA")

    # fallback: synthetic Markov DNA with local regimes
    rng = random.Random(args.seed)
    parts = []
    regimes = ["AT", "GC", "MOTIF", "BAL"]
    motif = "TATAAT"
    while len("".join(parts)) < args.dna_bases:
        reg = rng.choice(regimes)
        length = rng.randint(1000, 6000)
        s = []
        for i in range(length):
            if reg == "AT":
                probs = [("A", 0.38), ("T", 0.38), ("C", 0.12), ("G", 0.12)]
            elif reg == "GC":
                probs = [("G", 0.36), ("C", 0.36), ("A", 0.14), ("T", 0.14)]
            elif reg == "MOTIF" and rng.random() < 0.03:
                s.append(motif)
                continue
            else:
                probs = [("A", 0.25), ("C", 0.25), ("G", 0.25), ("T", 0.25)]
            r = rng.random()
            acc = 0.0
            for ch, p in probs:
                acc += p
                if r <= acc:
                    s.append(ch)
                    break
        parts.append("".join(s))
    seq = "".join(parts)[: args.dna_bases]
    return seq


def mutate_seq(seq: str, rate: float) -> str:
    out = list(seq)
    for i, ch in enumerate(out):
        if random.random() < rate:
            choices = [b for b in DNA_BASES if b != ch]
            out[i] = random.choice(choices)
    return "".join(out)


def shuffle_seq(seq: str) -> str:
    x = list(seq)
    random.shuffle(x)
    return "".join(x)


def revcomp(seq: str) -> str:
    return seq.translate(RC_BASE)[::-1].upper()


def kmer_indices(seq: str, k: int) -> List[int]:
    vals: List[int] = []
    code = 0
    valid = 0
    mask = 4 ** k - 1
    for ch in seq.upper():
        b = BASE_TO_INT.get(ch, -1)
        if b < 0:
            code = 0
            valid = 0
            continue
        code = ((code * 4) + b) & mask
        valid += 1
        if valid >= k:
            vals.append(code)
    return vals


def rc_index(idx: int, k: int) -> int:
    out = 0
    x = idx
    for _ in range(k):
        b = x & 3
        x >>= 2
        rb = 3 - b
        out = (out << 2) | rb
    return out


def gc_count_idx(idx: int, k: int) -> int:
    c = 0
    x = idx
    for _ in range(k):
        b = x & 3
        if b in (1, 2):
            c += 1
        x >>= 2
    return c


def hamming_idx(i: int, j: int, k: int) -> int:
    d = 0
    for _ in range(k):
        if (i & 3) != (j & 3):
            d += 1
        i >>= 2
        j >>= 2
    return d


def dna_relation_from_seq(seq: str, k: int, rel_dim: int) -> torch.Tensor:
    ids = kmer_indices(seq, k)
    n = 4 ** k
    mat = torch.zeros(n, n, rel_dim, dtype=torch.float32)
    if len(ids) < 4:
        return mat

    counts = torch.zeros(n, n, dtype=torch.float32)
    freq = torch.zeros(n, dtype=torch.float32)
    for x in ids:
        freq[x] += 1
    for a, b in zip(ids[:-1], ids[1:]):
        counts[a, b] += 1

    total = counts.sum().clamp_min(1.0)
    row = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    col = counts.sum(dim=0, keepdim=True).clamp_min(1.0)
    expected = row @ col / total
    pmi = torch.log1p(counts) - torch.log1p(expected)
    sym = 0.5 * (counts + counts.t())
    anti = counts - counts.t()
    outer_freq = torch.outer(freq, freq) / freq.sum().clamp_min(1.0)

    rc_mat = torch.zeros(n, n)
    gc_mat = torch.zeros(n, n)
    ham_mat = torch.zeros(n, n)
    for i in range(n):
        rc_mat[i, rc_index(i, k)] = 1.0 + math.log1p(float(freq[i].item()))
        gi = gc_count_idx(i, k)
        for j in range(n):
            gj = gc_count_idx(j, k)
            gc_mat[i, j] = 1.0 - abs(gi - gj) / max(1, k)
            ham_mat[i, j] = 1.0 - hamming_idx(i, j, k) / max(1, k)

    chans = [
        torch.log1p(counts),
        torch.log1p(counts.t()),
        torch.log1p(sym),
        anti / anti.abs().amax().clamp_min(1.0),
        pmi,
        torch.log1p(outer_freq),
        rc_mat,
        gc_mat,
        ham_mat,
    ]
    for c in range(rel_dim):
        mat[:, :, c] = chans[c % len(chans)]
    return mat


class RelationSampler:
    def __init__(self, args, device: str):
        self.args = args
        self.device = device
        self.seq: Optional[str] = None
        if args.data in ("dna", "mixed"):
            self.seq = download_dna_sequence(args)
            print(f"DNA sequence ready: bases={len(self.seq):,} k={args.kmer} N={4 ** args.kmer}")

    @property
    def n(self) -> int:
        if self.args.data == "dna":
            return 4 ** self.args.kmer
        return self.args.n

    def _dna_chunk(self) -> str:
        assert self.seq is not None
        w = min(self.args.dna_window, len(self.seq))
        if len(self.seq) <= w:
            return self.seq
        start = random.randint(0, len(self.seq) - w)
        return self.seq[start:start+w]

    def _dna_batch(self, batch: int, far_mode: str = "mixed") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, nears, fars = [], [], []
        for _ in range(batch):
            s = self._dna_chunk()
            if len(s) < self.args.dna_window and len(s) > 0:
                s = (s * (self.args.dna_window // len(s) + 1))[: self.args.dna_window]
            near = mutate_seq(s, self.args.dna_mut_rate)
            mode = far_mode
            if mode == "mixed":
                mode = random.choice(["shuffle", "other", "revcomp", "chimera"])
            if mode == "shuffle":
                far = shuffle_seq(s)
            elif mode == "revcomp":
                far = revcomp(s)
            elif mode == "chimera":
                s2 = self._dna_chunk()
                cut = len(s) // 2
                far = s[:cut] + s2[cut:]
            else:
                far = self._dna_chunk()
            xs.append(dna_relation_from_seq(s, self.args.kmer, self.args.rel_dim))
            nears.append(dna_relation_from_seq(near, self.args.kmer, self.args.rel_dim))
            fars.append(dna_relation_from_seq(far, self.args.kmer, self.args.rel_dim))
        r0 = normalize_rel(torch.stack(xs, dim=0)).to(self.device)
        rn = normalize_rel(torch.stack(nears, dim=0)).to(self.device)
        rf = normalize_rel(torch.stack(fars, dim=0)).to(self.device)
        return r0, rn, rf

    def sample_triplet(self, batch: int, n_override: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.args.data == "dna":
            return self._dna_batch(batch)
        if self.args.data == "mixed" and random.random() < self.args.mixed_dna_prob:
            return self._dna_batch(batch)
        n = n_override or self.args.n
        r0 = random_relation_batch_hard(batch, n, self.args.rel_dim, self.device, self.args.structure_mix)
        rn = perturb_relation(r0, self.args.input_noise)
        rf = random_relation_batch_hard(batch, n, self.args.rel_dim, self.device, self.args.structure_mix)
        return r0, rn, rf


# ------------------------- model -------------------------

class ClosureQuestionOp(nn.Module):
    """Memory-safe contractive closure operator.

    Unlike v2 residual-delta op, this op proposes a *closure target* from
    coordinate-free relational statistics. Repeated self-application can then
    become a contraction mapping instead of a lazy identity.

    It never materializes a huge concat [B,N,N,kD]; each component is projected
    separately and summed.
    """
    def __init__(self, dim: int, hidden_mult: int = 1):
        super().__init__()
        hidden = dim * hidden_mult
        self.left = nn.Linear(dim, dim, bias=False)
        self.right = nn.Linear(dim, dim, bias=False)
        self.proj = nn.ModuleList([nn.Linear(dim, hidden, bias=False) for _ in range(12)])
        self.bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(hidden, dim)
        # candidate strength. Positive enough to move, not explode.
        self.gate = nn.Parameter(torch.tensor(-0.8))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        b, n, _, d = r.shape
        rij = r
        rji = r.transpose(1, 2)
        row_base = r.mean(dim=2, keepdim=True)
        col_base = r.mean(dim=1, keepdim=True)
        glob_base = r.mean(dim=(1, 2), keepdim=True)
        row = row_base.expand(b, n, n, d)
        col = col_base.expand(b, n, n, d)
        glob = glob_base.expand(b, n, n, d)
        sym = 0.5 * (rij + rji)
        anti = 0.5 * (rij - rji)
        row_std = r.std(dim=2, unbiased=False, keepdim=True).expand(b, n, n, d)
        col_std = r.std(dim=1, unbiased=False, keepdim=True).expand(b, n, n, d)
        # Bottleneck closure composition: row/col/global tell the system what
        # is mutually reachable; this is deliberately not a coordinate.
        comp = torch.tanh(self.left(row) * self.right(col))
        reciprocity = rij * rji
        # Smoothed relation components make near perturbations contract.
        smooth = 0.25 * rij + 0.25 * rji + 0.25 * row + 0.25 * col
        imbalance = row - col
        parts = (
            rij, rji, row, col, glob, sym, anti, comp, reciprocity,
            row_std - col_std, smooth, imbalance,
        )
        z = self.bias
        for layer, part in zip(self.proj, parts):
            z = z + layer(part)
        target = self.out(F.gelu(z))
        # target is a closure proposal, not a residual delta
        return torch.sigmoid(self.gate) * target


class SelfReferentialClosureField(nn.Module):
    def __init__(self, rel_dim: int = 8, dim: int = 48, n_ops: int = 8, iters: int = 8, controller_temp: float = 1.0, checkpoint_ops: bool = True):
        super().__init__()
        self.rel_dim = rel_dim
        self.dim = dim
        self.n_ops = n_ops
        self.iters = iters
        self.controller_temp = controller_temp
        self.checkpoint_ops = checkpoint_ops
        self.encoder = nn.Sequential(nn.Linear(rel_dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.ops = nn.ModuleList([ClosureQuestionOp(dim) for _ in range(n_ops)])
        self.norm = nn.LayerNorm(dim)
        # learned interpolation. Keeps self-application stable and lets curves decay.
        self.step_logit = nn.Parameter(torch.tensor(-1.15))
        self.residual_logit = nn.Parameter(torch.tensor(-1.50))
        summary_dim = dim * 10 + 6
        self.controller = nn.Sequential(nn.Linear(summary_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, n_ops))

    def invariant_summary(self, r: torch.Tensor) -> torch.Tensor:
        # channel-wise invariant statistics
        mean = r.mean(dim=(1, 2))
        std = r.std(dim=(1, 2), unbiased=False)
        mx = r.amax(dim=(1, 2))
        mn = r.amin(dim=(1, 2))
        sym = 0.5 * (r + r.transpose(1, 2))
        anti = 0.5 * (r - r.transpose(1, 2))
        sym_e = sym.pow(2).mean(dim=(1, 2)).sqrt()
        anti_e = anti.pow(2).mean(dim=(1, 2)).sqrt()
        row = r.mean(dim=2)
        col = r.mean(dim=1)
        row_disp = row.std(dim=1, unbiased=False)
        col_disp = col.std(dim=1, unbiased=False)
        closure_res = (row - col).abs().mean(dim=1)

        # scalar distribution stats of pair norms, helps descriptor not be empty
        pair_norm = r.float().pow(2).mean(dim=-1).sqrt()
        s_mean = pair_norm.mean(dim=(1, 2), keepdim=False).unsqueeze(-1)
        s_std = pair_norm.std(dim=(1, 2), unbiased=False).unsqueeze(-1)
        s_max = pair_norm.amax(dim=(1, 2)).unsqueeze(-1)
        s_min = pair_norm.amin(dim=(1, 2)).unsqueeze(-1)
        asym_scalar = anti.float().pow(2).mean(dim=(1, 2, 3), keepdim=False).sqrt().unsqueeze(-1)
        sym_scalar = sym.float().pow(2).mean(dim=(1, 2, 3), keepdim=False).sqrt().unsqueeze(-1)
        scalars = torch.cat([s_mean, s_std, s_max, s_min, asym_scalar, sym_scalar], dim=-1).to(r.dtype)
        return torch.cat([mean, std, mx, mn, sym_e, anti_e, row_disp, col_disp, closure_res, mean * anti_e, scalars], dim=-1)

    def descriptor(self, r: torch.Tensor) -> torch.Tensor:
        # normalize but keep variance diagnostics meaningful outside
        return F.normalize(self.invariant_summary(r).float(), dim=-1)

    def step_once(self, r: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        summary = self.invariant_summary(r)
        logits = self.controller(summary) / max(self.controller_temp, 1e-6)
        w = F.softmax(logits, dim=-1)
        target = None
        # Memory-safe accumulation: do not materialize [B,ops,N,N,D].
        for k, op in enumerate(self.ops):
            if self.checkpoint_ops and self.training and r.requires_grad:
                tk = checkpoint(op, r, use_reentrant=False)
            else:
                tk = op(r)
            wk = w[:, k, None, None, None]
            target = tk * wk if target is None else target + tk * wk

        # Contractive self-application: recompute a coordinate-free closure target
        # and interpolate toward it. This avoids the v2 lazy residual identity.
        step = torch.sigmoid(self.step_logit)
        residual = torch.sigmoid(self.residual_logit)
        candidate = self.norm(residual * r + target)
        out = self.norm((1.0 - step) * r + step * candidate)
        delta = out - r
        delta_norm = delta.float().pow(2).mean(dim=(1, 2, 3)).sqrt()
        return out, w, delta_norm

    def forward(self, r_in: torch.Tensor, iters: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.encoder(r_in)
        h = h0
        iters = self.iters if iters is None else int(iters)
        weights, deltas = [], []
        for _ in range(iters):
            h, w, dn = self.step_once(h)
            weights.append(w)
            deltas.append(dn)
        return h0, h, torch.stack(weights, dim=0), torch.stack(deltas, dim=0)

    def continue_from_hidden(self, h: torch.Tensor, iters: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights, deltas = [], []
        for _ in range(iters):
            h, w, dn = self.step_once(h)
            weights.append(w)
            deltas.append(dn)
        return h, torch.stack(weights, dim=0), torch.stack(deltas, dim=0)


# ------------------------- metrics/losses -------------------------

def desc_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).pow(2).sum(dim=-1).sqrt()


def offdiag_pairwise_dist(desc: torch.Tensor) -> torch.Tensor:
    b = desc.shape[0]
    d = torch.cdist(desc.float(), desc.float())
    mask = ~torch.eye(b, dtype=torch.bool, device=desc.device)
    return d[mask]


def op_effective_count(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_mean = w.mean(dim=(0, 1))
    ent = -(q_mean.clamp_min(1e-8) * q_mean.clamp_min(1e-8).log()).sum()
    return torch.exp(ent), ent


def intrinsic_losses(model: SelfReferentialClosureField, triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], args) -> Tuple[torch.Tensor, Dict[str, float]]:
    r0, r_near, r_far = triplet
    h0, h, w, dn = model(r0, args.iters)
    if getattr(args, "detach_pairs", False) and model.training:
        # Lowest-VRAM mode: near/far branches are targets.
        with torch.no_grad():
            h0n, hn, _, _ = model(r_near, args.iters)
            h0f, hf, _, _ = model(r_far, args.iters)
    elif model.training:
        # Default v3 mode: near branch keeps gradients, so contraction cannot be
        # solved only by moving the anchor. Far branch is target-only to save VRAM.
        h0n, hn, _, _ = model(r_near, args.iters)
        with torch.no_grad():
            h0f, hf, _, _ = model(r_far, args.iters)
    else:
        h0n, hn, _, _ = model(r_near, args.iters)
        h0f, hf, _, _ = model(r_far, args.iters)

    d0 = model.descriptor(h0)
    d0n = model.descriptor(h0n)
    d0f = model.descriptor(h0f)
    desc = model.descriptor(h)
    desc_n = model.descriptor(hn)
    desc_f = model.descriptor(hf)

    input_same = desc_dist(d0, d0n).detach()
    input_far = desc_dist(d0, d0f).detach()
    same_d = desc_dist(desc, desc_n)
    far_d = desc_dist(desc, desc_f)

    # Descriptor-space same-basin contraction.
    # v3 used a raw squared-margin only; when input_same was tiny the gradient was too weak.
    # v4 combines absolute target + relative ratio target.
    same_loss = same_d.pow(2).mean()
    desc_target = torch.maximum(args.same_contract * input_same, torch.full_like(input_same, args.same_abs_target))
    contract_loss = F.relu(same_d - desc_target).pow(2).mean()
    contract_ratio = same_d / input_same.clamp_min(args.contract_eps)
    contract_ratio_loss = F.relu(contract_ratio - args.same_contract).pow(2).mean()

    # Hidden-space basin contraction. This prevents a descriptor-only trick and forces
    # the actual self-applied relation state H* of near perturbations to come together.
    h_input_same = (h0 - h0n).float().pow(2).mean(dim=(1, 2, 3)).sqrt().detach()
    h_output_same = (h - hn).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    h_contract_ratio = h_output_same / h_input_same.clamp_min(args.contract_eps)
    hidden_same_loss = h_output_same.pow(2).mean()
    hidden_contract_loss = F.relu(h_contract_ratio - args.hidden_same_contract).pow(2).mean()

    sep_loss = F.relu(args.margin - far_d).pow(2).mean()
    far_preserve_loss = F.relu(args.far_keep * input_far - far_d).pow(2).mean()

    # fixed point after self-application
    h_next, _, dn_next = model.continue_from_hidden(h, iters=args.fixed_iters)
    fixed_loss = (h_next - h).pow(2).mean()

    # recovery from perturbing stabilized state
    h_pert = h.detach() + args.state_noise * torch.randn_like(h)
    h_rec, _, _ = model.continue_from_hidden(h_pert, iters=args.recovery_iters)
    recovery_loss = (h_rec - h.detach()).pow(2).mean()

    # non-identity pressure: final descriptor should not equal initial descriptor exactly
    move_d = desc_dist(desc, d0)
    move_loss = F.relu(args.min_move - move_d).pow(2).mean()

    # batch diversity
    offd = offdiag_pairwise_dist(desc)
    batch_sep_loss = F.relu(args.batch_margin - offd).pow(2).mean()

    # convergence curve: first half should move more than last half, but don't overconstrain
    early_delta = dn[: max(1, args.iters // 3)].mean()
    late_delta = dn[max(1, (2 * args.iters) // 3):].mean() if args.iters > 2 else dn[-1].mean()
    motion_floor_loss = F.relu(args.min_early_delta - early_delta).pow(2)
    converge_loss = F.relu(late_delta - args.delta_decay * early_delta.detach()).pow(2)
    late_ceiling_loss = F.relu(late_delta - args.max_late_delta).pow(2)

    eff_ops_t, op_ent = op_effective_count(w)
    op_usage_loss = -op_ent
    per_sample_ent = -(w.mean(dim=0).clamp_min(1e-8) * w.mean(dim=0).clamp_min(1e-8).log()).sum(dim=-1).mean()
    op_sparse_loss = per_sample_ent

    # Anti-collapse guards. These are weak by default, but prevent the model from
    # winning contraction by compressing all states into a tiny descriptor/state cloud.
    state_var_t = h.float().var(dim=(1, 2)).mean()
    desc_raw_t = model.invariant_summary(h).float()
    desc_var_t = desc_raw_t.var(dim=0, unbiased=False).mean()
    var_floor_loss = F.relu(args.state_var_floor - state_var_t).pow(2) + F.relu(args.desc_var_floor - desc_var_t).pow(2)

    total = (
        args.same_w * same_loss
        + args.contract_w * contract_loss
        + args.contract_ratio_w * contract_ratio_loss
        + args.hidden_same_w * hidden_same_loss
        + args.hidden_contract_w * hidden_contract_loss
        + args.sep_w * sep_loss
        + args.far_preserve_w * far_preserve_loss
        + args.fixed_w * fixed_loss
        + args.recovery_w * recovery_loss
        + args.move_w * move_loss
        + args.batch_sep_w * batch_sep_loss
        + args.motion_w * motion_floor_loss
        + args.converge_w * converge_loss
        + args.late_w * late_ceiling_loss
        + args.op_usage_w * op_usage_loss
        + args.op_sparse_w * op_sparse_loss
        + args.var_floor_w * var_floor_loss
    )

    with torch.no_grad():
        curve = [float(x) for x in dn.mean(dim=1).detach().float().cpu().tolist()]
        desc_var = desc_var_t.detach().item()
        h_var = state_var_t.detach().item()
        metrics = {
            "loss": float(total.detach().item()),
            "input_same": float(input_same.mean().item()),
            "input_far": float(input_far.mean().item()),
            "same_d": float(same_d.mean().detach().item()),
            "far_d": float(far_d.mean().detach().item()),
            "sep_ratio": float((far_d.mean() / (same_d.mean() + 1e-8)).detach().item()),
            "same_contract": float((same_d.mean() / (input_same.mean() + 1e-8)).detach().item()),
            "contract_ratio_loss": float(contract_ratio_loss.detach().item()),
            "h_input_same": float(h_input_same.mean().detach().item()),
            "h_output_same": float(h_output_same.mean().detach().item()),
            "h_contract": float(h_contract_ratio.mean().detach().item()),
            "far_keep": float((far_d.mean() / (input_far.mean() + 1e-8)).detach().item()),
            "fixed": float(fixed_loss.detach().item()),
            "recovery": float(recovery_loss.detach().item()),
            "move": float(move_d.mean().detach().item()),
            "batch_offd": float(offd.mean().detach().item()),
            "desc_var": desc_var,
            "state_var": h_var,
            "early_delta": float(early_delta.detach().item()),
            "late_delta": float(late_delta.detach().item()),
            "eff_ops": float(eff_ops_t.detach().item()),
            "op_entropy": float(op_ent.detach().item()),
            "curve": curve,
        }
    return total, metrics


@torch.no_grad()
def identity_baseline(model: SelfReferentialClosureField, triplet: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
    r0, rn, rf = triplet
    h0 = model.encoder(r0)
    hn = model.encoder(rn)
    hf = model.encoder(rf)
    d0 = model.descriptor(h0)
    dn = model.descriptor(hn)
    df = model.descriptor(hf)
    same = desc_dist(d0, dn).mean()
    far = desc_dist(d0, df).mean()
    offd = offdiag_pairwise_dist(d0).mean()
    return {
        "id_same": float(same.item()),
        "id_far": float(far.item()),
        "id_ratio": float((far / (same + 1e-8)).item()),
        "id_batch": float(offd.item()),
    }


@torch.no_grad()
def permutation_error(model: SelfReferentialClosureField, sampler: RelationSampler, args) -> float:
    model.eval()
    r, _, _ = sampler.sample_triplet(min(8, args.eval_batch_size), n_override=args.n)
    if r.shape[1] > 96:
        return -1.0
    _, h, _, _ = model(r, args.iters)
    p = torch.randperm(r.shape[1], device=args.device)
    inv = inverse_perm(p)
    rp = r[:, p][:, :, p]
    _, hp, _, _ = model(rp, args.iters)
    hp_unp = hp[:, inv][:, :, inv]
    return (h - hp_unp).abs().max().item()


@torch.no_grad()
def evaluate(model: SelfReferentialClosureField, sampler: RelationSampler, args, n_override: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    triplet = sampler.sample_triplet(args.eval_batch_size, n_override=n_override)
    _, metrics = intrinsic_losses(model, triplet, args)
    metrics.update(identity_baseline(model, triplet))
    if n_override is None or n_override == args.n:
        metrics["perm_err"] = permutation_error(model, sampler, args)
    else:
        metrics["perm_err"] = -1.0
    return metrics


def format_curve(curve: List[float], max_items: int = 8) -> str:
    vals = curve[:max_items]
    return "[" + ",".join(f"{v:.3f}" for v in vals) + ("..." if len(curve) > max_items else "") + "]"


# ------------------------- train -------------------------

def train(args) -> None:
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    sampler = RelationSampler(args, args.device)
    model = SelfReferentialClosureField(
        rel_dim=args.rel_dim,
        dim=args.dim,
        n_ops=args.ops,
        iters=args.iters,
        controller_temp=args.controller_temp,
        checkpoint_ops=args.checkpoint_ops,
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device.startswith("cuda") and args.amp == "fp16"))

    print("SRCF-v4 basin-contractive hard/DNA self-referential closure field / no supervised labels")
    print(f"device={args.device} amp={args.amp} data={args.data} n={sampler.n} rel_dim={args.rel_dim} dim={args.dim} ops={args.ops} iters={args.iters}")
    print(f"contractive/memsafe: checkpoint_ops={args.checkpoint_ops} detach_pairs={args.detach_pairs} batch={args.batch_size} eval_batch={args.eval_batch_size}")
    print(f"params={sum(p.numel() for p in model.parameters()):,}")
    print("metrics: out_same low, out_far high, contract<1 and h_contract<1, far_keep~>=0.7, move>0, fixed/recovery low, curve should decay, perm~0")

    # frozen/random model baseline before any training
    m0 = evaluate(model, sampler, args, n_override=(args.n if args.data != "dna" else None))
    print(
        "baseline random model | "
        f"id_same={m0['id_same']:.3f} id_far={m0['id_far']:.3f} id_ratio={m0['id_ratio']:.2f} | "
        f"out_same={m0['same_d']:.3f} out_far={m0['far_d']:.3f} ratio={m0['sep_ratio']:.2f} "
        f"contract={m0['same_contract']:.2f} h_contract={m0['h_contract']:.2f} far_keep={m0['far_keep']:.2f} move={m0['move']:.3f} curve={format_curve(m0['curve'])}"
    )
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    for step in range(1, args.steps + 1):
        model.train()
        triplet = sampler.sample_triplet(args.batch_size)
        opt.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            loss, _ = intrinsic_losses(model, triplet, args)
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
            n_override = args.n if args.data != "dna" else None
            m = evaluate(model, sampler, args, n_override=n_override)
            if csv_file is not None:
                row = {"step": step, **{k: v for k, v in m.items() if not isinstance(v, list)}, "curve": ";".join(f"{x:.6g}" for x in m.get("curve", []))}
                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                csv_file.flush()
            print(
                f"step {step:05d} loss={m['loss']:.4f} | "
                f"id={m['id_same']:.3f}/{m['id_far']:.3f}({m['id_ratio']:.1f}) "
                f"out={m['same_d']:.3f}/{m['far_d']:.3f}({m['sep_ratio']:.1f}) "
                f"contract={m['same_contract']:.2f} h_contract={m['h_contract']:.2f} far_keep={m['far_keep']:.2f} "
                f"fixed={m['fixed']:.4f} recovery={m['recovery']:.4f} move={m['move']:.3f} "
                f"batch={m['batch_offd']:.3f} state_var={m['state_var']:.3f} desc_var={m['desc_var']:.2e} "
                f"eff_ops={m['eff_ops']:.2f} perm={m['perm_err']:.1e} curve={format_curve(m['curve'])}"
            )
            if args.data != "dna" and args.ood_n and args.ood_n != args.n:
                o = evaluate(model, sampler, args, n_override=args.ood_n)
                print(
                    f"  OOD-N={args.ood_n}: id={o['id_same']:.3f}/{o['id_far']:.3f}({o['id_ratio']:.1f}) "
                    f"out={o['same_d']:.3f}/{o['far_d']:.3f}({o['sep_ratio']:.1f}) "
                    f"contract={o['same_contract']:.2f} h_contract={o['h_contract']:.2f} far_keep={o['far_keep']:.2f} "
                    f"fixed={o['fixed']:.4f} recovery={o['recovery']:.4f} move={o['move']:.3f} eff_ops={o['eff_ops']:.2f}"
                )
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

    if args.save_path:
        torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
        print(f"saved: {args.save_path}")
    if csv_file is not None:
        csv_file.close()


# ------------------------- args -------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)

    p.add_argument("--data", choices=["synthetic", "dna", "mixed"], default="synthetic")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--ood-n", type=int, default=48)
    p.add_argument("--rel-dim", type=int, default=8)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--ops", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--fixed-iters", type=int, default=2)
    p.add_argument("--recovery-iters", type=int, default=4)
    p.add_argument("--controller-temp", type=float, default=1.0)
    p.add_argument("--checkpoint-ops", action=argparse.BooleanOptionalAction, default=True, help="recompute heavy operator activations during backward to save VRAM")
    p.add_argument("--detach-pairs", action=argparse.BooleanOptionalAction, default=False, help="train near/far branches as targets to save VRAM; gradients still flow through anchor branch")

    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # synthetic
    p.add_argument("--structure-mix", type=float, default=0.70)
    p.add_argument("--input-noise", type=float, default=0.18)

    # DNA
    p.add_argument("--dna-accession", type=str, default="NC_000913.3")
    p.add_argument("--dna-bases", type=int, default=200000)
    p.add_argument("--dna-window", type=int, default=2048)
    p.add_argument("--dna-cache", type=str, default="./dna_NC_000913_200k.fasta")
    p.add_argument("--download-dna", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kmer", type=int, default=3)
    p.add_argument("--dna-mut-rate", type=float, default=0.040)
    p.add_argument("--mixed-dna-prob", type=float, default=0.5)

    # intrinsic loss hyperparams
    p.add_argument("--state-noise", type=float, default=0.08)
    p.add_argument("--margin", type=float, default=0.65)
    p.add_argument("--batch-margin", type=float, default=0.45)
    p.add_argument("--same-contract", type=float, default=0.70)
    p.add_argument("--same-abs-target", type=float, default=0.020)
    p.add_argument("--contract-eps", type=float, default=0.010)
    p.add_argument("--hidden-same-contract", type=float, default=0.75)
    p.add_argument("--far-keep", type=float, default=0.80)
    p.add_argument("--min-move", type=float, default=0.14)
    p.add_argument("--min-early-delta", type=float, default=0.020)
    p.add_argument("--delta-decay", type=float, default=0.65)
    p.add_argument("--max-late-delta", type=float, default=0.25)

    p.add_argument("--same-w", type=float, default=2.0)
    p.add_argument("--contract-w", type=float, default=8.0)
    p.add_argument("--contract-ratio-w", type=float, default=1.5)
    p.add_argument("--hidden-same-w", type=float, default=1.0)
    p.add_argument("--hidden-contract-w", type=float, default=2.0)
    p.add_argument("--sep-w", type=float, default=1.0)
    p.add_argument("--far-preserve-w", type=float, default=0.9)
    p.add_argument("--fixed-w", type=float, default=0.25)
    p.add_argument("--recovery-w", type=float, default=0.35)
    p.add_argument("--move-w", type=float, default=0.25)
    p.add_argument("--batch-sep-w", type=float, default=0.35)
    p.add_argument("--motion-w", type=float, default=0.15)
    p.add_argument("--converge-w", type=float, default=0.25)
    p.add_argument("--late-w", type=float, default=0.10)
    p.add_argument("--op-usage-w", type=float, default=0.004)
    p.add_argument("--op-sparse-w", type=float, default=0.001)

    p.add_argument("--save-path", type=str, default="")
    p.add_argument("--metrics-csv", type=str, default="", help="optional CSV path for step metrics")
    p.add_argument("--state-var-floor", type=float, default=0.20)
    p.add_argument("--desc-var-floor", type=float, default=1e-4)
    p.add_argument("--var-floor-w", type=float, default=0.15)
    args = p.parse_args()
    if args.data == "dna":
        args.n = 4 ** args.kmer
    if args.kmer < 2 or args.kmer > 4:
        raise ValueError("--kmer should be 2, 3, or 4; k=3 is recommended for P40")
    return args


if __name__ == "__main__":
    train(parse_args())
