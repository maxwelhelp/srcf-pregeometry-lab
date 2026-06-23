#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
circuit_birth_finetune_protect_v1.py
VERSION = v1.0-circuit-birth-finetune-protect-quick

Real quick A/B fine-tune benchmark:

  A) baseline LoRA fine-tune on tiny edit facts
  B) same LoRA + circuit-birth protected-subspace regularizer

Purpose
-------
This is the first test that asks:
  Does circuit-target signal birth help a real training/patch loop?

It does NOT claim big downstream generalization. It measures:
  - edit_loss / edit_token_acc
  - retain_loss / retain_token_acc
  - circuit protected coefficient drift
  - tradeoff: same edit learning with less retain/circuit drift?

Circuit protection
------------------
Before fine-tune, extract real Qwen attention circuit targets for selected layer(s):
  M_qk_aug[h,d] = Wq_aug[h].T @ R_delta @ Wk_aug[kv] / sqrt(D)
  C_vo_aug[h]   = Wo[h] @ Wv_aug[kv]

Build base operator dictionary, mine residual BirthOps:
  QK BirthOps: PCA across heads/deltas
  VO BirthOps: per-head residual direction / SVD direction

During protected fine-tune:
  penalize drift of projection coefficients of current M_qk/C_vo on these BirthOps.

This is intentionally small and fast. Use it as a direction check.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

VERSION = "v1.3-circuit-birth-finetune-protect-l1-stress"


# ---------------- basics ----------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name: str):
    name = str(name).lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(name)


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


def parse_head_list(s: str, n_heads: int) -> List[int]:
    s = str(s).strip().lower()
    if s in ("all", "*"):
        return list(range(n_heads))
    out = [int(x.strip()) for x in s.replace(";", ",").split(",") if x.strip()]
    return [h for h in out if 0 <= h < n_heads]


def get_layers(model: Any):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot find transformer layers")


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    rotary = getattr(model.model, "rotary_emb", None) if hasattr(model, "model") else None
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return rotary(position_ids)


def rotate_half_matrix(D: int, device, dtype=torch.float32) -> torch.Tensor:
    P = torch.zeros(D, D, device=device, dtype=dtype)
    h = D // 2
    for i in range(h):
        P[i, h + i] = -1.0
        P[h + i, i] = 1.0
    return P


def rope_col_matrix(cos_row: torch.Tensor, sin_row: torch.Tensor) -> torch.Tensor:
    cos_row = cos_row.detach().float()
    sin_row = sin_row.detach().float()
    D = int(cos_row.numel())
    P = rotate_half_matrix(D, cos_row.device, cos_row.dtype)
    return torch.diag(cos_row) + torch.diag(sin_row) @ P


@torch.no_grad()
def build_rope_pos_from_model(model, max_delta: int, device: str) -> Dict[int, torch.Tensor]:
    cfg = model.config
    H = int(cfg.hidden_size)
    T = max_delta + 1
    hidden = torch.zeros(1, T, H, device=device, dtype=next(model.parameters()).dtype)
    pos = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = compute_position_embeddings(model, hidden, pos)
    cos2 = cos[0] if cos.dim() == 3 else cos
    sin2 = sin[0] if sin.dim() == 3 else sin
    return {i: rope_col_matrix(cos2[i].float().cpu(), sin2[i].float().cpu()) for i in range(T)}


def rel_change(new: float, old: float, eps: float = 1e-12) -> float:
    return (old - new) / max(abs(old), eps)


# ---------------- LoRA ----------------

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.r)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        in_f = base.in_features
        out_f = base.out_features

        # IMPORTANT:
        # Keep trainable LoRA params in fp32 even when the model is fp16.
        # AdamW on fp16 LoRA params can explode to NaN in a few steps.
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout.to(device=dev)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # Compute LoRA branch in fp32 for stability, then cast back to model dtype.
        z = F.linear(self.dropout(x).float(), self.lora_A)
        z = F.linear(z, self.lora_B) * self.scaling
        return y + z.to(dtype=y.dtype)

    def effective_weight(self) -> torch.Tensor:
        # Circuit extraction uses fp32 effective weights.
        return self.base.weight.float() + (self.lora_B @ self.lora_A) * self.scaling

    def effective_bias(self):
        b = self.base.bias
        return None if b is None else b.float()


def apply_lora(model, layers: List[int], targets: List[str], r: int, alpha: float, dropout: float) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
    tr_layers = get_layers(model)
    for li in layers:
        attn = tr_layers[li].self_attn
        for name in targets:
            mod = getattr(attn, name)
            if not isinstance(mod, LoRALinear):
                wrapped = LoRALinear(mod, r=r, alpha=alpha, dropout=dropout)
                # Do NOT call wrapped.to(dtype=mod.weight.dtype), it would convert LoRA fp32 params to fp16.
                wrapped.to(device=mod.weight.device)
                setattr(attn, name, wrapped)
    model.train()


def trainable_params(model) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def module_weight_bias(mod: nn.Module) -> Tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(mod, LoRALinear):
        return mod.effective_weight(), mod.effective_bias()
    return mod.weight, getattr(mod, "bias", None)


# ---------------- operator dictionary and birth mining ----------------

def dct_matrix(n: int, device, dtype=torch.float32) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
    i = torch.arange(n, device=device, dtype=dtype).view(1, -1)
    C = torch.cos(math.pi / n * (i + 0.5) * k)
    C[0, :] *= math.sqrt(1.0 / n)
    if n > 1:
        C[1:, :] *= math.sqrt(2.0 / n)
    return C


def shift_square(n: int, shift: int, device, dtype=torch.float32) -> torch.Tensor:
    M = torch.zeros(n, n, device=device, dtype=dtype)
    for i in range(n):
        j = i - shift
        if 0 <= j < n:
            M[i, j] = 1.0
    return M


def shift_rect(rows: int, cols: int, shift: int, device, dtype=torch.float32) -> torch.Tensor:
    M = torch.zeros(rows, cols, device=device, dtype=dtype)
    usable = min(rows, cols)
    for i in range(rows):
        j = i - shift
        if 0 <= j < usable:
            M[i, j] = 1.0
    return M


def block_avg_square(n: int, block: int, device, dtype=torch.float32) -> torch.Tensor:
    M = torch.zeros(n, n, device=device, dtype=dtype)
    for s in range(0, n, block):
        e = min(n, s + block)
        M[s:e, s:e] = 1.0 / max(1, e - s)
    return M


def block_avg_rect(rows: int, cols: int, block: int, device, dtype=torch.float32) -> torch.Tensor:
    M = torch.zeros(rows, cols, device=device, dtype=dtype)
    usable = min(rows, cols)
    for s in range(0, usable, block):
        e = min(usable, s + block)
        M[s:e, s:e] = 1.0 / max(1, e - s)
    return M


class MatrixDict:
    def __init__(self, ops: List[Tuple[str, torch.Tensor]], ridge: float = 1e-4, device: str = "cpu"):
        self.names = [n for n, _ in ops]
        mats = [m.detach().float().to(device) for _, m in ops]
        self.ops = torch.stack(mats, dim=0).contiguous()
        self.P = int(self.ops.shape[0])
        flat = self.ops.reshape(self.P, -1).T.contiguous()
        norms = torch.linalg.norm(flat, dim=0).clamp_min(1e-12)
        self.A = flat / norms.view(1, -1)
        self.norms = norms
        eye = torch.eye(self.P, device=device)
        self.pinv = torch.linalg.solve(self.A.T @ self.A + ridge * eye, self.A.T)

    @torch.no_grad()
    def decode(self, targets: torch.Tensor) -> torch.Tensor:
        B = int(targets.shape[0])
        flat = targets.detach().float().reshape(B, -1).to(self.A.device)
        c_norm = flat @ self.pinv.T
        c_raw = c_norm / self.norms.view(1, -1)
        recon = torch.einsum("bp,pij->bij", c_raw, self.ops)
        return recon


def build_square_dict(n: int, device: str) -> MatrixDict:
    dev = torch.device(device)
    ops: List[Tuple[str, torch.Tensor]] = []
    I = torch.eye(n, device=dev)
    ops.append(("Identity", I))
    ramp = torch.linspace(-1, 1, n, device=dev)
    ops.append(("RampDiag", torch.diag(ramp)))
    ops.append(("MeanProject", torch.ones(n, n, device=dev) / n))
    for sh in [1, 2, 4, 8, 16, 32]:
        if sh < n:
            ops.append((f"ShiftR{sh}", shift_square(n, sh, dev)))
            ops.append((f"ShiftL{sh}", shift_square(n, -sh, dev)))
    for b in [2, 4, 8, 16, 32, 64, 128]:
        if b < n:
            ops.append((f"BlockAvg{b}", block_avg_square(n, b, dev)))
    try:
        C = dct_matrix(n, dev)
        k = max(1, n // 8)
        ops.append(("DCTLow", C[:k].T @ C[:k]))
        ops.append(("DCTHigh", C[-k:].T @ C[-k:]))
    except Exception:
        pass
    return MatrixDict(ops, device=device)


def build_rect_dict(rows: int, cols: int, device: str) -> MatrixDict:
    dev = torch.device(device)
    ops: List[Tuple[str, torch.Tensor]] = []
    M = torch.zeros(rows, cols, device=dev)
    d = min(rows, cols)
    M[torch.arange(d), torch.arange(d)] = 1.0
    ops.append(("IdentityRect", M))
    if cols == rows + 1:
        B = torch.zeros(rows, cols, device=dev)
        B[:, -1] = 1.0 / math.sqrt(rows)
        ops.append(("BiasColumn", B))
    ramp = torch.linspace(-1, 1, rows, device=dev)
    R = torch.zeros(rows, cols, device=dev)
    R[torch.arange(d), torch.arange(d)] = ramp[:d]
    ops.append(("RampDiagRect", R))
    ops.append(("MeanRect", torch.ones(rows, cols, device=dev) / math.sqrt(rows * cols)))
    for sh in [1, 2, 4, 8, 16, 32]:
        if sh < rows:
            ops.append((f"ShiftR{sh}Rect", shift_rect(rows, cols, sh, dev)))
            ops.append((f"ShiftL{sh}Rect", shift_rect(rows, cols, -sh, dev)))
    for b in [2, 4, 8, 16, 32, 64, 128]:
        if b < rows:
            ops.append((f"BlockAvg{b}Rect", block_avg_rect(rows, cols, b, dev)))
    return MatrixDict(ops, device=device)


@torch.no_grad()
def mine_pca_ops(residuals: torch.Tensor, k: int, device: str) -> List[torch.Tensor]:
    B = int(residuals.shape[0])
    flat = residuals.reshape(B, -1).float().to(device)
    norms = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    flatn = flat / norms
    q = min(max(1, int(k)), min(flatn.shape))
    try:
        _, S, V = torch.pca_lowrank(flatn, q=q, center=False, niter=2)
        comps = V.T.contiguous()
    except Exception:
        _, _, Vh = torch.linalg.svd(flatn, full_matrices=False)
        comps = Vh[:q]
    out = []
    m, n = residuals.shape[1], residuals.shape[2]
    for i in range(q):
        op = comps[i].reshape(m, n)
        op = op / torch.linalg.norm(op).clamp_min(1e-12)
        out.append(op.detach().to(device))
    return out


@torch.no_grad()
def first_svd_op(residual: torch.Tensor, device: str) -> torch.Tensor:
    residual = residual.detach().float().to(device)
    try:
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        op = torch.outer(U[:, 0], Vh[0])
    except Exception:
        op = residual
    return (op / torch.linalg.norm(op).clamp_min(1e-12)).detach()


# ---------------- circuit extraction/protection ----------------

@dataclass
class QKRef:
    layer: int
    head: int
    delta: int
    op: torch.Tensor
    c0: torch.Tensor


@dataclass
class VORef:
    layer: int
    head: int
    op: torch.Tensor
    c0: torch.Tensor


@dataclass
class CircuitSpec:
    qk_refs: List[QKRef]
    vo_refs: List[VORef]
    meta_by_layer: Dict[int, Dict[str, int]]
    Rpos_by_layer: Dict[int, Dict[int, torch.Tensor]]


def get_aug_weights(model, layer_idx: int, meta: Dict[str, int], device: str) -> Dict[str, torch.Tensor]:
    layer = get_layers(model)[layer_idx]
    attn = layer.self_attn
    H = int(meta["H"]); D = int(meta["D"]); n_heads = int(meta["n_heads"]); n_kv = int(meta["n_kv"])

    Wq_raw, bq_raw = module_weight_bias(attn.q_proj)
    Wk_raw, bk_raw = module_weight_bias(attn.k_proj)
    Wv_raw, bv_raw = module_weight_bias(attn.v_proj)
    Wo_raw, bo_raw = module_weight_bias(attn.o_proj)

    Wq = Wq_raw.float().to(device).view(n_heads, D, H)
    Wk = Wk_raw.float().to(device).view(n_kv, D, H)
    Wv = Wv_raw.float().to(device).view(n_kv, D, H)
    Wo_flat = Wo_raw.float().to(device)
    Wo = torch.stack([Wo_flat[:, h * D:(h + 1) * D] for h in range(n_heads)], dim=0)

    def bias_aug(b, count):
        if b is None:
            return torch.zeros(count, D, device=device)
        return b.float().to(device).view(count, D)

    bq = bias_aug(bq_raw, n_heads)
    bk = bias_aug(bk_raw, n_kv)
    bv = bias_aug(bv_raw, n_kv)
    Wq_aug = torch.cat([Wq, bq[:, :, None]], dim=2)
    Wk_aug = torch.cat([Wk, bk[:, :, None]], dim=2)
    Wv_aug = torch.cat([Wv, bv[:, :, None]], dim=2)
    return {"Wq_aug": Wq_aug, "Wk_aug": Wk_aug, "Wv_aug": Wv_aug, "Wo": Wo}


def qk_target_for(weights, Rpos, meta, h: int, d: int) -> torch.Tensor:
    kv = h // int(meta["kv_groups"])
    D = int(meta["D"])
    Rrel = Rpos[d].to(weights["Wq_aug"].device).float().T @ Rpos[0].to(weights["Wq_aug"].device).float()
    return (weights["Wq_aug"][h].T @ Rrel @ weights["Wk_aug"][kv]) / math.sqrt(D)


def vo_target_for(weights, meta, h: int) -> torch.Tensor:
    kv = h // int(meta["kv_groups"])
    return weights["Wo"][h] @ weights["Wv_aug"][kv]


@torch.no_grad()
def build_layer_targets(model, layer_idx: int, meta: Dict[str, int], Rpos: Dict[int, torch.Tensor],
                        heads: List[int], deltas: List[int], device: str) -> Tuple[torch.Tensor, List[Tuple[int, int]], torch.Tensor]:
    weights = get_aug_weights(model, layer_idx, meta, device)
    Ms = []
    midx = []
    for h in heads:
        for d in deltas:
            Ms.append(qk_target_for(weights, Rpos, meta, h, d).detach())
            midx.append((h, d))
    Cs = [vo_target_for(weights, meta, h).detach() for h in heads]
    return torch.stack(Ms, dim=0), midx, torch.stack(Cs, dim=0)


@torch.no_grad()
def build_circuit_spec(model, layers: List[int], heads_mode: str, max_delta: int,
                       qk_births: int, vo_births: int, vo_mode: str, device: str) -> CircuitSpec:
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    H = int(cfg.hidden_size)
    D = int(getattr(cfg, "head_dim", H // n_heads))
    kv_groups = n_heads // n_kv
    heads = parse_head_list(heads_mode, n_heads)
    deltas = list(range(max_delta + 1))

    qk_refs: List[QKRef] = []
    vo_refs: List[VORef] = []
    meta_by_layer: Dict[int, Dict[str, int]] = {}
    Rpos_by_layer: Dict[int, Dict[int, torch.Tensor]] = {}

    for li in layers:
        meta = {"H": H, "D": D, "n_heads": n_heads, "n_kv": n_kv, "kv_groups": kv_groups}
        meta_by_layer[li] = meta
        Rpos = build_rope_pos_from_model(model, max_delta, device)
        Rpos_by_layer[li] = Rpos

        M, midx, C = build_layer_targets(model, li, meta, Rpos, heads, deltas, device)
        qk_dict = build_square_dict(H + 1, device)
        M_recon = qk_dict.decode(M)
        M_res = M - M_recon
        qk_ops = mine_pca_ops(M_res, qk_births, device)

        for op in qk_ops:
            for idx, (h, d) in enumerate(midx):
                c0 = torch.sum(M[idx] * op).detach()
                qk_refs.append(QKRef(li, h, d, op.detach(), c0))

        vo_dict = build_rect_dict(H, H + 1, device)
        C_recon = vo_dict.decode(C)
        C_res = C - C_recon
        for hi, h in enumerate(heads):
            if vo_mode == "none" or vo_births <= 0:
                continue
            if vo_mode == "full":
                op = C_res[hi] / torch.linalg.norm(C_res[hi]).clamp_min(1e-12)
                c0 = torch.sum(C[hi] * op).detach()
                vo_refs.append(VORef(li, h, op.detach(), c0))
            else:
                # svd1
                op = first_svd_op(C_res[hi], device)
                c0 = torch.sum(C[hi] * op).detach()
                vo_refs.append(VORef(li, h, op.detach(), c0))

        print(f"[protect spec] L{li}: heads={len(heads)} deltas={len(deltas)} qk_refs={len(qk_refs)} vo_refs={len(vo_refs)}")
    return CircuitSpec(qk_refs, vo_refs, meta_by_layer, Rpos_by_layer)


def circuit_protect_loss(model, spec: CircuitSpec, qk_weight: float, vo_weight: float, loss_kind: str = "l2") -> torch.Tensor:
    device = next(model.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    count = 0
    cache_weights: Dict[int, Dict[str, torch.Tensor]] = {}

    def penalty(x: torch.Tensor) -> torch.Tensor:
        if loss_kind == "l1":
            return x.abs()
        if loss_kind == "huber":
            return F.smooth_l1_loss(x, torch.zeros_like(x), reduction="none")
        return x.pow(2)

    for ref in spec.qk_refs:
        if ref.layer not in cache_weights:
            cache_weights[ref.layer] = get_aug_weights(model, ref.layer, spec.meta_by_layer[ref.layer], str(device))
        weights = cache_weights[ref.layer]
        M = qk_target_for(weights, spec.Rpos_by_layer[ref.layer], spec.meta_by_layer[ref.layer], ref.head, ref.delta)
        c = torch.sum(M * ref.op.to(device))
        denom = (ref.c0.to(device).abs() + 1e-3).detach()
        total = total + qk_weight * penalty((c - ref.c0.to(device)) / denom)
        count += 1

    for ref in spec.vo_refs:
        if ref.layer not in cache_weights:
            cache_weights[ref.layer] = get_aug_weights(model, ref.layer, spec.meta_by_layer[ref.layer], str(device))
        weights = cache_weights[ref.layer]
        C = vo_target_for(weights, spec.meta_by_layer[ref.layer], ref.head)
        c = torch.sum(C * ref.op.to(device))
        denom = (ref.c0.to(device).abs() + 1e-3).detach()
        total = total + vo_weight * penalty((c - ref.c0.to(device)) / denom)
        count += 1

    if count == 0:
        return total
    return total / count


@torch.no_grad()
def circuit_drift(model, spec: CircuitSpec) -> Dict[str, float]:
    device = next(model.parameters()).device
    q_vals = []
    v_vals = []
    cache_weights: Dict[int, Dict[str, torch.Tensor]] = {}
    for ref in spec.qk_refs:
        if ref.layer not in cache_weights:
            cache_weights[ref.layer] = get_aug_weights(model, ref.layer, spec.meta_by_layer[ref.layer], str(device))
        weights = cache_weights[ref.layer]
        M = qk_target_for(weights, spec.Rpos_by_layer[ref.layer], spec.meta_by_layer[ref.layer], ref.head, ref.delta)
        c = torch.sum(M * ref.op.to(device))
        denom = ref.c0.to(device).abs() + 1e-3
        q_vals.append(float(((c - ref.c0.to(device)).abs() / denom).detach().cpu()))
    for ref in spec.vo_refs:
        if ref.layer not in cache_weights:
            cache_weights[ref.layer] = get_aug_weights(model, ref.layer, spec.meta_by_layer[ref.layer], str(device))
        weights = cache_weights[ref.layer]
        C = vo_target_for(weights, spec.meta_by_layer[ref.layer], ref.head)
        c = torch.sum(C * ref.op.to(device))
        denom = ref.c0.to(device).abs() + 1e-3
        v_vals.append(float(((c - ref.c0.to(device)).abs() / denom).detach().cpu()))
    def avg(xs):
        return sum(xs) / max(1, len(xs))
    return {"qk_coeff_drift": avg(q_vals), "vo_coeff_drift": avg(v_vals), "n_qk": len(q_vals), "n_vo": len(v_vals)}


# ---------------- dataset ----------------

FACT_WORDS = [
    ("Aster", "zorbax"), ("Beryl", "nuvium"), ("Cedar", "lomtek"), ("Dora", "vexlan"),
    ("Elm", "praxor"), ("Fable", "quintor"), ("Garnet", "mivora"), ("Haven", "seldin"),
    ("Iris", "kavrol"), ("Juno", "tarnex"), ("Kite", "orbix"), ("Lumen", "fendro"),
    ("Mango", "zelpin"), ("Nova", "crayth"), ("Opal", "vornix"), ("Piper", "draxil"),
]

RETAIN = [
    ("Question: What is 2 + 3?\nAnswer:", " 5"),
    ("Question: What color is the sky on a clear day?\nAnswer:", " blue"),
    ("Question: What is the capital of France?\nAnswer:", " Paris"),
    ("Question: Opposite of hot is?\nAnswer:", " cold"),
    ("Question: How many days are in a week?\nAnswer:", " 7"),
    ("Question: What language is primarily spoken in Spain?\nAnswer:", " Spanish"),
    ("Question: 10 minus 4 equals?\nAnswer:", " 6"),
    ("Question: Water freezes at 0 degrees Celsius. True or false?\nAnswer:", " true"),
    ("Question: The chemical symbol for water is?\nAnswer:", " H2O"),
    ("Question: A triangle has how many sides?\nAnswer:", " 3"),
]


def make_edit_examples(n: int) -> List[Tuple[str, str]]:
    rows = []
    for i in range(n):
        name, word = FACT_WORDS[i % len(FACT_WORDS)]
        key = f"{name}-{100+i}"
        prompt = (
            f"User: Memorize this artificial fact. The secret codeword for {key} is {word}.\n"
            f"Question: What is the secret codeword for {key}?\nAnswer:"
        )
        rows.append((prompt, " " + word))
    return rows


def encode_example(tok, prompt: str, answer: str, max_len: int) -> Dict[str, List[int]]:
    eos = tok.eos_token or ""
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(answer + eos, add_special_tokens=False).input_ids
    ids = p_ids + a_ids
    labels = [-100] * len(p_ids) + a_ids
    if len(ids) > max_len:
        ids = ids[-max_len:]
        labels = labels[-max_len:]
    return {"input_ids": ids, "labels": labels}


def collate(tok, batch: List[Dict[str, List[int]]], device: str) -> Dict[str, torch.Tensor]:
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attn = []
    for x in batch:
        l = len(x["input_ids"])
        pad_n = max_len - l
        input_ids.append(x["input_ids"] + [pad] * pad_n)
        labels.append(x["labels"] + [-100] * pad_n)
        attn.append([1] * l + [0] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, device=device, dtype=torch.long),
        "labels": torch.tensor(labels, device=device, dtype=torch.long),
        "attention_mask": torch.tensor(attn, device=device, dtype=torch.long),
    }


def cycle_batches(data: List[Dict[str, List[int]]], batch_size: int, seed: int):
    rng = random.Random(seed)
    while True:
        idx = list(range(len(data)))
        rng.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            yield [data[j] for j in idx[i:i+batch_size]]


@torch.no_grad()
def eval_loss_acc(model, tok, data: List[Dict[str, List[int]]], batch_size: int, device: str) -> Dict[str, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    for i in range(0, len(data), batch_size):
        batch = collate(tok, data[i:i+batch_size], device)
        out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
        logits = out.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        pred = logits.argmax(dim=-1)
        correct += int(((pred == labels) & mask).sum().detach().cpu())
        total += int(mask.sum().detach().cpu())
    model.train()
    return {"loss": sum(losses) / max(1, len(losses)), "tok_acc": correct / max(1, total)}


# ---------------- training ----------------

def load_model_tok(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    model.config.use_cache = False
    return model, tok


def run_one(mode: str, args, edit_data, retain_data) -> Dict[str, Any]:
    print(f"\n=== RUN {mode} ===")
    set_seed(args.seed)
    model, tok = load_model_tok(args)
    layers = parse_int_list(args.layers)
    apply_lora(model, layers, ["q_proj", "k_proj", "v_proj", "o_proj"], args.lora_r, args.lora_alpha, args.lora_dropout)

    spec = build_circuit_spec(
        model, layers=layers, heads_mode=args.protect_heads, max_delta=args.protect_max_delta,
        qk_births=args.protect_qk_births, vo_births=args.protect_vo_births,
        vo_mode=args.protect_vo_mode, device=args.device
    )

    opt = torch.optim.AdamW(trainable_params(model), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    batches = cycle_batches(edit_data, args.batch_size, args.seed + 17)

    before_edit = eval_loss_acc(model, tok, edit_data, args.eval_batch_size, args.device)
    before_retain = eval_loss_acc(model, tok, retain_data, args.eval_batch_size, args.device)
    before_drift = circuit_drift(model, spec)
    print(f"before edit loss={before_edit['loss']:.4f} acc={before_edit['tok_acc']:.3f} | retain loss={before_retain['loss']:.4f} acc={before_retain['tok_acc']:.3f} | drift={before_drift}")

    last = {}
    for step in range(1, args.steps + 1):
        batch = collate(tok, next(batches), args.device)
        out = model(**batch)
        ce = out.loss
        reg = torch.zeros((), device=args.device)
        if mode == "protect":
            reg = circuit_protect_loss(model, spec, args.lambda_qk, args.lambda_vo, loss_kind=args.protect_loss)
        loss = ce + args.lambda_protect * reg
        if not torch.isfinite(loss):
            print(f"{mode} step {step:04d}/{args.steps} NONFINITE ce={float(ce.detach().cpu())} reg={float(reg.detach().cpu())} loss={float(loss.detach().cpu())}; stopping run")
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params(model), args.grad_clip)
        opt.step()
        last = {"ce": float(ce.detach().cpu()), "reg": float(reg.detach().cpu()), "loss": float(loss.detach().cpu())}
        if step % args.log_every == 0 or step == 1 or step == args.steps:
            print(f"{mode} step {step:04d}/{args.steps} ce={last['ce']:.4f} reg={last['reg']:.3e} loss={last['loss']:.4f}")

    after_edit = eval_loss_acc(model, tok, edit_data, args.eval_batch_size, args.device)
    after_retain = eval_loss_acc(model, tok, retain_data, args.eval_batch_size, args.device)
    after_drift = circuit_drift(model, spec)
    print(f"after  edit loss={after_edit['loss']:.4f} acc={after_edit['tok_acc']:.3f} | retain loss={after_retain['loss']:.4f} acc={after_retain['tok_acc']:.3f} | drift={after_drift}")

    return {
        "mode": mode,
        "before_edit": before_edit, "before_retain": before_retain, "before_drift": before_drift,
        "after_edit": after_edit, "after_retain": after_retain, "after_drift": after_drift,
        "last": last,
    }


def pct_improve_smaller(new, old):
    return 100.0 * (old - new) / max(1e-12, abs(old))


def main():
    args = parse_args()
    print(f"=== Circuit Birth Fine-tune Protect {VERSION} ===")
    print(f"model={args.model} device={args.device} dtype={args.dtype}")
    print(f"layers={args.layers} steps={args.steps} batch={args.batch_size} lr={args.lr}")
    print(f"lambda_protect={args.lambda_protect} lambda_qk={args.lambda_qk} lambda_vo={args.lambda_vo}")
    print("")

    # Need tokenizer for encoding, but avoid keeping first model.
    model_tmp, tok = load_model_tok(args)
    del model_tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    edit_pairs = make_edit_examples(args.edit_examples)
    retain_pairs = RETAIN[:args.retain_examples]
    edit_data = [encode_example(tok, p, a, args.max_seq_len) for p, a in edit_pairs]
    retain_data = [encode_example(tok, p, a, args.max_seq_len) for p, a in retain_pairs]

    base = run_one("baseline", args, edit_data, retain_data)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prot = run_one("protect", args, edit_data, retain_data)

    print("\n=== SUMMARY ===")
    for key in ["edit", "retain"]:
        b0 = base[f"before_{key}"]["loss"]
        b1 = base[f"after_{key}"]["loss"]
        p1 = prot[f"after_{key}"]["loss"]
        print(f"{key}_loss: before={b0:.4f} baseline_after={b1:.4f} protect_after={p1:.4f} protect_vs_baseline_delta={p1-b1:+.4f}")
        print(f"{key}_acc:  before={base[f'before_{key}']['tok_acc']:.3f} baseline_after={base[f'after_{key}']['tok_acc']:.3f} protect_after={prot[f'after_{key}']['tok_acc']:.3f}")

    for drift_key in ["qk_coeff_drift", "vo_coeff_drift"]:
        bd = base["after_drift"][drift_key]
        pd = prot["after_drift"][drift_key]
        print(f"{drift_key}: baseline={bd:.6f} protect={pd:.6f} reduction={pct_improve_smaller(pd, bd):+.2f}%")

    # Simple interpretation
    edit_gap = prot["after_edit"]["loss"] - base["after_edit"]["loss"]
    retain_gain = base["after_retain"]["loss"] - prot["after_retain"]["loss"]
    qk_red = pct_improve_smaller(prot["after_drift"]["qk_coeff_drift"], base["after_drift"]["qk_coeff_drift"])
    vo_red = pct_improve_smaller(prot["after_drift"]["vo_coeff_drift"], base["after_drift"]["vo_coeff_drift"])
    print("\nInterpretation:")
    print("  Good outcome: protect keeps edit_loss close to baseline, lowers retain_loss, and reduces circuit drift.")
    print(f"  edit_loss_gap protect-baseline = {edit_gap:+.4f} (<= small is good)")
    print(f"  retain_loss_gain baseline-protect = {retain_gain:+.4f} (>0 is good)")
    print(f"  circuit drift reduction QK={qk_red:+.2f}% VO={vo_red:+.2f}%")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--attn-implementation", default="eager")

    p.add_argument("--layers", default="23")
    p.add_argument("--protect-heads", default="0,1,2,3,4,5,6")
    p.add_argument("--protect-max-delta", type=int, default=2)
    p.add_argument("--protect-qk-births", type=int, default=1)
    p.add_argument("--protect-vo-births", type=int, default=1)
    p.add_argument("--protect-vo-mode", choices=["svd1", "full", "none"], default="svd1")

    p.add_argument("--lora-r", type=int, default=4)
    p.add_argument("--lora-alpha", type=float, default=8.0)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--adam-eps", type=float, default=1e-6)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-clip", type=float, default=0.3)

    p.add_argument("--lambda-protect", type=float, default=0.05)
    p.add_argument("--lambda-qk", type=float, default=1.0)
    p.add_argument("--lambda-vo", type=float, default=1.0)
    p.add_argument("--protect-loss", choices=["l1", "l2", "huber"], default="l1")

    p.add_argument("--edit-examples", type=int, default=12)
    p.add_argument("--retain-examples", type=int, default=10)
    p.add_argument("--max-seq-len", type=int, default=96)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


if __name__ == "__main__":
    main()
