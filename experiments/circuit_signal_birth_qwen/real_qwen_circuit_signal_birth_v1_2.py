#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_qwen_circuit_signal_birth_v1_2.py
VERSION = v1.2-real-qwen-circuit-signal-birth-vo-per-head-prompts

REAL TEST: no planted hidden operator.

Question tested
---------------
Can signal_birth use REAL transformer circuit targets as input, instead of
activation/blame toy features?

Pipeline:
  1. Load real Qwen.
  2. Extract exact all-head circuit targets for a layer:
       M_qk_aug[h,d] = Wq_aug[h].T @ R_delta @ Wk_aug[kv] / sqrt(head_dim)
       C_vo_aug[h]   = Wo[h] @ Wv_aug[kv]
  3. Decode these targets with a fixed readable operator dictionary.
  4. Compute residuals.
  5. Mine new BirthOp candidates from residuals by PCA/SVD.
  6. Accept BirthOps only if train-target and heldout-target projection improve.
  7. Evaluate functional prompt behavior:
       score_rel / A_rel / KL / top1 / Y_all_rel / H_after_rel

This is NOT raw-vs-circuit SVD compression.
This is base operator dictionary vs base dictionary + mined residual BirthOps.

Example quick:
  python real_qwen_circuit_signal_birth_v1_2.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --device cuda --eval-device cuda \
    --dtype fp16 --attn-implementation eager --layers 2 \
    --max-length 48 --prompts-per-suite 1 \
    --qk-births 2 --vo-births 2
"""
from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

VERSION = "v1.2-real-qwen-circuit-signal-birth-vo-per-head-prompts"


# ---------------- utils ----------------

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


def rel_num_den(pred: torch.Tensor, true: torch.Tensor) -> Tuple[float, float]:
    pred = pred.detach().float()
    true = true.detach().float()
    d = pred - true
    return float((d * d).sum().item()), float((true * true).sum().item())


def rel_from(num: float, den: float, eps: float = 1e-12) -> float:
    return math.sqrt(num / max(eps, den))


def rel_err(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-12) -> float:
    n, d = rel_num_den(pred, true)
    return rel_from(n, d, eps=eps)


def cosine_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    av = a.detach().float().reshape(-1)
    bv = b.detach().float().reshape(-1)
    return float(torch.dot(av, bv) / ((torch.linalg.norm(av) * torch.linalg.norm(bv)).clamp_min(eps)))


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


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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


def apply_rope_rows(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return q * cos + rotate_half(q) * sin


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
    usable = min(rows, cols - 1) if cols > rows else min(rows, cols)
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
    usable = min(rows, cols - 1) if cols > rows else min(rows, cols)
    for s in range(0, usable, block):
        e = min(usable, s + block)
        M[s:e, s:e] = 1.0 / max(1, e - s)
    return M


def build_prompts(prompts_per_suite: int) -> List[Dict[str, str]]:
    base = [
        ("plain", "Explain why rivers are important for cities."),
        ("plain", "The quick brown fox jumps over the lazy dog."),
        ("code", "Write a Python function that computes factorial iteratively."),
        ("code", "Given a list of numbers, return the indices of the two largest values."),
        ("math", "Solve the equation 3x + 5 = 20 and explain each step."),
        ("math", "If a triangle has sides 3, 4, 5, what is its area?"),
        ("reason", "Alice has more apples than Bob. Bob has more apples than Carol. Who has the fewest apples?"),
        ("json", "JSON: {\"user\":\"Alice\", \"score\":42, \"active\":true}"),
        ("mixed", "Translate to French: the weather is cold today."),
        ("symbols", "Sequence: A -> B -> C, then C -> D. What comes after B?"),
        ("long", "In a small village near the mountains, people used a river to power mills, irrigate fields, and move goods."),
        ("qa", "Question: What is photosynthesis? Answer in one short paragraph."),
    ]
    rows = []
    for rep in range(max(1, prompts_per_suite)):
        for suite, text in base:
            rows.append({"suite": suite, "text": text + ("" if rep == 0 else f" Repeat {rep}.")})
    return rows


# ---------------- operator dictionaries ----------------

class MatrixDict:
    def __init__(self, ops: List[Tuple[str, torch.Tensor]], ridge: float = 1e-4):
        self.names = [n for n, _ in ops]
        mats = [m.detach().float() for _, m in ops]
        self.ops = torch.stack(mats, dim=0).contiguous()
        self.P = int(self.ops.shape[0])
        self.shape = tuple(self.ops.shape[1:])
        A_raw = self.ops.reshape(self.P, -1).T.contiguous()
        norms = torch.linalg.norm(A_raw, dim=0).clamp_min(1e-12)
        self.A = A_raw / norms.view(1, -1)
        self.norms = norms
        eye = torch.eye(self.P, device=self.A.device, dtype=self.A.dtype)
        self.pinv = torch.linalg.solve(self.A.T @ self.A + ridge * eye, self.A.T)  # [P,N]

    def to(self, device: str) -> "MatrixDict":
        self.ops = self.ops.to(device)
        self.A = self.A.to(device)
        self.norms = self.norms.to(device)
        self.pinv = self.pinv.to(device)
        return self

    def append(self, name: str, mat: torch.Tensor, ridge: float = 1e-4) -> "MatrixDict":
        ops = [(n, self.ops[i].detach().cpu()) for i, n in enumerate(self.names)]
        ops.append((name, mat.detach().float().cpu()))
        return MatrixDict(ops, ridge=ridge).to(str(self.ops.device))

    @torch.no_grad()
    def decode(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # targets [B,*shape]
        B = int(targets.shape[0])
        flat = targets.reshape(B, -1).float()
        c_norm = flat @ self.pinv.T
        c_raw = c_norm / self.norms.view(1, -1)
        recon = torch.einsum("bp,pij->bij", c_raw, self.ops)
        return c_raw, recon

    @torch.no_grad()
    def rel(self, targets: torch.Tensor) -> float:
        _, recon = self.decode(targets)
        return rel_err(recon, targets)


def build_square_dict(n: int, device: str, dtype=torch.float32) -> MatrixDict:
    dev = torch.device(device)
    ops: List[Tuple[str, torch.Tensor]] = []
    I = torch.eye(n, device=dev, dtype=dtype)
    ops.append(("Identity", I))
    ramp = torch.linspace(-1, 1, n, device=dev, dtype=dtype)
    ops.append(("RampDiag", torch.diag(ramp)))
    ops.append(("MeanProject", torch.ones(n, n, device=dev, dtype=dtype) / n))
    for sh in [1, 2, 4, 8, 16, 32]:
        if sh < n:
            ops.append((f"ShiftR{sh}", shift_square(n, sh, dev, dtype)))
            ops.append((f"ShiftL{sh}", shift_square(n, -sh, dev, dtype)))
    for b in [2, 4, 8, 16, 32, 64, 128]:
        if b < n:
            ops.append((f"BlockAvg{b}", block_avg_square(n, b, dev, dtype)))
    # spectral projections, cheap enough at n~897
    try:
        C = dct_matrix(n, dev, dtype)
        k = max(1, n // 8)
        ops.append(("DCTLow", C[:k].T @ C[:k]))
        ops.append(("DCTHigh", C[-k:].T @ C[-k:]))
    except Exception:
        pass
    return MatrixDict(ops).to(device)


def build_rect_dict(rows: int, cols: int, device: str, dtype=torch.float32) -> MatrixDict:
    dev = torch.device(device)
    ops: List[Tuple[str, torch.Tensor]] = []
    M = torch.zeros(rows, cols, device=dev, dtype=dtype)
    diag = min(rows, cols)
    M[torch.arange(diag), torch.arange(diag)] = 1.0
    ops.append(("IdentityRect", M))
    # bias-column operator if homogeneous column exists
    if cols == rows + 1:
        B = torch.zeros(rows, cols, device=dev, dtype=dtype)
        B[:, -1] = 1.0 / math.sqrt(rows)
        ops.append(("BiasColumn", B))
    ramp = torch.linspace(-1, 1, rows, device=dev, dtype=dtype)
    R = torch.zeros(rows, cols, device=dev, dtype=dtype)
    d = min(rows, cols)
    R[torch.arange(d), torch.arange(d)] = ramp[:d]
    ops.append(("RampDiagRect", R))
    ops.append(("MeanRect", torch.ones(rows, cols, device=dev, dtype=dtype) / math.sqrt(rows * cols)))
    for sh in [1, 2, 4, 8, 16, 32]:
        if sh < rows:
            ops.append((f"ShiftR{sh}Rect", shift_rect(rows, cols, sh, dev, dtype)))
            ops.append((f"ShiftL{sh}Rect", shift_rect(rows, cols, -sh, dev, dtype)))
    for b in [2, 4, 8, 16, 32, 64, 128]:
        if b < rows:
            ops.append((f"BlockAvg{b}Rect", block_avg_rect(rows, cols, b, dev, dtype)))
    return MatrixDict(ops).to(device)


@torch.no_grad()
def mine_birth_candidates(residuals: torch.Tensor, k: int, prefix: str, device: str) -> List[Tuple[str, torch.Tensor, float]]:
    # residuals [B,m,n]; PCA over repeated residuals.
    B = int(residuals.shape[0])
    flat = residuals.reshape(B, -1).float().to(device)
    # remove target-wise mean scale but do not center across features: residual direction matters.
    norms = torch.linalg.norm(flat, dim=1, keepdim=True).clamp_min(1e-12)
    flatn = flat / norms
    q = min(max(1, int(k)), min(flatn.shape))
    try:
        # V: [N,q]
        _, S, V = torch.pca_lowrank(flatn, q=q, center=False, niter=2)
        comps = V.T.contiguous()
        vals = S[:q]
    except Exception:
        U, S, Vh = torch.linalg.svd(flatn, full_matrices=False)
        comps = Vh[:q].contiguous()
        vals = S[:q]
    out = []
    m, n = residuals.shape[1], residuals.shape[2]
    for i in range(q):
        mat = comps[i].reshape(m, n)
        mat = mat / torch.linalg.norm(mat).clamp_min(1e-12)
        energy = float((vals[i] ** 2 / (S[:q].pow(2).sum().clamp_min(1e-12))).detach().cpu()) if q > 0 else 0.0
        out.append((f"{prefix}_BirthPCA{i}", mat.detach().cpu(), energy))
    return out


# ---------------- model extraction ----------------

@dataclass
class LayerSeq:
    prompt_id: int
    suite: str
    text: str
    Xraw: torch.Tensor
    Xn: torch.Tensor
    Xaug: torch.Tensor
    Q: torch.Tensor          # [heads,T,D]
    K_kv: torch.Tensor       # [kv,T,D]
    A: torch.Tensor          # [heads,T,T]
    Y_heads: torch.Tensor    # [heads,T,H]
    Y_all: torch.Tensor
    H_after: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


@torch.no_grad()
def collect_layer_data(model, tokenizer, prompts: List[Dict[str, str]], layer_idx: int, max_length: int, device: str) -> Tuple[List[LayerSeq], Dict[str, int]]:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    D = int(getattr(cfg, "head_dim", H // n_heads))
    kv_groups = n_heads // n_kv
    Wo_all = attn.o_proj.weight.detach().float().to(device)
    bo = getattr(attn.o_proj, "bias", None)
    bo = torch.zeros(H, device=device) if bo is None else bo.detach().float().to(device)

    rows: List[LayerSeq] = []
    for pi, pr in enumerate(prompts):
        enc = tokenizer(pr["text"], return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        T = int(input_ids.shape[1])
        if T < 2:
            continue
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        Xraw = outputs.hidden_states[layer_idx].detach()
        Xn = layer.input_layernorm(Xraw).detach()

        q = attn.q_proj(Xn).view(1, T, n_heads, D).transpose(1, 2).contiguous()
        k = attn.k_proj(Xn).view(1, T, n_kv, D).transpose(1, 2).contiguous()
        v = attn.v_proj(Xn).view(1, T, n_kv, D).transpose(1, 2).contiguous()
        pos = torch.arange(T, device=device).unsqueeze(0)
        cos, sin = compute_position_embeddings(model, Xn, pos)
        cos2 = cos[0] if cos.dim() == 3 else cos
        sin2 = sin[0] if sin.dim() == 3 else sin
        q_rot = apply_rope_rows(q, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))[0].float()
        k_rot = apply_rope_rows(k, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))[0].float()
        V_kv = v[0].float()

        A_heads = []
        Y_heads = []
        for h in range(n_heads):
            kv = h // kv_groups
            scores = (q_rot[h] @ k_rot[kv].T) / math.sqrt(D)
            A = causal_softmax(scores.float())
            Z = A @ V_kv[kv]
            Wo_h = Wo_all[:, h * D:(h + 1) * D]
            Yh = Z @ Wo_h.T
            A_heads.append(A.cpu())
            Y_heads.append(Yh.cpu())
        A_heads_t = torch.stack(A_heads, dim=0)
        Y_heads_t = torch.stack(Y_heads, dim=0)
        Y_all = Y_heads_t.sum(dim=0) + bo.detach().cpu().view(1, -1)
        Xraw0 = Xraw[0].float().cpu()
        H_after = Xraw0 + Y_all
        Xn0 = Xn[0].float().cpu()
        Xaug = torch.cat([Xn0, torch.ones(T, 1)], dim=1)
        rows.append(LayerSeq(
            prompt_id=pi, suite=str(pr.get("suite", "?")), text=pr["text"],
            Xraw=Xraw0, Xn=Xn0, Xaug=Xaug, Q=q_rot.cpu(), K_kv=k_rot.cpu(),
            A=A_heads_t, Y_heads=Y_heads_t, Y_all=Y_all, H_after=H_after,
            cos=cos2.float().cpu(), sin=sin2.float().cpu()
        ))
        del outputs
    meta = {"H": H, "D": D, "n_heads": n_heads, "n_kv": n_kv, "kv_groups": kv_groups}
    return rows, meta


def slice_bias(module: Any, start: int, end: int) -> torch.Tensor:
    b = getattr(module, "bias", None)
    if b is None:
        return torch.zeros(end - start, dtype=torch.float32)
    return b.detach().float()[start:end].cpu()


def build_all_weights(model, layer_idx: int, meta: Dict[str, int]) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    H = int(meta["H"]); D = int(meta["D"]); n_heads = int(meta["n_heads"]); n_kv = int(meta["n_kv"])
    Wq = attn.q_proj.weight.detach().float().cpu().view(n_heads, D, H)
    Wk = attn.k_proj.weight.detach().float().cpu().view(n_kv, D, H)
    Wv = attn.v_proj.weight.detach().float().cpu().view(n_kv, D, H)
    Wo_flat = attn.o_proj.weight.detach().float().cpu()
    Wo = torch.stack([Wo_flat[:, h * D:(h + 1) * D] for h in range(n_heads)], dim=0)
    bq = torch.stack([slice_bias(attn.q_proj, h * D, (h + 1) * D) for h in range(n_heads)], dim=0)
    bk = torch.stack([slice_bias(attn.k_proj, kv * D, (kv + 1) * D) for kv in range(n_kv)], dim=0)
    bv = torch.stack([slice_bias(attn.v_proj, kv * D, (kv + 1) * D) for kv in range(n_kv)], dim=0)
    Wq_aug = torch.cat([Wq, bq[:, :, None]], dim=2)
    Wk_aug = torch.cat([Wk, bk[:, :, None]], dim=2)
    Wv_aug = torch.cat([Wv, bv[:, :, None]], dim=2)
    bo = getattr(attn.o_proj, "bias", None)
    bo = torch.zeros(H) if bo is None else bo.detach().float().cpu()
    return {"Wq_aug": Wq_aug, "Wk_aug": Wk_aug, "Wv_aug": Wv_aug, "Wo": Wo, "bo": bo}


def build_rope_pos(seqs: List[LayerSeq], max_pos: int) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for s in seqs:
        T = int(s.cos.shape[0])
        for p in range(min(T, max_pos + 1)):
            if p not in out:
                out[p] = rope_col_matrix(s.cos[p], s.sin[p]).cpu()
    return out


@torch.no_grad()
def build_circuit_targets(weights: Dict[str, torch.Tensor], Rpos: Dict[int, torch.Tensor], meta: Dict[str, int], max_delta: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
    """
    Exact affine circuit targets, matching qwen_circuit_matrix_targets_v2_affine_basis.py:

      score_ij = x_aug_i^T M_qk_aug[d] x_aug_j
      M_qk_aug[d] = Wq_aug.T @ (R_i.T @ R_j) @ Wk_aug / sqrt(D), d=i-j

      payload_j = x_aug_j @ C_vo_aug.T
      C_vo_aug = Wo_head @ Wv_aug

    For RoPE relative delta we use representative pair i=d, j=0:
      Rrel[d] = Rpos[d].T @ Rpos[0]
    """
    n_heads = int(meta["n_heads"])
    kv_groups = int(meta["kv_groups"])
    H = int(meta["H"])
    D = int(meta["D"])

    Wq_aug = weights["Wq_aug"].to(device).float()  # [heads,D,H+1]
    Wk_aug = weights["Wk_aug"].to(device).float()  # [kv,D,H+1]
    Wv_aug = weights["Wv_aug"].to(device).float()  # [kv,D,H+1]
    Wo = weights["Wo"].to(device).float()          # [heads,H,D]

    assert Wq_aug.shape[-1] == H + 1, f"Wq_aug must be [heads,D,H+1], got {tuple(Wq_aug.shape)}"
    assert Wk_aug.shape[-1] == H + 1, f"Wk_aug must be [kv,D,H+1], got {tuple(Wk_aug.shape)}"
    assert Wv_aug.shape[-1] == H + 1, f"Wv_aug must be [kv,D,H+1], got {tuple(Wv_aug.shape)}"
    assert Wo.shape[1:] == (H, D), f"Wo must be [heads,H,D], got {tuple(Wo.shape)}"

    R0 = Rpos[0].to(device).float()
    M_list = []
    M_index = []
    for h in range(n_heads):
        kv = h // kv_groups
        for d in range(max_delta + 1):
            if d not in Rpos:
                continue
            Ri = Rpos[d].to(device).float()
            Rj = R0
            Rrel = Ri.T @ Rj
            M = (Wq_aug[h].T @ Rrel @ Wk_aug[kv]) / math.sqrt(D)
            assert M.shape == (H + 1, H + 1), f"M_qk_aug bad shape {tuple(M.shape)}"
            M_list.append(M)
            M_index.append((h, d))

    C_list = []
    for h in range(n_heads):
        kv = h // kv_groups
        C_vo_aug = Wo[h] @ Wv_aug[kv]  # [H,H+1], includes bv in last column
        assert C_vo_aug.shape == (H, H + 1), f"C_vo_aug must be [H,H+1], got {tuple(C_vo_aug.shape)}"
        C_list.append(C_vo_aug)

    M_stack = torch.stack(M_list, dim=0)
    C_stack = torch.stack(C_list, dim=0)
    assert M_stack.shape[-2:] == (H + 1, H + 1), f"M_stack bad shape {tuple(M_stack.shape)}"
    assert C_stack.shape[-2:] == (H, H + 1), f"C_stack bad shape {tuple(C_stack.shape)}"
    return M_stack, C_stack, M_index


# ---------------- target split and eval ----------------

def split_target_indices(M_index: List[Tuple[int, int]], n_heads: int, mode: str = "heads") -> Tuple[torch.Tensor, torch.Tensor]:
    train = []
    held = []
    if mode == "deltas":
        for i, (_, d) in enumerate(M_index):
            (train if d % 2 == 0 else held).append(i)
    else:
        for i, (h, _) in enumerate(M_index):
            (train if h < n_heads // 2 else held).append(i)
    if not held:
        held = train[::2]
        train = train[1::2]
    return torch.tensor(train, dtype=torch.long), torch.tensor(held, dtype=torch.long)


def split_head_indices(n_heads: int) -> Tuple[torch.Tensor, torch.Tensor]:
    tr = list(range(0, n_heads // 2))
    he = list(range(n_heads // 2, n_heads))
    return torch.tensor(tr, dtype=torch.long), torch.tensor(he, dtype=torch.long)


@torch.no_grad()
def target_metrics(name: str, qk_dict: MatrixDict, vo_dict: MatrixDict, M: torch.Tensor, C: torch.Tensor,
                   M_train_idx: torch.Tensor, M_held_idx: torch.Tensor, C_train_idx: torch.Tensor, C_held_idx: torch.Tensor) -> Dict[str, float]:
    out = {"name": name}
    for split_name, mi, ci in [("train", M_train_idx, C_train_idx), ("held", M_held_idx, C_held_idx)]:
        out[f"{split_name}_M_rel"] = qk_dict.rel(M[mi.to(M.device)])
        out[f"{split_name}_C_rel"] = vo_dict.rel(C[ci.to(C.device)])
        out[f"{split_name}_both_rel"] = 0.5 * (out[f"{split_name}_M_rel"] + out[f"{split_name}_C_rel"])
    return out


@torch.no_grad()
def reconstruct_all_targets(qk_dict: MatrixDict, vo_dict: MatrixDict, M: torch.Tensor, C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    _, Mh = qk_dict.decode(M)
    _, Ch = vo_dict.decode(C)
    return Mh.detach(), Ch.detach()


@torch.no_grad()
def target_metrics_from_hat(name: str, M_hat: torch.Tensor, C_hat: torch.Tensor, M: torch.Tensor, C: torch.Tensor,
                            M_train_idx: torch.Tensor, M_held_idx: torch.Tensor,
                            C_train_idx: torch.Tensor, C_held_idx: torch.Tensor) -> Dict[str, float]:
    out = {"name": name}
    for split_name, mi, ci in [("train", M_train_idx, C_train_idx), ("held", M_held_idx, C_held_idx)]:
        mi = mi.to(M.device)
        ci = ci.to(C.device)
        out[f"{split_name}_M_rel"] = rel_err(M_hat[mi], M[mi])
        out[f"{split_name}_C_rel"] = rel_err(C_hat[ci], C[ci])
        out[f"{split_name}_both_rel"] = 0.5 * (out[f"{split_name}_M_rel"] + out[f"{split_name}_C_rel"])
    return out


@torch.no_grad()
def fit_atom_update(current: torch.Tensor, target: torch.Tensor, atom: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Best scalar update current + alpha*atom for one target matrix."""
    atom = atom.to(target.device).float()
    atom = atom / torch.linalg.norm(atom).clamp_min(1e-12)
    resid = (target - current).float()
    alpha = torch.sum(resid * atom) / torch.sum(atom * atom).clamp_min(1e-12)
    return current + alpha * atom, float(alpha.detach().cpu())


@torch.no_grad()
def mine_vo_head_atoms(residual: torch.Tensor, max_candidates: int, include_full_residual: bool = False) -> List[Tuple[str, torch.Tensor, float]]:
    """
    Per-head VO candidates from one head's C_vo residual.

    Unlike shared VO mining, this does not require residual to transfer across heads.
    Acceptance is functional: train prompts vs heldout prompts.
    """
    residual = residual.detach().float()
    out: List[Tuple[str, torch.Tensor, float]] = []
    total = float((residual * residual).sum().clamp_min(1e-12).item())
    if include_full_residual:
        mat = residual / torch.linalg.norm(residual).clamp_min(1e-12)
        out.append(("VOHeadFullResidual", mat.cpu(), 1.0))

    try:
        U, S, Vh = torch.linalg.svd(residual, full_matrices=False)
        k = min(max_candidates, int(S.numel()))
        for i in range(k):
            # rank-1 direction. Scaling fitted later by fit_atom_update.
            mat = torch.outer(U[:, i], Vh[i])
            mat = mat / torch.linalg.norm(mat).clamp_min(1e-12)
            e = float(((S[i] * S[i]) / max(total, 1e-12)).detach().cpu())
            out.append((f"VOHeadSVD{i}", mat.cpu(), e))
    except Exception:
        mat = residual / torch.linalg.norm(residual).clamp_min(1e-12)
        out.append(("VOHeadFallbackResidual", mat.cpu(), 1.0))
    return out[:max_candidates + (1 if include_full_residual else 0)]


@torch.no_grad()
def accept_vo_per_head_prompts(
    C: torch.Tensor,
    C_start_hat: torch.Tensor,
    M_hat: torch.Tensor,
    M_index: List[Tuple[int, int]],
    weights: Dict[str, torch.Tensor],
    meta: Dict[str, int],
    max_delta: int,
    train_seqs: List[LayerSeq],
    held_seqs: List[LayerSeq],
    max_candidates: int,
    births_per_head: int,
    min_held_y_gain: float,
    min_held_h_gain: float,
    include_full_residual: bool,
    device: str,
) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
    """
    VO birth with prompt split, not head split.

    For each head:
      current C_hat[h] -> mine residual atoms from C[h]-C_hat[h]
      accept candidate only if full-layer functional Y_all/H_after improves on heldout prompts.
    """
    C_hat = C_start_hat.detach().clone().to(device).float()
    C_true = C.to(device).float()
    accepted: List[Dict[str, float]] = []
    n_heads = int(meta["n_heads"])

    cur_train = eval_functional(train_seqs, M_hat, C_hat, M_index, weights, meta, max_delta, device)
    cur_held = eval_functional(held_seqs, M_hat, C_hat, M_index, weights, meta, max_delta, device)

    for h in range(n_heads):
        accepted_for_head = 0
        for cand_name, atom, energy in mine_vo_head_atoms(C_true[h] - C_hat[h], max_candidates, include_full_residual=include_full_residual):
            if accepted_for_head >= births_per_head:
                break
            new_head, alpha = fit_atom_update(C_hat[h], C_true[h], atom.to(device))
            C_try = C_hat.clone()
            C_try[h] = new_head

            tr = eval_functional(train_seqs, M_hat, C_try, M_index, weights, meta, max_delta, device)
            he = eval_functional(held_seqs, M_hat, C_try, M_index, weights, meta, max_delta, device)

            train_y_gain = cur_train["Y_all_rel"] - tr["Y_all_rel"]
            held_y_gain = cur_held["Y_all_rel"] - he["Y_all_rel"]
            train_h_gain = cur_train["H_after_rel"] - tr["H_after_rel"]
            held_h_gain = cur_held["H_after_rel"] - he["H_after_rel"]
            ok = (
                train_y_gain >= -1e-6
                and held_y_gain >= min_held_y_gain
                and held_h_gain >= min_held_h_gain
            )
            print(
                f"  VO head H{h:02d} cand {cand_name}: energy={energy:.3f} alpha={alpha:+.4g} "
                f"heldY_gain={held_y_gain:+.5f} heldH_gain={held_h_gain:+.5f} ok={ok}"
            )
            if ok:
                C_hat = C_try
                cur_train = tr
                cur_held = he
                accepted_for_head += 1
                accepted.append({
                    "head": h,
                    "name": cand_name,
                    "energy": energy,
                    "alpha": alpha,
                    "held_y_gain": held_y_gain,
                    "held_h_gain": held_h_gain,
                    "train_y_gain": train_y_gain,
                    "train_h_gain": train_h_gain,
                })
    return C_hat.detach(), accepted


@torch.no_grad()
def eval_functional(seqs: List[LayerSeq], Mhat: torch.Tensor, Chat: torch.Tensor, M_index: List[Tuple[int, int]],
                    weights: Dict[str, torch.Tensor], meta: Dict[str, int], max_delta: int, device: str) -> Dict[str, float]:
    n_heads = int(meta["n_heads"]); kv_groups = int(meta["kv_groups"]); D = int(meta["D"])
    bo = weights["bo"].to(device).float()
    # Map [head,delta] -> matrix index
    mdict = {(h, d): i for i, (h, d) in enumerate(M_index)}
    Mdev = Mhat.to(device).float()
    Cdev = Chat.to(device).float()
    acc = {"score_n":0.0, "score_d":0.0, "A_n":0.0, "A_d":0.0, "Y_n":0.0, "Y_d":0.0, "H_n":0.0, "H_d":0.0,
           "KL":0.0, "top":0, "rows":0, "covered":0, "allrows":0}
    for s in seqs:
        X = s.Xaug.to(device).float()
        Xraw = s.Xraw.to(device).float()
        T = int(X.shape[0])
        A_true_all = s.A.to(device).float()
        Y_true_all = s.Y_all.to(device).float()
        H_true = s.H_after.to(device).float()
        Q_true = s.Q.to(device).float()
        K_true_kv = s.K_kv.to(device).float()
        Y_pred_all = torch.zeros_like(Y_true_all)

        for h in range(n_heads):
            kv = h // kv_groups
            S = torch.full((T, T), torch.finfo(torch.float32).min, device=device)
            S_true = (Q_true[h] @ K_true_kv[kv].T) / math.sqrt(D)
            full_rows = torch.ones(T, dtype=torch.bool, device=device)
            for d in range(min(max_delta, T - 1) + 1):
                key = (h, d)
                if key not in mdict:
                    if d <= T - 1:
                        full_rows[d:T] = False
                    continue
                M = Mdev[mdict[key]]
                Xi = X[d:T]
                Xj = X[:T-d]
                vals = ((Xi @ M) * Xj).sum(dim=-1)
                rows = torch.arange(d, T, device=device)
                cols = torch.arange(0, T-d, device=device)
                S[rows, cols] = vals
            if max_delta < T - 1:
                full_rows[max_delta+1:] = False
            Ahat = torch.softmax(S, dim=-1)
            acc["allrows"] += T
            acc["covered"] += int(full_rows.sum().item())
            if int(full_rows.sum()) > 0:
                pair_mask = torch.zeros(T, T, dtype=torch.bool, device=device)
                for i in range(T):
                    if full_rows[i]:
                        pair_mask[i, :i+1] = True
                sn, sd = rel_num_den(S[pair_mask], S_true[pair_mask])
                an, ad = rel_num_den(Ahat[full_rows], A_true_all[h][full_rows])
                acc["score_n"] += sn; acc["score_d"] += sd
                acc["A_n"] += an; acc["A_d"] += ad
                acc["KL"] += float(F.kl_div((Ahat[full_rows] + 1e-12).log(), A_true_all[h][full_rows], reduction="sum").item())
                acc["top"] += int((Ahat[full_rows].argmax(dim=-1) == A_true_all[h][full_rows].argmax(dim=-1)).sum().item())
                acc["rows"] += int(full_rows.sum().item())
            assert Cdev[h].shape[-1] == X.shape[-1], f"C_vo_aug/Xaug mismatch: C={tuple(Cdev[h].shape)} X={tuple(X.shape)}"
            payload = X @ Cdev[h].T
            Y_pred_all += Ahat @ payload

        Y_pred_all = Y_pred_all + bo.view(1, -1)
        H_pred = Xraw + Y_pred_all
        yn, yd = rel_num_den(Y_pred_all, Y_true_all)
        hn, hd = rel_num_den(H_pred, H_true)
        acc["Y_n"] += yn; acc["Y_d"] += yd
        acc["H_n"] += hn; acc["H_d"] += hd

    rows = max(1, acc["rows"])
    return {
        "score_rel": rel_from(acc["score_n"], acc["score_d"]),
        "A_rel": rel_from(acc["A_n"], acc["A_d"]),
        "KL": acc["KL"] / rows,
        "top1": acc["top"] / rows,
        "Y_all_rel": rel_from(acc["Y_n"], acc["Y_d"]),
        "H_after_rel": rel_from(acc["H_n"], acc["H_d"]),
        "coverage": acc["covered"] / max(1, acc["allrows"]),
    }


def fmt_target(m: Dict[str, float]) -> str:
    return (f"{m['name']:<10} trainM={m['train_M_rel']:.4f} trainC={m['train_C_rel']:.4f} "
            f"heldM={m['held_M_rel']:.4f} heldC={m['held_C_rel']:.4f} heldBoth={m['held_both_rel']:.4f}")


def fmt_func(name: str, m: Dict[str, float]) -> str:
    return (f"{name:<10} score={m['score_rel']:.4f} A={m['A_rel']:.4f} KL={m['KL']:.4f} "
            f"top1={m['top1']:.3f} Y_all={m['Y_all_rel']:.4f} H_after={m['H_after_rel']:.4f} cov={m['coverage']:.2f}")


def try_accept_births(base_dict: MatrixDict, targets: torch.Tensor, train_idx: torch.Tensor, held_idx: torch.Tensor,
                      births: List[Tuple[str, torch.Tensor, float]], max_accept: int, min_held_gain: float,
                      label: str, device: str) -> Tuple[MatrixDict, List[Dict[str, float]]]:
    cur = base_dict
    accepted: List[Dict[str, float]] = []
    train_idx = train_idx.to(targets.device)
    held_idx = held_idx.to(targets.device)
    for name, mat, energy in births:
        if len(accepted) >= max_accept:
            break
        before_train = cur.rel(targets[train_idx])
        before_held = cur.rel(targets[held_idx])
        cand = cur.append(name, mat)
        after_train = cand.rel(targets[train_idx])
        after_held = cand.rel(targets[held_idx])
        gain_train = before_train - after_train
        gain_held = before_held - after_held
        ok = gain_train > 0 and gain_held >= min_held_gain
        print(f"  {label} candidate {name}: energy={energy:.3f} train_gain={gain_train:+.5f} held_gain={gain_held:+.5f} ok={ok}")
        if ok:
            cur = cand
            accepted.append({"name": name, "energy": energy, "train_gain": gain_train, "held_gain": gain_held})
    return cur, accepted


# ---------------- main ----------------

def run(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    dtype = get_dtype(args.dtype)
    print(f"=== Real Qwen Circuit Signal Birth {VERSION} ===")
    print(f"model={args.model} device={args.device} eval_device={args.eval_device} dtype={args.dtype}")
    print("NO planted hidden. BirthOps mined from real residuals of M_qk_aug/C_vo_aug.")
    print("")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    model.eval()

    prompts = build_prompts(args.prompts_per_suite)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    split = max(2, int(len(prompts) * args.train_frac))
    train_prompts = prompts[:split]
    held_prompts = prompts[split:]
    if len(held_prompts) < 2:
        held_prompts = prompts[::2]
        train_prompts = prompts[1::2]

    for layer_idx in parse_int_list(args.layers):
        t0 = time.perf_counter()
        print(f"\n--- layer L{layer_idx} ---")
        train_seqs, meta = collect_layer_data(model, tok, train_prompts, layer_idx, args.max_length, args.device)
        held_seqs, _ = collect_layer_data(model, tok, held_prompts, layer_idx, args.max_length, args.device)
        allseqs = train_seqs + held_seqs
        max_observed = max(int(s.Xaug.shape[0]) for s in allseqs)
        max_delta = min(int(args.max_delta), max_observed - 1)
        Rpos = build_rope_pos(allseqs, max_delta)
        weights = build_all_weights(model, layer_idx, meta)

        print(f"seqs train={len(train_seqs)} held={len(held_seqs)} T_max={max_observed} max_delta={max_delta}")
        print(f"model heads={meta['n_heads']} kv={meta['n_kv']} H={meta['H']} D={meta['D']}")

        M, C, M_index = build_circuit_targets(weights, Rpos, meta, max_delta, args.eval_device)
        n_heads = int(meta["n_heads"])
        M_train_idx, M_held_idx = split_target_indices(M_index, n_heads, mode=args.target_split)
        C_train_idx, C_held_idx = split_head_indices(n_heads)

        print(f"targets: M_qk={tuple(M.shape)} C_vo={tuple(C.shape)} split={args.target_split} M_train={len(M_train_idx)} M_held={len(M_held_idx)} C_train={len(C_train_idx)} C_held={len(C_held_idx)}")
        assert M.shape[-1] == int(meta["H"]) + 1 and M.shape[-2] == int(meta["H"]) + 1, "M_qk_aug must be [*,H+1,H+1]"
        assert C.shape[-1] == int(meta["H"]) + 1 and C.shape[-2] == int(meta["H"]), "C_vo_aug must be [*,H,H+1]"

        # Exact affine target sanity check. If VO affine/bias is wrong, this will not be near zero.
        exact_train_f = eval_functional(train_seqs, M, C, M_index, weights, meta, max_delta, args.eval_device)
        exact_held_f = eval_functional(held_seqs, M, C, M_index, weights, meta, max_delta, args.eval_device)
        print(fmt_func("exact_tr", exact_train_f))
        print(fmt_func("exact_he", exact_held_f))

        qk_base = build_square_dict(M.shape[-1], args.eval_device)
        vo_base = build_rect_dict(C.shape[-2], C.shape[-1], args.eval_device)
        print(f"base dict: qk_ops={qk_base.P} vo_ops={vo_base.P}")

        base_target = target_metrics("base", qk_base, vo_base, M, C, M_train_idx, M_held_idx, C_train_idx, C_held_idx)
        print(fmt_target(base_target))

        # Mine residuals.
        _, M_base_recon = qk_base.decode(M)
        _, C_base_recon = vo_base.decode(C)
        M_res = M[M_train_idx.to(M.device)] - M_base_recon[M_train_idx.to(M.device)]
        C_res = C[C_train_idx.to(C.device)] - C_base_recon[C_train_idx.to(C.device)]

        print("mining qk residual births...")
        qk_candidates = mine_birth_candidates(M_res, args.qk_candidates, "QK", args.eval_device)
        qk_birth, qk_acc = try_accept_births(
            qk_base, M, M_train_idx, M_held_idx, qk_candidates,
            max_accept=args.qk_births, min_held_gain=args.min_held_gain_qk,
            label="QK", device=args.eval_device
        )

        print("functional eval on prompts...")
        M_base_hat, C_base_hat = reconstruct_all_targets(qk_base, vo_base, M, C)
        M_qk_birth_hat, C_shared_base_hat = reconstruct_all_targets(qk_birth, vo_base, M, C)

        vo_acc = []
        if args.vo_mode == "shared-heads":
            print("mining vo residual births... mode=shared-heads")
            vo_candidates = mine_birth_candidates(C_res, args.vo_candidates, "VO", args.eval_device)
            vo_birth, vo_acc = try_accept_births(
                vo_base, C, C_train_idx, C_held_idx, vo_candidates,
                max_accept=args.vo_births, min_held_gain=args.min_held_gain_vo,
                label="VO", device=args.eval_device
            )
            M_birth_hat, C_birth_hat = reconstruct_all_targets(qk_birth, vo_birth, M, C)
            birth_target = target_metrics("birth", qk_birth, vo_birth, M, C, M_train_idx, M_held_idx, C_train_idx, C_held_idx)
        else:
            print("mining vo residual births... mode=per-head-prompts")
            # Start from base VO reconstruction, then accept private per-head updates by prompt-heldout functional gain.
            C_birth_hat, vo_acc = accept_vo_per_head_prompts(
                C=C,
                C_start_hat=C_shared_base_hat,
                M_hat=M_qk_birth_hat,
                M_index=M_index,
                weights=weights,
                meta=meta,
                max_delta=max_delta,
                train_seqs=train_seqs,
                held_seqs=held_seqs,
                max_candidates=args.vo_per_head_candidates,
                births_per_head=args.vo_per_head_births,
                min_held_y_gain=args.vo_min_held_y_gain,
                min_held_h_gain=args.vo_min_held_h_gain,
                include_full_residual=args.vo_include_full_residual,
                device=args.eval_device,
            )
            M_birth_hat = M_qk_birth_hat
            birth_target = target_metrics_from_hat("birth", M_birth_hat, C_birth_hat, M, C, M_train_idx, M_held_idx, C_train_idx, C_held_idx)

        print(fmt_target(birth_target))

        base_train_f = eval_functional(train_seqs, M_base_hat, C_base_hat, M_index, weights, meta, max_delta, args.eval_device)
        base_held_f = eval_functional(held_seqs, M_base_hat, C_base_hat, M_index, weights, meta, max_delta, args.eval_device)
        qkonly_train_f = eval_functional(train_seqs, M_qk_birth_hat, C_shared_base_hat, M_index, weights, meta, max_delta, args.eval_device)
        qkonly_held_f = eval_functional(held_seqs, M_qk_birth_hat, C_shared_base_hat, M_index, weights, meta, max_delta, args.eval_device)
        birth_train_f = eval_functional(train_seqs, M_birth_hat, C_birth_hat, M_index, weights, meta, max_delta, args.eval_device)
        birth_held_f = eval_functional(held_seqs, M_birth_hat, C_birth_hat, M_index, weights, meta, max_delta, args.eval_device)

        print(fmt_func("base_tr", base_train_f))
        print(fmt_func("base_he", base_held_f))
        print(fmt_func("qkonly_tr", qkonly_train_f))
        print(fmt_func("qkonly_he", qkonly_held_f))
        print(fmt_func("birth_tr", birth_train_f))
        print(fmt_func("birth_he", birth_held_f))
        print(f"accepted QK={len(qk_acc)} VO={len(vo_acc)} mode={args.vo_mode}")
        print(f"delta held target: M={birth_target['held_M_rel']-base_target['held_M_rel']:+.5f} C={birth_target['held_C_rel']-base_target['held_C_rel']:+.5f}")
        print(f"delta held func vs base: A={birth_held_f['A_rel']-base_held_f['A_rel']:+.5f} Y_all={birth_held_f['Y_all_rel']-base_held_f['Y_all_rel']:+.5f} H_after={birth_held_f['H_after_rel']-base_held_f['H_after_rel']:+.5f}")
        print(f"delta held func vs qkonly: Y_all={birth_held_f['Y_all_rel']-qkonly_held_f['Y_all_rel']:+.5f} H_after={birth_held_f['H_after_rel']-qkonly_held_f['H_after_rel']:+.5f}")
        print(f"time={time.perf_counter()-t0:.1f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"])
    p.add_argument("--attn-implementation", default="eager")
    p.add_argument("--layers", default="2")
    p.add_argument("--max-length", type=int, default=48)
    p.add_argument("--max-delta", type=int, default=47)
    p.add_argument("--prompts-per-suite", type=int, default=1)
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--target-split", choices=["heads","deltas"], default="heads")
    p.add_argument("--qk-candidates", type=int, default=4)
    p.add_argument("--vo-candidates", type=int, default=4)
    p.add_argument("--qk-births", type=int, default=2)
    p.add_argument("--vo-births", type=int, default=2)
    p.add_argument("--min-held-gain-qk", type=float, default=0.001)
    p.add_argument("--min-held-gain-vo", type=float, default=0.001)

    p.add_argument("--vo-mode", choices=["shared-heads", "per-head-prompts"], default="per-head-prompts")
    p.add_argument("--vo-per-head-candidates", type=int, default=2)
    p.add_argument("--vo-per-head-births", type=int, default=1)
    p.add_argument("--vo-min-held-y-gain", type=float, default=0.001)
    p.add_argument("--vo-min-held-h-gain", type=float, default=-1e-5)
    p.add_argument("--vo-include-full-residual", action="store_true",
                   help="allow one full residual atom per head; useful as upper-bound/private-operator test")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
