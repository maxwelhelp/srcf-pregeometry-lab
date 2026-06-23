#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
real_qwen_joint_heads_circuit_compare_v2.py
VERSION = v2.0-real-joint-all-heads-circuit-vs-raw

Real transformer test, no planted hidden operator.

This version fixes the wrong level of v1:
  v1 compared one head in isolation.
  v2 compares the whole attention layer as one differentiable all-head circuit:

      Y_all = sum_h A_h(X, M_qk_h) @ X_aug @ C_vo_h.T
      H_after_attn = H_before + Y_all

Methods:
  exact:
    real model head outputs reconstructed from Wq/Wk/Wv/Wo and real X.

  raw:
    low-rank SVD Wq/Wk/Wv/Wo separately, then run all heads and sum.

  circuit:
    build exact circuit targets per head:
      M_qk_h[d] = Wq_aug_h.T @ R_delta @ Wk_aug_kv / sqrt(head_dim)
      C_vo_h    = Wo_h @ Wv_aug_kv
    low-rank those targets directly, then run all heads and sum.

Metrics:
  score_rel/A_rel/KL/top1 are per-head routing metrics.
  Y_all_rel is the real joint all-head attention-output metric.
  H_after_rel is after residual addition.
  head_credit prints true per-head contribution to Y_all.

No hidden true_corr exists here because this is real Qwen.
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

VERSION = "v2.0-real-joint-all-heads-circuit-vs-raw"


# ---------------- utilities ----------------

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
    raise ValueError(f"bad dtype: {name}")


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in str(s).replace(";", ",").split(",") if x.strip()]


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


def rel_num_den(pred: torch.Tensor, true: torch.Tensor) -> Tuple[float, float]:
    pred = pred.detach().float()
    true = true.detach().float()
    d = pred - true
    return float((d * d).sum().item()), float((true * true).sum().item())


def rel_from(num: float, den: float, eps: float = 1e-12) -> float:
    return math.sqrt(num / max(eps, den))


def cosine_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    av = a.detach().float().reshape(-1)
    bv = b.detach().float().reshape(-1)
    return float(torch.dot(av, bv).item() / max(eps, float(torch.linalg.norm(av) * torch.linalg.norm(bv))))


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


def svd_reconstruct(M: torch.Tensor, rank: int) -> torch.Tensor:
    M = M.detach().float()
    r = min(max(0, int(rank)), min(M.shape))
    if r <= 0:
        return torch.zeros_like(M)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


def lowrank_factors(M: torch.Tensor, rank: int, device: str = "cuda", randomized: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns A,B where M ~= A @ B.T.
    A: [m,r], B: [n,r]
    """
    M = M.detach().float().to(device)
    r = min(max(0, int(rank)), min(M.shape))
    if r <= 0:
        return torch.zeros(M.shape[0], 1, device=device), torch.zeros(M.shape[1], 1, device=device)
    if randomized and r + 8 < min(M.shape):
        try:
            q = min(min(M.shape) - 1, max(r + 8, r))
            U, S, V = torch.svd_lowrank(M, q=q, niter=2)
            A = U[:, :r] * S[:r]
            B = V[:, :r]
            return A.contiguous(), B.contiguous()
        except Exception:
            pass
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    A = U[:, :r] * S[:r]
    B = Vh[:r].T
    return A.contiguous(), B.contiguous()


def lowrank_param_count(shape: Tuple[int, int], rank: int) -> int:
    m, n = int(shape[0]), int(shape[1])
    r = min(max(0, int(rank)), min(m, n))
    return r * (m + n)


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


# ---------------- data structures ----------------

@dataclass
class LayerSeq:
    prompt_id: int
    suite: str
    text: str
    Xraw: torch.Tensor       # [T,H]
    Xn: torch.Tensor         # [T,H]
    Xaug: torch.Tensor       # [T,H+1]
    Q: torch.Tensor          # [heads,T,D]
    K_kv: torch.Tensor       # [kv,T,D]
    V_kv: torch.Tensor       # [kv,T,D]
    A: torch.Tensor          # [heads,T,T]
    Y_heads: torch.Tensor    # [heads,T,H]
    Y_all: torch.Tensor      # [T,H]
    H_after: torch.Tensor    # [T,H]
    cos: torch.Tensor        # [T,D]
    sin: torch.Tensor        # [T,D]


# ---------------- extraction ----------------

def slice_bias(module: Any, start: int, end: int) -> torch.Tensor:
    b = getattr(module, "bias", None)
    if b is None:
        return torch.zeros(end - start, dtype=torch.float32)
    return b.detach().float()[start:end].cpu()


@torch.no_grad()
def collect_layer_data(model, tokenizer, prompts: List[Dict[str, str]], layer_idx: int,
                       max_length: int, device: str) -> Tuple[List[LayerSeq], Dict[str, int]]:
    layers = get_layers(model)
    layer = layers[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    D = int(getattr(cfg, "head_dim", H // n_heads))
    kv_groups = n_heads // n_kv
    Wo_all = attn.o_proj.weight.detach().float().to(device)  # [H, heads*D]
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
        Xraw = outputs.hidden_states[layer_idx].detach()  # [1,T,H]
        Xn = layer.input_layernorm(Xraw).detach()

        q = attn.q_proj(Xn).view(1, T, n_heads, D).transpose(1, 2).contiguous()  # [1,heads,T,D]
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
        A_heads_t = torch.stack(A_heads, dim=0)           # [heads,T,T]
        Y_heads_t = torch.stack(Y_heads, dim=0)           # [heads,T,H]
        Y_all = Y_heads_t.sum(dim=0) + bo.detach().cpu().view(1, -1)
        Xraw0 = Xraw[0].float().cpu()
        H_after = Xraw0 + Y_all
        Xn0 = Xn[0].float().cpu()
        Xaug = torch.cat([Xn0, torch.ones(T, 1)], dim=1)

        rows.append(LayerSeq(
            prompt_id=pi, suite=str(pr.get("suite", "?")), text=pr["text"],
            Xraw=Xraw0, Xn=Xn0, Xaug=Xaug, Q=q_rot.cpu(), K_kv=k_rot.cpu(), V_kv=V_kv.cpu(),
            A=A_heads_t, Y_heads=Y_heads_t, Y_all=Y_all, H_after=H_after,
            cos=cos2.float().cpu(), sin=sin2.float().cpu(),
        ))
        del outputs
    meta = {"H": H, "D": D, "n_heads": n_heads, "n_kv": n_kv, "kv_groups": kv_groups}
    return rows, meta


def build_all_weights(model, layer_idx: int, meta: Dict[str, int]) -> Dict[str, torch.Tensor]:
    layers = get_layers(model)
    attn = layers[layer_idx].self_attn
    H = int(meta["H"])
    D = int(meta["D"])
    n_heads = int(meta["n_heads"])
    n_kv = int(meta["n_kv"])

    Wq = attn.q_proj.weight.detach().float().cpu().view(n_heads, D, H)
    Wk = attn.k_proj.weight.detach().float().cpu().view(n_kv, D, H)
    Wv = attn.v_proj.weight.detach().float().cpu().view(n_kv, D, H)
    Wo_flat = attn.o_proj.weight.detach().float().cpu()  # [H, heads*D]
    Wo = torch.stack([Wo_flat[:, h * D:(h + 1) * D] for h in range(n_heads)], dim=0)  # [heads,H,D]

    bq = []
    for h in range(n_heads):
        bq.append(slice_bias(attn.q_proj, h * D, (h + 1) * D))
    bk = []
    bv = []
    for kv in range(n_kv):
        bk.append(slice_bias(attn.k_proj, kv * D, (kv + 1) * D))
        bv.append(slice_bias(attn.v_proj, kv * D, (kv + 1) * D))
    bq = torch.stack(bq, dim=0)
    bk = torch.stack(bk, dim=0)
    bv = torch.stack(bv, dim=0)

    Wq_aug = torch.cat([Wq, bq[:, :, None]], dim=2)  # [heads,D,H+1]
    Wk_aug = torch.cat([Wk, bk[:, :, None]], dim=2)  # [kv,D,H+1]
    Wv_aug = torch.cat([Wv, bv[:, :, None]], dim=2)  # [kv,D,H+1]
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


# ---------------- approximations ----------------

def make_raw_approx(weights: Dict[str, torch.Tensor], rank: int, device: str) -> Dict[str, torch.Tensor]:
    Wq = torch.stack([svd_reconstruct(w, rank) for w in weights["Wq_aug"]], dim=0).to(device)
    Wk = torch.stack([svd_reconstruct(w, rank) for w in weights["Wk_aug"]], dim=0).to(device)
    Wv = torch.stack([svd_reconstruct(w, rank) for w in weights["Wv_aug"]], dim=0).to(device)
    Wo = torch.stack([svd_reconstruct(w, rank) for w in weights["Wo"]], dim=0).to(device)
    return {"Wq_aug": Wq, "Wk_aug": Wk, "Wv_aug": Wv, "Wo": Wo, "bo": weights["bo"].to(device)}


def raw_budget(weights: Dict[str, torch.Tensor], rank: int) -> int:
    total = 0
    for w in weights["Wq_aug"]:
        total += lowrank_param_count(tuple(w.shape), rank)
    for w in weights["Wk_aug"]:
        total += lowrank_param_count(tuple(w.shape), rank)
    for w in weights["Wv_aug"]:
        total += lowrank_param_count(tuple(w.shape), rank)
    for w in weights["Wo"]:
        total += lowrank_param_count(tuple(w.shape), rank)
    return total


def make_circuit_factors(weights: Dict[str, torch.Tensor], Rpos: Dict[int, torch.Tensor],
                         meta: Dict[str, int], max_delta: int, rank_qk: int, rank_vo: int,
                         device: str, randomized: bool = True) -> Dict[str, Any]:
    n_heads = int(meta["n_heads"])
    kv_groups = int(meta["kv_groups"])
    D = int(meta["D"])
    if 0 not in Rpos:
        raise RuntimeError("Rpos missing 0")
    R0 = Rpos[0].to(device).float()
    Wq = weights["Wq_aug"].to(device).float()
    Wk = weights["Wk_aug"].to(device).float()
    Wv = weights["Wv_aug"].to(device).float()
    Wo = weights["Wo"].to(device).float()

    qk_A: List[List[torch.Tensor]] = []
    qk_B: List[List[torch.Tensor]] = []
    deltas = [d for d in range(max_delta + 1) if d in Rpos]
    for h in range(n_heads):
        kv = h // kv_groups
        Ah_list = []
        Bh_list = []
        for d in deltas:
            Rrel = Rpos[d].to(device).float().T @ R0
            M = (Wq[h].T @ Rrel @ Wk[kv]) / math.sqrt(D)  # [H+1,H+1]
            A, B = lowrank_factors(M, rank_qk, device=device, randomized=randomized)
            Ah_list.append(A)
            Bh_list.append(B)
        qk_A.append(Ah_list)
        qk_B.append(Bh_list)

    vo_A: List[torch.Tensor] = []
    vo_B: List[torch.Tensor] = []
    for h in range(n_heads):
        kv = h // kv_groups
        C = Wo[h] @ Wv[kv]  # [H,H+1]
        A, B = lowrank_factors(C, rank_vo, device=device, randomized=randomized)
        vo_A.append(A)
        vo_B.append(B)

    return {"qk_A": qk_A, "qk_B": qk_B, "vo_A": vo_A, "vo_B": vo_B, "deltas": deltas, "bo": weights["bo"].to(device).float()}


def circuit_budget(meta: Dict[str, int], max_delta_present: int, Hplus1: int, rank_qk: int, rank_vo: int) -> int:
    n_heads = int(meta["n_heads"])
    H = int(meta["H"])
    # M qk: [H+1,H+1] for each head*delta
    qk = n_heads * (max_delta_present + 1) * lowrank_param_count((Hplus1, Hplus1), rank_qk)
    # C vo: [H,H+1] for each head
    vo = n_heads * lowrank_param_count((H, Hplus1), rank_vo)
    return qk + vo


# ---------------- evaluation ----------------

@torch.no_grad()
def eval_exact(seqs: List[LayerSeq], meta: Dict[str, int], device: str) -> Dict[str, float]:
    # sanity: exact is by definition true stored tensors, so errors are zero.
    total_rows = sum(int(s.A.shape[1]) * int(meta["n_heads"]) for s in seqs)
    return {
        "score_rel": 0.0,
        "A_rel": 0.0,
        "KL": 0.0,
        "top1": 1.0,
        "Y_all_rel": 0.0,
        "H_after_rel": 0.0,
        "coverage": 1.0,
        "rows": float(total_rows),
    }


@torch.no_grad()
def eval_raw(seqs: List[LayerSeq], weights_hat: Dict[str, torch.Tensor], meta: Dict[str, int],
             max_delta: int, device: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    n_heads = int(meta["n_heads"])
    n_kv = int(meta["n_kv"])
    kv_groups = int(meta["kv_groups"])
    D = int(meta["D"])
    H = int(meta["H"])

    Wq = weights_hat["Wq_aug"]
    Wk = weights_hat["Wk_aug"]
    Wv = weights_hat["Wv_aug"]
    Wo = weights_hat["Wo"]
    bo = weights_hat["bo"]

    acc = {"score_n":0.0,"score_d":0.0,"A_n":0.0,"A_d":0.0,"Y_n":0.0,"Y_d":0.0,"H_n":0.0,"H_d":0.0,
           "KL":0.0,"top":0,"rows":0,"covered":0,"allrows":0}
    head_rows = [{"Y_n":0.0,"Y_d":0.0,"norm":0.0,"align":0.0} for _ in range(n_heads)]

    for s in seqs:
        X = s.Xaug.to(device).float()
        Xraw = s.Xraw.to(device).float()
        T = int(X.shape[0])
        cos = s.cos.to(device).float()
        sin = s.sin.to(device).float()
        A_true_all = s.A.to(device).float()
        Y_true_heads = s.Y_heads.to(device).float()
        Y_true_all = s.Y_all.to(device).float()
        H_true = s.H_after.to(device).float()
        Q_true = s.Q.to(device).float()
        K_true_kv = s.K_kv.to(device).float()

        Khat = []
        Vhat = []
        for kv in range(n_kv):
            kpre = X @ Wk[kv].T
            kh = apply_rope_rows(kpre, cos, sin)
            Khat.append(kh)
            Vhat.append(X @ Wv[kv].T)
        Khat = torch.stack(Khat, dim=0)
        Vhat = torch.stack(Vhat, dim=0)

        Y_pred_all = torch.zeros_like(Y_true_all)
        Y_pred_heads = []
        for h in range(n_heads):
            kv = h // kv_groups
            qpre = X @ Wq[h].T
            qh = apply_rope_rows(qpre, cos, sin)
            S = (qh @ Khat[kv].T) / math.sqrt(D)
            S_true = (Q_true[h] @ K_true_kv[kv].T) / math.sqrt(D)

            full_rows = torch.ones(T, dtype=torch.bool, device=device)
            if max_delta < T - 1:
                for i in range(T):
                    if i > max_delta:
                        full_rows[i] = False
            Ahat = causal_softmax(S)
            rows = full_rows
            acc["allrows"] += T
            acc["covered"] += int(rows.sum().item())
            if int(rows.sum().item()) > 0:
                tril = torch.tril(torch.ones(T,T,device=device,dtype=torch.bool))
                if max_delta < T - 1:
                    delta_ok = torch.zeros(T,T,device=device,dtype=torch.bool)
                    for i in range(T):
                        j0 = max(0, i - max_delta)
                        delta_ok[i, j0:i+1] = True
                    pair_mask = tril & delta_ok
                else:
                    pair_mask = tril
                sn, sd = rel_num_den(S[pair_mask], S_true[pair_mask])
                acc["score_n"] += sn; acc["score_d"] += sd
                an, ad = rel_num_den(Ahat[rows], A_true_all[h][rows])
                acc["A_n"] += an; acc["A_d"] += ad
                acc["KL"] += float(F.kl_div((Ahat[rows] + 1e-12).log(), A_true_all[h][rows], reduction="sum").item())
                acc["top"] += int((Ahat[rows].argmax(dim=-1) == A_true_all[h][rows].argmax(dim=-1)).sum().item())
                acc["rows"] += int(rows.sum().item())

            Z = Ahat @ Vhat[kv]
            Yh = Z @ Wo[h].T
            Y_pred_heads.append(Yh)
            Y_pred_all += Yh

        Y_pred_all = Y_pred_all + bo.view(1, -1)
        H_pred = Xraw + Y_pred_all
        yn, yd = rel_num_den(Y_pred_all, Y_true_all)
        hn, hd = rel_num_den(H_pred, H_true)
        acc["Y_n"] += yn; acc["Y_d"] += yd
        acc["H_n"] += hn; acc["H_d"] += hd

        Y_pred_heads_t = torch.stack(Y_pred_heads, dim=0)
        for h in range(n_heads):
            n, d = rel_num_den(Y_pred_heads_t[h], Y_true_heads[h])
            head_rows[h]["Y_n"] += n
            head_rows[h]["Y_d"] += d
            head_rows[h]["norm"] += float(torch.linalg.norm(Y_true_heads[h]).item())
            head_rows[h]["align"] += cosine_flat(Y_true_heads[h], Y_true_all)

    rows = max(1, acc["rows"])
    out = {
        "score_rel": rel_from(acc["score_n"], acc["score_d"]),
        "A_rel": rel_from(acc["A_n"], acc["A_d"]),
        "KL": acc["KL"] / rows,
        "top1": acc["top"] / rows,
        "Y_all_rel": rel_from(acc["Y_n"], acc["Y_d"]),
        "H_after_rel": rel_from(acc["H_n"], acc["H_d"]),
        "coverage": acc["covered"] / max(1, acc["allrows"]),
        "rows": float(acc["rows"]),
    }
    head = []
    for h, r in enumerate(head_rows):
        head.append({
            "head": h,
            "Y_head_rel": rel_from(r["Y_n"], r["Y_d"]),
            "true_norm_sum": r["norm"],
            "align_to_Y_all_sum": r["align"],
        })
    return out, head


@torch.no_grad()
def eval_circuit(seqs: List[LayerSeq], factors: Dict[str, Any], meta: Dict[str, int],
                 max_delta: int, device: str) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    n_heads = int(meta["n_heads"])
    kv_groups = int(meta["kv_groups"])
    D = int(meta["D"])

    qk_A = factors["qk_A"]
    qk_B = factors["qk_B"]
    vo_A = factors["vo_A"]
    vo_B = factors["vo_B"]
    deltas = factors["deltas"]
    d_to_idx = {d:i for i,d in enumerate(deltas)}
    bo = factors["bo"]

    acc = {"score_n":0.0,"score_d":0.0,"A_n":0.0,"A_d":0.0,"Y_n":0.0,"Y_d":0.0,"H_n":0.0,"H_d":0.0,
           "KL":0.0,"top":0,"rows":0,"covered":0,"allrows":0}
    head_rows = [{"Y_n":0.0,"Y_d":0.0,"norm":0.0,"align":0.0} for _ in range(n_heads)]

    for s in seqs:
        X = s.Xaug.to(device).float()
        Xraw = s.Xraw.to(device).float()
        T = int(X.shape[0])
        A_true_all = s.A.to(device).float()
        Y_true_heads = s.Y_heads.to(device).float()
        Y_true_all = s.Y_all.to(device).float()
        H_true = s.H_after.to(device).float()
        Q_true = s.Q.to(device).float()
        K_true_kv = s.K_kv.to(device).float()

        Y_pred_all = torch.zeros_like(Y_true_all)
        Y_pred_heads = []

        for h in range(n_heads):
            kv = h // kv_groups
            S = torch.full((T, T), torch.finfo(torch.float32).min, device=device)
            S_true = (Q_true[h] @ K_true_kv[kv].T) / math.sqrt(D)
            full_rows = torch.ones(T, dtype=torch.bool, device=device)

            for d in range(min(max_delta, T - 1) + 1):
                if d not in d_to_idx:
                    if d <= T - 1:
                        for i in range(d, T):
                            full_rows[i] = False
                    continue
                idx = d_to_idx[d]
                A_fac = qk_A[h][idx]
                B_fac = qk_B[h][idx]
                Xi = X[d:T]      # i=d..T-1
                Xj = X[:T-d]     # j=0..T-d-1
                li = Xi @ A_fac
                rj = Xj @ B_fac
                vals = (li * rj).sum(dim=-1)
                rows_i = torch.arange(d, T, device=device)
                cols_j = torch.arange(0, T-d, device=device)
                S[rows_i, cols_j] = vals

            if max_delta < T - 1:
                for i in range(max_delta + 1, T):
                    full_rows[i] = False

            # replace non-causal future by -inf already; but row 0.. covered have all causal entries filled
            Ahat = torch.softmax(S, dim=-1)
            rows = full_rows
            acc["allrows"] += T
            acc["covered"] += int(rows.sum().item())
            if int(rows.sum().item()) > 0:
                pair_mask = torch.zeros(T, T, dtype=torch.bool, device=device)
                for i in range(T):
                    if rows[i]:
                        pair_mask[i, :i+1] = True
                sn, sd = rel_num_den(S[pair_mask], S_true[pair_mask])
                acc["score_n"] += sn; acc["score_d"] += sd
                an, ad = rel_num_den(Ahat[rows], A_true_all[h][rows])
                acc["A_n"] += an; acc["A_d"] += ad
                acc["KL"] += float(F.kl_div((Ahat[rows] + 1e-12).log(), A_true_all[h][rows], reduction="sum").item())
                acc["top"] += int((Ahat[rows].argmax(dim=-1) == A_true_all[h][rows].argmax(dim=-1)).sum().item())
                acc["rows"] += int(rows.sum().item())

            # payload = X @ C.T, C ~= A @ B.T, A:[H,r], B:[H+1,r]
            payload = (X @ vo_B[h]) @ vo_A[h].T
            Yh = Ahat @ payload
            Y_pred_heads.append(Yh)
            Y_pred_all += Yh

        Y_pred_all = Y_pred_all + bo.view(1, -1)
        H_pred = Xraw + Y_pred_all
        yn, yd = rel_num_den(Y_pred_all, Y_true_all)
        hn, hd = rel_num_den(H_pred, H_true)
        acc["Y_n"] += yn; acc["Y_d"] += yd
        acc["H_n"] += hn; acc["H_d"] += hd

        Y_pred_heads_t = torch.stack(Y_pred_heads, dim=0)
        for h in range(n_heads):
            n, d = rel_num_den(Y_pred_heads_t[h], Y_true_heads[h])
            head_rows[h]["Y_n"] += n
            head_rows[h]["Y_d"] += d
            head_rows[h]["norm"] += float(torch.linalg.norm(Y_true_heads[h]).item())
            head_rows[h]["align"] += cosine_flat(Y_true_heads[h], Y_true_all)

    rows = max(1, acc["rows"])
    out = {
        "score_rel": rel_from(acc["score_n"], acc["score_d"]),
        "A_rel": rel_from(acc["A_n"], acc["A_d"]),
        "KL": acc["KL"] / rows,
        "top1": acc["top"] / rows,
        "Y_all_rel": rel_from(acc["Y_n"], acc["Y_d"]),
        "H_after_rel": rel_from(acc["H_n"], acc["H_d"]),
        "coverage": acc["covered"] / max(1, acc["allrows"]),
        "rows": float(acc["rows"]),
    }
    head = []
    for h, r in enumerate(head_rows):
        head.append({
            "head": h,
            "Y_head_rel": rel_from(r["Y_n"], r["Y_d"]),
            "true_norm_sum": r["norm"],
            "align_to_Y_all_sum": r["align"],
        })
    return out, head


def fmt(name: str, m: Dict[str, float]) -> str:
    return (
        f"{name:<16} score={m['score_rel']:.5f} A={m['A_rel']:.5f} KL={m['KL']:.5f} "
        f"top1={m['top1']:.3f} Y_all={m['Y_all_rel']:.5f} H_after={m['H_after_rel']:.5f} cov={m['coverage']:.2f}"
    )


def print_top_heads(title: str, head_rows: List[Dict[str, float]], n: int = 5) -> None:
    rows = sorted(head_rows, key=lambda r: r["true_norm_sum"], reverse=True)[:n]
    print(f"{title}:")
    for r in rows:
        print(f"  H{int(r['head']):02d} Y_head_rel={r['Y_head_rel']:.4f} true_norm_sum={r['true_norm_sum']:.3f} align_sum={r['align_to_Y_all_sum']:.3f}")


# ---------------- main run ----------------

def run(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(args.seed)
    dtype = get_dtype(args.dtype)
    print(f"=== Real Qwen JOINT all-head circuit compare {VERSION} ===")
    print(f"model={args.model} device={args.device} eval_device={args.eval_device} dtype={args.dtype}")
    print("NO planted hidden operator. Real layer, all attention heads summed together.")
    print("metrics: score/A per head; Y_all/H_after for whole layer.")
    print("")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(args.device)
    model.eval()

    cfg = model.config
    layers = parse_int_list(args.layers)
    prompts = build_prompts(args.prompts_per_suite)
    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
    split = max(2, int(len(prompts) * args.train_frac))
    train_prompts = prompts[:split]
    held_prompts = prompts[split:]
    if len(held_prompts) < 2:
        held_prompts = prompts[::2]
        train_prompts = prompts[1::2]
    retain_prompts = [p for p in prompts if p["suite"] in ("plain", "math", "code", "reason")][::2]
    if len(retain_prompts) < 2:
        retain_prompts = held_prompts

    agg = []
    for layer_idx in layers:
        t0 = time.perf_counter()
        print(f"\n--- layer L{layer_idx} all heads ---")
        train_seqs, meta = collect_layer_data(model, tok, train_prompts, layer_idx, args.max_length, args.device)
        held_seqs, _ = collect_layer_data(model, tok, held_prompts, layer_idx, args.max_length, args.device)
        retain_seqs, _ = collect_layer_data(model, tok, retain_prompts, layer_idx, args.max_length, args.device)
        allseqs = train_seqs + held_seqs + retain_seqs
        max_observed = max(int(s.Xaug.shape[0]) for s in allseqs)
        max_delta = min(int(args.max_delta), max_observed - 1)
        Rpos = build_rope_pos(allseqs, max_delta)
        weights = build_all_weights(model, layer_idx, meta)

        rb = raw_budget(weights, args.rank_raw)
        cb = circuit_budget(meta, max_delta, int(meta["H"]) + 1, args.rank_circuit_qk, args.rank_circuit_vo)

        print(f"seqs train={len(train_seqs)} held={len(held_seqs)} retain={len(retain_seqs)} T_max={max_observed} max_delta={max_delta}")
        print(f"model heads={meta['n_heads']} kv={meta['n_kv']} H={meta['H']} D={meta['D']}")
        print(f"budget raw_rank={args.rank_raw} raw_params≈{rb:,} | circuit_qk_rank={args.rank_circuit_qk} circuit_vo_rank={args.rank_circuit_vo} circuit_params≈{cb:,}")

        exact_train = eval_exact(train_seqs, meta, args.eval_device)
        exact_held = eval_exact(held_seqs, meta, args.eval_device)
        print(fmt("exact_train", exact_train))
        print(fmt("exact_held", exact_held))

        print("building raw approximation...", flush=True)
        raw_hat = make_raw_approx(weights, args.rank_raw, args.eval_device)
        raw_train, raw_head_train = eval_raw(train_seqs, raw_hat, meta, max_delta, args.eval_device)
        raw_held, raw_head_held = eval_raw(held_seqs, raw_hat, meta, max_delta, args.eval_device)
        raw_retain, raw_head_retain = eval_raw(retain_seqs, raw_hat, meta, max_delta, args.eval_device)

        print("building circuit approximation...", flush=True)
        circ_fac = make_circuit_factors(
            weights, Rpos, meta, max_delta,
            args.rank_circuit_qk, args.rank_circuit_vo,
            args.eval_device, randomized=(not args.full_svd_circuit)
        )
        circ_train, circ_head_train = eval_circuit(train_seqs, circ_fac, meta, max_delta, args.eval_device)
        circ_held, circ_head_held = eval_circuit(held_seqs, circ_fac, meta, max_delta, args.eval_device)
        circ_retain, circ_head_retain = eval_circuit(retain_seqs, circ_fac, meta, max_delta, args.eval_device)

        print(fmt("raw_train", raw_train))
        print(fmt("raw_held", raw_held))
        print(fmt("raw_retain", raw_retain))
        print(fmt("circuit_train", circ_train))
        print(fmt("circuit_held", circ_held))
        print(fmt("circuit_retain", circ_retain))
        print(f"delta held: circuit-raw A={circ_held['A_rel']-raw_held['A_rel']:+.5f} Y_all={circ_held['Y_all_rel']-raw_held['Y_all_rel']:+.5f} H_after={circ_held['H_after_rel']-raw_held['H_after_rel']:+.5f}")
        print_top_heads("raw held top true-norm heads", raw_head_held, n=args.print_heads)
        print_top_heads("circuit held top true-norm heads", circ_head_held, n=args.print_heads)
        elapsed = time.perf_counter() - t0
        print(f"time={elapsed:.1f}s")

        agg.append({
            "layer": layer_idx,
            "raw_held_A": raw_held["A_rel"],
            "circ_held_A": circ_held["A_rel"],
            "raw_held_Y": raw_held["Y_all_rel"],
            "circ_held_Y": circ_held["Y_all_rel"],
            "raw_held_H": raw_held["H_after_rel"],
            "circ_held_H": circ_held["H_after_rel"],
            "raw_retain_Y": raw_retain["Y_all_rel"],
            "circ_retain_Y": circ_retain["Y_all_rel"],
            "elapsed": elapsed,
        })

    if agg:
        def mean(k: str) -> float:
            return sum(r[k] for r in agg) / len(agg)
        print("\n=== aggregate ===")
        print(f"raw_held_A_rel       {mean('raw_held_A'):.5f}")
        print(f"circuit_held_A_rel   {mean('circ_held_A'):.5f}")
        print(f"raw_held_Y_all_rel   {mean('raw_held_Y'):.5f}")
        print(f"circuit_held_Y_all_rel {mean('circ_held_Y'):.5f}")
        print(f"raw_held_H_after_rel {mean('raw_held_H'):.5f}")
        print(f"circuit_held_H_after_rel {mean('circ_held_H'):.5f}")
        print(f"raw_retain_Y_all_rel {mean('raw_retain_Y'):.5f}")
        print(f"circuit_retain_Y_all_rel {mean('circ_retain_Y'):.5f}")
        print("\nInterpretation:")
        print("  This is the correct joint all-head test.")
        print("  If circuit improves A but not Y_all/H_after, QK reading helps but VO/all-head write still needs better decoder/budget.")
        print("  If circuit improves Y_all/H_after too, circuit-target reading transfers to real layer behavior.")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="2")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=63)
    ap.add_argument("--prompts-per-suite", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=0)
    ap.add_argument("--train-frac", type=float, default=0.5)
    ap.add_argument("--rank-raw", type=int, default=16)
    ap.add_argument("--rank-circuit-qk", type=int, default=1)
    ap.add_argument("--rank-circuit-vo", type=int, default=16)
    ap.add_argument("--full-svd-circuit", action="store_true", help="slower but deterministic full SVD for circuit matrices")
    ap.add_argument("--print-heads", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
