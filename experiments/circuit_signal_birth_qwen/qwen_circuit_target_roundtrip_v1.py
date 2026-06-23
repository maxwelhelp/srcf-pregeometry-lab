#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_circuit_target_roundtrip_v1.py

Next step after synthetic_program_roundtrip_v1_1.py.

Purpose:
1) Extract real Qwen QK/VO affine circuit targets using the known-good formula:
   M_qk_aug[d] = Wq_aug.T @ (R_i.T @ R_j) @ Wk_aug / sqrt(D), d=i-j
   C_vo_aug    = Wo_head @ Wv_aug
2) Verify targets against real forward Q/K/V/A/Y.
3) Run Level-0 analytic decode on selected targets and report residual.

No training, no KL, no alpha sweep, no learned dictionary.
"""
from __future__ import annotations

import argparse, json, math, os, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise RuntimeError("Please install transformers to run Qwen target extraction") from e

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import (
    build_qk_primitives, build_vo_primitives, primitive_manifest,
    decode_greedy_analytic, make_program, rel_err
)

VERSION = "qwen_circuit_target_roundtrip_v1.0"


@dataclass
class CircuitSeq:
    prompt_id: int
    text: str
    token_ids: List[int]
    Xn: torch.Tensor
    Xaug: torch.Tensor
    Q_pre: torch.Tensor
    K_pre: torch.Tensor
    Q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    A: torch.Tensor
    Y: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor


def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)


def get_dtype(name: str):
    name = name.lower()
    if name in ("fp16", "float16"): return torch.float16
    if name in ("bf16", "bfloat16"): return torch.bfloat16
    return torch.float32


def get_layers(model: Any):
    return model.model.layers


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rotate_half_matrix(D: int, device=None, dtype=torch.float32) -> torch.Tensor:
    device = device or torch.device("cpu")
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


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    T = scores.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=scores.device, dtype=torch.bool), diagonal=1)
    return torch.softmax(scores.masked_fill(mask, torch.finfo(scores.dtype).min), dim=-1)


def compute_position_embeddings(model: Any, hidden_states: torch.Tensor, position_ids: torch.Tensor):
    rotary = getattr(model.model, "rotary_emb", None)
    if rotary is None:
        raise RuntimeError("model.model.rotary_emb not found")
    try:
        return rotary(hidden_states, position_ids)
    except TypeError:
        return rotary(position_ids)


def _slice_bias(module: Any, start: int, end: int) -> torch.Tensor:
    b = getattr(module, "bias", None)
    if b is None:
        return torch.zeros(end - start, dtype=torch.float32)
    return b.detach().float()[start:end].cpu()


def build_weight_slices(model: Any, layer_idx: int, head_idx: int, meta: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    attn = get_layers(model)[layer_idx].self_attn
    H = int(meta["hidden_size"]); D = int(meta["head_dim"]); kv_idx = int(meta["kv_idx"])
    Wq = attn.q_proj.weight.detach().float()[head_idx * D:(head_idx + 1) * D, :].cpu()
    Wk = attn.k_proj.weight.detach().float()[kv_idx * D:(kv_idx + 1) * D, :].cpu()
    Wv = attn.v_proj.weight.detach().float()[kv_idx * D:(kv_idx + 1) * D, :].cpu()
    Wo = attn.o_proj.weight.detach().float()[:, head_idx * D:(head_idx + 1) * D].cpu()
    bq = _slice_bias(attn.q_proj, head_idx * D, (head_idx + 1) * D)
    bk = _slice_bias(attn.k_proj, kv_idx * D, (kv_idx + 1) * D)
    bv = _slice_bias(attn.v_proj, kv_idx * D, (kv_idx + 1) * D)
    Wq_aug = torch.cat([Wq, bq[:, None]], dim=1)
    Wk_aug = torch.cat([Wk, bk[:, None]], dim=1)
    Wv_aug = torch.cat([Wv, bv[:, None]], dim=1)
    return {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo, "bq": bq, "bk": bk, "bv": bv, "Wq_aug": Wq_aug, "Wk_aug": Wk_aug, "Wv_aug": Wv_aug}


def build_prompts(n: int) -> List[str]:
    base = [
        "The capital of France is",
        "Write a Python function that adds two numbers.",
        "If x = 2 and y = 3, x + y =",
        "Translate to English: привіт світ",
        "A list in Python can be indexed with",
        "The quick brown fox jumps over",
        "def factorial(n):",
        "Question: What color is the sky? Answer:",
    ]
    out = []
    while len(out) < n:
        out.extend(base)
    return out[:n]


@torch.no_grad()
def collect_head_data(model: Any, tokenizer: Any, prompts: List[str], layer_idx: int, head_idx: int, max_length: int, device: str):
    layer = get_layers(model)[layer_idx]
    attn = layer.self_attn
    cfg = model.config
    H = int(cfg.hidden_size)
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, "num_key_value_heads", n_heads))
    D = int(getattr(cfg, "head_dim", H // n_heads))
    kv_groups = n_heads // n_kv
    kv_idx = int(head_idx) // kv_groups
    meta = {"hidden_size": H, "num_heads": n_heads, "num_kv_heads": n_kv, "head_dim": D, "kv_groups": kv_groups, "kv_idx": kv_idx}

    o_w = attn.o_proj.weight.detach().float()[:, head_idx * D:(head_idx + 1) * D].to(device)
    seqs: List[CircuitSeq] = []
    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None: attn_mask = attn_mask.to(device)
        T = int(input_ids.shape[1])
        if T < 2: continue
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
        q_rot = apply_rope_rows(q, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))
        k_rot = apply_rope_rows(k, cos2.unsqueeze(0).unsqueeze(0), sin2.unsqueeze(0).unsqueeze(0))
        Qh = q_rot[0, head_idx].float()
        Kh = k_rot[0, kv_idx].float()
        Vh = v[0, kv_idx].float()
        scores = (Qh @ Kh.T) / math.sqrt(D)
        A = causal_softmax(scores.float())
        Y = (A @ Vh) @ o_w.T
        Xn_cpu = Xn[0].float().cpu()
        Xaug_cpu = torch.cat([Xn_cpu, torch.ones(T, 1)], dim=1)
        seqs.append(CircuitSeq(pi, text, input_ids[0].detach().cpu().tolist(), Xn_cpu, Xaug_cpu, q[0, head_idx].float().cpu(), k[0, kv_idx].float().cpu(), Qh.cpu(), Kh.cpu(), Vh.cpu(), A.cpu(), Y.cpu(), cos2.float().cpu(), sin2.float().cpu()))
    return seqs, meta


def build_rope_by_pos(seqs: List[CircuitSeq], max_pos: int) -> Dict[int, torch.Tensor]:
    pos_cos: Dict[int, torch.Tensor] = {}; pos_sin: Dict[int, torch.Tensor] = {}
    for s in seqs:
        for p in range(min(int(s.cos.shape[0]), max_pos + 1)):
            if p not in pos_cos:
                pos_cos[p] = s.cos[p]; pos_sin[p] = s.sin[p]
    return {p: rope_col_matrix(pos_cos[p], pos_sin[p]).cpu() for p in sorted(pos_cos)}


def qk_delta_matrices_affine(weights: Dict[str, torch.Tensor], Rpos: Dict[int, torch.Tensor], max_delta: int, head_dim: int) -> Dict[int, torch.Tensor]:
    Wq, Wk = weights["Wq_aug"], weights["Wk_aug"]
    out: Dict[int, torch.Tensor] = {}
    if 0 not in Rpos: return out
    R0 = Rpos[0]
    for d in range(max_delta + 1):
        if d not in Rpos: continue
        # For d=i-j choose pair i=d, j=0 => R_i.T @ R_j = Rpos[d].T @ Rpos[0]
        Rrel = Rpos[d].T @ R0
        out[d] = (Wq.T @ Rrel @ Wk) / math.sqrt(head_dim)
    return out


def verify_targets(seqs: List[CircuitSeq], weights: Dict[str, torch.Tensor], Mdelta: Dict[int, torch.Tensor], max_delta: int):
    Cvo = weights["Wo"] @ weights["Wv_aug"]
    acc = {k: 0.0 for k in ["score_num", "score_den", "A_num", "A_den", "V_num", "V_den", "Y_num", "Y_den"]}
    rows = []
    for s in seqs:
        T = int(s.Xn.shape[0]); D = int(s.Q.shape[-1])
        scores_true = (s.Q @ s.K.T) / math.sqrt(D)
        scores_hat = torch.zeros_like(scores_true)
        mask_eval = torch.zeros_like(scores_true, dtype=torch.bool)
        for i in range(T):
            for j in range(i + 1):
                d = i - j
                if d <= max_delta and d in Mdelta:
                    scores_hat[i, j] = s.Xaug[i] @ Mdelta[d] @ s.Xaug[j]
                    mask_eval[i, j] = True
        if int(mask_eval.sum()) > 0:
            diff = (scores_hat[mask_eval] - scores_true[mask_eval]).float()
            acc["score_num"] += float((diff * diff).sum()); acc["score_den"] += float((scores_true[mask_eval].float() ** 2).sum())
        Ahat = causal_softmax(scores_hat.masked_fill(~mask_eval, torch.finfo(scores_hat.dtype).min))
        row_covered = torch.tensor([all(((i-j) <= max_delta and (i-j) in Mdelta) for j in range(i+1)) for i in range(T)], dtype=torch.bool)
        if int(row_covered.sum()) > 0:
            da = (Ahat[row_covered] - s.A[row_covered]).float()
            acc["A_num"] += float((da * da).sum()); acc["A_den"] += float((s.A[row_covered].float() ** 2).sum())
        V_hat = s.Xaug @ weights["Wv_aug"].T
        Y_hat = s.A @ (s.Xaug @ Cvo.T)
        for key, pred, true in [("V", V_hat, s.V), ("Y", Y_hat, s.Y)]:
            diff = (pred - true).float()
            acc[f"{key}_num"] += float((diff * diff).sum()); acc[f"{key}_den"] += float((true.float() * true.float()).sum())
        rows.append({"prompt_id": s.prompt_id, "T": T, "text": s.text[:120], "score_rel": rel_err(scores_hat[mask_eval], scores_true[mask_eval]) if int(mask_eval.sum()) else None, "Y_rel": rel_err(Y_hat, s.Y)})
    summary = {
        "score_rel": math.sqrt(acc["score_num"] / max(1e-12, acc["score_den"])),
        "A_rel": math.sqrt(acc["A_num"] / max(1e-12, acc["A_den"])) if acc["A_den"] else None,
        "V_rel": math.sqrt(acc["V_num"] / max(1e-12, acc["V_den"])),
        "Y_rel": math.sqrt(acc["Y_num"] / max(1e-12, acc["Y_den"])),
    }
    return summary, rows


def parse_ints(s: str) -> List[int]:
    out = []
    for part in s.split(','):
        part = part.strip()
        if not part: continue
        out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="23")
    ap.add_argument("--heads", default="0,1")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--out", default="runs/qwen_circuit_target_roundtrip_v1")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds) if args.thresholds else load_thresholds(None)
    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=None, attn_implementation=args.attn_implementation, trust_remote_code=True).to(args.device)
    model.eval()

    prompts = build_prompts(args.prompts)
    layer_ids = parse_ints(args.layers); head_ids = parse_ints(args.heads)
    per_head = []
    all_prog_rows = []
    for li in layer_ids:
        for hi in head_ids:
            seqs, meta = collect_head_data(model, tok, prompts, li, hi, args.max_length, args.device)
            if not seqs:
                continue
            weights = build_weight_slices(model, li, hi, meta)
            max_pos = min(args.max_delta, max(int(s.Xn.shape[0]) - 1 for s in seqs))
            Rpos = build_rope_by_pos(seqs, max_pos)
            Mdelta = qk_delta_matrices_affine(weights, Rpos, max_pos, int(meta["head_dim"]))
            exact, rows = verify_targets(seqs, weights, Mdelta, max_pos)
            H = int(meta["hidden_size"])
            qk_prims = build_qk_primitives(H)
            vo_prims = build_vo_primitives(H)
            # Level-0 decode report for C_vo and first few deltas. Expected partial on real Qwen.
            Cvo = weights["Wo"] @ weights["Wv_aug"]
            vo_ops, vo_rec, vo_m = decode_greedy_analytic(Cvo, vo_prims, "vo", thresholds, max_ops=64, device="cpu")
            vo_prog = make_program(f"L{li}H{hi}_vo_level0", "vo", H, vo_ops, Cvo, vo_rec, vo_m, model_name=args.model, layer=li, head=hi)
            write_json(out / f"L{li}H{hi}_vo_program_level0.json", vo_prog)
            qk_decode = []
            for d in sorted(Mdelta.keys())[: min(4, len(Mdelta))]:
                ops, rec, met = decode_greedy_analytic(Mdelta[d], qk_prims, "qk", thresholds, max_ops=64, device="cpu")
                prog = make_program(f"L{li}H{hi}_qk_delta{d}_level0", "qk", H, ops, Mdelta[d], rec, met, model_name=args.model, layer=li, head=hi)
                write_json(out / f"L{li}H{hi}_qk_delta{d}_program_level0.json", prog)
                qk_decode.append({"delta": d, "roundtrip_error": met["roundtrip_error"], "typed_coverage": met["typed_coverage"], "ops_count": met["ops_count"]})
            row = {"layer": li, "head": hi, "kv_head": meta["kv_idx"], "seqs": len(seqs), "max_delta_used": max_pos, **exact, "vo_level0_error": vo_m["roundtrip_error"], "vo_level0_coverage": vo_m["typed_coverage"], "qk_level0": qk_decode}
            per_head.append(row)
            all_prog_rows.extend(rows)
            print(f"L{li}H{hi}: score={exact['score_rel']:.3e} A={exact['A_rel']:.3e} V={exact['V_rel']:.3e} Y={exact['Y_rel']:.3e} | vo_L0={vo_m['roundtrip_error']:.3f}")
    target_closed = all((r["score_rel"] < 5e-3 and r["V_rel"] < 5e-3 and r["Y_rel"] < 5e-3) for r in per_head)
    manifest = {"version": VERSION, "mode": "qwen_circuit_target_roundtrip", "model": args.model, "closure_level": "circuit_target", "target_extraction_closed": target_closed, "level0_decode_expected_partial_on_real_qwen": True, "thresholds_loaded_from": args.thresholds or "DEFAULT_THRESHOLDS", "raw_weight_passthrough_used": False, "kl_distillation_used_as_main_method": False, "alpha_sweep_used_as_main_method": False, "gradient_used_as_main_method": False, "mlp_in_v1_scope": False, "level2_promotion_in_v1_scope": False, "per_head_count": len(per_head)}
    write_json(out / "manifest.json", manifest)
    write_json(out / "per_head_summary.json", per_head)
    write_jsonl(out / "per_prompt_verify.jsonl", all_prog_rows)
    write_json(out / "primitive_dictionary_manifest.json", primitive_manifest(build_qk_primitives(int(model.config.hidden_size)) + build_vo_primitives(int(model.config.hidden_size))))
    print("=== qwen target roundtrip summary ===")
    print(json.dumps(manifest, indent=2))
    if not target_closed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
