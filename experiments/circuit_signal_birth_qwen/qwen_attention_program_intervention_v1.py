#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_attention_program_intervention_v1.py

Last-layer attention program intervention.

This is stronger than offline replay:
  1) run real model and collect hidden before layer L
  2) replace attention output at L with saved QK+VO program output
  3) run the real post-attention MLP of layer L
  4) run final norm + lm_head
  5) compare hidden/logits/top tokens with the original model

Designed first for Qwen2.5-0.5B L23, the final layer.
No training, no KL distillation, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import build_prompts, collect_head_data, get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program, build_program_mdelta
from qwen_attention_block_program_replay_v1 import load_vo_program, build_vo_program_C, program_scores_A_for_seq

VERSION = "qwen_attention_program_intervention_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def model_dtype(model):
    return next(model.parameters()).dtype


@torch.no_grad()
def collect_truth(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str):
    truth = {}
    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        truth[pi] = {
            "text": text,
            "input_ids": input_ids[0].detach().cpu().tolist(),
            "H_before": out.hidden_states[layer_idx][0].detach().float().cpu(),
            "H_layer_true": out.hidden_states[layer_idx + 1][0].detach().float().cpu(),
            "logits_true": out.logits[0].detach().float().cpu(),
        }
    return truth


def build_program_attention_outputs(model, tok, prompts, layer_idx: int, heads: List[int], args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates):
    acc: Dict[int, Dict[str, torch.Tensor]] = {}
    per_head = []
    missing_all = []

    for hi in heads:
        seqs, meta = collect_head_data(model, tok, prompts, layer_idx, hi, args.max_length, args.device)
        if not seqs:
            continue
        Mprog, _Mexact, _delta_rows, qk_missing = build_program_mdelta(
            model, layer_idx, hi, meta, seqs, args, thresholds, qk_atoms, qk_gates
        )
        Cprog, _Cexact, vo_info = build_vo_program_C(model, layer_idx, hi, meta, thresholds, vo_atoms, vo_gates)
        missing_all.extend(qk_missing)
        missing_all.extend(vo_info["missing_atoms"])

        score_rels, A_rels, Y_rels, top1s = [], [], [], []
        for s in seqs:
            _scores_prog, A_prog, _scores_true, row_covered, mask_eval = program_scores_A_for_seq(s, Mprog, args.max_delta)
            payload_prog = s.Xaug @ Cprog.T
            Y_head_prog = A_prog @ payload_prog
            Y_head_true = s.Y

            if int(s.prompt_id) not in acc:
                acc[int(s.prompt_id)] = {
                    "Y_prog": torch.zeros_like(Y_head_prog.float().cpu()),
                    "Y_true": torch.zeros_like(Y_head_true.float().cpu()),
                }
            acc[int(s.prompt_id)]["Y_prog"] += Y_head_prog.float().cpu()
            acc[int(s.prompt_id)]["Y_true"] += Y_head_true.float().cpu()

            if int(mask_eval.sum()) > 0:
                score_rels.append(rel_err(_scores_prog[mask_eval], _scores_true[mask_eval]))
            if int(row_covered.sum()) > 0:
                A_rels.append(rel_err(A_prog[row_covered], s.A[row_covered]))
                top_true = torch.argmax(s.A[row_covered], dim=-1)
                top_prog = torch.argmax(A_prog[row_covered], dim=-1)
                top1s.append(float((top_true == top_prog).float().mean()))
            Y_rels.append(rel_err(Y_head_prog, Y_head_true))

        per_head.append({
            "layer": layer_idx,
            "head": hi,
            "kv_head": int(meta["kv_idx"]),
            "score_rel": mean(score_rels),
            "A_rel": mean(A_rels),
            "Y_head_rel": mean(Y_rels),
            "top1": mean(top1s),
            "vo_matrix_err": vo_info["program_matrix_err"],
        })
        print(f"L{layer_idx}H{hi}: score={per_head[-1]['score_rel']:.3e} A={per_head[-1]['A_rel']:.3e} Yh={per_head[-1]['Y_head_rel']:.3e} top1={per_head[-1]['top1']:.3f}")

    return acc, per_head, missing_all


@torch.no_grad()
def run_last_layer_intervention(model, truth: Dict[int, Dict[str, Any]], y_acc: Dict[int, Dict[str, torch.Tensor]], layer_idx: int, device: str):
    layer = model.model.layers[layer_idx]
    dtype = model_dtype(model)
    rows = []

    for pi, t in sorted(truth.items()):
        H_before = t["H_before"].to(device=device, dtype=dtype).unsqueeze(0)
        Y_prog = y_acc[pi]["Y_prog"].to(device=device, dtype=dtype).unsqueeze(0)
        Y_true_sum = y_acc[pi]["Y_true"].float().cpu()

        H_attn_prog = H_before + Y_prog
        mlp_in = layer.post_attention_layernorm(H_attn_prog)
        H_layer_prog = H_attn_prog + layer.mlp(mlp_in)

        H_final_prog = model.model.norm(H_layer_prog)
        logits_prog = model.lm_head(H_final_prog)[0].detach().float().cpu()

        H_layer_true = t["H_layer_true"]
        logits_true = t["logits_true"]

        # Last-token distribution metrics.
        lt = logits_true[-1]
        lp = logits_prog[-1]
        logp_true = F.log_softmax(lt, dim=-1)
        logp_prog = F.log_softmax(lp, dim=-1)
        p_true = torch.exp(logp_true)
        kl_true_prog = float(torch.sum(p_true * (logp_true - logp_prog)))
        top1_true = int(torch.argmax(lt).item())
        top1_prog = int(torch.argmax(lp).item())
        top5_true = set(torch.topk(lt, k=5).indices.tolist())
        top5_prog = set(torch.topk(lp, k=5).indices.tolist())

        rows.append({
            "prompt_id": pi,
            "text": t["text"][:120],
            "T": int(H_layer_true.shape[0]),
            "Y_all_rel": rel_err(y_acc[pi]["Y_prog"], Y_true_sum),
            "H_attn_after_rel": rel_err(t["H_before"] + y_acc[pi]["Y_prog"], t["H_before"] + Y_true_sum),
            "H_layer_after_mlp_rel": rel_err(H_layer_prog[0].detach().float().cpu(), H_layer_true),
            "logits_rel": rel_err(logits_prog, logits_true),
            "last_token_KL_true_to_prog": kl_true_prog,
            "last_top1_match": bool(top1_true == top1_prog),
            "last_top5_overlap": len(top5_true & top5_prog),
            "top1_true_id": top1_true,
            "top1_prog_id": top1_prog,
        })

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--qk-program-run", required=True)
    ap.add_argument("--vo-program-run", required=True)
    ap.add_argument("--h-tol", type=float, default=1e-3)
    ap.add_argument("--logits-tol", type=float, default=1e-2)
    ap.add_argument("--kl-tol", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/qwen_attention_program_intervention_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    qk_atoms, qk_gates = load_qk_program(Path(args.qk_program_run))
    vo_atoms, vo_gates = load_vo_program(Path(args.vo_program_run))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    n_layers = len(model.model.layers)
    if args.layer != n_layers - 1:
        raise SystemExit(f"v1 supports only final layer intervention. Got layer={args.layer}, final={n_layers-1}")

    prompts = build_prompts(args.prompts)
    heads = parse_ints(args.heads)

    truth = collect_truth(model, tok, prompts, args.layer, args.max_length, args.device)
    y_acc, per_head, missing = build_program_attention_outputs(
        model, tok, prompts, args.layer, heads, args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates
    )
    rows = run_last_layer_intervention(model, truth, y_acc, args.layer, args.device)

    h_mean = mean([r["H_layer_after_mlp_rel"] for r in rows])
    h_max = max([r["H_layer_after_mlp_rel"] for r in rows])
    logits_mean = mean([r["logits_rel"] for r in rows])
    logits_max = max([r["logits_rel"] for r in rows])
    kl_mean = mean([r["last_token_KL_true_to_prog"] for r in rows])
    kl_max = max([r["last_token_KL_true_to_prog"] for r in rows])
    top1_rate = mean([1.0 if r["last_top1_match"] else 0.0 for r in rows])

    report = {
        "version": VERSION,
        "mode": "last_layer_attention_program_intervention",
        "model": args.model,
        "layer": args.layer,
        "qk_program_run": args.qk_program_run,
        "vo_program_run": args.vo_program_run,
        "qk_atom_count": len(qk_atoms),
        "vo_atom_count": len(vo_atoms),
        "prompts": len(rows),
        "heads": len(per_head),
        "score_rel_mean": mean([r["score_rel"] for r in per_head]),
        "A_rel_mean": mean([r["A_rel"] for r in per_head]),
        "Y_head_rel_mean": mean([r["Y_head_rel"] for r in per_head]),
        "Y_all_rel_mean": mean([r["Y_all_rel"] for r in rows]),
        "H_attn_after_rel_mean": mean([r["H_attn_after_rel"] for r in rows]),
        "H_layer_after_mlp_rel_mean": h_mean,
        "H_layer_after_mlp_rel_max": h_max,
        "logits_rel_mean": logits_mean,
        "logits_rel_max": logits_max,
        "last_token_KL_mean": kl_mean,
        "last_token_KL_max": kl_max,
        "last_top1_match_rate": top1_rate,
        "h_tol": args.h_tol,
        "logits_tol": args.logits_tol,
        "kl_tol": args.kl_tol,
        "hidden_closed": h_mean <= args.h_tol,
        "logits_closed": logits_mean <= args.logits_tol,
        "kl_closed": kl_mean <= args.kl_tol,
        "status": "ATTENTION_PROGRAM_INTERVENTION_CLOSED" if (h_mean <= args.h_tol and logits_mean <= args.logits_tol and top1_rate >= 1.0) else "ATTENTION_PROGRAM_INTERVENTION_PARTIAL",
        "no_training": True,
        "closure_level": "last_layer_attention_intervention_logits",
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "universal_claim_allowed": False,
        "missing_atom_count": len(set(missing)),
        "missing_atoms": sorted(set(missing))[:50],
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_head_intervention_attention.jsonl", per_head)
    write_jsonl(out / "per_prompt_intervention_logits.jsonl", rows)

    print("=== Qwen Attention Program Intervention ===")
    print(json.dumps({
        "qk_atom_count": report["qk_atom_count"],
        "vo_atom_count": report["vo_atom_count"],
        "prompts": report["prompts"],
        "heads": report["heads"],
        "Y_all_rel_mean": report["Y_all_rel_mean"],
        "H_attn_after_rel_mean": report["H_attn_after_rel_mean"],
        "H_layer_after_mlp_rel_mean": report["H_layer_after_mlp_rel_mean"],
        "logits_rel_mean": report["logits_rel_mean"],
        "last_token_KL_mean": report["last_token_KL_mean"],
        "last_top1_match_rate": report["last_top1_match_rate"],
        "status": report["status"],
        "missing_atom_count": report["missing_atom_count"],
    }, indent=2))
    print(f"out={out}")

    if report["status"] != "ATTENTION_PROGRAM_INTERVENTION_CLOSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
