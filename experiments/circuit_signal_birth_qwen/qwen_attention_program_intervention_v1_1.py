#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_attention_program_intervention_v1_1.py

Fix over v1:
  For the final layer, HuggingFace output_hidden_states[layer+1] is the final
  normalized hidden state, not the raw pre-final-norm layer output.

v1 compared pre-final-norm H_layer_prog against final-norm hidden_states[-1],
so H_layer_after_mlp_rel was artificially huge while logits/KL were closed.

This v1.1 compares:
  H_final_prog = model.model.norm(H_layer_prog)
against:
  H_final_true = outputs.hidden_states[layer+1]

No training, no KL distillation, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import build_prompts, get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program
import qwen_attention_program_intervention_v1 as base

VERSION = "qwen_attention_program_intervention_v1.1-final-norm-hidden-fix"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def model_dtype(model):
    return next(model.parameters()).dtype


@torch.no_grad()
def run_last_layer_intervention_fixed(model, truth: Dict[int, Dict[str, Any]], y_acc: Dict[int, Dict[str, torch.Tensor]], layer_idx: int, device: str):
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
        H_final_prog_cpu = H_final_prog[0].detach().float().cpu()

        # For final layer in HF output_hidden_states, this is already final norm hidden.
        H_final_true = t["H_layer_true"]
        logits_true = t["logits_true"]

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
            "T": int(H_final_true.shape[0]),
            "Y_all_rel": rel_err(y_acc[pi]["Y_prog"], Y_true_sum),
            "H_attn_after_rel": rel_err(t["H_before"] + y_acc[pi]["Y_prog"], t["H_before"] + Y_true_sum),
            "H_final_norm_rel": rel_err(H_final_prog_cpu, H_final_true),
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
    ap.add_argument("--h-final-tol", type=float, default=1e-3)
    ap.add_argument("--logits-tol", type=float, default=1e-2)
    ap.add_argument("--kl-tol", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/qwen_attention_program_intervention_v1_1")
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
        raise SystemExit(f"v1.1 supports only final layer intervention. Got layer={args.layer}, final={n_layers-1}")

    prompts = build_prompts(args.prompts)
    heads = parse_ints(args.heads)

    truth = base.collect_truth(model, tok, prompts, args.layer, args.max_length, args.device)
    y_acc, per_head, missing = base.build_program_attention_outputs(
        model, tok, prompts, args.layer, heads, args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates
    )
    rows = run_last_layer_intervention_fixed(model, truth, y_acc, args.layer, args.device)

    h_final_mean = mean([r["H_final_norm_rel"] for r in rows])
    h_final_max = max([r["H_final_norm_rel"] for r in rows])
    logits_mean = mean([r["logits_rel"] for r in rows])
    logits_max = max([r["logits_rel"] for r in rows])
    kl_mean = mean([r["last_token_KL_true_to_prog"] for r in rows])
    kl_max = max([r["last_token_KL_true_to_prog"] for r in rows])
    top1_rate = mean([1.0 if r["last_top1_match"] else 0.0 for r in rows])

    report = {
        "version": VERSION,
        "mode": "last_layer_attention_program_intervention_final_norm_fixed",
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
        "H_final_norm_rel_mean": h_final_mean,
        "H_final_norm_rel_max": h_final_max,
        "logits_rel_mean": logits_mean,
        "logits_rel_max": logits_max,
        "last_token_KL_mean": kl_mean,
        "last_token_KL_max": kl_max,
        "last_top1_match_rate": top1_rate,
        "h_final_tol": args.h_final_tol,
        "logits_tol": args.logits_tol,
        "kl_tol": args.kl_tol,
        "final_hidden_closed": h_final_mean <= args.h_final_tol,
        "logits_closed": logits_mean <= args.logits_tol,
        "kl_closed": kl_mean <= args.kl_tol,
        "status": "ATTENTION_PROGRAM_INTERVENTION_CLOSED" if (h_final_mean <= args.h_final_tol and logits_mean <= args.logits_tol and top1_rate >= 1.0) else "ATTENTION_PROGRAM_INTERVENTION_PARTIAL",
        "bugfix_note": "v1 H_layer_after_mlp_rel compared pre-final-norm hidden to HF final-norm hidden_states[-1]; v1.1 compares final-norm hidden to final-norm hidden.",
        "no_training": True,
        "closure_level": "last_layer_attention_intervention_final_norm_logits",
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "universal_claim_allowed": False,
        "missing_atom_count": len(set(missing)),
        "missing_atoms": sorted(set(missing))[:50],
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_head_intervention_attention.jsonl", per_head)
    write_jsonl(out / "per_prompt_intervention_logits.jsonl", rows)

    print("=== Qwen Attention Program Intervention v1.1 ===")
    print(json.dumps({
        "qk_atom_count": report["qk_atom_count"],
        "vo_atom_count": report["vo_atom_count"],
        "prompts": report["prompts"],
        "heads": report["heads"],
        "Y_all_rel_mean": report["Y_all_rel_mean"],
        "H_attn_after_rel_mean": report["H_attn_after_rel_mean"],
        "H_final_norm_rel_mean": report["H_final_norm_rel_mean"],
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
