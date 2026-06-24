#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_mlp_delta_scan_v1.py

Fast no-training Coder -> Instruct final-layer MLP delta scan.

Purpose:
  Compare L23 attention-VO transfer against L23 MLP/SwiGLU transfer.

This is an output-level causal probe, not a final weight rewrite:
  H_layer_student_patched = H_layer_student + alpha * (MLP_out_teacher - MLP_out_student)
  logits_patched = lm_head(final_norm(H_layer_student_patched))

Why output-level first:
  We need a fast answer to: is the code-specific signal cleaner in MLP than in
  L23 attention VO? If yes, then implement weight/program-level MLP transplant.

No model training, no distillation, no LoRA.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
from qwen_teacher_student_code_transfer_v2 import select_prompts

VERSION = "qwen_teacher_student_mlp_delta_scan_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def crop_logits(a: torch.Tensor, b: torch.Tensor):
    T = min(a.shape[0], b.shape[0])
    V = min(a.shape[-1], b.shape[-1])
    return a[-T:, :V].float(), b[-T:, :V].float()


def logits_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    """Metrics from distribution a to b, robust to small length/vocab mismatch."""
    aa, bb = crop_logits(a, b)
    lt_a = aa[-1]
    lt_b = bb[-1]
    logp_a = F.log_softmax(lt_a, dim=-1)
    logp_b = F.log_softmax(lt_b, dim=-1)
    p_a = torch.exp(logp_a)
    kl = float(torch.sum(p_a * (logp_a - logp_b)))
    top1_a = int(torch.argmax(lt_a).item())
    top1_b = int(torch.argmax(lt_b).item())
    top5_a = set(torch.topk(lt_a, k=5).indices.tolist())
    top5_b = set(torch.topk(lt_b, k=5).indices.tolist())
    return {
        "logits_rel": rel_err(aa, bb),
        "last_token_KL_a_to_b": kl,
        "last_top1_match": bool(top1_a == top1_b),
        "last_top5_overlap": len(top5_a & top5_b),
        "top1_a_id": top1_a,
        "top1_b_id": top1_b,
    }


@torch.no_grad()
def collect_mlp_trace(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str):
    layer = model.model.layers[layer_idx]
    traces: Dict[int, Dict[str, Any]] = {}

    for pi, text in enumerate(prompts):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        captured: Dict[str, torch.Tensor] = {}

        def mlp_hook(_mod, _inp, out):
            captured["mlp_out"] = out.detach()[0].float().cpu()

        def layer_hook(_mod, _inp, out):
            # Qwen decoder layer returns tuple(hidden_states, ...)
            hs = out[0] if isinstance(out, (tuple, list)) else out
            captured["layer_out_pre_final_norm"] = hs.detach()[0].float().cpu()

        h1 = layer.mlp.register_forward_hook(mlp_hook)
        h2 = layer.register_forward_hook(layer_hook)
        try:
            outputs = model(**encoded, use_cache=False, output_hidden_states=False)
        finally:
            h1.remove()
            h2.remove()

        if "mlp_out" not in captured or "layer_out_pre_final_norm" not in captured:
            raise RuntimeError("failed to capture MLP/layer output hooks")

        traces[pi] = {
            "text": text,
            "input_len": int(encoded["input_ids"].shape[1]),
            "logits": outputs.logits.detach()[0].float().cpu(),
            "mlp_out": captured["mlp_out"],
            "layer_out_pre_final_norm": captured["layer_out_pre_final_norm"],
        }
    return traces


@torch.no_grad()
def patch_student_logits(student, student_trace: Dict[str, Any], teacher_trace: Dict[str, Any], alpha: float, device: str):
    T = min(student_trace["layer_out_pre_final_norm"].shape[0], teacher_trace["mlp_out"].shape[0], student_trace["mlp_out"].shape[0])
    dtype = next(student.parameters()).dtype

    Hs = student_trace["layer_out_pre_final_norm"][-T:].to(device=device, dtype=dtype)
    Ms = student_trace["mlp_out"][-T:].float()
    Mt = teacher_trace["mlp_out"][-T:].float()
    delta = (Mt - Ms).to(device=device, dtype=dtype)
    H_patch = Hs + float(alpha) * delta
    H_final = student.model.norm(H_patch.unsqueeze(0))
    logits = student.lm_head(H_final)[0].detach().float().cpu()

    return logits, {
        "T": int(T),
        "mlp_delta_rel": rel_err(Mt, Ms),
        "applied_delta_rel_to_student_layer": rel_err((float(alpha) * (Mt - Ms)), student_trace["layer_out_pre_final_norm"][-T:]),
        "student_mlp_norm": float(torch.linalg.norm(Ms.float()).item()),
        "teacher_mlp_norm": float(torch.linalg.norm(Mt.float()).item()),
        "delta_norm": float(torch.linalg.norm((Mt - Ms).float()).item()),
    }


def summarize(rows: List[Dict[str, Any]]):
    before_rel = mean([r["teacher_student_before_logits_rel"] for r in rows])
    after_rel = mean([r["teacher_student_after_logits_rel"] for r in rows])
    before_kl = mean([r["teacher_student_before_KL"] for r in rows])
    after_kl = mean([r["teacher_student_after_KL"] for r in rows])
    return {
        "before_logits_rel": before_rel,
        "after_logits_rel": after_rel,
        "logits_improvement": before_rel - after_rel,
        "logits_improvement_pct_of_before": (before_rel - after_rel) / max(1e-12, before_rel),
        "before_KL": before_kl,
        "after_KL": after_kl,
        "KL_improvement": before_kl - after_kl,
        "KL_improvement_pct_of_before": (before_kl - after_kl) / max(1e-12, before_kl),
        "student_shift_logits_rel": mean([r["student_before_after_logits_rel"] for r in rows]),
        "student_shift_KL": mean([r["student_before_after_KL"] for r in rows]),
        "mlp_delta_rel_mean": mean([r["mlp_delta_rel"] for r in rows]),
        "applied_delta_rel_to_student_layer_mean": mean([r["applied_delta_rel_to_student_layer"] for r in rows]),
        "after_top1_match_rate": mean([1.0 if r["teacher_student_after_top1_match"] else 0.0 for r in rows]),
    }


def score_candidate(sums: Dict[str, float], kl_weight: float, shift_weight: float, delta_weight: float):
    kl_damage = max(0.0, -float(sums["KL_improvement"]))
    return float(sums["logits_improvement"]) - kl_weight * kl_damage - shift_weight * float(sums["student_shift_logits_rel"]) - delta_weight * float(sums["applied_delta_rel_to_student_layer_mean"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-mode", default="code", choices=["code", "retain", "mixed"])
    ap.add_argument("--alphas", default="0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--kl-weight", type=float, default=0.25)
    ap.add_argument("--shift-weight", type=float, default=0.05)
    ap.add_argument("--delta-weight", type=float, default=0.02)
    ap.add_argument("--base-sanity-eps", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/qwen_teacher_student_mlp_delta_scan_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    prompts = select_prompts(args.prompt_mode, args.prompts)
    alphas = parse_floats(args.alphas)

    tok_teacher = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    tok_student = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    teacher.eval()
    teacher_trace = collect_mlp_trace(teacher, tok_teacher, prompts, args.layer, args.max_length, args.device)
    del teacher
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    student.eval()
    student_trace = collect_mlp_trace(student, tok_student, prompts, args.layer, args.max_length, args.device)

    # sanity: reconstruct student final logits from captured layer_out via final norm + lm_head.
    sanity_rows = []
    for pi in sorted(student_trace.keys()):
        H = student_trace[pi]["layer_out_pre_final_norm"].to(args.device, dtype=next(student.parameters()).dtype)
        logits_re = student.lm_head(student.model.norm(H.unsqueeze(0)))[0].detach().float().cpu()
        shift = logits_metrics(student_trace[pi]["logits"], logits_re)
        sanity_rows.append({
            "prompt_id": pi,
            "student_recompute_logits_rel": shift["logits_rel"],
            "student_recompute_KL": shift["last_token_KL_a_to_b"],
        })
    sanity_shift = mean([r["student_recompute_logits_rel"] for r in sanity_rows])
    sanity_pass = sanity_shift <= args.base_sanity_eps
    print(f"base_recompute: shift={sanity_shift:.6e} sanity={sanity_pass}")

    summary_rows = []
    per_prompt_all = []

    for alpha in alphas:
        rows = []
        for pi in sorted(student_trace.keys()):
            logits_patch, patch_info = patch_student_logits(student, student_trace[pi], teacher_trace[pi], alpha, args.device)
            before = logits_metrics(teacher_trace[pi]["logits"], student_trace[pi]["logits"])
            after = logits_metrics(teacher_trace[pi]["logits"], logits_patch)
            shift = logits_metrics(student_trace[pi]["logits"], logits_patch)
            row = {
                "prompt_id": pi,
                "prompt_mode": args.prompt_mode,
                "alpha": float(alpha),
                "text": student_trace[pi]["text"][:180],
                "teacher_student_before_logits_rel": before["logits_rel"],
                "teacher_student_after_logits_rel": after["logits_rel"],
                "teacher_student_logits_rel_improvement": before["logits_rel"] - after["logits_rel"],
                "teacher_student_before_KL": before["last_token_KL_a_to_b"],
                "teacher_student_after_KL": after["last_token_KL_a_to_b"],
                "teacher_student_KL_improvement": before["last_token_KL_a_to_b"] - after["last_token_KL_a_to_b"],
                "teacher_student_before_top1_match": before["last_top1_match"],
                "teacher_student_after_top1_match": after["last_top1_match"],
                "student_before_after_logits_rel": shift["logits_rel"],
                "student_before_after_KL": shift["last_token_KL_a_to_b"],
                **patch_info,
            }
            rows.append(row)
            per_prompt_all.append(row)
        sums = summarize(rows)
        score = score_candidate(sums, args.kl_weight, args.shift_weight, args.delta_weight)
        item = {
            "candidate": f"MLP_a{alpha:g}",
            "alpha": float(alpha),
            "prompt_mode": args.prompt_mode,
            **sums,
            "selection_score": score,
            "useful_logits": sums["logits_improvement"] > 0,
            "useful_KL": sums["KL_improvement"] > 0,
        }
        summary_rows.append(item)
        print(
            f"MLP_a{alpha:g}: logit_gain={sums['logits_improvement']:+.5f} "
            f"KL_gain={sums['KL_improvement']:+.5f} "
            f"shift={sums['student_shift_logits_rel']:.5f} "
            f"delta={sums['applied_delta_rel_to_student_layer_mean']:.5f} "
            f"score={score:+.5f}"
        )

    summary_rows = sorted(summary_rows, key=lambda r: r["selection_score"], reverse=True)
    report = {
        "version": VERSION,
        "mode": "final_layer_mlp_output_delta_transfer_scan",
        "prompt_mode": args.prompt_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "prompts": len(prompts),
        "alphas": alphas,
        "base_recompute_sanity": {
            "student_recompute_logits_rel_mean": sanity_shift,
            "pass": sanity_pass,
            "eps": args.base_sanity_eps,
        },
        "best_candidates": summary_rows,
        "status": "MLP_DELTA_SCAN_RAN" if sanity_pass else "BASELINE_SANITY_FAILED",
        "no_training": True,
        "closure_level": "output_level_mlp_delta_causal_probe",
        "note": "This is not yet a reusable weight/program patch. It tests whether MLP output delta is a cleaner causal signal than L23 attention VO delta.",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_mlp_delta_scan.jsonl", summary_rows)
    write_jsonl(out / "per_prompt_mlp_delta_scan.jsonl", per_prompt_all)
    write_jsonl(out / "student_recompute_sanity.jsonl", sanity_rows)

    print("=== Qwen Teacher -> Student MLP Delta Scan v1 ===")
    print(json.dumps({
        "prompt_mode": report["prompt_mode"],
        "teacher_model": report["teacher_model"],
        "student_model": report["student_model"],
        "layer": report["layer"],
        "prompts": report["prompts"],
        "base_recompute_sanity": report["base_recompute_sanity"],
        "top": [{
            "candidate": r["candidate"],
            "logits_improvement": r["logits_improvement"],
            "KL_improvement": r["KL_improvement"],
            "student_shift_logits_rel": r["student_shift_logits_rel"],
            "applied_delta_rel_to_student_layer_mean": r["applied_delta_rel_to_student_layer_mean"],
            "selection_score": r["selection_score"],
        } for r in summary_rows[:10]],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")

    if not sanity_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
