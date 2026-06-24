#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_head_pseudocode_term_transfer_scan_v1_1.py

FIXED evaluator for term-level pseudocode transfer scan on arbitrary layers.

v1 used qwen_teacher_student_attention_transfer_v1.run_student_final_layer_with_program,
which is explicitly final-layer only. For L6/L12/L18 it skipped all subsequent
layers, so base_no_patch already shifted logits by ~1.4. That invalidated only
those term-scan outputs, not the expensive QK/VO autoexpand program-runs.

v1.1 patches the actual layer.self_attn output with a forward hook and then lets
the real model run through the layer MLP and all later layers. Therefore
base_no_patch must be near zero for any layer.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program
from qwen_teacher_student_code_transfer_v2 import select_prompts
import qwen_teacher_student_attention_transfer_v1 as ts
import qwen_teacher_student_delta_head_scan_v4 as v4
from qwen_head_pseudocode_term_transfer_scan_v1 import (
    parse_floats,
    parse_delta_range,
    parse_candidates,
    expand_spec,
    build_y_acc_for_term_patch,
    candidate_score,
    sanity_check_base,
)

VERSION = "qwen_head_pseudocode_term_transfer_scan_v1.1"


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


@torch.no_grad()
def run_student_with_attention_output_patch(student, tokenizer, prompts: List[str], y_acc: Dict[int, Dict[str, torch.Tensor]], layer_idx: int, max_length: int, device: str):
    """Patch layer.self_attn output for one prompt, then run the full model.

    This works for non-final layers because the whole model forward continues
    through the layer MLP and all later layers.
    """
    dtype = next(student.parameters()).dtype
    layer = student.model.layers[layer_idx]
    out: Dict[int, Dict[str, torch.Tensor]] = {}

    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        if pi not in y_acc:
            raise RuntimeError(f"missing y_acc for prompt_id={pi}")
        Y_prog_cpu = y_acc[pi]["Y_prog"].float().cpu()
        Y_student_cpu = y_acc[pi]["Y_student_true"].float().cpu()

        def hook(_module, _inputs, output):
            # Qwen self_attn returns a tuple with attn_output as first item.
            if isinstance(output, tuple):
                attn_out = output[0]
                Y = Y_prog_cpu.to(device=device, dtype=attn_out.dtype).unsqueeze(0)
                if Y.shape[1] != attn_out.shape[1] or Y.shape[2] != attn_out.shape[2]:
                    raise RuntimeError(f"Y_prog shape {tuple(Y.shape)} does not match attn_out {tuple(attn_out.shape)}")
                return (Y,) + tuple(output[1:])
            Y = Y_prog_cpu.to(device=device, dtype=output.dtype).unsqueeze(0)
            if tuple(Y.shape) != tuple(output.shape):
                raise RuntimeError(f"Y_prog shape {tuple(Y.shape)} does not match attn_out {tuple(output.shape)}")
            return Y

        handle = layer.self_attn.register_forward_hook(hook)
        try:
            res = student(**enc, use_cache=False)
        finally:
            handle.remove()
        out[pi] = {
            "logits_prog": res.logits[0].detach().float().cpu(),
            "Y_prog": Y_prog_cpu,
            "Y_student_true": Y_student_cpu,
        }
    return out


@torch.no_grad()
def evaluate_patch_full_forward(student, tokenizer, prompts, student_truth, teacher_truth, y_acc, layer_idx: int, max_length: int, device: str):
    student_prog = run_student_with_attention_output_patch(student, tokenizer, prompts, y_acc, layer_idx, max_length, device)
    rows = []
    for pi in sorted(student_truth.keys()):
        teacher_logits = teacher_truth[pi]["logits_true"]
        student_logits = student_truth[pi]["logits_true"]
        prog_logits = student_prog[pi]["logits_prog"]
        before = ts.logits_metrics(teacher_logits, student_logits)
        after = ts.logits_metrics(teacher_logits, prog_logits)
        shift = ts.logits_metrics(student_logits, prog_logits)
        rows.append({
            "prompt_id": pi,
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
            "Y_prog_vs_student_attention_rel": rel_err(student_prog[pi]["Y_prog"], student_prog[pi]["Y_student_true"]),
        })
    return rows


def summarize_rows(rows):
    return {
        "before_logits_rel": mean([r["teacher_student_before_logits_rel"] for r in rows]),
        "after_logits_rel": mean([r["teacher_student_after_logits_rel"] for r in rows]),
        "logits_improvement": mean([r["teacher_student_logits_rel_improvement"] for r in rows]),
        "before_KL": mean([r["teacher_student_before_KL"] for r in rows]),
        "after_KL": mean([r["teacher_student_after_KL"] for r in rows]),
        "KL_improvement": mean([r["teacher_student_KL_improvement"] for r in rows]),
        "student_shift_logits_rel": mean([r["student_before_after_logits_rel"] for r in rows]),
        "student_shift_KL": mean([r["student_before_after_KL"] for r in rows]),
        "Y_prog_vs_student_attention_rel": mean([r["Y_prog_vs_student_attention_rel"] for r in rows]),
        "after_top1_match_rate": mean([1.0 if r["teacher_student_after_top1_match"] else 0.0 for r in rows]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--all-heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--candidate-heads", default="1")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--medium-deltas", default="7:13")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-mode", default="code", choices=["code", "retain", "mixed"])
    ap.add_argument("--alphas", default="0.025,0.05,0.075,0.1")
    ap.add_argument("--term-candidates", default="vo_linear;vo_bias;q_affine_medium;k_affine_medium;content_medium;const_medium;qk_affine_medium;qk_cond_medium;vo_linear+qk_affine_medium;vo_linear+qk_cond_medium")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--teacher-qk-program-run", required=True)
    ap.add_argument("--teacher-vo-program-run", required=True)
    ap.add_argument("--student-qk-program-run", required=True)
    ap.add_argument("--student-vo-program-run", required=True)
    ap.add_argument("--include-all-candidates", action="store_true")
    ap.add_argument("--kl-weight", type=float, default=0.25)
    ap.add_argument("--shift-weight", type=float, default=0.05)
    ap.add_argument("--y-weight", type=float, default=0.02)
    ap.add_argument("--base-sanity-eps", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/head_pseudocode_term_transfer_scan_v1_1")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds)
    tqk_atoms, tqk_gates = load_qk_program(Path(args.teacher_qk_program_run))
    tvo_atoms, tvo_gates = load_vo_program(Path(args.teacher_vo_program_run))
    sqk_atoms, sqk_gates = load_qk_program(Path(args.student_qk_program_run))
    svo_atoms, svo_gates = load_vo_program(Path(args.student_vo_program_run))

    all_heads = parse_ints(args.all_heads)
    candidate_heads = parse_ints(args.candidate_heads)
    alphas = parse_floats(args.alphas)
    term_candidates = parse_candidates(args.term_candidates)
    medium_deltas = parse_delta_range(args.medium_deltas)
    prompts = select_prompts(args.prompt_mode, args.prompts)

    bad = sorted(set(candidate_heads) - set(all_heads))
    if bad:
        raise SystemExit(f"candidate heads not included in all-head set: {bad}")

    tok_teacher = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    tok_student = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=get_dtype(args.dtype), device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True,
    ).to(args.device).eval()
    teacher_truth = ts.collect_model_truth(teacher, tok_teacher, prompts, args.layer, args.max_length, args.device)
    compiled_teacher, teacher_compile_rows, missing_teacher = ts.compile_teacher_attention_program(
        teacher, tok_teacher, prompts, args.layer, all_heads, args, thresholds, tqk_atoms, tqk_gates, tvo_atoms, tvo_gates
    )
    del teacher; gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=get_dtype(args.dtype), device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True,
    ).to(args.device).eval()
    student_truth = ts.collect_model_truth(student, tok_student, prompts, args.layer, args.max_length, args.device)
    compiled_student, student_compile_rows, missing_student = ts.compile_teacher_attention_program(
        student, tok_student, prompts, args.layer, all_heads, args, thresholds, sqk_atoms, sqk_gates, svo_atoms, svo_gates
    )
    student_head_data = v4.precompute_student_heads(student, tok_student, prompts, args.layer, all_heads, args)

    base_y_acc, _base_head_rows, _base_missing = build_y_acc_for_term_patch(
        student_head_data, compiled_student, compiled_teacher, set(), 0.0, "vo_linear", medium_deltas, args
    )
    base_rows = evaluate_patch_full_forward(student, tok_student, prompts, student_truth, teacher_truth, base_y_acc, args.layer, args.max_length, args.device)
    base_sums = summarize_rows(base_rows)
    base_sanity = sanity_check_base(base_sums, eps=args.base_sanity_eps)
    print(f"base_no_patch: shift={base_sums['student_shift_logits_rel']:.6e} Y={base_sums['Y_prog_vs_student_attention_rel']:.6e} sanity={base_sanity['base_sanity_pass']}")
    if not base_sanity["base_sanity_pass"]:
        print("WARNING: baseline failed; candidate results will be marked invalid")

    candidates = []
    for alpha in alphas:
        for spec in term_candidates:
            for h in candidate_heads:
                candidates.append((f"H{h}_{spec}_a{alpha:g}", {h}, spec, float(alpha)))
            if args.include_all_candidates:
                candidates.append((f"CANDIDATE_ALL_{spec}_a{alpha:g}", set(candidate_heads), spec, float(alpha)))

    summary_rows = []
    per_prompt_all = []
    per_head_metrics_all = []
    for name, patch_set, spec, alpha in candidates:
        y_acc, patched_head_rows, missing = build_y_acc_for_term_patch(
            student_head_data, compiled_student, compiled_teacher, patch_set, alpha, spec, medium_deltas, args
        )
        rows = evaluate_patch_full_forward(student, tok_student, prompts, student_truth, teacher_truth, y_acc, args.layer, args.max_length, args.device)
        sums = summarize_rows(rows)
        score = candidate_score(sums, args.kl_weight, args.shift_weight, args.y_weight)
        terms, delta_mode = expand_spec(spec)
        item = {
            "candidate": name,
            "patch_heads": sorted(list(patch_set)),
            "alpha": float(alpha),
            "candidate_spec": spec,
            "terms": sorted(terms),
            "delta_mode": delta_mode,
            "medium_deltas": sorted(medium_deltas),
            "prompt_mode": args.prompt_mode,
            **sums,
            "selection_score": score,
            "useful_logits": sums["logits_improvement"] > 0,
            "useful_KL": sums["KL_improvement"] > 0,
            "missing_heads": missing,
        }
        summary_rows.append(item)
        for r in rows:
            r.update({"candidate": name, "candidate_spec": spec, "patch_heads": sorted(list(patch_set)), "alpha": float(alpha), "prompt_mode": args.prompt_mode})
            per_prompt_all.append(r)
        for r in patched_head_rows:
            r.update({"candidate": name, "prompt_mode": args.prompt_mode})
            per_head_metrics_all.append(r)
        print(
            f"{name}: logit_gain={sums['logits_improvement']:+.5f} "
            f"KL_gain={sums['KL_improvement']:+.5f} "
            f"shift={sums['student_shift_logits_rel']:.5f} "
            f"Y={sums['Y_prog_vs_student_attention_rel']:.5f} "
            f"score={score:+.5f}"
        )

    summary_rows = sorted(summary_rows, key=lambda r: r["selection_score"], reverse=True)
    report = {
        "version": VERSION,
        "mode": "head_pseudocode_term_level_transfer_scan_full_forward_hook",
        "prompt_mode": args.prompt_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "all_heads": all_heads,
        "candidate_heads": candidate_heads,
        "alphas": alphas,
        "term_candidates": term_candidates,
        "medium_deltas": sorted(medium_deltas),
        "prompts": len(prompts),
        "base_no_patch": base_sums,
        "base_sanity": base_sanity,
        "best_candidates": summary_rows[:20],
        "status": "TERM_LEVEL_TRANSFER_SCAN_RAN" if base_sanity["base_sanity_pass"] else "BASELINE_SANITY_FAILED",
        "no_training": True,
        "closure_level": "attention_head_matrix_pseudocode_terms",
        "uses_raw_weight_diff": False,
        "evaluator": "self_attn_forward_hook_full_model_forward",
        "pseudocode_terms": ["content", "q_affine", "k_affine", "const", "vo_linear", "vo_bias"],
        "missing_teacher_atom_count": len(set(missing_teacher)),
        "missing_student_atom_count": len(set(missing_student)),
    }
    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_term_transfer_scan.jsonl", summary_rows)
    write_jsonl(out / "per_prompt_term_transfer_scan.jsonl", per_prompt_all)
    write_jsonl(out / "per_patched_head_term_metrics.jsonl", per_head_metrics_all)
    write_jsonl(out / "per_head_compiled_teacher_program.jsonl", teacher_compile_rows)
    write_jsonl(out / "per_head_compiled_student_program.jsonl", student_compile_rows)

    print("=== Qwen Head Pseudocode Term Transfer Scan v1.1 ===")
    print(json.dumps({
        "prompt_mode": report["prompt_mode"],
        "layer": report["layer"],
        "candidate_heads": report["candidate_heads"],
        "medium_deltas": report["medium_deltas"],
        "base_no_patch": report["base_no_patch"],
        "base_sanity": report["base_sanity"],
        "candidates": len(summary_rows),
        "top10": [{
            "candidate": r["candidate"],
            "spec": r["candidate_spec"],
            "alpha": r["alpha"],
            "logits_improvement": r["logits_improvement"],
            "KL_improvement": r["KL_improvement"],
            "student_shift_logits_rel": r["student_shift_logits_rel"],
            "Y_prog_vs_student_attention_rel": r["Y_prog_vs_student_attention_rel"],
            "selection_score": r["selection_score"],
        } for r in summary_rows[:10]],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
