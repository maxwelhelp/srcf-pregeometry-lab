#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_delta_head_scan_v4_1.py

Fix over v4:
  v4 used --heads both as:
    1) heads included in the student forward reconstruction;
    2) candidate heads to patch.

If --heads was a subset, the no-patch baseline silently dropped the other native
student heads. That made base_no_patch shift logits even when Y_prog_vs_student_attention_rel=0.

v4.1 separates:
  --all-heads        full attention heads included in every forward replay
  --candidate-heads  subset scanned for delta patches

Core transfer remains:
  M_patch = M_student + alpha * (M_teacher - M_student)
  C_patch = C_student + alpha * (C_teacher - C_student)

No model training, no distillation, no LoRA.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program
from qwen_teacher_student_code_transfer_v2 import select_prompts
import qwen_teacher_student_attention_transfer_v1 as ts
import qwen_teacher_student_delta_head_scan_v4 as v4

VERSION = "qwen_teacher_student_delta_head_scan_v4.1-all-heads-candidate-heads-fix"


def sanity_check_base(base_sums: dict, eps: float = 1e-5) -> dict:
    """A true no-patch baseline should reproduce native student attention.

    It may still differ from teacher, but student_before_after shift must be ~0
    and Y_prog_vs_student_attention_rel must be ~0.
    """
    y = abs(float(base_sums.get("Y_prog_vs_student_attention_rel", 999.0)))
    shift = abs(float(base_sums.get("student_shift_logits_rel", 999.0)))
    return {
        "base_y_closed": y <= eps,
        "base_student_shift_closed": shift <= eps,
        "base_sanity_pass": (y <= eps and shift <= eps),
        "base_y_eps": eps,
        "base_shift_eps": eps,
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
    ap.add_argument("--candidate-heads", default="0,1,5,6,11,13")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-mode", default="code", choices=["code", "retain", "mixed"])
    ap.add_argument("--alphas", default="0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--transfer-mode", default="both", choices=["both", "qk", "vo"])
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--teacher-qk-program-run", required=True)
    ap.add_argument("--teacher-vo-program-run", required=True)
    ap.add_argument("--student-qk-program-run", required=True)
    ap.add_argument("--student-vo-program-run", required=True)
    ap.add_argument("--include-all-candidates", action="store_true")
    ap.add_argument("--kl-weight", type=float, default=0.25)
    ap.add_argument("--shift-weight", type=float, default=0.05)
    ap.add_argument("--y-weight", type=float, default=0.02)
    ap.add_argument("--base-sanity-eps", type=float, default=1e-5)
    ap.add_argument("--out", default="runs/qwen_teacher_student_delta_head_scan_v4_1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    tqk_atoms, tqk_gates = load_qk_program(Path(args.teacher_qk_program_run))
    tvo_atoms, tvo_gates = load_vo_program(Path(args.teacher_vo_program_run))
    sqk_atoms, sqk_gates = load_qk_program(Path(args.student_qk_program_run))
    svo_atoms, svo_gates = load_vo_program(Path(args.student_vo_program_run))

    all_heads = parse_ints(args.all_heads)
    candidate_heads = parse_ints(args.candidate_heads)
    alphas = v4.parse_floats(args.alphas)
    prompts = select_prompts(args.prompt_mode, args.prompts)

    bad_candidates = sorted(set(candidate_heads) - set(all_heads))
    if bad_candidates:
        raise SystemExit(f"candidate heads not included in all-head forward set: {bad_candidates}")

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
    teacher_truth = ts.collect_model_truth(teacher, tok_teacher, prompts, args.layer, args.max_length, args.device)
    compiled_teacher, teacher_compile_rows, missing_teacher = ts.compile_teacher_attention_program(
        teacher, tok_teacher, prompts, args.layer, all_heads, args, thresholds, tqk_atoms, tqk_gates, tvo_atoms, tvo_gates
    )
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
    student_truth = ts.collect_model_truth(student, tok_student, prompts, args.layer, args.max_length, args.device)
    compiled_student, student_compile_rows, missing_student = ts.compile_teacher_attention_program(
        student, tok_student, prompts, args.layer, all_heads, args, thresholds, sqk_atoms, sqk_gates, svo_atoms, svo_gates
    )
    student_head_data = v4.precompute_student_heads(student, tok_student, prompts, args.layer, all_heads, args)

    base_y_acc, _base_head_rows, _base_missing = v4.build_y_acc_for_delta_patch(
        student_head_data, compiled_student, compiled_teacher, set(), 0.0, args.transfer_mode, args
    )
    base_rows = v4.evaluate_patch(student, student_truth, teacher_truth, base_y_acc, args.layer, args.device)
    base_sums = v4.summarize_rows(base_rows)
    base_sanity = sanity_check_base(base_sums, eps=args.base_sanity_eps)
    print(
        "base_no_patch: "
        f"shift={base_sums['student_shift_logits_rel']:.6e} "
        f"Y={base_sums['Y_prog_vs_student_attention_rel']:.6e} "
        f"sanity={base_sanity['base_sanity_pass']}"
    )

    candidates = []
    for a in alphas:
        for h in candidate_heads:
            candidates.append((f"H{h}_a{a:g}", {h}, float(a)))
        if args.include_all_candidates:
            candidates.append((f"CANDIDATE_ALL_a{a:g}", set(candidate_heads), float(a)))
            candidates.append((f"FULL_ALL_a{a:g}", set(all_heads), float(a)))

    summary_rows = []
    per_prompt_all = []
    per_head_metrics_all = []

    for name, patch_set, alpha in candidates:
        y_acc, patched_head_rows, missing = v4.build_y_acc_for_delta_patch(
            student_head_data, compiled_student, compiled_teacher, patch_set, alpha, args.transfer_mode, args
        )
        rows = v4.evaluate_patch(student, student_truth, teacher_truth, y_acc, args.layer, args.device)
        sums = v4.summarize_rows(rows)
        score = v4.candidate_score(sums, args.kl_weight, args.shift_weight, args.y_weight)
        item = {
            "candidate": name,
            "patch_heads": sorted(list(patch_set)),
            "alpha": float(alpha),
            "transfer_mode": args.transfer_mode,
            "prompt_mode": args.prompt_mode,
            **sums,
            "selection_score": score,
            "useful_logits": sums["logits_improvement"] > 0,
            "useful_KL": sums["KL_improvement"] > 0,
            "missing_heads": missing,
        }
        summary_rows.append(item)
        for r in rows:
            r.update({"candidate": name, "patch_heads": sorted(list(patch_set)), "alpha": float(alpha), "prompt_mode": args.prompt_mode})
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
        "mode": "teacher_student_delta_per_head_transfer_scan_fixed_all_heads",
        "prompt_mode": args.prompt_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "all_heads": all_heads,
        "candidate_heads": candidate_heads,
        "alphas": alphas,
        "transfer_mode": args.transfer_mode,
        "prompts": len(prompts),
        "teacher_qk_program_run": args.teacher_qk_program_run,
        "teacher_vo_program_run": args.teacher_vo_program_run,
        "student_qk_program_run": args.student_qk_program_run,
        "student_vo_program_run": args.student_vo_program_run,
        "teacher_qk_atom_count": len(tqk_atoms),
        "teacher_vo_atom_count": len(tvo_atoms),
        "student_qk_atom_count": len(sqk_atoms),
        "student_vo_atom_count": len(svo_atoms),
        "base_no_patch": base_sums,
        "base_sanity": base_sanity,
        "best_candidates": summary_rows[:15],
        "status": "DELTA_PER_HEAD_TRANSFER_SCAN_FIXED_RAN" if base_sanity["base_sanity_pass"] else "BASELINE_SANITY_FAILED",
        "no_training": True,
        "closure_level": "all_native_student_heads_plus_selected_delta_patch",
        "missing_teacher_atom_count": len(set(missing_teacher)),
        "missing_student_atom_count": len(set(missing_student)),
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_delta_head_scan.jsonl", summary_rows)
    write_jsonl(out / "per_prompt_delta_head_scan.jsonl", per_prompt_all)
    write_jsonl(out / "per_patched_head_delta_attention_metrics.jsonl", per_head_metrics_all)
    write_jsonl(out / "per_head_compiled_teacher_program.jsonl", teacher_compile_rows)
    write_jsonl(out / "per_head_compiled_student_program.jsonl", student_compile_rows)

    print("=== Qwen Teacher -> Student Delta Per-Head Scan v4.1 ===")
    print(json.dumps({
        "prompt_mode": report["prompt_mode"],
        "transfer_mode": report["transfer_mode"],
        "prompts": report["prompts"],
        "all_heads": report["all_heads"],
        "candidate_heads": report["candidate_heads"],
        "base_no_patch": report["base_no_patch"],
        "base_sanity": report["base_sanity"],
        "candidates": len(summary_rows),
        "top10": [{
            "candidate": r["candidate"],
            "logits_improvement": r["logits_improvement"],
            "KL_improvement": r["KL_improvement"],
            "student_shift_logits_rel": r["student_shift_logits_rel"],
            "Y_prog_vs_student_attention_rel": r["Y_prog_vs_student_attention_rel"],
            "selection_score": r["selection_score"],
        } for r in summary_rows[:10]],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")

    if not base_sanity["base_sanity_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
