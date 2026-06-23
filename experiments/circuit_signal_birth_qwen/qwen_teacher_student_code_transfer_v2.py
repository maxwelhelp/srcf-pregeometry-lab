#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_code_transfer_v2.py

Teacher -> student no-training attention transfer with prompt modes:
  --prompt-mode code
  --prompt-mode retain
  --prompt-mode mixed

Purpose:
  The v1 transfer proved that Coder L23 attention program can move Instruct logits
  toward Coder in full-logit space, but KL worsened on generic prompts.

This v2 asks the right question:
  Does the teacher Coder program help more on code prompts than on retain prompts?

No training, no KL distillation, no LoRA, no alpha sweep.
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
import qwen_teacher_student_attention_transfer_v1 as ts

VERSION = "qwen_teacher_student_code_transfer_v2.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def prompts_code():
    return [
        "Write a Python function `is_prime(n)` and explain the edge cases.",
        "Complete this Python code:\n\ndef fibonacci(n):\n    # return the nth Fibonacci number\n",
        "Fix the bug in this code:\n\nfor i in range(len(items)):\n    print(items[i+1])\n",
        "Implement binary search in Python for a sorted list of integers.",
        "Given a list of strings, write Python code to group them by their first letter.",
        "Explain what this JavaScript code does:\n\nconst xs = arr.map(x => x * 2).filter(x => x > 10);",
        "Write a SQL query to find users who made more than 3 orders last month.",
        "Convert this recursive factorial function to an iterative version in Python.",
    ]


def prompts_retain():
    return [
        "Explain why the sky looks blue in simple terms.",
        "Write a short summary of the causes of the French Revolution.",
        "What are three healthy habits for better sleep?",
        "Translate this sentence into Ukrainian: The weather is warm today.",
        "Give a simple explanation of photosynthesis.",
        "What is the difference between a planet and a star?",
        "Summarize the plot of a classic adventure story in one paragraph.",
        "Explain compound interest with a small numerical example.",
    ]


def prompts_mixed():
    c = prompts_code()
    r = prompts_retain()
    out = []
    for a, b in zip(c, r):
        out.append(a)
        out.append(b)
    return out


def select_prompts(mode: str, n: int):
    if mode == "code":
        base = prompts_code()
    elif mode == "retain":
        base = prompts_retain()
    elif mode == "mixed":
        base = prompts_mixed()
    else:
        raise ValueError(mode)
    return base[:max(1, min(n, len(base)))]


def summarize(per_prompt):
    before_rel = mean([r["teacher_student_before_logits_rel"] for r in per_prompt])
    after_rel = mean([r["teacher_student_after_logits_rel"] for r in per_prompt])
    before_kl = mean([r["teacher_student_before_KL"] for r in per_prompt])
    after_kl = mean([r["teacher_student_after_KL"] for r in per_prompt])
    shift_rel = mean([r["student_before_after_logits_rel"] for r in per_prompt])
    y_vs_student = mean([r["Y_prog_vs_student_attention_rel"] for r in per_prompt])
    top1_before = mean([1.0 if r["teacher_student_before_top1_match"] else 0.0 for r in per_prompt])
    top1_after = mean([1.0 if r["teacher_student_after_top1_match"] else 0.0 for r in per_prompt])
    return {
        "teacher_student_before_logits_rel_mean": before_rel,
        "teacher_student_after_logits_rel_mean": after_rel,
        "teacher_student_logits_rel_improvement": before_rel - after_rel,
        "teacher_student_logits_rel_improvement_pct_of_before": (before_rel - after_rel) / max(1e-12, before_rel),
        "teacher_student_before_KL_mean": before_kl,
        "teacher_student_after_KL_mean": after_kl,
        "teacher_student_KL_improvement": before_kl - after_kl,
        "teacher_student_KL_improvement_pct_of_before": (before_kl - after_kl) / max(1e-12, before_kl),
        "student_before_after_logits_rel_mean": shift_rel,
        "Y_prog_vs_student_attention_rel_mean": y_vs_student,
        "teacher_student_before_top1_match_rate": top1_before,
        "teacher_student_after_top1_match_rate": top1_after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-mode", default="code", choices=["code", "retain", "mixed"])
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--qk-program-run", required=True)
    ap.add_argument("--vo-program-run", required=True)
    ap.add_argument("--out", default="runs/qwen_teacher_student_code_transfer_v2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    qk_atoms, qk_gates = load_qk_program(Path(args.qk_program_run))
    vo_atoms, vo_gates = load_vo_program(Path(args.vo_program_run))
    prompt_list = select_prompts(args.prompt_mode, args.prompts)
    heads = parse_ints(args.heads)

    tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)

    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    teacher.eval()
    if args.layer != len(teacher.model.layers) - 1:
        raise SystemExit(f"v2 supports final layer only. layer={args.layer}, final={len(teacher.model.layers)-1}")

    teacher_truth = ts.collect_model_truth(teacher, tok, prompt_list, args.layer, args.max_length, args.device)
    compiled, compile_rows, missing_compile = ts.compile_teacher_attention_program(
        teacher, tok, prompt_list, args.layer, heads, args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates
    )
    del teacher
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("loading student...")
    student_tok = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    student.eval()
    if args.layer != len(student.model.layers) - 1:
        raise SystemExit(f"v2 supports final layer only. layer={args.layer}, student_final={len(student.model.layers)-1}")

    student_truth = ts.collect_model_truth(student, student_tok, prompt_list, args.layer, args.max_length, args.device)
    y_acc, transfer_head_rows, missing_heads = ts.apply_compiled_teacher_attention_to_student(
        student, student_tok, prompt_list, compiled, args.layer, heads, args
    )
    student_prog = ts.run_student_final_layer_with_program(student, student_truth, y_acc, args.layer, args.device)

    per_prompt = []
    for pi in sorted(student_truth.keys()):
        teacher_logits = teacher_truth[pi]["logits_true"]
        student_logits = student_truth[pi]["logits_true"]
        prog_logits = student_prog[pi]["logits_prog"]
        before = ts.logits_metrics(teacher_logits, student_logits)
        after = ts.logits_metrics(teacher_logits, prog_logits)
        student_shift = ts.logits_metrics(student_logits, prog_logits)
        per_prompt.append({
            "prompt_id": pi,
            "prompt_mode": args.prompt_mode,
            "text": student_truth[pi]["text"][:180],
            "teacher_student_before_logits_rel": before["logits_rel"],
            "teacher_student_after_logits_rel": after["logits_rel"],
            "teacher_student_logits_rel_improvement": before["logits_rel"] - after["logits_rel"],
            "teacher_student_before_KL": before["last_token_KL_a_to_b"],
            "teacher_student_after_KL": after["last_token_KL_a_to_b"],
            "teacher_student_KL_improvement": before["last_token_KL_a_to_b"] - after["last_token_KL_a_to_b"],
            "teacher_student_before_top1_match": before["last_top1_match"],
            "teacher_student_after_top1_match": after["last_top1_match"],
            "student_before_after_logits_rel": student_shift["logits_rel"],
            "student_before_after_KL": student_shift["last_token_KL_a_to_b"],
            "Y_prog_vs_student_attention_rel": ts.rel_err(student_prog[pi]["Y_prog"], student_prog[pi]["Y_student_true"]),
        })

    sums = summarize(per_prompt)
    report = {
        "version": VERSION,
        "mode": "teacher_student_code_retain_prompt_transfer",
        "prompt_mode": args.prompt_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "qk_program_run": args.qk_program_run,
        "vo_program_run": args.vo_program_run,
        "qk_atom_count": len(qk_atoms),
        "vo_atom_count": len(vo_atoms),
        "compiled_heads": len(compiled),
        "prompts": len(per_prompt),
        **sums,
        "useful_code_direction_logits": sums["teacher_student_logits_rel_improvement"] > 0,
        "useful_code_direction_KL": sums["teacher_student_KL_improvement"] > 0,
        "self_transfer_case": args.teacher_model == args.student_model,
        "status": "TEACHER_STUDENT_CODE_TRANSFER_RAN",
        "no_training": True,
        "closure_level": "teacher_program_inserted_into_student_final_attention_prompt_mode",
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "missing_compile_atom_count": len(set(missing_compile)),
        "missing_student_heads": missing_heads,
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_prompt_teacher_student_code_transfer.jsonl", per_prompt)
    write_jsonl(out / "per_head_compiled_teacher_program.jsonl", compile_rows)
    write_jsonl(out / "per_head_student_transfer_attention.jsonl", transfer_head_rows)

    print("=== Qwen Teacher -> Student Code/Retain Transfer v2 ===")
    print(json.dumps({
        "prompt_mode": report["prompt_mode"],
        "teacher_model": report["teacher_model"],
        "student_model": report["student_model"],
        "qk_atom_count": report["qk_atom_count"],
        "vo_atom_count": report["vo_atom_count"],
        "compiled_heads": report["compiled_heads"],
        "prompts": report["prompts"],
        "teacher_student_before_logits_rel_mean": report["teacher_student_before_logits_rel_mean"],
        "teacher_student_after_logits_rel_mean": report["teacher_student_after_logits_rel_mean"],
        "teacher_student_logits_rel_improvement": report["teacher_student_logits_rel_improvement"],
        "teacher_student_logits_rel_improvement_pct_of_before": report["teacher_student_logits_rel_improvement_pct_of_before"],
        "teacher_student_before_KL_mean": report["teacher_student_before_KL_mean"],
        "teacher_student_after_KL_mean": report["teacher_student_after_KL_mean"],
        "teacher_student_KL_improvement": report["teacher_student_KL_improvement"],
        "student_before_after_logits_rel_mean": report["student_before_after_logits_rel_mean"],
        "Y_prog_vs_student_attention_rel_mean": report["Y_prog_vs_student_attention_rel_mean"],
        "useful_code_direction_logits": report["useful_code_direction_logits"],
        "useful_code_direction_KL": report["useful_code_direction_KL"],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
