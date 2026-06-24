#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_term_transfer_heldout_top_v1.py

Heldout test for top term-level pseudocode transfer candidates.

This is the follow-up after the full train map:
  - train map found candidate term patches on 8 code + 8 retain prompts
  - this script evaluates selected candidates on separate heldout code/retain prompts

No QK/VO autoexpand recomputation. It reuses saved program-runs:
  qwen_coder_L{L}_autoexpand_closure_v2
  qwen_coder_L{L}_vo_autoexpand_replay_v1
  qwen_L{L}_autoexpand_closure_v2
  qwen_L{L}_vo_autoexpand_replay_v1

Candidate format:
  L:H:spec:alpha
example:
  6:11:vo_linear:0.05
  12:9:k_affine_medium:0.025

It reports train_code/train_retain/heldout_code/heldout_retain for each candidate.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program
import qwen_teacher_student_attention_transfer_v1 as ts
import qwen_teacher_student_delta_head_scan_v4 as v4
from qwen_head_pseudocode_term_transfer_scan_v1 import (
    parse_delta_range,
    build_y_acc_for_term_patch,
)
from qwen_head_pseudocode_term_transfer_scan_v1_1 import (
    evaluate_patch_full_forward,
    summarize_rows,
)

VERSION = "qwen_term_transfer_heldout_top_v1.0"


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def train_code_prompts() -> List[str]:
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


def train_retain_prompts() -> List[str]:
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


def heldout_code_prompts() -> List[str]:
    return [
        "Write a Python function that checks whether a string has balanced parentheses.",
        "Complete this code:\n\ndef merge_sorted(a, b):\n    # return one sorted list\n",
        "Find the bug:\n\nwhile i < len(nums):\n    total += nums[i]\n",
        "Explain what this Python list comprehension does: [x*x for x in xs if x % 2 == 0]",
        "Write a JavaScript function that removes duplicate values from an array.",
        "Create a SQL query that returns the top 5 products by total sales.",
        "Refactor this code into a helper function:\n\nprint(user.name)\nprint(user.email)\n",
        "Given a dictionary of counts, write Python code to return the key with the largest value.",
    ]


def heldout_retain_prompts() -> List[str]:
    return [
        "Explain how rainbows form in simple words.",
        "Give a short overview of why exercise is good for health.",
        "Describe the water cycle for a middle school student.",
        "Translate this sentence into Ukrainian: I will visit my friend tomorrow.",
        "What are the main differences between reptiles and mammals?",
        "Summarize why people use maps in geography.",
        "Explain what a budget is with a simple example.",
        "Write a short paragraph about why libraries are useful.",
    ]


def parse_candidate(s: str) -> Dict[str, Any]:
    # L:H:spec:alpha, where spec may not contain ':'
    parts = s.split(":")
    if len(parts) != 4:
        raise ValueError(f"Bad candidate {s!r}; expected L:H:spec:alpha")
    return {
        "layer": int(parts[0]),
        "head": int(parts[1]),
        "spec": parts[2],
        "alpha": float(parts[3]),
        "name": f"L{parts[0]}H{parts[1]}_{parts[2]}_a{parts[3]}",
    }


def candidate_score(sums: Dict[str, float]) -> float:
    code_kl = float(sums.get("heldout_code_KL_gain", 0.0))
    retain_kl = float(sums.get("heldout_retain_KL_gain", 0.0))
    code_logit = float(sums.get("heldout_code_logit_gain", 0.0))
    retain_logit = float(sums.get("heldout_retain_logit_gain", 0.0))
    retain_damage = max(0.0, -retain_kl)
    return code_kl + 0.25 * code_logit - 2.0 * retain_damage - 0.25 * max(0.0, retain_logit)


def pick_layer_candidates(candidates: List[Dict[str, Any]], layer: int) -> List[Dict[str, Any]]:
    return [c for c in candidates if int(c["layer"]) == int(layer)]


def program_run_paths(root: Path, layer: int) -> Dict[str, Path]:
    return {
        "teacher_qk": root / f"qwen_coder_L{layer}_autoexpand_closure_v2",
        "teacher_vo": root / f"qwen_coder_L{layer}_vo_autoexpand_replay_v1",
        "student_qk": root / f"qwen_L{layer}_autoexpand_closure_v2",
        "student_vo": root / f"qwen_L{layer}_vo_autoexpand_replay_v1",
    }


def eval_one_dataset(student, tok_student, teacher, tok_teacher, prompts: List[str], layer: int, all_heads: List[int], cand: Dict[str, Any], compiled_teacher, compiled_student, thresholds, args, medium_deltas: Set[int]) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
    teacher_truth = ts.collect_model_truth(teacher, tok_teacher, prompts, layer, args.max_length, args.device)
    student_truth = ts.collect_model_truth(student, tok_student, prompts, layer, args.max_length, args.device)
    student_head_data = v4.precompute_student_heads(student, tok_student, prompts, layer, all_heads, args)

    base_y_acc, _, _ = build_y_acc_for_term_patch(
        student_head_data, compiled_student, compiled_teacher, set(), 0.0, cand["spec"], medium_deltas, args
    )
    base_rows = evaluate_patch_full_forward(student, tok_student, prompts, student_truth, teacher_truth, base_y_acc, layer, args.max_length, args.device)
    base_sums = summarize_rows(base_rows)

    y_acc, patched_head_rows, missing = build_y_acc_for_term_patch(
        student_head_data, compiled_student, compiled_teacher, {cand["head"]}, cand["alpha"], cand["spec"], medium_deltas, args
    )
    rows = evaluate_patch_full_forward(student, tok_student, prompts, student_truth, teacher_truth, y_acc, layer, args.max_length, args.device)
    sums = summarize_rows(rows)
    info = {
        "base": base_sums,
        "missing": missing,
        "patched_head_rows": patched_head_rows,
    }
    return sums, rows, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--medium-deltas", default="7:13")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--program-root", default="runs/exact_program_transplant_v1")
    ap.add_argument("--all-heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--candidates", default="6:11:vo_linear:0.05;6:11:k_affine_medium:0.05;6:11:content_medium:0.05;12:9:k_affine_medium:0.025;12:9:q_affine_medium:0.025;12:9:content_medium:0.025;12:12:vo_linear:0.075;18:7:k_affine_medium:0.05")
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/heldout_top_term_transfer_v1")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds)
    medium_deltas = parse_delta_range(args.medium_deltas)
    all_heads = [int(x) for x in args.all_heads.split(",") if x.strip()]
    candidates = [parse_candidate(x.strip()) for x in args.candidates.split(";") if x.strip()]
    layers = sorted(set(c["layer"] for c in candidates))
    program_root = Path(args.program_root)

    tok_teacher = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    tok_student = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=get_dtype(args.dtype), device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True,
    ).to(args.device).eval()
    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=get_dtype(args.dtype), device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True,
    ).to(args.device).eval()

    datasets = {
        "train_code": train_code_prompts(),
        "train_retain": train_retain_prompts(),
        "heldout_code": heldout_code_prompts(),
        "heldout_retain": heldout_retain_prompts(),
    }

    all_results = []
    per_prompt_all = []
    base_all = []

    for layer in layers:
        paths = program_run_paths(program_root, layer)
        print(f"\n=== layer {layer} ===")
        tqk_atoms, tqk_gates = load_qk_program(paths["teacher_qk"])
        tvo_atoms, tvo_gates = load_vo_program(paths["teacher_vo"])
        sqk_atoms, sqk_gates = load_qk_program(paths["student_qk"])
        svo_atoms, svo_gates = load_vo_program(paths["student_vo"])

        # Compile per dataset because exact sequence traces are prompt-dependent.
        compiled_cache = {}
        for ds_name, prompts in datasets.items():
            print(f"compiling {ds_name} L{layer}...")
            compiled_teacher, _teacher_rows, _missing_teacher = ts.compile_teacher_attention_program(
                teacher, tok_teacher, prompts, layer, all_heads, args, thresholds, tqk_atoms, tqk_gates, tvo_atoms, tvo_gates
            )
            compiled_student, _student_rows, _missing_student = ts.compile_teacher_attention_program(
                student, tok_student, prompts, layer, all_heads, args, thresholds, sqk_atoms, sqk_gates, svo_atoms, svo_gates
            )
            compiled_cache[ds_name] = (compiled_teacher, compiled_student)

        for cand in pick_layer_candidates(candidates, layer):
            row = {"candidate": cand["name"], "layer": layer, "head": cand["head"], "spec": cand["spec"], "alpha": cand["alpha"]}
            print(f"\n--- {cand['name']} ---")
            for ds_name, prompts in datasets.items():
                compiled_teacher, compiled_student = compiled_cache[ds_name]
                sums, per_prompt, info = eval_one_dataset(
                    student, tok_student, teacher, tok_teacher, prompts, layer, all_heads, cand,
                    compiled_teacher, compiled_student, thresholds, args, medium_deltas
                )
                prefix = ds_name
                row[f"{prefix}_logit_gain"] = sums["logits_improvement"]
                row[f"{prefix}_KL_gain"] = sums["KL_improvement"]
                row[f"{prefix}_shift"] = sums["student_shift_logits_rel"]
                row[f"{prefix}_Y_rel"] = sums["Y_prog_vs_student_attention_rel"]
                row[f"{prefix}_top1"] = sums["after_top1_match_rate"]
                row[f"{prefix}_base_shift"] = info["base"]["student_shift_logits_rel"]
                row[f"{prefix}_base_Y"] = info["base"]["Y_prog_vs_student_attention_rel"]
                base_all.append({"candidate": cand["name"], "dataset": ds_name, **info["base"]})
                for r in per_prompt:
                    r.update({"candidate": cand["name"], "dataset": ds_name, "layer": layer, "head": cand["head"], "spec": cand["spec"], "alpha": cand["alpha"]})
                    per_prompt_all.append(r)
                print(f"{ds_name}: logit={sums['logits_improvement']:+.5f} KL={sums['KL_improvement']:+.5f} shift={sums['student_shift_logits_rel']:.5f} Y={sums['Y_prog_vs_student_attention_rel']:.5f}")

            row["heldout_score"] = candidate_score(row)
            row["heldout_clean"] = bool(
                row["heldout_code_KL_gain"] > 0
                and row["heldout_code_logit_gain"] >= 0
                and max(0.0, -row["heldout_retain_KL_gain"]) <= max(0.002, 0.25 * row["heldout_code_KL_gain"])
            )
            all_results.append(row)

    all_results.sort(key=lambda r: r["heldout_score"], reverse=True)
    clean = [r for r in all_results if r["heldout_clean"]]
    report = {
        "version": VERSION,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layers": layers,
        "candidates": candidates,
        "medium_deltas": sorted(medium_deltas),
        "result_count": len(all_results),
        "heldout_clean_count": len(clean),
        "top": all_results,
        "top_clean": clean,
    }
    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_heldout.jsonl", all_results)
    write_jsonl(out / "per_prompt_heldout.jsonl", per_prompt_all)
    write_jsonl(out / "base_sanity_by_candidate_dataset.jsonl", base_all)

    print("\n=== Qwen Term Transfer Heldout Top v1 ===")
    print(json.dumps({
        "result_count": len(all_results),
        "heldout_clean_count": len(clean),
        "top": all_results[:20],
        "top_clean": clean[:20],
        "out": str(out),
    }, indent=2))

    del teacher, student
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
