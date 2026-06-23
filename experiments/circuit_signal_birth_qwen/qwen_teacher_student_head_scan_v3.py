#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_head_scan_v3.py

Per-head Coder -> Student no-training attention transfer scan.

Why this exists:
  v2 patched all L23 heads at once. It moved Instruct logits toward Coder, but
  KL worsened. We need to know which heads are useful and which heads are harmful.

This v3 tests one patched teacher head at a time while all other heads remain
native student heads. It also optionally tests ALL heads.

No training, no KL distillation, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import collect_head_data, get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program, program_scores_A_for_seq
import qwen_teacher_student_attention_transfer_v1 as ts
from qwen_teacher_student_code_transfer_v2 import select_prompts

VERSION = "qwen_teacher_student_head_scan_v3.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


@torch.no_grad()
def precompute_student_heads(student, tokenizer, prompts, layer_idx: int, heads: List[int], args):
    data = {}
    for hi in heads:
        seqs, meta = collect_head_data(student, tokenizer, prompts, layer_idx, hi, args.max_length, args.device)
        if seqs:
            data[hi] = {"seqs": seqs, "meta": meta}
    return data


@torch.no_grad()
def build_y_acc_for_patch(student_head_data, compiled_teacher, patch_heads: set[int], args):
    y_acc: Dict[int, Dict[str, torch.Tensor]] = {}
    per_patched_head = []

    for hi, pack in student_head_data.items():
        seqs = pack["seqs"]
        use_patch = hi in patch_heads
        if use_patch and hi not in compiled_teacher:
            use_patch = False

        score_rels, A_rels, Y_rels, top1s = [], [], [], []

        for s in seqs:
            if int(s.prompt_id) not in y_acc:
                y_acc[int(s.prompt_id)] = {
                    "Y_prog": torch.zeros_like(s.Y.float().cpu()),
                    "Y_student_true": torch.zeros_like(s.Y.float().cpu()),
                }

            if use_patch:
                Mdelta = compiled_teacher[hi]["Mdelta"]
                Cvo = compiled_teacher[hi]["Cvo"]
                scores_prog, A_prog, scores_student, row_covered, mask_eval = program_scores_A_for_seq(s, Mdelta, args.max_delta)
                payload_prog = s.Xaug @ Cvo.T
                Y_head_prog = A_prog @ payload_prog

                if int(mask_eval.sum()) > 0:
                    score_rels.append(rel_err(scores_prog[mask_eval], scores_student[mask_eval]))
                if int(row_covered.sum()) > 0:
                    A_rels.append(rel_err(A_prog[row_covered], s.A[row_covered]))
                    top_true = torch.argmax(s.A[row_covered], dim=-1)
                    top_prog = torch.argmax(A_prog[row_covered], dim=-1)
                    top1s.append(float((top_true == top_prog).float().mean()))
                Y_rels.append(rel_err(Y_head_prog, s.Y))
            else:
                # Keep native student head output unchanged.
                Y_head_prog = s.Y

            y_acc[int(s.prompt_id)]["Y_prog"] += Y_head_prog.float().cpu()
            y_acc[int(s.prompt_id)]["Y_student_true"] += s.Y.float().cpu()

        if hi in patch_heads:
            per_patched_head.append({
                "head": hi,
                "score_vs_student": mean(score_rels) if score_rels else None,
                "A_vs_student": mean(A_rels) if A_rels else None,
                "Yh_vs_student": mean(Y_rels) if Y_rels else None,
                "top1_vs_student": mean(top1s) if top1s else None,
            })

    return y_acc, per_patched_head


@torch.no_grad()
def evaluate_patch(student, student_truth, teacher_truth, y_acc, layer_idx: int, device: str):
    student_prog = ts.run_student_final_layer_with_program(student, student_truth, y_acc, layer_idx, device)
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
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-mode", default="code", choices=["code", "retain", "mixed"])
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--qk-program-run", required=True)
    ap.add_argument("--vo-program-run", required=True)
    ap.add_argument("--include-all", action="store_true")
    ap.add_argument("--out", default="runs/qwen_teacher_student_head_scan_v3")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    qk_atoms, qk_gates = load_qk_program(Path(args.qk_program_run))
    vo_atoms, vo_gates = load_vo_program(Path(args.vo_program_run))
    heads = parse_ints(args.heads)
    prompts = select_prompts(args.prompt_mode, args.prompts)

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
    teacher_truth = ts.collect_model_truth(teacher, tok, prompts, args.layer, args.max_length, args.device)
    compiled, compile_rows, missing_compile = ts.compile_teacher_attention_program(
        teacher, tok, prompts, args.layer, heads, args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates
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
    student_truth = ts.collect_model_truth(student, student_tok, prompts, args.layer, args.max_length, args.device)
    student_head_data = precompute_student_heads(student, student_tok, prompts, args.layer, heads, args)

    candidates = [(f"H{h}", {h}) for h in heads]
    if args.include_all:
        candidates.append(("ALL", set(heads)))

    summary_rows = []
    per_prompt_all = []
    per_patched_head_all = []

    for name, patch_set in candidates:
        y_acc, patched_head_rows = build_y_acc_for_patch(student_head_data, compiled, patch_set, args)
        rows = evaluate_patch(student, student_truth, teacher_truth, y_acc, args.layer, args.device)
        sums = summarize_rows(rows)
        score = sums["logits_improvement"] - 0.25 * max(0.0, -sums["KL_improvement"]) - 0.05 * sums["student_shift_logits_rel"]
        item = {
            "candidate": name,
            "patch_heads": sorted(list(patch_set)),
            "prompt_mode": args.prompt_mode,
            **sums,
            "selection_score": score,
            "useful_logits": sums["logits_improvement"] > 0,
            "useful_KL": sums["KL_improvement"] > 0,
        }
        summary_rows.append(item)
        for r in rows:
            r.update({"candidate": name, "patch_heads": sorted(list(patch_set)), "prompt_mode": args.prompt_mode})
            per_prompt_all.append(r)
        for r in patched_head_rows:
            r.update({"candidate": name, "prompt_mode": args.prompt_mode})
            per_patched_head_all.append(r)
        print(f"{name}: logit_gain={sums['logits_improvement']:+.5f} KL_gain={sums['KL_improvement']:+.5f} shift={sums['student_shift_logits_rel']:.5f} score={score:+.5f}")

    summary_rows = sorted(summary_rows, key=lambda r: r["selection_score"], reverse=True)
    report = {
        "version": VERSION,
        "mode": "per_head_teacher_student_transfer_scan",
        "prompt_mode": args.prompt_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "heads": heads,
        "prompts": len(prompts),
        "qk_atom_count": len(qk_atoms),
        "vo_atom_count": len(vo_atoms),
        "compiled_heads": len(compiled),
        "best_candidates": summary_rows[:10],
        "status": "PER_HEAD_TRANSFER_SCAN_RAN",
        "no_training": True,
        "missing_compile_atom_count": len(set(missing_compile)),
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_head_scan.jsonl", summary_rows)
    write_jsonl(out / "per_prompt_head_scan.jsonl", per_prompt_all)
    write_jsonl(out / "per_patched_head_attention_metrics.jsonl", per_patched_head_all)
    write_jsonl(out / "per_head_compiled_teacher_program.jsonl", compile_rows)

    print("=== Qwen Teacher -> Student Per-Head Transfer Scan v3 ===")
    print(json.dumps({
        "prompt_mode": args.prompt_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "prompts": len(prompts),
        "candidates": len(summary_rows),
        "top5": [{
            "candidate": r["candidate"],
            "logits_improvement": r["logits_improvement"],
            "KL_improvement": r["KL_improvement"],
            "student_shift_logits_rel": r["student_shift_logits_rel"],
            "selection_score": r["selection_score"],
        } for r in summary_rows[:5]],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
