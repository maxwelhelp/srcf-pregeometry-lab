#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_aligned_transfer_v1.py

Teacher -> student no-training attention transfer with residual-basis Procrustes alignment.

Motivation:
  Direct Coder -> Instruct L23 attention transfer moved logits toward Coder, but
  KL could worsen. Likely reason: Coder and Instruct residual/RMSNorm spaces are
  not exactly the same basis.

This script:
  1) collects same-prompt Xn activations from teacher and student at layer L;
  2) solves orthogonal Procrustes R such that X_student @ R ~= X_teacher;
  3) compiles teacher program matrices M_qk[h,d], C_vo[h];
  4) rotates them into student basis:
       M_student = R_aug @ M_teacher @ R_aug.T
       C_student = R.T @ C_teacher @ R_aug.T   (row convention explanation below)
     In code with row vectors, teacher hidden ~= student hidden @ R.
     For VO payload y_teacher = x_teacher_aug @ C_teacher.T.
     Student-basis payload is y_student ~= y_teacher @ R.T, so:
       C_student.T = R_aug @ C_teacher.T @ R.T
       C_student   = R @ C_teacher @ R_aug.T
  5) inserts aligned teacher program into student final attention layer;
  6) compares student_before vs teacher and student_after vs teacher.

No training, no KL distillation, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import collect_head_data, get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program
from qwen_teacher_student_code_transfer_v2 import select_prompts
import qwen_teacher_student_attention_transfer_v1 as ts

VERSION = "qwen_teacher_student_aligned_transfer_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


@torch.no_grad()
def collect_xn_for_alignment(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str, head_for_probe: int = 0):
    # collect_head_data already extracts Xn exactly in the same normalized space used by circuit targets.
    seqs, _meta = collect_head_data(model, tokenizer, prompts, layer_idx, head_for_probe, max_length, device)
    by_prompt = {int(s.prompt_id): s.Xn.detach().float().cpu() for s in seqs}
    return by_prompt


def paired_stack(student_xn: Dict[int, torch.Tensor], teacher_xn: Dict[int, torch.Tensor], center: bool):
    xs_list, xt_list = [], []
    for pid in sorted(set(student_xn) & set(teacher_xn)):
        xs = student_xn[pid]
        xt = teacher_xn[pid]
        T = min(xs.shape[0], xt.shape[0])
        if T <= 0:
            continue
        xs_list.append(xs[:T])
        xt_list.append(xt[:T])
    if not xs_list:
        raise RuntimeError("no paired activation rows for alignment")
    Xs = torch.cat(xs_list, dim=0).float()
    Xt = torch.cat(xt_list, dim=0).float()
    if center:
        Xs = Xs - Xs.mean(dim=0, keepdim=True)
        Xt = Xt - Xt.mean(dim=0, keepdim=True)
    return Xs, Xt


def procrustes_row_map(Xs: torch.Tensor, Xt: torch.Tensor):
    # Row convention: find R minimizing ||Xs @ R - Xt||_F, orthogonal R.
    C = Xs.T @ Xt
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    R = U @ Vh
    return R.float(), S.float()


def cka_linear(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic = torch.linalg.norm(X.T @ Y) ** 2
    var1 = torch.linalg.norm(X.T @ X) ** 2
    var2 = torch.linalg.norm(Y.T @ Y) ** 2
    return float(hsic / (torch.sqrt(var1 * var2).clamp_min(eps)))


def make_R_aug(R: torch.Tensor):
    H = R.shape[0]
    R_aug = torch.eye(H + 1, dtype=R.dtype)
    R_aug[:H, :H] = R
    return R_aug


def align_compiled_program(compiled: Dict[int, Dict[str, Any]], R: torch.Tensor):
    R = R.cpu().float()
    R_aug = make_R_aug(R)
    out: Dict[int, Dict[str, Any]] = {}
    for h, pack in compiled.items():
        M_aligned = {}
        for d, M in pack["Mdelta"].items():
            M_aligned[int(d)] = (R_aug @ M.cpu().float() @ R_aug.T).cpu()
        C = pack["Cvo"].cpu().float()
        C_aligned = (R @ C @ R_aug.T).cpu()
        out[int(h)] = {
            "Mdelta": M_aligned,
            "Cvo": C_aligned,
            "kv_head": int(pack.get("kv_head", -1)),
        }
    return out


def evaluate_transfer(student, student_truth, teacher_truth, y_acc, layer_idx: int, device: str):
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
            "text": student_truth[pi]["text"][:180],
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


def summarize(rows):
    before_rel = mean([r["teacher_student_before_logits_rel"] for r in rows])
    after_rel = mean([r["teacher_student_after_logits_rel"] for r in rows])
    before_kl = mean([r["teacher_student_before_KL"] for r in rows])
    after_kl = mean([r["teacher_student_after_KL"] for r in rows])
    return {
        "teacher_student_before_logits_rel_mean": before_rel,
        "teacher_student_after_logits_rel_mean": after_rel,
        "teacher_student_logits_rel_improvement": before_rel - after_rel,
        "teacher_student_logits_rel_improvement_pct_of_before": (before_rel - after_rel) / max(1e-12, before_rel),
        "teacher_student_before_KL_mean": before_kl,
        "teacher_student_after_KL_mean": after_kl,
        "teacher_student_KL_improvement": before_kl - after_kl,
        "teacher_student_KL_improvement_pct_of_before": (before_kl - after_kl) / max(1e-12, before_kl),
        "student_before_after_logits_rel_mean": mean([r["student_before_after_logits_rel"] for r in rows]),
        "student_before_after_KL_mean": mean([r["student_before_after_KL"] for r in rows]),
        "Y_prog_vs_student_attention_rel_mean": mean([r["Y_prog_vs_student_attention_rel"] for r in rows]),
        "teacher_student_before_top1_match_rate": mean([1.0 if r["teacher_student_before_top1_match"] else 0.0 for r in rows]),
        "teacher_student_after_top1_match_rate": mean([1.0 if r["teacher_student_after_top1_match"] else 0.0 for r in rows]),
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
    ap.add_argument("--align-prompts", type=int, default=8)
    ap.add_argument("--align-prompt-mode", default="mixed", choices=["code", "retain", "mixed"])
    ap.add_argument("--no-center", action="store_true")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--qk-program-run", required=True)
    ap.add_argument("--vo-program-run", required=True)
    ap.add_argument("--out", default="runs/qwen_teacher_student_aligned_transfer_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    qk_atoms, qk_gates = load_qk_program(Path(args.qk_program_run))
    vo_atoms, vo_gates = load_vo_program(Path(args.vo_program_run))
    heads = parse_ints(args.heads)
    eval_prompts = select_prompts(args.prompt_mode, args.prompts)
    align_prompts = select_prompts(args.align_prompt_mode, args.align_prompts)

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
    if args.layer != len(teacher.model.layers) - 1:
        raise SystemExit(f"v1 supports final layer only. layer={args.layer}, final={len(teacher.model.layers)-1}")

    teacher_truth = ts.collect_model_truth(teacher, tok_teacher, eval_prompts, args.layer, args.max_length, args.device)
    teacher_xn = collect_xn_for_alignment(teacher, tok_teacher, align_prompts, args.layer, args.max_length, args.device)
    compiled, compile_rows, missing_compile = ts.compile_teacher_attention_program(
        teacher, tok_teacher, eval_prompts, args.layer, heads, args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates
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
    if args.layer != len(student.model.layers) - 1:
        raise SystemExit(f"v1 supports final layer only. layer={args.layer}, student_final={len(student.model.layers)-1}")

    student_truth = ts.collect_model_truth(student, tok_student, eval_prompts, args.layer, args.max_length, args.device)
    student_xn = collect_xn_for_alignment(student, tok_student, align_prompts, args.layer, args.max_length, args.device)

    Xs, Xt = paired_stack(student_xn, teacher_xn, center=(not args.no_center))
    R, S = procrustes_row_map(Xs, Xt)
    align_before = rel_err(Xs, Xt)
    align_after = rel_err(Xs @ R, Xt)
    cka_before = cka_linear(Xs, Xt)
    cka_after = cka_linear(Xs @ R, Xt)
    ortho_err = rel_err(R.T @ R, torch.eye(R.shape[0]))

    print(f"alignment: rel {align_before:.4f} -> {align_after:.4f} | CKA {cka_before:.4f} -> {cka_after:.4f} | ortho={ortho_err:.3e}")

    compiled_aligned = align_compiled_program(compiled, R)
    y_acc_aligned, transfer_head_rows, missing_heads = ts.apply_compiled_teacher_attention_to_student(
        student, tok_student, eval_prompts, compiled_aligned, args.layer, heads, args
    )
    rows_aligned = evaluate_transfer(student, student_truth, teacher_truth, y_acc_aligned, args.layer, args.device)
    sums_aligned = summarize(rows_aligned)

    # Also run direct compiled teacher program as baseline inside the same script.
    y_acc_direct, transfer_head_rows_direct, missing_heads_direct = ts.apply_compiled_teacher_attention_to_student(
        student, tok_student, eval_prompts, compiled, args.layer, heads, args
    )
    rows_direct = evaluate_transfer(student, student_truth, teacher_truth, y_acc_direct, args.layer, args.device)
    sums_direct = summarize(rows_direct)

    report = {
        "version": VERSION,
        "mode": "procrustes_aligned_teacher_student_attention_transfer",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "prompt_mode": args.prompt_mode,
        "align_prompt_mode": args.align_prompt_mode,
        "qk_program_run": args.qk_program_run,
        "vo_program_run": args.vo_program_run,
        "qk_atom_count": len(qk_atoms),
        "vo_atom_count": len(vo_atoms),
        "compiled_heads": len(compiled),
        "eval_prompts": len(eval_prompts),
        "align_rows": int(Xs.shape[0]),
        "centered_alignment": not args.no_center,
        "alignment_rel_before": align_before,
        "alignment_rel_after": align_after,
        "alignment_rel_gain": align_before - align_after,
        "cka_before": cka_before,
        "cka_after": cka_after,
        "R_orthogonality_err": ortho_err,
        "singular_values_top10": [float(x) for x in S[:10]],
        "direct": sums_direct,
        "aligned": sums_aligned,
        "aligned_vs_direct_logits_gain_delta": sums_aligned["teacher_student_logits_rel_improvement"] - sums_direct["teacher_student_logits_rel_improvement"],
        "aligned_vs_direct_KL_gain_delta": sums_aligned["teacher_student_KL_improvement"] - sums_direct["teacher_student_KL_improvement"],
        "no_training": True,
        "closure_level": "procrustes_aligned_teacher_program_inserted_into_student_final_attention",
        "same_basis_transplant_allowed": False,
        "cross_model_claim_allowed": True,
        "missing_compile_atom_count": len(set(missing_compile)),
        "missing_student_heads_aligned": missing_heads,
        "missing_student_heads_direct": missing_heads_direct,
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_prompt_aligned_transfer.jsonl", rows_aligned)
    write_jsonl(out / "per_prompt_direct_transfer_baseline.jsonl", rows_direct)
    write_jsonl(out / "per_head_aligned_student_attention.jsonl", transfer_head_rows)
    write_jsonl(out / "per_head_direct_student_attention.jsonl", transfer_head_rows_direct)
    write_jsonl(out / "per_head_compiled_teacher_program.jsonl", compile_rows)

    print("=== Qwen Teacher -> Student Aligned Transfer v1 ===")
    print(json.dumps({
        "prompt_mode": report["prompt_mode"],
        "align_prompt_mode": report["align_prompt_mode"],
        "qk_atom_count": report["qk_atom_count"],
        "vo_atom_count": report["vo_atom_count"],
        "compiled_heads": report["compiled_heads"],
        "eval_prompts": report["eval_prompts"],
        "align_rows": report["align_rows"],
        "alignment_rel_before": report["alignment_rel_before"],
        "alignment_rel_after": report["alignment_rel_after"],
        "cka_before": report["cka_before"],
        "cka_after": report["cka_after"],
        "direct_logits_gain": report["direct"]["teacher_student_logits_rel_improvement"],
        "aligned_logits_gain": report["aligned"]["teacher_student_logits_rel_improvement"],
        "direct_KL_gain": report["direct"]["teacher_student_KL_improvement"],
        "aligned_KL_gain": report["aligned"]["teacher_student_KL_improvement"],
        "aligned_vs_direct_logits_gain_delta": report["aligned_vs_direct_logits_gain_delta"],
        "aligned_vs_direct_KL_gain_delta": report["aligned_vs_direct_KL_gain_delta"],
        "aligned_Y_prog_vs_student_attention_rel": report["aligned"]["Y_prog_vs_student_attention_rel_mean"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
