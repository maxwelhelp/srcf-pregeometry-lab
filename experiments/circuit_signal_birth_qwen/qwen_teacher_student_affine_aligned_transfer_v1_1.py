#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_affine_aligned_transfer_v1_1.py

Full-rank affine/ridge alignment for Coder -> Instruct program transfer.

Why not PCA here:
  PCA in the old qwen_circuit_matrix_targets_v2_affine_basis.py was mainly a
  low-rank functional diagnostic/compression path. Cross-model transfer does not
  need low-rank compression first. We need a full-rank coordinate map between
  residual/RMSNorm spaces.

This script solves a ridge affine map:
  X_teacher ~= [X_student, 1] @ T
where T maps student augmented coordinates into teacher augmented coordinates:
  x_t_aug = x_s_aug @ T_aug

Then transforms teacher circuit programs into student coordinates:
  QK: M_s = T_aug @ M_t @ T_aug.T
  VO: C_s = A^{-T} @ C_t @ T_aug.T
where x_t = x_s @ A + b and y_s @ A ~= y_t, hence y_s = y_t @ A^{-1}.

It compares direct teacher program transfer vs affine-aligned transfer.
No training of the model, no KL distillation, no LoRA, no alpha sweep.
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
from qwen_circuit_target_roundtrip_v1 import get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program
from qwen_teacher_student_code_transfer_v2 import select_prompts
import qwen_teacher_student_attention_transfer_v1 as ts
import qwen_teacher_student_aligned_transfer_v1 as ortho

VERSION = "qwen_teacher_student_affine_aligned_transfer_v1.1"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def cka_linear(X: torch.Tensor, Y: torch.Tensor, eps: float = 1e-12) -> float:
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic = torch.linalg.norm(X.T @ Y) ** 2
    var1 = torch.linalg.norm(X.T @ X) ** 2
    var2 = torch.linalg.norm(Y.T @ Y) ** 2
    return float(hsic / (torch.sqrt(var1 * var2).clamp_min(eps)))


def fit_affine_ridge_student_to_teacher(Xs: torch.Tensor, Xt: torch.Tensor, ridge: float):
    """Fit Xt ~= [Xs,1] @ B. Return T_aug and A.

    Row convention:
      x_t_aug = x_s_aug @ T_aug
      T_aug = [[A, 0], [b, 1]]
    """
    Xs = Xs.float()
    Xt = Xt.float()
    N, H = Xs.shape
    Xaug = torch.cat([Xs, torch.ones(N, 1)], dim=1)
    I = torch.eye(H + 1, dtype=torch.float32)
    I[-1, -1] = 0.0  # do not regularize bias
    # B: [H+1,H]
    B = torch.linalg.solve(Xaug.T @ Xaug + float(ridge) * I, Xaug.T @ Xt)
    A = B[:H, :].contiguous()     # [H,H]
    b = B[H, :].contiguous()      # [H]
    T = torch.eye(H + 1, dtype=torch.float32)
    T[:H, :H] = A
    T[H, :H] = b
    return T, A, b


def condition_number(A: torch.Tensor, eps: float = 1e-12) -> float:
    try:
        s = torch.linalg.svdvals(A.float())
        return float(s.max() / s.min().clamp_min(eps))
    except Exception:
        return float("inf")


def align_compiled_program_affine(compiled: Dict[int, Dict[str, Any]], T_aug: torch.Tensor, A: torch.Tensor, pinv_rcond: float):
    T_aug = T_aug.cpu().float()
    A = A.cpu().float()
    A_inv = torch.linalg.pinv(A, rcond=float(pinv_rcond))
    A_inv_T = A_inv.T.contiguous()
    out: Dict[int, Dict[str, Any]] = {}
    for h, pack in compiled.items():
        M_aligned = {}
        for d, M in pack["Mdelta"].items():
            M_aligned[int(d)] = (T_aug @ M.cpu().float() @ T_aug.T).cpu()
        C = pack["Cvo"].cpu().float()
        C_aligned = (A_inv_T @ C @ T_aug.T).cpu()
        out[int(h)] = {
            "Mdelta": M_aligned,
            "Cvo": C_aligned,
            "kv_head": int(pack.get("kv_head", -1)),
        }
    return out


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
    ap.add_argument("--align-prompts", type=int, default=16)
    ap.add_argument("--align-prompt-mode", default="mixed", choices=["code", "retain", "mixed"])
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--pinv-rcond", type=float, default=1e-4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--qk-program-run", required=True)
    ap.add_argument("--vo-program-run", required=True)
    ap.add_argument("--out", default="runs/qwen_teacher_student_affine_aligned_transfer_v1_1")
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
        raise SystemExit(f"v1.1 supports final layer only. layer={args.layer}, final={len(teacher.model.layers)-1}")

    teacher_truth = ts.collect_model_truth(teacher, tok_teacher, eval_prompts, args.layer, args.max_length, args.device)
    teacher_xn = ortho.collect_xn_for_alignment(teacher, tok_teacher, align_prompts, args.layer, args.max_length, args.device)
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
        raise SystemExit(f"v1.1 supports final layer only. layer={args.layer}, student_final={len(student.model.layers)-1}")

    student_truth = ts.collect_model_truth(student, tok_student, eval_prompts, args.layer, args.max_length, args.device)
    student_xn = ortho.collect_xn_for_alignment(student, tok_student, align_prompts, args.layer, args.max_length, args.device)

    # No centering here: affine map handles bias explicitly.
    Xs, Xt = ortho.paired_stack(student_xn, teacher_xn, center=False)
    T_aug, A, b = fit_affine_ridge_student_to_teacher(Xs, Xt, args.ridge)
    align_before = rel_err(Xs, Xt)
    align_after = rel_err(torch.cat([Xs, torch.ones(Xs.shape[0], 1)], dim=1) @ T_aug[:, :Xs.shape[1]], Xt)
    cka_before = cka_linear(Xs, Xt)
    cka_after = cka_linear(torch.cat([Xs, torch.ones(Xs.shape[0], 1)], dim=1) @ T_aug[:, :Xs.shape[1]], Xt)
    cond_A = condition_number(A)

    print(f"affine alignment: rel {align_before:.4f} -> {align_after:.4f} | CKA {cka_before:.4f} -> {cka_after:.4f} | cond(A)={cond_A:.2e}")

    compiled_affine = align_compiled_program_affine(compiled, T_aug, A, args.pinv_rcond)
    y_acc_affine, head_rows_affine, missing_heads_affine = ts.apply_compiled_teacher_attention_to_student(
        student, tok_student, eval_prompts, compiled_affine, args.layer, heads, args
    )
    rows_affine = ortho.evaluate_transfer(student, student_truth, teacher_truth, y_acc_affine, args.layer, args.device)
    sums_affine = summarize(rows_affine)

    y_acc_direct, head_rows_direct, missing_heads_direct = ts.apply_compiled_teacher_attention_to_student(
        student, tok_student, eval_prompts, compiled, args.layer, heads, args
    )
    rows_direct = ortho.evaluate_transfer(student, student_truth, teacher_truth, y_acc_direct, args.layer, args.device)
    sums_direct = summarize(rows_direct)

    report = {
        "version": VERSION,
        "mode": "full_rank_affine_ridge_aligned_teacher_student_attention_transfer",
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
        "alignment_type": "full_rank_affine_ridge",
        "ridge": args.ridge,
        "pinv_rcond": args.pinv_rcond,
        "alignment_rel_before": align_before,
        "alignment_rel_after": align_after,
        "alignment_rel_gain": align_before - align_after,
        "cka_before": cka_before,
        "cka_after": cka_after,
        "A_condition_number": cond_A,
        "bias_norm": float(torch.linalg.norm(b).item()),
        "direct": sums_direct,
        "affine_aligned": sums_affine,
        "affine_vs_direct_logits_gain_delta": sums_affine["teacher_student_logits_rel_improvement"] - sums_direct["teacher_student_logits_rel_improvement"],
        "affine_vs_direct_KL_gain_delta": sums_affine["teacher_student_KL_improvement"] - sums_direct["teacher_student_KL_improvement"],
        "no_training": True,
        "closure_level": "full_rank_affine_aligned_teacher_program_inserted_into_student_final_attention",
        "same_basis_transplant_allowed": False,
        "cross_model_claim_allowed": True,
        "missing_compile_atom_count": len(set(missing_compile)),
        "missing_student_heads_affine": missing_heads_affine,
        "missing_student_heads_direct": missing_heads_direct,
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_prompt_affine_aligned_transfer.jsonl", rows_affine)
    write_jsonl(out / "per_prompt_direct_transfer_baseline.jsonl", rows_direct)
    write_jsonl(out / "per_head_affine_aligned_student_attention.jsonl", head_rows_affine)
    write_jsonl(out / "per_head_direct_student_attention.jsonl", head_rows_direct)
    write_jsonl(out / "per_head_compiled_teacher_program.jsonl", compile_rows)

    print("=== Qwen Teacher -> Student Affine Aligned Transfer v1.1 ===")
    print(json.dumps({
        "prompt_mode": report["prompt_mode"],
        "align_prompt_mode": report["align_prompt_mode"],
        "alignment_type": report["alignment_type"],
        "ridge": report["ridge"],
        "pinv_rcond": report["pinv_rcond"],
        "align_rows": report["align_rows"],
        "alignment_rel_before": report["alignment_rel_before"],
        "alignment_rel_after": report["alignment_rel_after"],
        "cka_before": report["cka_before"],
        "cka_after": report["cka_after"],
        "A_condition_number": report["A_condition_number"],
        "direct_logits_gain": report["direct"]["teacher_student_logits_rel_improvement"],
        "affine_logits_gain": report["affine_aligned"]["teacher_student_logits_rel_improvement"],
        "direct_KL_gain": report["direct"]["teacher_student_KL_improvement"],
        "affine_KL_gain": report["affine_aligned"]["teacher_student_KL_improvement"],
        "affine_vs_direct_logits_gain_delta": report["affine_vs_direct_logits_gain_delta"],
        "affine_vs_direct_KL_gain_delta": report["affine_vs_direct_KL_gain_delta"],
        "affine_Y_prog_vs_student_attention_rel": report["affine_aligned"]["Y_prog_vs_student_attention_rel_mean"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
