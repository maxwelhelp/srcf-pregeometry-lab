#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_contrastive_gradient_weight_transfer_heldout_v1_2.py

Heldout evaluation for directional contrastive gradient weight transfer.

Build mask on train/gradient prompts:
  train_code, train_retain

Apply selective teacher-student delta and evaluate on both:
  train_code/train_retain and heldout_code/heldout_retain

This checks whether the clean code-KL / low-retain-damage result generalizes
beyond the same prompts used to build the gradient mask.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
from qwen_teacher_student_code_transfer_v2 import select_prompts
import qwen_contrastive_gradient_weight_transfer_v1_1 as base

VERSION = "qwen_contrastive_gradient_weight_transfer_heldout_v1.2"


def prompt_split(mode: str, grad_n: int, eval_n: int, eval_offset: int) -> tuple[list[str], list[str]]:
    need = max(grad_n, eval_offset + eval_n)
    prompts = select_prompts(mode, need)
    if len(prompts) < need:
        raise RuntimeError(f"select_prompts({mode}, {need}) returned only {len(prompts)} prompts")
    grad = prompts[:grad_n]
    heldout = prompts[eval_offset:eval_offset + eval_n]
    overlap = set(grad) & set(heldout)
    if overlap:
        raise RuntimeError(f"train/heldout overlap for {mode}: {len(overlap)} prompts")
    return grad, heldout


def prefixed_gain(before: Dict[str, Any], after: Dict[str, Any], label: str) -> Dict[str, float]:
    return base.gain(before, after, label)


def heldout_score(row: Dict[str, float], retain_damage_weight: float, retain_shift_weight: float) -> float:
    retain_damage = max(0.0, -float(row["heldout_retain_KL_gain"]))
    retain_shift = abs(float(row["heldout_retain_logits_gain"]))
    return (
        float(row["heldout_code_KL_gain"])
        + float(row["heldout_code_logits_gain"])
        - retain_damage_weight * retain_damage
        - retain_shift_weight * retain_shift
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="18,23")
    ap.add_argument("--components", default="vo,mlp")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--grad-prompts", type=int, default=8)
    ap.add_argument("--eval-prompts", type=int, default=8)
    ap.add_argument("--eval-offset", type=int, default=8)
    ap.add_argument("--alphas", default="0.025,0.05,0.075")
    ap.add_argument("--mask-top-frac", type=float, default=0.001)
    ap.add_argument("--retain-ratio", type=float, default=0.25)
    ap.add_argument("--eps-scale", type=float, default=0.05)
    ap.add_argument("--retain-damage-weight", type=float, default=2.0)
    ap.add_argument("--retain-shift-weight", type=float, default=0.25)
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/contrastive_gradient_weight_transfer_heldout_v1_2")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = base.parse_ints(args.layers)
    components = base.parse_strs(args.components)
    alphas = base.parse_floats(args.alphas)

    train_code, heldout_code = prompt_split("code", args.grad_prompts, args.eval_prompts, args.eval_offset)
    train_retain, heldout_retain = prompt_split("retain", args.grad_prompts, args.eval_prompts, args.eval_offset)

    tok_t = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    student.eval()

    names = base.select_parameter_names(student, layers, components)
    if not names:
        raise SystemExit("no selected parameters; check --layers/--components")
    print(f"selected params: {len(names)}")
    for n in names[:30]:
        print(" ", n)
    if len(names) > 30:
        print(" ...")

    params = base.set_trainable_only(student, names)

    print("baseline eval...")
    base_train_code, base_train_code_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, train_code, args.max_length, args.device, "train_code")
    base_train_retain, base_train_retain_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, train_retain, args.max_length, args.device, "train_retain")
    base_heldout_code, base_heldout_code_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, heldout_code, args.max_length, args.device, "heldout_code")
    base_heldout_retain, base_heldout_retain_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, heldout_retain, args.max_length, args.device, "heldout_retain")

    print("code gradients on train prompts...")
    grads_code, code_loss = base.compute_grad_map(teacher, student, tok_t, tok_s, train_code, params, args.max_length, args.device)
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("retain gradients on train prompts...")
    grads_retain, retain_loss = base.compute_grad_map(teacher, student, tok_t, tok_s, train_retain, params, args.max_length, args.device)

    print("directional masks...")
    masks, mask_rows = base.make_directional_masks(teacher, student, names, grads_code, grads_retain, args.mask_top_frac, args.retain_ratio, args.eps_scale)
    total = sum(r["numel"] for r in mask_rows)
    selected = sum(r["selected"] for r in mask_rows)
    print(f"selected weights: {selected}/{total} = {selected / max(1, total):.8f}")

    backup = base.backup_params(student, names)
    candidate_rows: List[Dict[str, Any]] = []
    per_prompt_rows: List[Dict[str, Any]] = []
    patch_rows_all: List[Dict[str, Any]] = []

    for alpha in alphas:
        base.restore_params(student, backup)
        patch_rows = base.apply_masked_delta(student, teacher, names, masks, alpha)
        patch_rows_all.extend(patch_rows)

        after_train_code, after_train_code_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, train_code, args.max_length, args.device, "train_code")
        after_train_retain, after_train_retain_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, train_retain, args.max_length, args.device, "train_retain")
        after_heldout_code, after_heldout_code_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, heldout_code, args.max_length, args.device, "heldout_code")
        after_heldout_retain, after_heldout_retain_rows = base.eval_prompt_set(teacher, student, tok_t, tok_s, heldout_retain, args.max_length, args.device, "heldout_retain")

        row: Dict[str, Any] = {
            "candidate": f"heldout_directional_contrastive_grad_a{alpha:g}",
            "alpha": float(alpha),
            **prefixed_gain(base_train_code, after_train_code, "train_code"),
            **prefixed_gain(base_train_retain, after_train_retain, "train_retain"),
            **prefixed_gain(base_heldout_code, after_heldout_code, "heldout_code"),
            **prefixed_gain(base_heldout_retain, after_heldout_retain, "heldout_retain"),
            "selected_weights": selected,
            "selected_frac": selected / max(1, total),
        }
        row["heldout_selection_score"] = heldout_score(row, args.retain_damage_weight, args.retain_shift_weight)
        row["train_selection_score"] = base.candidate_score({
            "code_KL_gain": row["train_code_KL_gain"],
            "code_logits_gain": row["train_code_logits_gain"],
            "retain_KL_gain": row["train_retain_KL_gain"],
            "retain_logits_gain": row["train_retain_logits_gain"],
        }, args.retain_damage_weight, args.retain_shift_weight)
        candidate_rows.append(row)

        for rr in after_train_code_rows + after_train_retain_rows + after_heldout_code_rows + after_heldout_retain_rows:
            rr["alpha"] = float(alpha)
            rr["candidate"] = row["candidate"]
            per_prompt_rows.append(rr)

        print(
            f"a={alpha:g}: "
            f"train_code_KL={row['train_code_KL_gain']:+.5f} "
            f"train_retain_KL={row['train_retain_KL_gain']:+.5f} "
            f"heldout_code_KL={row['heldout_code_KL_gain']:+.5f} "
            f"heldout_retain_KL={row['heldout_retain_KL_gain']:+.5f} "
            f"heldout_score={row['heldout_selection_score']:+.5f}"
        )

    base.restore_params(student, backup)
    candidate_rows = sorted(candidate_rows, key=lambda x: x["heldout_selection_score"], reverse=True)

    report = {
        "version": VERSION,
        "mode": "heldout_directional_contrastive_gradient_weight_delta",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layers": layers if layers else "all",
        "components": components,
        "grad_prompts": args.grad_prompts,
        "eval_prompts": args.eval_prompts,
        "eval_offset": args.eval_offset,
        "alphas": alphas,
        "mask_top_frac": args.mask_top_frac,
        "retain_ratio": args.retain_ratio,
        "eps_scale": args.eps_scale,
        "selected_parameter_count": len(names),
        "selected_weights": selected,
        "selected_frac": selected / max(1, total),
        "gradient_losses": {"train_code_loss": code_loss, "train_retain_loss": retain_loss},
        "baseline": {
            **base_train_code,
            **base_train_retain,
            **base_heldout_code,
            **base_heldout_retain,
        },
        "best_candidates": candidate_rows,
        "status": "HELDOUT_DIRECTIONAL_CONTRASTIVE_GRADIENT_TRANSFER_RAN",
        "no_training": True,
        "closure_level": "weight_level_directional_masked_delta_heldout_eval",
        "note": "Mask is built only on train_code/train_retain prompts; heldout_code/heldout_retain are evaluated after patching.",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_parameter_directional_mask_stats.jsonl", mask_rows)
    write_jsonl(out / "per_candidate_heldout_weight_transfer.jsonl", candidate_rows)
    write_jsonl(out / "per_prompt_after_heldout_weight_transfer.jsonl", per_prompt_rows)
    write_jsonl(out / "per_parameter_patch_stats.jsonl", patch_rows_all)
    write_jsonl(out / "baseline_train_code.jsonl", base_train_code_rows)
    write_jsonl(out / "baseline_train_retain.jsonl", base_train_retain_rows)
    write_jsonl(out / "baseline_heldout_code.jsonl", base_heldout_code_rows)
    write_jsonl(out / "baseline_heldout_retain.jsonl", base_heldout_retain_rows)

    print("=== Qwen Heldout Directional Contrastive Gradient Transfer v1.2 ===")
    print(json.dumps({
        "layers": report["layers"],
        "components": report["components"],
        "grad_prompts": report["grad_prompts"],
        "eval_prompts": report["eval_prompts"],
        "eval_offset": report["eval_offset"],
        "selected_frac": report["selected_frac"],
        "gradient_losses": report["gradient_losses"],
        "top": [
            {
                "candidate": r["candidate"],
                "train_code_KL_gain": r["train_code_KL_gain"],
                "train_retain_KL_gain": r["train_retain_KL_gain"],
                "heldout_code_KL_gain": r["heldout_code_KL_gain"],
                "heldout_retain_KL_gain": r["heldout_retain_KL_gain"],
                "heldout_code_logits_gain": r["heldout_code_logits_gain"],
                "heldout_retain_logits_gain": r["heldout_retain_logits_gain"],
                "heldout_selection_score": r["heldout_selection_score"],
            }
            for r in candidate_rows[:10]
        ],
        "status": report["status"],
    }, indent=2))
    print("out=", out)


if __name__ == "__main__":
    main()
