#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_attention_transfer_v1.py

Teacher -> student no-training attention transfer test.

Important distinction from replay/intervention v1:
  replay v1 rebuilds Level0 from the same model, so it is self-replay.
  real teacher->student transfer must compile teacher program matrices first.

This script:
  1) loads teacher model and teacher saved program bundle;
  2) compiles teacher M_qk_program[h,d] and C_vo_program[h];
  3) loads student model;
  4) applies teacher compiled attention program inside the student's final layer;
  5) compares:
       student_before vs teacher
       student_after_program vs teacher
       student_after_program vs student_before

No training, no KL distillation, no LoRA, no alpha sweep.
First real use-case: same architecture / same residual basis.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import build_prompts, collect_head_data, get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program, build_program_mdelta
from qwen_attention_block_program_replay_v1 import load_vo_program, build_vo_program_C, program_scores_A_for_seq
import qwen_attention_program_intervention_v1 as last_layer_tools

VERSION = "qwen_teacher_student_attention_transfer_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def logits_metrics(logits_a: torch.Tensor, logits_b: torch.Tensor) -> Dict[str, float]:
    # Metrics over full logits plus last-token KL/top-k.
    la = logits_a.float().cpu()
    lb = logits_b.float().cpu()
    lt_a = la[-1]
    lt_b = lb[-1]
    logp_a = F.log_softmax(lt_a, dim=-1)
    logp_b = F.log_softmax(lt_b, dim=-1)
    p_a = torch.exp(logp_a)
    kl_a_to_b = float(torch.sum(p_a * (logp_a - logp_b)))
    top1_a = int(torch.argmax(lt_a).item())
    top1_b = int(torch.argmax(lt_b).item())
    top5_a = set(torch.topk(lt_a, k=5).indices.tolist())
    top5_b = set(torch.topk(lt_b, k=5).indices.tolist())
    return {
        "logits_rel": rel_err(la, lb),
        "last_token_KL_a_to_b": kl_a_to_b,
        "last_top1_match": bool(top1_a == top1_b),
        "last_top5_overlap": len(top5_a & top5_b),
        "top1_a_id": top1_a,
        "top1_b_id": top1_b,
    }


@torch.no_grad()
def collect_model_truth(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str):
    return last_layer_tools.collect_truth(model, tokenizer, prompts, layer_idx, max_length, device)


@torch.no_grad()
def compile_teacher_attention_program(teacher, tokenizer, prompts, layer_idx: int, heads: List[int], args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates):
    compiled: Dict[int, Dict[str, Any]] = {}
    per_head = []
    missing_all = []
    for hi in heads:
        seqs, meta = collect_head_data(teacher, tokenizer, prompts, layer_idx, hi, args.max_length, args.device)
        if not seqs:
            continue
        Mprog, _Mexact, _delta_rows, qk_missing = build_program_mdelta(
            teacher, layer_idx, hi, meta, seqs, args, thresholds, qk_atoms, qk_gates
        )
        Cprog, _Cexact, vo_info = build_vo_program_C(teacher, layer_idx, hi, meta, thresholds, vo_atoms, vo_gates)
        compiled[hi] = {
            "Mdelta": {int(k): v.cpu().float() for k, v in Mprog.items()},
            "Cvo": Cprog.cpu().float(),
            "kv_head": int(meta["kv_idx"]),
        }
        missing_all.extend(qk_missing)
        missing_all.extend(vo_info["missing_atoms"])
        per_head.append({
            "head": hi,
            "kv_head": int(meta["kv_idx"]),
            "qk_deltas": len(Mprog),
            "vo_matrix_err": float(vo_info["program_matrix_err"]),
            "vo_gates": int(vo_info["num_gates"]),
        })
    return compiled, per_head, missing_all


@torch.no_grad()
def apply_compiled_teacher_attention_to_student(student, tokenizer, prompts, compiled, layer_idx: int, heads: List[int], args):
    y_acc: Dict[int, Dict[str, torch.Tensor]] = {}
    per_head = []
    missing_heads = []

    for hi in heads:
        if hi not in compiled:
            missing_heads.append(hi)
            continue
        seqs, meta = collect_head_data(student, tokenizer, prompts, layer_idx, hi, args.max_length, args.device)
        if not seqs:
            continue
        Mdelta = compiled[hi]["Mdelta"]
        Cvo = compiled[hi]["Cvo"]
        score_rels, A_rels, Y_rels, top1s = [], [], [], []
        for s in seqs:
            scores_prog, A_prog, scores_true_student, row_covered, mask_eval = program_scores_A_for_seq(s, Mdelta, args.max_delta)
            payload_prog = s.Xaug @ Cvo.T
            Y_head_prog = A_prog @ payload_prog
            Y_head_student = s.Y
            if int(s.prompt_id) not in y_acc:
                y_acc[int(s.prompt_id)] = {
                    "Y_prog": torch.zeros_like(Y_head_prog.float().cpu()),
                    "Y_student_true": torch.zeros_like(Y_head_student.float().cpu()),
                }
            y_acc[int(s.prompt_id)]["Y_prog"] += Y_head_prog.float().cpu()
            y_acc[int(s.prompt_id)]["Y_student_true"] += Y_head_student.float().cpu()
            if int(mask_eval.sum()) > 0:
                score_rels.append(rel_err(scores_prog[mask_eval], scores_true_student[mask_eval]))
            if int(row_covered.sum()) > 0:
                A_rels.append(rel_err(A_prog[row_covered], s.A[row_covered]))
                top_true = torch.argmax(s.A[row_covered], dim=-1)
                top_prog = torch.argmax(A_prog[row_covered], dim=-1)
                top1s.append(float((top_true == top_prog).float().mean()))
            Y_rels.append(rel_err(Y_head_prog, Y_head_student))
        per_head.append({
            "head": hi,
            "student_score_rel_vs_student_attention": mean(score_rels),
            "student_A_rel_vs_student_attention": mean(A_rels),
            "student_Y_head_rel_vs_student_attention": mean(Y_rels),
            "student_top1_vs_student_attention": mean(top1s),
        })
        print(f"transfer H{hi}: score_vs_student={per_head[-1]['student_score_rel_vs_student_attention']:.3e} A_vs_student={per_head[-1]['student_A_rel_vs_student_attention']:.3e} Yh_vs_student={per_head[-1]['student_Y_head_rel_vs_student_attention']:.3e}")
    return y_acc, per_head, missing_heads


@torch.no_grad()
def run_student_final_layer_with_program(student, student_truth, y_acc, layer_idx: int, device: str):
    layer = student.model.layers[layer_idx]
    dtype = next(student.parameters()).dtype
    out = {}
    for pi, t in sorted(student_truth.items()):
        H_before = t["H_before"].to(device=device, dtype=dtype).unsqueeze(0)
        Y_prog = y_acc[pi]["Y_prog"].to(device=device, dtype=dtype).unsqueeze(0)
        H_attn_prog = H_before + Y_prog
        mlp_in = layer.post_attention_layernorm(H_attn_prog)
        H_layer_prog = H_attn_prog + layer.mlp(mlp_in)
        H_final_prog = student.model.norm(H_layer_prog)
        logits_prog = student.lm_head(H_final_prog)[0].detach().float().cpu()
        out[pi] = {
            "H_final_prog": H_final_prog[0].detach().float().cpu(),
            "logits_prog": logits_prog,
            "Y_prog": y_acc[pi]["Y_prog"].float().cpu(),
            "Y_student_true": y_acc[pi]["Y_student_true"].float().cpu(),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
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
    ap.add_argument("--teacher-improve-eps", type=float, default=1e-6)
    ap.add_argument("--out", default="runs/qwen_teacher_student_attention_transfer_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds)
    qk_atoms, qk_gates = load_qk_program(Path(args.qk_program_run))
    vo_atoms, vo_gates = load_vo_program(Path(args.vo_program_run))
    prompts = build_prompts(args.prompts)
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
        raise SystemExit(f"v1 supports final layer only. layer={args.layer}, final={len(teacher.model.layers)-1}")
    teacher_truth = collect_model_truth(teacher, tok, prompts, args.layer, args.max_length, args.device)
    compiled, compile_rows, missing_compile = compile_teacher_attention_program(
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
    if args.layer != len(student.model.layers) - 1:
        raise SystemExit(f"v1 supports final layer only. layer={args.layer}, student_final={len(student.model.layers)-1}")
    if student.config.hidden_size != len(next(iter(compiled.values()))["Cvo"]):
        # Cvo shape is [H, H+1], len(Cvo)=H rows; this check catches obvious mismatch.
        raise SystemExit("student hidden_size does not match compiled teacher program C_vo rows")

    student_truth = collect_model_truth(student, student_tok, prompts, args.layer, args.max_length, args.device)
    y_acc, transfer_head_rows, missing_heads = apply_compiled_teacher_attention_to_student(
        student, student_tok, prompts, compiled, args.layer, heads, args
    )
    student_prog = run_student_final_layer_with_program(student, student_truth, y_acc, args.layer, args.device)

    per_prompt = []
    for pi in sorted(student_truth.keys()):
        teacher_logits = teacher_truth[pi]["logits_true"]
        student_logits = student_truth[pi]["logits_true"]
        prog_logits = student_prog[pi]["logits_prog"]
        before = logits_metrics(teacher_logits, student_logits)
        after = logits_metrics(teacher_logits, prog_logits)
        student_shift = logits_metrics(student_logits, prog_logits)
        per_prompt.append({
            "prompt_id": pi,
            "text": student_truth[pi]["text"][:120],
            "teacher_student_before_logits_rel": before["logits_rel"],
            "teacher_student_after_logits_rel": after["logits_rel"],
            "teacher_student_before_KL": before["last_token_KL_a_to_b"],
            "teacher_student_after_KL": after["last_token_KL_a_to_b"],
            "teacher_student_before_top1_match": before["last_top1_match"],
            "teacher_student_after_top1_match": after["last_top1_match"],
            "student_before_after_logits_rel": student_shift["logits_rel"],
            "student_before_after_KL": student_shift["last_token_KL_a_to_b"],
            "Y_prog_vs_student_attention_rel": rel_err(student_prog[pi]["Y_prog"], student_prog[pi]["Y_student_true"]),
        })

    before_rel = mean([r["teacher_student_before_logits_rel"] for r in per_prompt])
    after_rel = mean([r["teacher_student_after_logits_rel"] for r in per_prompt])
    before_kl = mean([r["teacher_student_before_KL"] for r in per_prompt])
    after_kl = mean([r["teacher_student_after_KL"] for r in per_prompt])
    shift_rel = mean([r["student_before_after_logits_rel"] for r in per_prompt])
    y_vs_student = mean([r["Y_prog_vs_student_attention_rel"] for r in per_prompt])

    report = {
        "version": VERSION,
        "mode": "teacher_student_attention_program_transfer",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "qk_program_run": args.qk_program_run,
        "vo_program_run": args.vo_program_run,
        "qk_atom_count": len(qk_atoms),
        "vo_atom_count": len(vo_atoms),
        "compiled_heads": len(compiled),
        "prompts": len(per_prompt),
        "teacher_student_before_logits_rel_mean": before_rel,
        "teacher_student_after_logits_rel_mean": after_rel,
        "teacher_student_logits_rel_improvement": before_rel - after_rel,
        "teacher_student_before_KL_mean": before_kl,
        "teacher_student_after_KL_mean": after_kl,
        "teacher_student_KL_improvement": before_kl - after_kl,
        "student_before_after_logits_rel_mean": shift_rel,
        "Y_prog_vs_student_attention_rel_mean": y_vs_student,
        "teacher_transfer_improved_logits": after_rel < before_rel - args.teacher_improve_eps,
        "teacher_transfer_improved_KL": after_kl < before_kl - args.teacher_improve_eps,
        "self_transfer_case": args.teacher_model == args.student_model,
        "status": "TEACHER_STUDENT_ATTENTION_TRANSFER_RAN",
        "no_training": True,
        "closure_level": "teacher_program_inserted_into_student_final_attention",
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "missing_compile_atom_count": len(set(missing_compile)),
        "missing_student_heads": missing_heads,
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_prompt_teacher_student_transfer.jsonl", per_prompt)
    write_jsonl(out / "per_head_compiled_teacher_program.jsonl", compile_rows)
    write_jsonl(out / "per_head_student_transfer_attention.jsonl", transfer_head_rows)

    print("=== Qwen Teacher -> Student Attention Transfer ===")
    print(json.dumps({
        "teacher_model": report["teacher_model"],
        "student_model": report["student_model"],
        "qk_atom_count": report["qk_atom_count"],
        "vo_atom_count": report["vo_atom_count"],
        "compiled_heads": report["compiled_heads"],
        "prompts": report["prompts"],
        "teacher_student_before_logits_rel_mean": report["teacher_student_before_logits_rel_mean"],
        "teacher_student_after_logits_rel_mean": report["teacher_student_after_logits_rel_mean"],
        "teacher_student_logits_rel_improvement": report["teacher_student_logits_rel_improvement"],
        "teacher_student_before_KL_mean": report["teacher_student_before_KL_mean"],
        "teacher_student_after_KL_mean": report["teacher_student_after_KL_mean"],
        "teacher_student_KL_improvement": report["teacher_student_KL_improvement"],
        "student_before_after_logits_rel_mean": report["student_before_after_logits_rel_mean"],
        "Y_prog_vs_student_attention_rel_mean": report["Y_prog_vs_student_attention_rel_mean"],
        "self_transfer_case": report["self_transfer_case"],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
