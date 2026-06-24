#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_head_pseudocode_term_transfer_scan_v1.py

Term-level MATRIX PSEUDOCODE transfer scan for Qwen attention heads.

This is not raw Wq/Wk/Wv/Wo transfer and not whole-head transfer.
It patches the student's head-circuit program by individual pseudocode terms:

  score[i,j] = const_delta[d]
             + q_affine[d](x_i)
             + k_affine[d](x_j)
             + content_bilinear[d](x_i, x_j)

  payload[j] = VO_linear @ x_j + VO_bias
  Y_head[i]  = sum_j softmax(score[i,j]) * payload[j]

For a selected term set, it applies:

  term_student += alpha * (term_teacher - term_student)

Supported candidate specs:
  vo_linear
  vo_bias
  q_affine_all / q_affine_medium
  k_affine_all / k_affine_medium
  content_all / content_medium
  const_all / const_medium
  qk_affine_medium          = q_affine_medium + k_affine_medium
  qk_cond_medium            = content_medium + q_affine_medium + k_affine_medium
  vo_linear+qk_affine_medium
  vo_linear+qk_cond_medium

"medium" means d in --medium-deltas, default 7:13.
All non-patched heads remain native student, using the known-good v4.1 replay
path. No training, no distillation, no LoRA.
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
from qwen_circuit_target_roundtrip_v1 import get_dtype, parse_ints
from qwen_qk_program_attention_replay_v1 import load_saved_program as load_qk_program
from qwen_attention_block_program_replay_v1 import load_vo_program, program_scores_A_for_seq
from qwen_teacher_student_code_transfer_v2 import select_prompts
import qwen_teacher_student_attention_transfer_v1 as ts
import qwen_teacher_student_delta_head_scan_v4 as v4

VERSION = "qwen_head_pseudocode_term_transfer_scan_v1.0"


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_delta_range(s: str) -> Set[int]:
    out: Set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def parse_candidates(s: str) -> List[str]:
    # Semicolon is preferred because + and comma are meaningful inside specs.
    return [x.strip() for x in s.split(";") if x.strip()]


def expand_spec(spec: str) -> Tuple[Set[str], str]:
    """Return base terms and delta mode for a candidate spec."""
    terms: Set[str] = set()
    delta_mode = "all"
    for token in spec.split("+"):
        t = token.strip()
        if not t:
            continue
        if t == "vo_linear":
            terms.add("vo_linear")
        elif t == "vo_bias":
            terms.add("vo_bias")
        elif t == "qk_affine_medium":
            terms.update(["q_affine", "k_affine"]); delta_mode = "medium"
        elif t == "qk_affine_all":
            terms.update(["q_affine", "k_affine"])
        elif t == "qk_cond_medium":
            terms.update(["content", "q_affine", "k_affine"]); delta_mode = "medium"
        elif t == "qk_cond_all":
            terms.update(["content", "q_affine", "k_affine"])
        elif t.endswith("_medium"):
            terms.add(t[:-7]); delta_mode = "medium"
        elif t.endswith("_all"):
            terms.add(t[:-4])
        else:
            terms.add(t)
    allowed = {"content", "q_affine", "k_affine", "const", "vo_linear", "vo_bias"}
    bad = sorted(terms - allowed)
    if bad:
        raise ValueError(f"Unknown term(s) in candidate {spec!r}: {bad}")
    return terms, delta_mode


def split_dims(M: torch.Tensor) -> int:
    return int(M.shape[0] - 1)


def mix_program_terms(student_pack: Dict[str, Any], teacher_pack: Dict[str, Any], alpha: float, candidate_spec: str, medium_deltas: Set[int]) -> Dict[str, Any]:
    terms, delta_mode = expand_spec(candidate_spec)
    out = {
        "Mdelta": {},
        "Cvo": student_pack["Cvo"].clone().float().cpu(),
        "kv_head": int(student_pack.get("kv_head", -1)),
        "candidate_spec": candidate_spec,
        "terms": sorted(terms),
        "delta_mode": delta_mode,
    }

    ds = sorted(set(student_pack["Mdelta"].keys()) & set(teacher_pack["Mdelta"].keys()))
    for d in ds:
        Ms = student_pack["Mdelta"][d].clone().float().cpu()
        Mt = teacher_pack["Mdelta"][d].float().cpu()
        H = split_dims(Ms)
        use_d = True if delta_mode == "all" else (int(d) in medium_deltas)
        if use_d:
            if "content" in terms:
                Ms[:H, :H] = Ms[:H, :H] + float(alpha) * (Mt[:H, :H] - Ms[:H, :H])
            if "q_affine" in terms:
                Ms[:H, H] = Ms[:H, H] + float(alpha) * (Mt[:H, H] - Ms[:H, H])
            if "k_affine" in terms:
                Ms[H, :H] = Ms[H, :H] + float(alpha) * (Mt[H, :H] - Ms[H, :H])
            if "const" in terms:
                Ms[H, H] = Ms[H, H] + float(alpha) * (Mt[H, H] - Ms[H, H])
        out["Mdelta"][int(d)] = Ms

    Cs = student_pack["Cvo"].clone().float().cpu()
    Ct = teacher_pack["Cvo"].float().cpu()
    H = int(Cs.shape[1] - 1)
    if "vo_linear" in terms:
        Cs[:, :H] = Cs[:, :H] + float(alpha) * (Ct[:, :H] - Cs[:, :H])
    if "vo_bias" in terms:
        Cs[:, H] = Cs[:, H] + float(alpha) * (Ct[:, H] - Cs[:, H])
    out["Cvo"] = Cs
    return out


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


@torch.no_grad()
def build_y_acc_for_term_patch(student_head_data, compiled_student, compiled_teacher, patch_heads: Set[int], alpha: float, candidate_spec: str, medium_deltas: Set[int], args):
    y_acc: Dict[int, Dict[str, torch.Tensor]] = {}
    per_patched_head = []
    missing = []

    patch_programs = {}
    for hi in patch_heads:
        if hi not in compiled_student or hi not in compiled_teacher:
            missing.append(hi)
            continue
        patch_programs[hi] = mix_program_terms(compiled_student[hi], compiled_teacher[hi], alpha, candidate_spec, medium_deltas)

    for hi, pack in student_head_data.items():
        seqs = pack["seqs"]
        use_patch = hi in patch_programs
        score_rels, A_rels, Y_rels, top1s = [], [], [], []

        for s in seqs:
            pid = int(s.prompt_id)
            if pid not in y_acc:
                y_acc[pid] = {
                    "Y_prog": torch.zeros_like(s.Y.float().cpu()),
                    "Y_student_true": torch.zeros_like(s.Y.float().cpu()),
                }

            if use_patch:
                Mdelta = patch_programs[hi]["Mdelta"]
                Cvo = patch_programs[hi]["Cvo"]
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
                Y_head_prog = s.Y

            y_acc[pid]["Y_prog"] += Y_head_prog.float().cpu()
            y_acc[pid]["Y_student_true"] += s.Y.float().cpu()

        if use_patch:
            per_patched_head.append({
                "head": hi,
                "alpha": float(alpha),
                "candidate_spec": candidate_spec,
                "score_vs_student": mean(score_rels) if score_rels else None,
                "A_vs_student": mean(A_rels) if A_rels else None,
                "Yh_vs_student": mean(Y_rels) if Y_rels else None,
                "top1_vs_student": mean(top1s) if top1s else None,
            })

    return y_acc, per_patched_head, missing


def candidate_score(sums, kl_weight: float, shift_weight: float, y_weight: float):
    kl_damage = max(0.0, -float(sums["KL_improvement"]))
    return float(sums["logits_improvement"]) - kl_weight * kl_damage - shift_weight * float(sums["student_shift_logits_rel"]) - y_weight * float(sums["Y_prog_vs_student_attention_rel"])


def sanity_check_base(base_sums: dict, eps: float = 1e-5) -> dict:
    y = abs(float(base_sums.get("Y_prog_vs_student_attention_rel", 999.0)))
    shift = abs(float(base_sums.get("student_shift_logits_rel", 999.0)))
    return {
        "base_y_closed": y <= eps,
        "base_student_shift_closed": shift <= eps,
        "base_sanity_pass": y <= eps and shift <= eps,
        "eps": eps,
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
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/head_pseudocode_term_transfer_scan_v1")
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
        args.teacher_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device).eval()
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
    ).to(args.device).eval()
    student_truth = ts.collect_model_truth(student, tok_student, prompts, args.layer, args.max_length, args.device)
    compiled_student, student_compile_rows, missing_student = ts.compile_teacher_attention_program(
        student, tok_student, prompts, args.layer, all_heads, args, thresholds, sqk_atoms, sqk_gates, svo_atoms, svo_gates
    )
    student_head_data = v4.precompute_student_heads(student, tok_student, prompts, args.layer, all_heads, args)

    base_y_acc, _base_head_rows, _base_missing = build_y_acc_for_term_patch(
        student_head_data, compiled_student, compiled_teacher, set(), 0.0, "vo_linear", medium_deltas, args
    )
    base_rows = v4.evaluate_patch(student, student_truth, teacher_truth, base_y_acc, args.layer, args.device)
    base_sums = v4.summarize_rows(base_rows)
    base_sanity = sanity_check_base(base_sums, eps=args.base_sanity_eps)
    print(f"base_no_patch: shift={base_sums['student_shift_logits_rel']:.6e} Y={base_sums['Y_prog_vs_student_attention_rel']:.6e} sanity={base_sanity['base_sanity_pass']}")

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
        rows = v4.evaluate_patch(student, student_truth, teacher_truth, y_acc, args.layer, args.device)
        sums = v4.summarize_rows(rows)
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
        "mode": "head_pseudocode_term_level_transfer_scan",
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

    print("=== Qwen Head Pseudocode Term Transfer Scan v1 ===")
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
