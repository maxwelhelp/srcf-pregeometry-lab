#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_head_circuit_similarity_report_v1.py

Visual/readable similarity report for Qwen teacher/student head-circuit programs.

This is a report layer on top of qwen_head_circuit_program_diff_v1.py. It does
not compare raw Wq/Wk/Wv/Wo. It recomputes each head as a functional circuit:

  QK: M_qk_aug[h,d]
  VO: C_vo_aug[h]

then compares teacher heads against student heads, including cross-head matching.

Outputs:
  - manifest.json
  - head_pair_similarity.jsonl
  - focus_head_delta_by_delta.jsonl
  - head_circuit_similarity_report.md

Use this when you want to "look with your eyes" at which heads/functions are
similar and where the real program delta lives.
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
from qwen_circuit_target_roundtrip_v1 import build_prompts, get_dtype, parse_ints
from qwen_head_circuit_program_diff_v1 import (
    exact_head_targets,
    compile_head_program,
    load_optional_programs,
    rel_err,
    cosine,
    decode_level0_matrix,
)

VERSION = "qwen_head_circuit_similarity_report_v1.0"


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def safe(x, digits=4):
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def pair_metrics(t_compiled: Dict[str, Any], s_compiled: Dict[str, Any], max_delta: int):
    qk_rows = []
    for d in range(max_delta + 1):
        if d not in t_compiled["M_prog"] or d not in s_compiled["M_prog"]:
            continue
        tm = t_compiled["M_prog"][d]
        sm = s_compiled["M_prog"][d]
        qk_rows.append({
            "delta": d,
            "qk_rel": rel_err(tm, sm),
            "qk_cos": cosine(tm, sm),
            "qk_delta_norm": float(torch.linalg.norm((tm - sm).float()).item()),
            "teacher_norm": float(torch.linalg.norm(tm.float()).item()),
            "student_norm": float(torch.linalg.norm(sm.float()).item()),
        })
    vc = cosine(t_compiled["C_prog"], s_compiled["C_prog"])
    vr = rel_err(t_compiled["C_prog"], s_compiled["C_prog"])
    return {
        "qk_rows": qk_rows,
        "qk_rel_mean": mean([r["qk_rel"] for r in qk_rows]),
        "qk_cos_mean": mean([r["qk_cos"] for r in qk_rows]),
        "qk_delta_norm_mean": mean([r["qk_delta_norm"] for r in qk_rows]),
        "vo_rel": vr,
        "vo_cos": vc,
        "vo_delta_norm": float(torch.linalg.norm((t_compiled["C_prog"] - s_compiled["C_prog"]).float()).item()),
    }


def combined_score(row: Dict[str, Any], qk_weight: float, vo_weight: float):
    # high is more similar: cosine high, rel low. VO usually has stronger write meaning.
    qk = float(row["qk_cos_mean"]) - 0.25 * float(row["qk_rel_mean"])
    vo = float(row["vo_cos"]) - 0.25 * float(row["vo_rel"])
    return qk_weight * qk + vo_weight * vo


def make_md(report: Dict[str, Any], pair_rows: List[Dict[str, Any]], focus_rows: List[Dict[str, Any]], top: int):
    lines = []
    lines.append("# Qwen head-circuit similarity report")
    lines.append("")
    lines.append("Compares teacher/student attention heads as full circuit programs, not raw projection weights.")
    lines.append("")
    lines.append("## Summary")
    lines.append("```json")
    lines.append(json.dumps(report["summary"], indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## Best student match for each teacher head")
    lines.append("| teacher head | best student head | score | QK cos | QK rel | VO cos | VO rel |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for h in report["heads"]:
        cand = [r for r in pair_rows if r["teacher_head"] == h]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["similarity_score"])
        lines.append(
            f"| H{h} | H{best['student_head']} | {safe(best['similarity_score'])} | {safe(best['qk_cos_mean'])} | {safe(best['qk_rel_mean'])} | {safe(best['vo_cos'])} | {safe(best['vo_rel'])} |"
        )
    lines.append("")

    lines.append(f"## Top {top} most similar head pairs")
    lines.append("| rank | teacher | student | score | QK cos | QK rel | VO cos | VO rel |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(sorted(pair_rows, key=lambda x: x["similarity_score"], reverse=True)[:top], 1):
        lines.append(
            f"| {i} | H{r['teacher_head']} | H{r['student_head']} | {safe(r['similarity_score'])} | {safe(r['qk_cos_mean'])} | {safe(r['qk_rel_mean'])} | {safe(r['vo_cos'])} | {safe(r['vo_rel'])} |"
        )
    lines.append("")

    lines.append(f"## Top {top} most different same-index heads")
    same = [r for r in pair_rows if r["teacher_head"] == r["student_head"]]
    lines.append("| rank | head | score | QK cos | QK rel | VO cos | VO rel |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for i, r in enumerate(sorted(same, key=lambda x: x["similarity_score"])[:top], 1):
        lines.append(
            f"| {i} | H{r['teacher_head']} | {safe(r['similarity_score'])} | {safe(r['qk_cos_mean'])} | {safe(r['qk_rel_mean'])} | {safe(r['vo_cos'])} | {safe(r['vo_rel'])} |"
        )
    lines.append("")

    lines.append("## Focus head delta by delta")
    lines.append("| teacher | student | delta | QK cos | QK rel | delta norm | teacher norm | student norm |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in focus_rows[: max(1, top * 3)]:
        lines.append(
            f"| H{r['teacher_head']} | H{r['student_head']} | d={r['delta']} | {safe(r['qk_cos'])} | {safe(r['qk_rel'])} | {safe(r['qk_delta_norm'])} | {safe(r['teacher_norm'])} | {safe(r['student_norm'])} |"
        )
    lines.append("")

    lines.append("## Interpretation hints")
    lines.append("- If same-index head has weak VO cosine but another student head has stronger VO cosine, head roles may be permuted or distributed.")
    lines.append("- If QK cosine is high but VO cosine is low, heads route similarly but write different residual features.")
    lines.append("- If VO differs strongly across all matches, code/style difference may be in write-space, not routing-space.")
    lines.append("- Low Level-0 coverage in the previous diff means the current analytic primitive dictionary is too weak for the delta; visual matrix/head similarity is more informative than primitive names.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--focus-heads", default="1")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--teacher-qk-program-run", default=None)
    ap.add_argument("--teacher-vo-program-run", default=None)
    ap.add_argument("--student-qk-program-run", default=None)
    ap.add_argument("--student-vo-program-run", default=None)
    ap.add_argument("--qk-weight", type=float, default=0.4)
    ap.add_argument("--vo-weight", type=float, default=0.6)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/head_circuit_similarity_report_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    heads = parse_ints(args.heads)
    focus_heads = parse_ints(args.focus_heads)
    prompts = build_prompts(args.prompts)
    thresholds = load_thresholds(args.thresholds)

    tqk_atoms, tqk_gates, tvo_atoms, tvo_gates = load_optional_programs(args.teacher_qk_program_run, args.teacher_vo_program_run)
    sqk_atoms, sqk_gates, svo_atoms, svo_gates = load_optional_programs(args.student_qk_program_run, args.student_vo_program_run)

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

    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    student.eval()

    teacher_compiled: Dict[int, Dict[str, Any]] = {}
    student_compiled: Dict[int, Dict[str, Any]] = {}

    class Obj: pass
    shim = Obj()
    shim.max_delta = args.max_delta
    shim.max_length = args.max_length
    shim.device = args.device
    shim.max_level0_ops = 64

    print("building head-circuit programs...")
    for h in heads:
        tt = exact_head_targets(teacher, tok_t, prompts, args.layer, h, args.max_length, args.max_delta, args.device)
        st = exact_head_targets(student, tok_s, prompts, args.layer, h, args.max_length, args.max_delta, args.device)
        if tt is None or st is None:
            continue
        teacher_compiled[h] = compile_head_program(teacher, args.layer, h, tt, shim, thresholds, tqk_atoms, tqk_gates, tvo_atoms, tvo_gates)
        student_compiled[h] = compile_head_program(student, args.layer, h, st, shim, thresholds, sqk_atoms, sqk_gates, svo_atoms, svo_gates)
        print(f"H{h}: built teacher/student M_qk + C_vo programs")

    pair_rows = []
    focus_rows = []
    for ht in heads:
        if ht not in teacher_compiled:
            continue
        for hs in heads:
            if hs not in student_compiled:
                continue
            pm = pair_metrics(teacher_compiled[ht], student_compiled[hs], args.max_delta)
            row = {
                "layer": args.layer,
                "teacher_head": ht,
                "student_head": hs,
                "qk_rel_mean": pm["qk_rel_mean"],
                "qk_cos_mean": pm["qk_cos_mean"],
                "qk_delta_norm_mean": pm["qk_delta_norm_mean"],
                "vo_rel": pm["vo_rel"],
                "vo_cos": pm["vo_cos"],
                "vo_delta_norm": pm["vo_delta_norm"],
            }
            row["similarity_score"] = combined_score(row, args.qk_weight, args.vo_weight)
            pair_rows.append(row)
            if ht in focus_heads or hs in focus_heads:
                for qr in pm["qk_rows"]:
                    fr = dict(qr)
                    fr.update({"layer": args.layer, "teacher_head": ht, "student_head": hs, "pair_similarity_score": row["similarity_score"]})
                    focus_rows.append(fr)

    best_by_teacher = {}
    for h in heads:
        cand = [r for r in pair_rows if r["teacher_head"] == h]
        if cand:
            best_by_teacher[str(h)] = max(cand, key=lambda r: r["similarity_score"])

    same_index = [r for r in pair_rows if r["teacher_head"] == r["student_head"]]
    summary = {
        "heads_compiled": len(teacher_compiled),
        "pair_count": len(pair_rows),
        "same_index_qk_cos_mean": mean([r["qk_cos_mean"] for r in same_index]),
        "same_index_vo_cos_mean": mean([r["vo_cos"] for r in same_index]),
        "same_index_qk_rel_mean": mean([r["qk_rel_mean"] for r in same_index]),
        "same_index_vo_rel_mean": mean([r["vo_rel"] for r in same_index]),
        "best_pair": max(pair_rows, key=lambda r: r["similarity_score"]) if pair_rows else None,
        "worst_same_index_pair": min(same_index, key=lambda r: r["similarity_score"]) if same_index else None,
    }

    report = {
        "version": VERSION,
        "mode": "visual_head_circuit_similarity_report",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "heads": heads,
        "focus_heads": focus_heads,
        "max_delta": args.max_delta,
        "prompts": args.prompts,
        "qk_weight": args.qk_weight,
        "vo_weight": args.vo_weight,
        "uses_head_circuit_targets": True,
        "uses_raw_weight_diff": False,
        "summary": summary,
        "best_by_teacher": best_by_teacher,
        "status": "HEAD_CIRCUIT_SIMILARITY_REPORT_RAN",
        "no_training": True,
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "head_pair_similarity.jsonl", sorted(pair_rows, key=lambda r: r["similarity_score"], reverse=True))
    write_jsonl(out / "focus_head_delta_by_delta.jsonl", sorted(focus_rows, key=lambda r: (r["teacher_head"], r["student_head"], r["delta"])))
    md = make_md(report, pair_rows, focus_rows, args.top)
    (out / "head_circuit_similarity_report.md").write_text(md, encoding="utf-8")

    del teacher, student
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("=== Qwen Head-Circuit Similarity Report v1 ===")
    print(json.dumps({
        "summary": summary,
        "top_pairs": sorted(pair_rows, key=lambda r: r["similarity_score"], reverse=True)[: min(args.top, 10)],
        "status": report["status"],
    }, indent=2))
    print(f"markdown={out / 'head_circuit_similarity_report.md'}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
