#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_readable_nonprivate_program_diff_v1.py

Readable prompt-independent program diff for Coder vs Instruct using decoded
QK/VO program runs, but excluding private_exact atoms.

Why:
  The full autoexpand diff is dominated by model-specific private residual atoms.
  This script removes that tail and builds a joint canonical basis over only
  non-private atoms, then prints per-head Coder vs Student operator coefficients.

It is not a transfer experiment. It is an inspection/report script.

Inputs are local run directories produced by:
  qwen_joint_autoexpand_closure_v2.py   for QK
  qwen_vo_autoexpand_replay_v1.py       for VO

Outputs:
  manifest.json
  canonical_clusters.jsonl
  qk_by_head_diff.jsonl
  vo_by_head_diff.jsonl
  readable_nonprivate_program_diff.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from program_dsl_v1 import write_json, write_jsonl

VERSION = "qwen_readable_nonprivate_program_diff_v1.0"


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def flat_unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    f = x.float().reshape(-1)
    return f / torch.linalg.norm(f).clamp_min(eps)


def infer_stage(atom_id: str, kind: str) -> str:
    a = atom_id.lower()
    if kind == "vo":
        if "private_exact" in a:
            return "vo_private_exact"
        if "shared" in a:
            return "vo_shared"
        return "vo_nonprivate_unknown"
    if "private_exact" in a:
        return "qk_private_exact"
    if "shared" in a:
        return "qk_shared"
    if "head_specific" in a:
        return "qk_head_specific"
    if "delta_bucket" in a:
        return "qk_delta_bucket"
    return "qk_nonprivate_unknown"


def is_private(atom_id: str) -> bool:
    return "private_exact" in atom_id.lower()


def is_style_like_stage(stage: str) -> bool:
    # Triage heuristic only: shared broad atoms are more style/global-like.
    return "shared" in stage


def is_knowledge_like_stage(stage: str) -> bool:
    return not is_style_like_stage(stage)


def load_qk_program(run: Path):
    atoms_path = run / "qk_autoexpand_atoms.pt"
    gates_path = run / "per_matrix_autoexpand_closure.jsonl"
    if not atoms_path.exists():
        raise FileNotFoundError(atoms_path)
    if not gates_path.exists():
        raise FileNotFoundError(gates_path)
    atoms: Dict[str, torch.Tensor] = torch.load(atoms_path, map_location="cpu")
    gates: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for row in read_jsonl(gates_path):
        key = (int(row["layer"]), int(row["head"]), int(row["delta"]))
        gates[key] = {str(k): float(v) for k, v in row.get("gates", {}).items()}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    return atoms, gates, manifest


def load_vo_program(run: Path):
    atoms_path = run / "vo_autoexpand_atoms.pt"
    gates_path = run / "per_head_vo_matrix_closure.jsonl"
    if not atoms_path.exists():
        raise FileNotFoundError(atoms_path)
    if not gates_path.exists():
        raise FileNotFoundError(gates_path)
    atoms: Dict[str, torch.Tensor] = torch.load(atoms_path, map_location="cpu")
    gates: Dict[Tuple[int, int], Dict[str, float]] = {}
    for row in read_jsonl(gates_path):
        key = (int(row["layer"]), int(row["head"]))
        gates[key] = {str(k): float(v) for k, v in row.get("gates", {}).items()}
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    return atoms, gates, manifest


def make_atom_entries(kind: str, model: str, atoms: Dict[str, torch.Tensor], include_private: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for atom_id, tensor in atoms.items():
        if (not include_private) and is_private(atom_id):
            continue
        stage = infer_stage(atom_id, kind)
        out.append({
            "kind": kind,
            "model": model,
            "atom_id": atom_id,
            "stage": stage,
            "shape": tuple(tensor.shape),
            "tensor": tensor.float().cpu(),
            "unit": flat_unit(tensor),
        })
    return out


def joint_cluster(entries: List[Dict[str, Any]], kind: str, min_cos: float) -> Tuple[Dict[Tuple[str, str], Tuple[str, float]], List[Dict[str, Any]]]:
    """Greedy shape-aware signed clustering. Returns atom map: (model, atom_id)->(cluster_id, sign)."""
    clusters: List[Dict[str, Any]] = []
    atom_map: Dict[Tuple[str, str], Tuple[str, float]] = {}

    # Prefer shared/promoted entries first, then head/delta. This makes cluster labels easier to read.
    stage_order = {"qk_shared": 0, "vo_shared": 0, "qk_head_specific": 1, "qk_delta_bucket": 2}
    entries = sorted(entries, key=lambda e: (stage_order.get(e["stage"], 9), e["model"], e["atom_id"]))

    for e in entries:
        best_i = None
        best_cos = 0.0
        for i, c in enumerate(clusters):
            if e["shape"] != c["shape"]:
                continue
            cos = float(e["unit"] @ c["rep_unit"])
            if abs(cos) > abs(best_cos):
                best_i = i
                best_cos = cos
        if best_i is not None and abs(best_cos) >= min_cos:
            cid = clusters[best_i]["cluster_id"]
            sign = 1.0 if best_cos >= 0 else -1.0
            clusters[best_i]["members"].append({
                "model": e["model"],
                "atom_id": e["atom_id"],
                "stage": e["stage"],
                "cos_to_rep": best_cos,
                "sign": sign,
            })
            clusters[best_i]["stage_counts"][e["stage"]] += 1
            clusters[best_i]["model_counts"][e["model"]] += 1
            atom_map[(e["model"], e["atom_id"])] = (cid, sign)
        else:
            cid = f"{kind}_canon_{len(clusters):04d}"
            clusters.append({
                "cluster_id": cid,
                "kind": kind,
                "shape": e["shape"],
                "rep_model": e["model"],
                "rep_atom": e["atom_id"],
                "rep_stage": e["stage"],
                "rep_unit": e["unit"],
                "stage_counts": Counter({e["stage"]: 1}),
                "model_counts": Counter({e["model"]: 1}),
                "members": [{
                    "model": e["model"],
                    "atom_id": e["atom_id"],
                    "stage": e["stage"],
                    "cos_to_rep": 1.0,
                    "sign": 1.0,
                }],
            })
            atom_map[(e["model"], e["atom_id"])] = (cid, 1.0)

    rows: List[Dict[str, Any]] = []
    for c in clusters:
        stage_counts = dict(c["stage_counts"])
        model_counts = dict(c["model_counts"])
        stage = max(stage_counts.items(), key=lambda kv: kv[1])[0]
        rows.append({
            "cluster_id": c["cluster_id"],
            "kind": c["kind"],
            "shape": list(c["shape"]),
            "rep_model": c["rep_model"],
            "rep_atom": c["rep_atom"],
            "rep_stage": c["rep_stage"],
            "stage": stage,
            "stage_counts": stage_counts,
            "model_counts": model_counts,
            "has_teacher": model_counts.get("teacher", 0) > 0,
            "has_student": model_counts.get("student", 0) > 0,
            "style_like": is_style_like_stage(stage),
            "knowledge_like": is_knowledge_like_stage(stage),
            "members": c["members"],
        })
    return atom_map, rows


def aggregate_qk_by_head(
    model: str,
    gates: Dict[Tuple[int, int, int], Dict[str, float]],
    atom_map: Dict[Tuple[str, str], Tuple[str, float]],
    layer: int,
    heads: List[int],
):
    sums = defaultdict(float)
    abs_sums = defaultdict(float)
    counts = Counter()
    delta_counts = Counter()
    for (li, h, d), gs in gates.items():
        if li != layer or h not in heads:
            continue
        delta_counts[h] += 1
        for atom_id, coeff in gs.items():
            key = (model, atom_id)
            if key not in atom_map:
                continue
            cid, sign = atom_map[key]
            c = float(coeff) * sign
            sums[(h, cid)] += c
            abs_sums[(h, cid)] += abs(c)
            counts[(h, cid)] += 1
    rows = {}
    for h in heads:
        denom = max(1, delta_counts[h])
        rows[h] = {}
        for (hh, cid), s in sums.items():
            if hh != h:
                continue
            rows[h][cid] = {
                "coeff_mean_over_deltas": s / denom,
                "coeff_abs_mean_over_deltas": abs_sums[(hh, cid)] / denom,
                "usage_count": counts[(hh, cid)],
                "delta_count": delta_counts[h],
            }
    return rows


def aggregate_vo_by_head(
    model: str,
    gates: Dict[Tuple[int, int], Dict[str, float]],
    atom_map: Dict[Tuple[str, str], Tuple[str, float]],
    layer: int,
    heads: List[int],
):
    rows = {h: {} for h in heads}
    for (li, h), gs in gates.items():
        if li != layer or h not in heads:
            continue
        for atom_id, coeff in gs.items():
            key = (model, atom_id)
            if key not in atom_map:
                continue
            cid, sign = atom_map[key]
            c = float(coeff) * sign
            rows[h][cid] = {
                "coeff": c,
                "coeff_abs": abs(c),
                "usage_count": 1,
            }
    return rows


def by_head_diff(kind: str, teacher_by_head: Dict[int, Dict[str, Dict[str, Any]]], student_by_head: Dict[int, Dict[str, Dict[str, Any]]], clusters: Dict[str, Dict[str, Any]], heads: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for h in heads:
        cids = set(teacher_by_head.get(h, {}).keys()) | set(student_by_head.get(h, {}).keys())
        for cid in sorted(cids):
            ci = clusters[cid]
            if kind == "qk":
                tc = teacher_by_head.get(h, {}).get(cid, {}).get("coeff_mean_over_deltas", 0.0)
                sc = student_by_head.get(h, {}).get(cid, {}).get("coeff_mean_over_deltas", 0.0)
                ts = teacher_by_head.get(h, {}).get(cid, {}).get("coeff_abs_mean_over_deltas", 0.0)
                ss = student_by_head.get(h, {}).get(cid, {}).get("coeff_abs_mean_over_deltas", 0.0)
            else:
                tc = teacher_by_head.get(h, {}).get(cid, {}).get("coeff", 0.0)
                sc = student_by_head.get(h, {}).get(cid, {}).get("coeff", 0.0)
                ts = abs(tc)
                ss = abs(sc)
            delta = tc - sc
            rows.append({
                "kind": kind,
                "head": h,
                "cluster_id": cid,
                "stage": ci["stage"],
                "rep_atom": ci["rep_atom"],
                "has_teacher_atom": ci["has_teacher"],
                "has_student_atom": ci["has_student"],
                "style_like": ci["style_like"],
                "knowledge_like": ci["knowledge_like"],
                "teacher_coeff": tc,
                "student_coeff": sc,
                "delta_coeff": delta,
                "abs_delta_coeff": abs(delta),
                "teacher_strength": ts,
                "student_strength": ss,
                "combined_strength": ts + ss,
                "priority": abs(delta) * (1.0 if ci["knowledge_like"] else 0.25) + 0.05 * (ts + ss),
            })
    return sorted(rows, key=lambda r: r["priority"], reverse=True)


def fmt(x: float) -> str:
    return f"{x:+.4f}"


def make_markdown(qk_rows: List[Dict[str, Any]], vo_rows: List[Dict[str, Any]], heads: List[int], top_k: int, focus_heads: List[int], counts: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Readable non-private Coder vs Instruct program diff")
    lines.append("")
    lines.append("Private exact residual atoms are excluded. Coefficients are in a joint canonical basis built by signed cosine clustering of non-private decoded atoms.")
    lines.append("")
    lines.append("## Counts")
    lines.append("```json")
    lines.append(json.dumps(counts, indent=2))
    lines.append("```")
    lines.append("")

    def section(title: str, rows: List[Dict[str, Any]]):
        lines.append(f"## {title}")
        for h in heads:
            hr = [r for r in rows if r["head"] == h]
            hr = sorted(hr, key=lambda r: r["abs_delta_coeff"], reverse=True)[:top_k]
            lines.append("")
            mark = " ⭐" if h in focus_heads else ""
            lines.append(f"### H{h}{mark}")
            if not hr:
                lines.append("No non-private operators.")
                continue
            lines.append("| rank | stage | cluster | Coder | Student | Δ | style | knowledge | rep atom |")
            lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")
            for i, r in enumerate(hr, 1):
                rep = str(r["rep_atom"])
                if len(rep) > 44:
                    rep = rep[:41] + "..."
                lines.append(
                    f"| {i} | {r['stage']} | `{r['cluster_id']}` | {fmt(r['teacher_coeff'])} | {fmt(r['student_coeff'])} | {fmt(r['delta_coeff'])} | {str(r['style_like'])} | {str(r['knowledge_like'])} | `{rep}` |"
                )
        lines.append("")

    section("QK by-head top non-private operator deltas", qk_rows)
    section("VO by-head top non-private operator deltas", vo_rows)

    lines.append("## Focus-head summary")
    for h in focus_heads:
        lines.append("")
        lines.append(f"### H{h} biggest non-private deltas")
        top = sorted([r for r in qk_rows + vo_rows if r["head"] == h], key=lambda r: r["abs_delta_coeff"], reverse=True)[: top_k * 2]
        lines.append("| kind | stage | cluster | Coder | Student | Δ | style | knowledge | rep atom |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---|")
        for r in top:
            rep = str(r["rep_atom"])
            if len(rep) > 44:
                rep = rep[:41] + "..."
            lines.append(
                f"| {r['kind']} | {r['stage']} | `{r['cluster_id']}` | {fmt(r['teacher_coeff'])} | {fmt(r['student_coeff'])} | {fmt(r['delta_coeff'])} | {str(r['style_like'])} | {str(r['knowledge_like'])} | `{rep}` |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-qk-run", required=True)
    ap.add_argument("--student-qk-run", required=True)
    ap.add_argument("--teacher-vo-run", required=True)
    ap.add_argument("--student-vo-run", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--focus-heads", default="1")
    ap.add_argument("--min-cos", type=float, default=0.90)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--include-private", action="store_true")
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/readable_nonprivate_program_diff_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    heads = parse_ints(args.heads)
    focus_heads = parse_ints(args.focus_heads)

    tqk_atoms, tqk_gates, tqk_manifest = load_qk_program(Path(args.teacher_qk_run))
    sqk_atoms, sqk_gates, sqk_manifest = load_qk_program(Path(args.student_qk_run))
    tvo_atoms, tvo_gates, tvo_manifest = load_vo_program(Path(args.teacher_vo_run))
    svo_atoms, svo_gates, svo_manifest = load_vo_program(Path(args.student_vo_run))

    qk_entries = make_atom_entries("qk", "teacher", tqk_atoms, args.include_private) + make_atom_entries("qk", "student", sqk_atoms, args.include_private)
    vo_entries = make_atom_entries("vo", "teacher", tvo_atoms, args.include_private) + make_atom_entries("vo", "student", svo_atoms, args.include_private)

    qk_map, qk_clusters = joint_cluster(qk_entries, "qk", args.min_cos)
    vo_map, vo_clusters = joint_cluster(vo_entries, "vo", args.min_cos)
    qk_cluster_by_id = {c["cluster_id"]: c for c in qk_clusters}
    vo_cluster_by_id = {c["cluster_id"]: c for c in vo_clusters}

    tqk_by_head = aggregate_qk_by_head("teacher", tqk_gates, qk_map, args.layer, heads)
    sqk_by_head = aggregate_qk_by_head("student", sqk_gates, qk_map, args.layer, heads)
    tvo_by_head = aggregate_vo_by_head("teacher", tvo_gates, vo_map, args.layer, heads)
    svo_by_head = aggregate_vo_by_head("student", svo_gates, vo_map, args.layer, heads)

    qk_diff = by_head_diff("qk", tqk_by_head, sqk_by_head, qk_cluster_by_id, heads)
    vo_diff = by_head_diff("vo", tvo_by_head, svo_by_head, vo_cluster_by_id, heads)

    all_clusters = qk_clusters + vo_clusters
    counts = {
        "include_private": bool(args.include_private),
        "teacher_qk_atoms_total": len(tqk_atoms),
        "student_qk_atoms_total": len(sqk_atoms),
        "teacher_vo_atoms_total": len(tvo_atoms),
        "student_vo_atoms_total": len(svo_atoms),
        "qk_atoms_used_nonprivate": len(qk_entries),
        "vo_atoms_used_nonprivate": len(vo_entries),
        "qk_canonical_clusters": len(qk_clusters),
        "vo_canonical_clusters": len(vo_clusters),
        "qk_clusters_with_both_models": sum(1 for c in qk_clusters if c["has_teacher"] and c["has_student"]),
        "vo_clusters_with_both_models": sum(1 for c in vo_clusters if c["has_teacher"] and c["has_student"]),
        "qk_stage_counts": dict(Counter(c["stage"] for c in qk_clusters)),
        "vo_stage_counts": dict(Counter(c["stage"] for c in vo_clusters)),
    }

    md = make_markdown(qk_diff, vo_diff, heads, args.top_k, focus_heads, counts)
    (out / "readable_nonprivate_program_diff.md").write_text(md, encoding="utf-8")

    report = {
        "version": VERSION,
        "mode": "readable_nonprivate_joint_canonical_program_diff",
        "teacher_qk_run": args.teacher_qk_run,
        "student_qk_run": args.student_qk_run,
        "teacher_vo_run": args.teacher_vo_run,
        "student_vo_run": args.student_vo_run,
        "layer": args.layer,
        "heads": heads,
        "focus_heads": focus_heads,
        "min_cos": args.min_cos,
        "top_k": args.top_k,
        "counts": counts,
        "top_focus": sorted([r for r in qk_diff + vo_diff if r["head"] in focus_heads], key=lambda r: r["abs_delta_coeff"], reverse=True)[: args.top_k * 4],
        "status": "READABLE_NONPRIVATE_PROGRAM_DIFF_RAN",
        "prompt_independent": True,
        "no_training": True,
        "note": "Private exact atoms excluded by default. This report is for visual/operator inspection before transfer.",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "canonical_clusters.jsonl", all_clusters)
    write_jsonl(out / "qk_by_head_diff.jsonl", qk_diff)
    write_jsonl(out / "vo_by_head_diff.jsonl", vo_diff)

    print("=== Qwen Readable Non-Private Program Diff v1 ===")
    print(json.dumps({
        "layer": args.layer,
        "heads": heads,
        "focus_heads": focus_heads,
        "counts": counts,
        "top_focus": [
            {
                "kind": r["kind"],
                "head": r["head"],
                "stage": r["stage"],
                "cluster": r["cluster_id"],
                "coder": r["teacher_coeff"],
                "student": r["student_coeff"],
                "delta": r["delta_coeff"],
                "style": r["style_like"],
                "knowledge": r["knowledge_like"],
                "rep_atom": r["rep_atom"],
            }
            for r in report["top_focus"][:12]
        ],
        "status": report["status"],
    }, indent=2))
    print(f"markdown={out / 'readable_nonprivate_program_diff.md'}")
    print(f"out={out}")


if __name__ == "__main__":
    main()
