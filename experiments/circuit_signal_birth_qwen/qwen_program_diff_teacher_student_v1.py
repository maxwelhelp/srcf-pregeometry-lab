#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_program_diff_teacher_student_v1.py

Prompt-independent decoded-program diff between a teacher program bundle and a
student program bundle.

Why:
  Gradient masks are behavior/probe based and can overfit prompts. This script
  compares the decoded operator programs directly:

    teacher weights -> saved program atoms/gates
    student weights -> saved program atoms/gates
    diff at operator/gate level

It does not train, does not compute KL, and does not use prompts. It reads the
already decoded QK/VO program-runs produced by the autoexpand/replay scripts.

Outputs:
  - qk_atom_matches.jsonl / vo_atom_matches.jsonl
  - qk_program_diff.jsonl / vo_program_diff.jsonl
  - transfer_candidates.jsonl
  - manifest.json

Interpretation:
  - style_like: broad/shared atoms used across many scopes.
  - knowledge_like: head/delta/private/selective atoms or ops with narrow usage.
  These labels are heuristics for triage, not proof.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from program_dsl_v1 import write_json, write_jsonl

VERSION = "qwen_program_diff_teacher_student_v1.0"


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def flat_unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    f = x.float().reshape(-1)
    return f / torch.linalg.norm(f).clamp_min(eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return float("nan")
    return float(flat_unit(a) @ flat_unit(b))


def tensor_rank(x: torch.Tensor, tol: float = 1e-5) -> int:
    try:
        s = torch.linalg.svdvals(x.float())
        return int((s > tol * s.max().clamp_min(1e-12)).sum().item())
    except Exception:
        return -1


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_qk_program(run: Path):
    atoms_path = run / "qk_autoexpand_atoms.pt"
    gates_path = run / "per_matrix_autoexpand_closure.jsonl"
    if not atoms_path.exists():
        raise FileNotFoundError(atoms_path)
    if not gates_path.exists():
        raise FileNotFoundError(gates_path)
    atoms: Dict[str, torch.Tensor] = torch.load(atoms_path, map_location="cpu")
    gates_by_scope: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for row in read_jsonl(gates_path):
        key = (int(row["layer"]), int(row["head"]), int(row["delta"]))
        gates_by_scope[key] = {str(k): float(v) for k, v in row.get("gates", {}).items()}
    dictionary = read_jsonl(run / "qk_autoexpanded_dictionary.jsonl")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    return atoms, gates_by_scope, dictionary, manifest


def load_vo_program(run: Path):
    atoms_path = run / "vo_autoexpand_atoms.pt"
    gates_path = run / "per_head_vo_matrix_closure.jsonl"
    if not atoms_path.exists():
        raise FileNotFoundError(atoms_path)
    if not gates_path.exists():
        raise FileNotFoundError(gates_path)
    atoms: Dict[str, torch.Tensor] = torch.load(atoms_path, map_location="cpu")
    gates_by_scope: Dict[Tuple[int, int], Dict[str, float]] = {}
    for row in read_jsonl(gates_path):
        key = (int(row["layer"]), int(row["head"]))
        gates_by_scope[key] = {str(k): float(v) for k, v in row.get("gates", {}).items()}
    dictionary = read_jsonl(run / "vo_autoexpanded_dictionary.jsonl")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8")) if (run / "manifest.json").exists() else {}
    return atoms, gates_by_scope, dictionary, manifest


def infer_stage(atom_id: str, kind: str) -> str:
    a = atom_id.lower()
    if kind == "vo":
        if "shared" in a:
            return "vo_shared"
        if "private_exact" in a:
            return "vo_private_exact"
        return "vo_unknown"
    # qk
    if "shared" in a:
        return "qk_shared"
    if "head_specific" in a:
        return "qk_head_specific"
    if "delta_bucket" in a:
        return "qk_delta_bucket"
    if "private_exact" in a:
        return "qk_private_exact"
    return "qk_unknown"


def parse_scope_from_atom(atom_id: str) -> Dict[str, Optional[int]]:
    # supports ..._L23_H1_D5_... and ..._H1 ...
    out: Dict[str, Optional[int]] = {"layer": None, "head": None, "delta": None}
    m = re.search(r"_L(\d+)", atom_id)
    if m:
        out["layer"] = int(m.group(1))
    m = re.search(r"_H(\d+)", atom_id)
    if m:
        out["head"] = int(m.group(1))
    m = re.search(r"_D(\d+)", atom_id)
    if m:
        out["delta"] = int(m.group(1))
    return out


def usage_stats(gates_by_scope: Dict[Any, Dict[str, float]], atoms: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    total_scopes = max(1, len(gates_by_scope))
    vals = defaultdict(list)
    scopes = defaultdict(list)
    for scope, gates in gates_by_scope.items():
        for aid, c in gates.items():
            vals[aid].append(float(c))
            scopes[aid].append(scope)
    stats: Dict[str, Dict[str, Any]] = {}
    for aid in atoms.keys():
        cs = vals.get(aid, [])
        abs_cs = [abs(c) for c in cs]
        stats[aid] = {
            "usage_count": len(cs),
            "usage_frac": len(cs) / total_scopes,
            "coeff_abs_mean": sum(abs_cs) / max(1, len(abs_cs)),
            "coeff_abs_max": max(abs_cs) if abs_cs else 0.0,
            "scopes": [list(s) if isinstance(s, tuple) else s for s in scopes.get(aid, [])[:20]],
        }
    return stats


def classify_atom(atom_id: str, kind: str, usage_frac: float) -> Dict[str, Any]:
    stage = infer_stage(atom_id, kind)
    broad = usage_frac >= 0.50 or "shared" in stage
    private = "private" in stage
    head_specific = "head_specific" in stage
    delta_bucket = "delta_bucket" in stage
    style_like = bool(broad and not head_specific and not delta_bucket and not private)
    knowledge_like = bool((not style_like) or head_specific or delta_bucket or private)
    return {
        "stage": stage,
        "style_like": style_like,
        "knowledge_like": knowledge_like,
        "private_exact": private,
        "same_arch_safe": True,
        "cross_model_safe": bool(knowledge_like and not private),
    }


def build_atom_matches(
    teacher_atoms: Dict[str, torch.Tensor],
    student_atoms: Dict[str, torch.Tensor],
    teacher_usage: Dict[str, Dict[str, Any]],
    student_usage: Dict[str, Dict[str, Any]],
    kind: str,
    min_cos: float,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    rows = []
    best_map: Dict[str, str] = {}
    student_units = {k: flat_unit(v) for k, v in student_atoms.items()}
    for ta, tv in teacher_atoms.items():
        tu = flat_unit(tv)
        best_id = None
        best_cos = -2.0
        for sa, su in student_units.items():
            if tuple(tv.shape) != tuple(student_atoms[sa].shape):
                continue
            c = float(tu @ su)
            if abs(c) > abs(best_cos) if best_id is not None else True:
                best_id = sa
                best_cos = c
        if best_id is not None:
            best_map[ta] = best_id
        t_class = classify_atom(ta, kind, teacher_usage.get(ta, {}).get("usage_frac", 0.0))
        s_class = classify_atom(best_id or "", kind, student_usage.get(best_id or "", {}).get("usage_frac", 0.0)) if best_id else {}
        rows.append({
            "kind": kind,
            "teacher_atom": ta,
            "teacher_stage": t_class["stage"],
            "teacher_style_like": t_class["style_like"],
            "teacher_knowledge_like": t_class["knowledge_like"],
            "teacher_usage_frac": teacher_usage.get(ta, {}).get("usage_frac", 0.0),
            "teacher_coeff_abs_mean": teacher_usage.get(ta, {}).get("coeff_abs_mean", 0.0),
            "student_best_atom": best_id,
            "student_best_stage": s_class.get("stage"),
            "best_cosine": best_cos if best_id is not None else None,
            "abs_best_cosine": abs(best_cos) if best_id is not None else None,
            "structural_match": bool(best_id is not None and abs(best_cos) >= min_cos),
            "teacher_rank": tensor_rank(tv) if tv.ndim == 2 else -1,
            "student_rank": tensor_rank(student_atoms[best_id]) if best_id and student_atoms[best_id].ndim == 2 else -1,
            "shape": list(tv.shape),
            **{f"teacher_{k}": v for k, v in parse_scope_from_atom(ta).items()},
            **{f"student_{k}": v for k, v in parse_scope_from_atom(best_id or "").items()},
        })
    rows = sorted(rows, key=lambda r: (not r["structural_match"], -(r["abs_best_cosine"] or 0.0)))
    return best_map, rows


def get_gate(gates: Dict[str, float], atom_id: Optional[str]) -> float:
    if atom_id is None:
        return 0.0
    return float(gates.get(atom_id, 0.0))


def program_diff_rows(
    kind: str,
    teacher_gates: Dict[Any, Dict[str, float]],
    student_gates: Dict[Any, Dict[str, float]],
    teacher_atoms: Dict[str, torch.Tensor],
    student_atoms: Dict[str, torch.Tensor],
    best_map: Dict[str, str],
    match_rows: List[Dict[str, Any]],
    min_coeff: float,
) -> List[Dict[str, Any]]:
    match_by_teacher = {r["teacher_atom"]: r for r in match_rows}
    rows: List[Dict[str, Any]] = []
    scopes = sorted(set(teacher_gates.keys()) | set(student_gates.keys()))
    for scope in scopes:
        tg = teacher_gates.get(scope, {})
        sg = student_gates.get(scope, {})
        # teacher-side delta entries
        for ta, tc in tg.items():
            sa = best_map.get(ta)
            sc = get_gate(sg, sa)
            delta = float(tc) - float(sc)
            mr = match_by_teacher.get(ta, {})
            cls = classify_atom(ta, kind, 0.0)
            row = {
                "kind": kind,
                "scope": list(scope) if isinstance(scope, tuple) else scope,
                "layer": int(scope[0]) if isinstance(scope, tuple) and len(scope) >= 1 else None,
                "head": int(scope[1]) if isinstance(scope, tuple) and len(scope) >= 2 else None,
                "delta": int(scope[2]) if isinstance(scope, tuple) and len(scope) >= 3 else None,
                "side": "teacher_delta",
                "teacher_atom": ta,
                "student_matched_atom": sa,
                "teacher_coeff": float(tc),
                "student_coeff_on_matched_atom": float(sc),
                "delta_coeff": delta,
                "abs_delta_coeff": abs(delta),
                "best_cosine": mr.get("best_cosine"),
                "abs_best_cosine": mr.get("abs_best_cosine"),
                "structural_match": mr.get("structural_match", False),
                "stage": infer_stage(ta, kind),
                "style_like": cls["style_like"],
                "knowledge_like": cls["knowledge_like"],
                "private_exact": cls["private_exact"],
                "candidate_priority": abs(delta) * (1.0 if cls["knowledge_like"] else 0.25) * (float(mr.get("abs_best_cosine") or 0.5)),
            }
            if abs(delta) >= min_coeff:
                rows.append(row)
        # student-only entries not matched by any teacher atom in this scope
        matched_student_atoms = {best_map.get(ta) for ta in tg.keys() if best_map.get(ta) is not None}
        for sa, sc in sg.items():
            if sa in matched_student_atoms:
                continue
            cls = classify_atom(sa, kind, 0.0)
            if abs(float(sc)) < min_coeff:
                continue
            rows.append({
                "kind": kind,
                "scope": list(scope) if isinstance(scope, tuple) else scope,
                "layer": int(scope[0]) if isinstance(scope, tuple) and len(scope) >= 1 else None,
                "head": int(scope[1]) if isinstance(scope, tuple) and len(scope) >= 2 else None,
                "delta": int(scope[2]) if isinstance(scope, tuple) and len(scope) >= 3 else None,
                "side": "student_only",
                "teacher_atom": None,
                "student_matched_atom": sa,
                "teacher_coeff": 0.0,
                "student_coeff_on_matched_atom": float(sc),
                "delta_coeff": -float(sc),
                "abs_delta_coeff": abs(float(sc)),
                "best_cosine": None,
                "abs_best_cosine": None,
                "structural_match": False,
                "stage": infer_stage(sa, kind),
                "style_like": cls["style_like"],
                "knowledge_like": cls["knowledge_like"],
                "private_exact": cls["private_exact"],
                "candidate_priority": abs(float(sc)) * (1.0 if cls["knowledge_like"] else 0.25),
            })
    return sorted(rows, key=lambda r: r["candidate_priority"], reverse=True)


def summarize(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(Counter(str(r.get(key)) for r in rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-qk-run", required=True)
    ap.add_argument("--student-qk-run", required=True)
    ap.add_argument("--teacher-vo-run", required=True)
    ap.add_argument("--student-vo-run", required=True)
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--focus-heads", default="1")
    ap.add_argument("--min-cos", type=float, default=0.95)
    ap.add_argument("--min-coeff", type=float, default=1e-7)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/program_diff_teacher_student_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    heads = set(parse_ints(args.heads))
    focus_heads = set(parse_ints(args.focus_heads))

    tqk_atoms, tqk_gates, tqk_dict, tqk_manifest = load_qk_program(Path(args.teacher_qk_run))
    sqk_atoms, sqk_gates, sqk_dict, sqk_manifest = load_qk_program(Path(args.student_qk_run))
    tvo_atoms, tvo_gates, tvo_dict, tvo_manifest = load_vo_program(Path(args.teacher_vo_run))
    svo_atoms, svo_gates, svo_dict, svo_manifest = load_vo_program(Path(args.student_vo_run))

    # filter requested layer/heads
    tqk_gates = {k: v for k, v in tqk_gates.items() if k[0] == args.layer and k[1] in heads}
    sqk_gates = {k: v for k, v in sqk_gates.items() if k[0] == args.layer and k[1] in heads}
    tvo_gates = {k: v for k, v in tvo_gates.items() if k[0] == args.layer and k[1] in heads}
    svo_gates = {k: v for k, v in svo_gates.items() if k[0] == args.layer and k[1] in heads}

    tqk_usage = usage_stats(tqk_gates, tqk_atoms)
    sqk_usage = usage_stats(sqk_gates, sqk_atoms)
    tvo_usage = usage_stats(tvo_gates, tvo_atoms)
    svo_usage = usage_stats(svo_gates, svo_atoms)

    qk_best, qk_matches = build_atom_matches(tqk_atoms, sqk_atoms, tqk_usage, sqk_usage, "qk", args.min_cos)
    vo_best, vo_matches = build_atom_matches(tvo_atoms, svo_atoms, tvo_usage, svo_usage, "vo", args.min_cos)

    qk_diff = program_diff_rows("qk", tqk_gates, sqk_gates, tqk_atoms, sqk_atoms, qk_best, qk_matches, args.min_coeff)
    vo_diff = program_diff_rows("vo", tvo_gates, svo_gates, tvo_atoms, svo_atoms, vo_best, vo_matches, args.min_coeff)

    all_diff = qk_diff + vo_diff
    transfer_candidates = [r for r in all_diff if r.get("knowledge_like") and abs(float(r.get("delta_coeff", 0.0))) >= args.min_coeff]
    transfer_candidates = sorted(transfer_candidates, key=lambda r: r["candidate_priority"], reverse=True)
    focus_candidates = [r for r in transfer_candidates if r.get("head") in focus_heads]

    report = {
        "version": VERSION,
        "mode": "teacher_student_decoded_program_diff",
        "teacher_qk_run": args.teacher_qk_run,
        "student_qk_run": args.student_qk_run,
        "teacher_vo_run": args.teacher_vo_run,
        "student_vo_run": args.student_vo_run,
        "layer": args.layer,
        "heads": sorted(heads),
        "focus_heads": sorted(focus_heads),
        "min_cos": args.min_cos,
        "min_coeff": args.min_coeff,
        "counts": {
            "teacher_qk_atoms": len(tqk_atoms),
            "student_qk_atoms": len(sqk_atoms),
            "teacher_vo_atoms": len(tvo_atoms),
            "student_vo_atoms": len(svo_atoms),
            "qk_structural_matches": sum(1 for r in qk_matches if r["structural_match"]),
            "vo_structural_matches": sum(1 for r in vo_matches if r["structural_match"]),
            "qk_diff_rows": len(qk_diff),
            "vo_diff_rows": len(vo_diff),
            "transfer_candidates": len(transfer_candidates),
            "focus_transfer_candidates": len(focus_candidates),
        },
        "qk_match_stage_counts": summarize(qk_matches, "teacher_stage"),
        "vo_match_stage_counts": summarize(vo_matches, "teacher_stage"),
        "transfer_stage_counts": summarize(transfer_candidates, "stage"),
        "top_transfer_candidates": transfer_candidates[: args.top],
        "top_focus_candidates": focus_candidates[: args.top],
        "status": "PROGRAM_DIFF_RAN",
        "no_training": True,
        "prompt_independent": True,
        "note": "Diff is based on saved decoded QK/VO program atoms/gates, not on behavioral gradients. style_like/knowledge_like are triage heuristics.",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "qk_atom_matches.jsonl", qk_matches)
    write_jsonl(out / "vo_atom_matches.jsonl", vo_matches)
    write_jsonl(out / "qk_program_diff.jsonl", qk_diff)
    write_jsonl(out / "vo_program_diff.jsonl", vo_diff)
    write_jsonl(out / "transfer_candidates.jsonl", transfer_candidates)
    write_jsonl(out / "focus_transfer_candidates.jsonl", focus_candidates)

    print("=== Qwen Teacher/Student Decoded Program Diff v1 ===")
    print(json.dumps({
        "layer": report["layer"],
        "heads": report["heads"],
        "focus_heads": report["focus_heads"],
        "counts": report["counts"],
        "transfer_stage_counts": report["transfer_stage_counts"],
        "top_focus_candidates": [
            {
                "kind": r["kind"],
                "scope": r["scope"],
                "stage": r["stage"],
                "teacher_atom": r["teacher_atom"],
                "student_matched_atom": r["student_matched_atom"],
                "delta_coeff": r["delta_coeff"],
                "abs_best_cosine": r.get("abs_best_cosine"),
                "priority": r["candidate_priority"],
                "style_like": r["style_like"],
                "knowledge_like": r["knowledge_like"],
            }
            for r in focus_candidates[: min(args.top, 12)]
        ],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
