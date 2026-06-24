#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_term_map_compare_code_retain_v1.py

Compare code/retain outputs from qwen_head_pseudocode_term_transfer_scan_v1.py
across multiple layers.

Goal:
  Find pseudocode-term patches where code gain is positive and retain damage is
  small. This is a map of candidate code-specific mechanisms, not a transfer.

Expected run dirs by default:
  <root>/L6_code/manifest.json
  <root>/L6_retain/manifest.json
  <root>/L12_code/manifest.json
  <root>/L12_retain/manifest.json
  ...

It also accepts explicit --code-run/--retain-run pairs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_manifest(p: Path) -> Dict[str, Any]:
    if p.is_dir():
        p = p / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def candidate_key(r: Dict[str, Any]) -> Tuple[int, str, float, Tuple[int, ...]]:
    heads = tuple(int(x) for x in r.get("patch_heads", []))
    return (int(heads[0]) if len(heads) == 1 else -1, str(r.get("candidate_spec", r.get("spec", ""))), float(r.get("alpha", 0.0)), heads)


def score_row(code: Dict[str, Any], retain: Dict[str, Any], layer: int) -> Dict[str, Any]:
    code_kl = float(code.get("KL_improvement", 0.0))
    retain_kl = float(retain.get("KL_improvement", 0.0))
    code_logit = float(code.get("logits_improvement", 0.0))
    retain_logit = float(retain.get("logits_improvement", 0.0))
    retain_damage = max(0.0, -retain_kl)
    code_damage = max(0.0, -code_kl)
    shift_code = float(code.get("student_shift_logits_rel", 0.0))
    shift_retain = float(retain.get("student_shift_logits_rel", 0.0))
    y_code = float(code.get("Y_prog_vs_student_attention_rel", 0.0))
    y_retain = float(retain.get("Y_prog_vs_student_attention_rel", 0.0))

    # Prefer positive code KL/logit, penalize retain damage and global shifts.
    selection = (
        code_kl
        + 0.25 * code_logit
        - 2.0 * retain_damage
        - 0.25 * max(0.0, retain_logit)
        - 0.05 * shift_code
        - 0.05 * shift_retain
        - 0.02 * y_code
        - 0.02 * y_retain
        - 2.0 * code_damage
    )
    ratio = code_kl / (retain_damage + 1e-6)
    heads = tuple(int(x) for x in code.get("patch_heads", []))
    return {
        "layer": int(layer),
        "head": int(heads[0]) if len(heads) == 1 else -1,
        "patch_heads": list(heads),
        "candidate": str(code.get("candidate", "")),
        "candidate_spec": str(code.get("candidate_spec", code.get("spec", ""))),
        "terms": code.get("terms", []),
        "alpha": float(code.get("alpha", 0.0)),
        "code_logit_gain": code_logit,
        "code_KL_gain": code_kl,
        "retain_logit_gain": retain_logit,
        "retain_KL_gain": retain_kl,
        "retain_damage": retain_damage,
        "code_retain_KL_ratio": ratio,
        "code_shift": shift_code,
        "retain_shift": shift_retain,
        "code_Y_rel": y_code,
        "retain_Y_rel": y_retain,
        "selection_score": selection,
        "is_clean_positive": bool(code_kl > 0.0 and code_logit >= 0.0 and retain_damage <= max(0.002, 0.25 * code_kl)),
    }


def collect_pairs_from_root(root: Path, layers: List[int]) -> List[Tuple[int, Path, Path]]:
    pairs = []
    for L in layers:
        pairs.append((L, root / f"L{L}_code", root / f"L{L}_retain"))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Root containing L{layer}_code and L{layer}_retain dirs")
    ap.add_argument("--layers", default="6,12,18")
    ap.add_argument("--code-run", action="append", default=[], help="Explicit code run dir/manifest. Repeat with matching --retain-run.")
    ap.add_argument("--retain-run", action="append", default=[], help="Explicit retain run dir/manifest. Repeat with matching --code-run.")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/term_map_compare_code_retain_v1.json")
    args = ap.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    pairs: List[Tuple[int, Path, Path]] = []
    if args.root:
        pairs.extend(collect_pairs_from_root(Path(args.root), layers))
    if args.code_run or args.retain_run:
        if len(args.code_run) != len(args.retain_run):
            raise SystemExit("--code-run and --retain-run counts must match")
        for i, (c, r) in enumerate(zip(args.code_run, args.retain_run)):
            layer = layers[i] if i < len(layers) else -1
            pairs.append((layer, Path(c), Path(r)))
    if not pairs:
        raise SystemExit("Provide --root or explicit --code-run/--retain-run pairs")

    rows: List[Dict[str, Any]] = []
    diagnostics = []
    for layer, code_path, retain_path in pairs:
        cm = read_manifest(code_path)
        rm = read_manifest(retain_path)
        cb = cm.get("best_candidates", [])
        rb = rm.get("best_candidates", [])
        c_map = {candidate_key(r): r for r in cb}
        r_map = {candidate_key(r): r for r in rb}
        common = sorted(set(c_map) & set(r_map))
        diagnostics.append({
            "layer": layer,
            "code_run": str(code_path),
            "retain_run": str(retain_path),
            "code_candidates": len(cb),
            "retain_candidates": len(rb),
            "matched_candidates": len(common),
            "code_status": cm.get("status"),
            "retain_status": rm.get("status"),
            "code_base_sanity": cm.get("base_sanity"),
            "retain_base_sanity": rm.get("base_sanity"),
        })
        for k in common:
            rows.append(score_row(c_map[k], r_map[k], layer))

    rows = sorted(rows, key=lambda r: r["selection_score"], reverse=True)
    clean = [r for r in rows if r["is_clean_positive"]]
    report = {
        "layers": layers,
        "pairs": diagnostics,
        "row_count": len(rows),
        "clean_positive_count": len(clean),
        "top": rows[: args.top],
        "top_clean": clean[: args.top],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Qwen Term Map Code/Retain Compare v1 ===")
    print(json.dumps({
        "layers": layers,
        "row_count": len(rows),
        "clean_positive_count": len(clean),
        "diagnostics": diagnostics,
        "top10": rows[:10],
        "top_clean10": clean[:10],
        "out": str(out),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
