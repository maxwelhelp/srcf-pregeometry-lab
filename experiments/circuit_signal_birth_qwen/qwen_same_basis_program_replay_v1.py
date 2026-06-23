#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_same_basis_program_replay_v1.py

Same-basis no-training program replay.

Reloads:
  - qk_autoexpand_atoms.pt
  - per_matrix_autoexpand_closure.jsonl

Then re-extracts Qwen QK targets, rebuilds Level0 recon, applies saved gates,
and verifies that the saved autoexpanded program closes the same circuit targets.

This is same-basis / same-architecture proof, not cross-model transfer.
No training, no KL, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import build_prompts, get_dtype, parse_ints
import qwen_joint_all_heads_decode_v1 as joint

VERSION = "qwen_same_basis_program_replay_v1.0"


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def mean(xs):
    return sum(xs) / max(1, len(xs))


def load_saved_program(program_run: Path):
    atoms_path = program_run / "qk_autoexpand_atoms.pt"
    gates_path = program_run / "per_matrix_autoexpand_closure.jsonl"
    if not atoms_path.exists():
        raise FileNotFoundError(atoms_path)
    if not gates_path.exists():
        raise FileNotFoundError(gates_path)

    atoms = torch.load(atoms_path, map_location="cpu")
    gates_by_key: Dict[Tuple[int, int, int], Dict[str, float]] = {}
    for line in gates_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["layer"]), int(row["head"]), int(row["delta"]))
        gates_by_key[key] = {str(k): float(v) for k, v in row.get("gates", {}).items()}
    return atoms, gates_by_key


def apply_saved_gates(entry: Dict[str, Any], atoms: Dict[str, torch.Tensor], gates: Dict[str, float]):
    recon = entry["recon"].clone().float().cpu()
    missing = []
    for atom_key, coeff in gates.items():
        if atom_key not in atoms:
            missing.append(atom_key)
            continue
        recon = recon + float(coeff) * atoms[atom_key].float().cpu()
    return recon, rel_err(recon, entry["target"]), missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="23")
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--program-run", required=True)
    ap.add_argument("--target-tol", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/qwen_same_basis_program_replay_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    atoms, gates_by_key = load_saved_program(Path(args.program_run))
    thresholds = load_thresholds(args.thresholds)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    entries = joint.collect_entries(
        model,
        tok,
        parse_ints(args.layers),
        parse_ints(args.heads),
        build_prompts(args.prompts),
        args,
        thresholds,
    )

    rows = []
    missing_all = []
    for e in entries:
        key = (int(e["layer"]), int(e["head"]), int(e["delta"]))
        gates = gates_by_key.get(key, {})
        _recon, err, missing = apply_saved_gates(e, atoms, gates)
        missing_all.extend(missing)
        rows.append({
            "layer": key[0],
            "head": key[1],
            "delta": key[2],
            "base_err": float(e["base_err"]),
            "program_err": float(err),
            "gain": float(e["base_err"]) - float(err),
            "num_gates": len(gates),
            "missing_atoms": missing,
        })

    base_mean = mean([r["base_err"] for r in rows])
    program_mean = mean([r["program_err"] for r in rows])
    program_max = max([r["program_err"] for r in rows]) if rows else 999.0
    missing_set = sorted(set(missing_all))

    report = {
        "version": VERSION,
        "mode": "same_basis_saved_program_replay",
        "model": args.model,
        "program_run": args.program_run,
        "atom_count": len(atoms),
        "matrix_entries": len(rows),
        "base_err_mean": base_mean,
        "program_err_mean": program_mean,
        "program_err_max": program_max,
        "target_tol": args.target_tol,
        "closed_mean": program_mean <= args.target_tol,
        "closed_max": program_max <= args.target_tol,
        "status": "SAME_BASIS_PROGRAM_REPLAY_CLOSED" if program_mean <= args.target_tol else "SAME_BASIS_PROGRAM_REPLAY_PARTIAL",
        "no_training": True,
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "universal_claim_allowed": False,
        "weight_rewrite_done": False,
        "closure_level": "circuit_target",
        "missing_atom_count": len(missing_set),
        "missing_atoms": missing_set[:50],
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_matrix_program_replay.jsonl", rows)

    print("=== Qwen Same-Basis Program Replay ===")
    print(json.dumps({
        "atom_count": report["atom_count"],
        "matrix_entries": report["matrix_entries"],
        "base_err_mean": report["base_err_mean"],
        "program_err_mean": report["program_err_mean"],
        "program_err_max": report["program_err_max"],
        "target_tol": report["target_tol"],
        "closed_mean": report["closed_mean"],
        "closed_max": report["closed_max"],
        "status": report["status"],
        "missing_atom_count": report["missing_atom_count"],
    }, indent=2))
    print(f"out={out}")

    if program_mean > args.target_tol:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
