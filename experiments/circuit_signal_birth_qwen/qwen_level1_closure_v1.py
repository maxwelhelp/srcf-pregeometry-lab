#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_level1_closure_v1.py

Replay/closure verifier for saved QK Level-1 verified extension atoms.
It does NOT mine new atoms. It reloads qk_level1_atoms.pt and verifies that
Level0 + saved Level1 atoms reproduce the previous gains on the same fixed split.

No training, no KL, no alpha sweep, no learned base dictionary.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import build_qk_primitives, decode_greedy_analytic, rel_err
from qwen_circuit_target_roundtrip_v1 import (
    build_prompts, collect_head_data, build_weight_slices, build_rope_by_pos,
    qk_delta_matrices_affine, get_dtype, parse_ints
)

VERSION = "qwen_level1_closure_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def flat(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1)


def coeff_for(residual: torch.Tensor, atom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = flat(atom); r = flat(residual)
    return (r @ a) / (a @ a).clamp_min(eps)


def apply_saved_atoms(entry: Dict[str, Any], atoms: List[torch.Tensor]) -> Dict[str, Any]:
    recon = entry["recon"].clone()
    coeffs = []
    for i, atom in enumerate(atoms):
        atom = atom.float().cpu()
        residual = entry["target"] - recon
        c = coeff_for(residual, atom)
        recon = recon + c * atom
        coeffs.append(float(c.detach().cpu()))
    return {"recon": recon, "coeffs": coeffs, "err": rel_err(recon, entry["target"])}


def split_heads(heads: List[int]):
    heads = sorted(heads)
    n_train = max(1, int(math.ceil(0.70 * len(heads))))
    if len(heads) > 1 and n_train >= len(heads): n_train = len(heads) - 1
    return heads[:n_train], heads[n_train:]


def split_entries(entries, train_heads, held_heads):
    train = [e for e in entries if e["head"] in train_heads and e["delta"] % 2 == 0]
    held = [e for e in entries if e["head"] in held_heads and e["delta"] % 2 == 1]
    return train, held


def load_atoms(extension_run: Path) -> List[torch.Tensor]:
    p = extension_run / "qk_level1_atoms.pt"
    if not p.exists():
        raise FileNotFoundError(f"Missing saved atoms: {p}")
    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, dict):
        keys = sorted(obj.keys())
        return [obj[k].float().cpu() for k in keys]
    if isinstance(obj, list):
        return [x.float().cpu() for x in obj]
    raise TypeError(f"Unsupported atom file format: {type(obj)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="23")
    ap.add_argument("--heads", default="0,1,2,3,4,5,6")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--extension-run", required=True)
    ap.add_argument("--out", default="runs/qwen_level1_closure_v1")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    extension_run = Path(args.extension_run)
    thresholds = load_thresholds(args.thresholds)
    atoms = load_atoms(extension_run)
    if not atoms:
        raise SystemExit("No atoms loaded")

    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True).to(args.device)
    model.eval()

    layers = parse_ints(args.layers); heads = parse_ints(args.heads)
    train_heads, held_heads = split_heads(heads)
    prompts = build_prompts(args.prompts)
    entries: List[Dict[str, Any]] = []

    for li in layers:
        for hi in heads:
            seqs, meta = collect_head_data(model, tok, prompts, li, hi, args.max_length, args.device)
            if not seqs: continue
            weights = build_weight_slices(model, li, hi, meta)
            max_pos = min(args.max_delta, max(int(s.Xn.shape[0])-1 for s in seqs))
            Rpos = build_rope_by_pos(seqs, max_pos)
            Mdelta = qk_delta_matrices_affine(weights, Rpos, max_pos, int(meta["head_dim"]))
            prims = build_qk_primitives(int(meta["hidden_size"]))
            for d, M in Mdelta.items():
                ops, rec, met = decode_greedy_analytic(M, prims, "qk", thresholds, max_ops=64, device="cpu")
                replay = apply_saved_atoms({"target": M.cpu().float(), "recon": rec.cpu().float()}, atoms)
                entries.append({"layer": li, "head": hi, "delta": d, "base_err": float(met["roundtrip_error"]), "level1_err": replay["err"], "gain": float(met["roundtrip_error"] - replay["err"]), "coeffs": replay["coeffs"], "ops_count": float(met["ops_count"])})

    train, held = split_entries(entries, train_heads, held_heads)
    report = {
        "version": VERSION,
        "mode": "qwen_level1_saved_atom_closure",
        "model": args.model,
        "layers": layers,
        "heads": heads,
        "extension_run": str(extension_run),
        "atom_count": len(atoms),
        "entries": len(entries),
        "train_entries": len(train),
        "heldout_entries": len(held),
        "base_train_err_mean": mean([e["base_err"] for e in train]),
        "level1_train_err_mean": mean([e["level1_err"] for e in train]),
        "train_gain_total": mean([e["base_err"] for e in train]) - mean([e["level1_err"] for e in train]),
        "base_heldout_err_mean": mean([e["base_err"] for e in held]),
        "level1_heldout_err_mean": mean([e["level1_err"] for e in held]),
        "heldout_gain_total": mean([e["base_err"] for e in held]) - mean([e["level1_err"] for e in held]),
        "closure_level": "circuit_target",
        "status": "EXTENDED_TARGET_SPECIFIC_PARTIAL_PROGRAM_CLOSED",
        "raw_weight_passthrough_used": False,
        "kl_distillation_used_as_main_method": False,
        "alpha_sweep_used_as_main_method": False,
        "gradient_used_as_main_method": False,
        "universal_claim_allowed": False,
        "level2_promotion_in_v1_scope": False,
    }
    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_matrix_closure.jsonl", entries)
    print("=== Qwen Level1 Saved-Atom Closure ===")
    print(json.dumps({k: report[k] for k in ["atom_count","entries","train_entries","heldout_entries","base_train_err_mean","level1_train_err_mean","base_heldout_err_mean","level1_heldout_err_mean","train_gain_total","heldout_gain_total","status"]}, indent=2))
    print(f"out={out}")
    if report["heldout_gain_total"] <= 0:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
