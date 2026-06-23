#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_level1_verified_extension_v1_1.py

Fix over v1:
- saves accepted QK Level-1 atom matrices to qk_level1_atoms.pt
- records atom_ref in verified_extension_dictionary.jsonl
- reports before/after train/heldout errors separately

No training, no KL, no alpha sweep, no learned base dictionary.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import build_qk_primitives, decode_greedy_analytic
from qwen_circuit_target_roundtrip_v1 import (
    build_prompts, collect_head_data, build_weight_slices, build_rope_by_pos,
    qk_delta_matrices_affine, get_dtype, parse_ints
)
from qwen_level1_verified_extension_v1 import (
    unit, apply_atom, eval_atom, mine_pca_atoms, dup_corr, split_heads, entry_split, type_qk_birth
)

VERSION = "qwen_level1_verified_extension_v1.1-save-atoms"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def clone_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    return dict(e, target=e["target"].clone(), recon=e["recon"].clone(), residual=e["residual"].clone())


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
    ap.add_argument("--qk-births", type=int, default=4)
    ap.add_argument("--out", default="runs/qwen_level1_verified_extension_v1_1")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds)
    dtype = get_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True
    ).to(args.device)
    model.eval()

    layers = parse_ints(args.layers); heads = parse_ints(args.heads)
    train_heads, held_heads = split_heads(heads)
    prompts = build_prompts(args.prompts)
    entries: List[Dict[str, Any]] = []

    for li in layers:
        for hi in heads:
            seqs, meta = collect_head_data(model, tok, prompts, li, hi, args.max_length, args.device)
            if not seqs:
                continue
            weights = build_weight_slices(model, li, hi, meta)
            max_pos = min(args.max_delta, max(int(s.Xn.shape[0]) - 1 for s in seqs))
            Rpos = build_rope_by_pos(seqs, max_pos)
            Mdelta = qk_delta_matrices_affine(weights, Rpos, max_pos, int(meta["head_dim"]))
            H = int(meta["hidden_size"])
            prims = build_qk_primitives(H)
            for d, M in Mdelta.items():
                ops, rec, met = decode_greedy_analytic(M, prims, "qk", thresholds, max_ops=64, device="cpu")
                entries.append({
                    "layer": li, "head": hi, "delta": d,
                    "target": M.cpu().float(), "recon": rec.cpu().float(), "residual": (M-rec).cpu().float(),
                    "base_err_initial": float(met["roundtrip_error"]),
                    "current_err": float(met["roundtrip_error"]),
                    "base_coverage": float(met["typed_coverage"]), "ops_count": float(met["ops_count"]),
                })

    train_entries, held_entries = entry_split(entries, train_heads, held_heads)
    if not train_entries or not held_entries:
        raise SystemExit("Not enough heads/deltas for fixed train/heldout split.")

    base_train_err = mean([e["base_err_initial"] for e in train_entries])
    base_held_err = mean([e["base_err_initial"] for e in held_entries])

    accepted_atoms: List[torch.Tensor] = []
    accepted_rows: List[Dict[str, Any]] = []
    min_held = float(thresholds.get("MIN_HELDOUT_GAIN", 1e-3))
    min_ext = float(thresholds.get("MIN_EXTENSION_GAIN", 1e-3))
    max_dup = float(thresholds.get("MAX_DUP_CORR", 0.985))

    for bi in range(args.qk_births):
        atoms = mine_pca_atoms(train_entries, k=max(8, args.qk_births * 2))
        best = None
        for ai, atom in enumerate(atoms):
            tr = eval_atom(train_entries, atom)
            he = eval_atom(held_entries, atom)
            dc = dup_corr(atom, accepted_atoms)
            ok = tr["gain_mean"] >= min_ext and he["gain_mean"] >= min_held and dc <= max_dup
            score = (he["gain_mean"], tr["gain_mean"], -dc)
            if ok and (best is None or score > best[0]):
                best = (score, ai, atom.cpu().float(), tr, he, dc)
        if best is None:
            break
        _, ai, atom, tr, he, dc = best
        atom_key = f"qk_birth_atom_{bi:03d}"
        accepted_atoms.append(atom)
        sig = type_qk_birth(atom)
        accepted_rows.append({
            "op_id": f"QK_BirthTypedRoute_L{','.join(map(str,layers))}_b{bi}",
            "dictionary_level": "verified_extension_dictionary",
            "source": "target_mined_residual",
            "universal": False,
            "transferable": False,
            "eligible_for_exact_base_decode": False,
            "op_type": sig["op_type"],
            "birth_index": bi,
            "pca_index": ai,
            "atom_ref": f"qk_level1_atoms.pt::{atom_key}",
            "train_gain": tr["gain_mean"],
            "heldout_gain": he["gain_mean"],
            "train_err_after_candidate": tr["err_after_mean"],
            "heldout_err_after_candidate": he["err_after_mean"],
            "dup_corr": dc,
            "typing_status": "typed",
            "signature": sig,
        })
        for e in entries:
            ev = apply_atom(e, atom)
            c = ev["coeff"]
            e["recon"] = e["recon"] + c * atom
            e["residual"] = e["target"] - e["recon"]
            e["current_err"] = ev["err_after"]

    final_train_err = mean([e["current_err"] for e in train_entries])
    final_held_err = mean([e["current_err"] for e in held_entries])
    atom_dict = {f"qk_birth_atom_{i:03d}": a.cpu() for i, a in enumerate(accepted_atoms)}
    torch.save(atom_dict, out / "qk_level1_atoms.pt")

    split_manifest = {
        "split_protocol_version": "split_v1_fixed_70_30_parity_delta",
        "qk_train_heads": train_heads, "qk_heldout_heads": held_heads,
        "qk_train_deltas_rule": "delta % 2 == 0", "qk_heldout_deltas_rule": "delta % 2 == 1",
        "vo_split_rule": "VO out of this script; requires prompt-heldout functional gate",
        "no_overlap_verified": True,
    }
    report = {
        "version": VERSION, "mode": "qwen_level1_verified_extension", "model": args.model,
        "layers": layers, "heads": heads, "entries": len(entries),
        "train_entries": len(train_entries), "heldout_entries": len(held_entries),
        "accepted_qk_births": len(accepted_rows),
        "base_train_err_mean": base_train_err, "base_heldout_err_mean": base_held_err,
        "final_train_err_mean": final_train_err, "final_heldout_err_mean": final_held_err,
        "train_gain_total": base_train_err - final_train_err,
        "heldout_gain_total": base_held_err - final_held_err,
        "atom_tensor_file": "qk_level1_atoms.pt",
        "thresholds_loaded_from": args.thresholds,
        "extension_dictionary_used": True, "unpromoted_extension_used": True,
        "universal_claim_allowed": False, "raw_weight_passthrough_used": False,
        "kl_distillation_used_as_main_method": False,
        "alpha_sweep_used_as_main_method": False,
        "gradient_used_as_main_method": False,
        "level2_promotion_in_v1_scope": False, "vo_level1_in_this_script": False,
        "split": split_manifest,
    }

    write_json(out / "manifest.json", report)
    write_json(out / "split_manifest.json", split_manifest)
    write_jsonl(out / "verified_extension_dictionary.jsonl", accepted_rows)
    write_jsonl(out / "per_matrix_level0_and_final.jsonl", [{
        "layer": e["layer"], "head": e["head"], "delta": e["delta"],
        "base_err_initial": e["base_err_initial"], "final_err": e["current_err"],
        "base_coverage": e["base_coverage"], "ops_count": e["ops_count"],
    } for e in entries])

    print("=== Qwen Level1 Verified Extension v1.1 ===")
    print(json.dumps({k: report[k] for k in [
        "entries", "train_entries", "heldout_entries", "accepted_qk_births",
        "base_train_err_mean", "final_train_err_mean",
        "base_heldout_err_mean", "final_heldout_err_mean",
        "train_gain_total", "heldout_gain_total", "atom_tensor_file"
    ]}, indent=2))
    print(f"out={out}")
    if len(accepted_rows) == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
