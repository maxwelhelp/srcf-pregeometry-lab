#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_level1_verified_extension_v1.py

Real Qwen Level-1 verified extension mining for Exact Program Transplant.

Doctrine rules:
- Level 0 analytic dictionary is kept separate.
- Level 1 candidates are mined from residual only after Level 0 decode.
- QK split is fixed: first 70% heads + even deltas train, remaining heads + odd deltas heldout.
- No training, no KL, no alpha sweep, no learned base dictionary.
- Output is verified_extension_dictionary.jsonl, not universal primitives.

v1 scope: QK only. VO Level1 requires prompt-heldout functional gate and is deferred.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path
from typing import Any, Dict, List

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import build_qk_primitives, decode_greedy_analytic, rel_err, energy
from qwen_circuit_target_roundtrip_v1 import (
    build_prompts, collect_head_data, build_weight_slices, build_rope_by_pos,
    qk_delta_matrices_affine, get_dtype, parse_ints
)

VERSION = "qwen_level1_verified_extension_v1.0"


def flat(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1)


def unit(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.float() / torch.linalg.norm(x.float()).clamp_min(eps)


def coeff_for(residual: torch.Tensor, atom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = flat(atom); r = flat(residual)
    return (r @ a) / (a @ a).clamp_min(eps)


def apply_atom(entry: Dict[str, Any], atom: torch.Tensor) -> Dict[str, float]:
    c = coeff_for(entry["residual"], atom)
    rec1 = entry["recon"] + c * atom
    err1 = rel_err(rec1, entry["target"])
    gain = float(entry["base_err"] - err1)
    return {"coeff": float(c.detach().cpu()), "err_after": float(err1), "gain": gain}


def eval_atom(entries: List[Dict[str, Any]], atom: torch.Tensor) -> Dict[str, float]:
    vals = [apply_atom(e, atom) for e in entries]
    if not vals:
        return {"gain_mean": 0.0, "err_after_mean": 0.0, "abs_coeff_mean": 0.0}
    return {
        "gain_mean": sum(v["gain"] for v in vals) / len(vals),
        "err_after_mean": sum(v["err_after"] for v in vals) / len(vals),
        "abs_coeff_mean": sum(abs(v["coeff"]) for v in vals) / len(vals),
    }


def mine_pca_atoms(entries: List[Dict[str, Any]], k: int) -> List[torch.Tensor]:
    if not entries:
        return []
    X = torch.stack([flat(e["residual"]) for e in entries], dim=0).float()
    X = X - X.mean(dim=0, keepdim=True)
    if torch.linalg.norm(X) <= 1e-12:
        return []
    # Vh rows are principal directions over flattened residual matrices.
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    atoms = []
    shape = tuple(entries[0]["target"].shape)
    for i in range(min(k, Vh.shape[0])):
        atoms.append(unit(Vh[i].reshape(shape)))
    return atoms


def dup_corr(atom: torch.Tensor, accepted: List[torch.Tensor]) -> float:
    if not accepted:
        return 0.0
    a = unit(atom).reshape(-1)
    return max(float(abs(a @ unit(b).reshape(-1)).detach().cpu()) for b in accepted)


def split_heads(heads: List[int]):
    heads = sorted(heads)
    n_train = max(1, int(math.ceil(0.70 * len(heads))))
    if len(heads) > 1 and n_train >= len(heads):
        n_train = len(heads) - 1
    return heads[:n_train], heads[n_train:]


def entry_split(entries: List[Dict[str, Any]], train_heads: List[int], held_heads: List[int]):
    train = [e for e in entries if e["head"] in train_heads and e["delta"] % 2 == 0]
    held = [e for e in entries if e["head"] in held_heads and e["delta"] % 2 == 1]
    return train, held


def type_qk_birth(atom: torch.Tensor) -> Dict[str, Any]:
    M = atom.float(); total = float((M*M).sum().clamp_min(1e-12))
    diag_mass = float((torch.diag(M[:-1, :-1])**2).sum() / total)
    bias_col_mass = float((M[:-1, -1]**2).sum() / total)
    band = torch.zeros_like(M, dtype=torch.bool)
    H = M.shape[0] - 1
    for i in range(H):
        for j in range(max(0, i-8), min(H, i+9)):
            band[i, j] = True
    local_band_mass = float((M[band]**2).sum() / total)
    rank = int(torch.linalg.matrix_rank(M).item())
    return {"op_type": "QK_BirthTypedRoute", "rank": rank, "diagonal_mass": diag_mass, "bias_col_mass": bias_col_mass, "local_band_mass": local_band_mass}


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
    ap.add_argument("--out", default="runs/qwen_level1_verified_extension_v1")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    thresholds = load_thresholds(args.thresholds)
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
            H = int(meta["hidden_size"])
            prims = build_qk_primitives(H)
            for d, M in Mdelta.items():
                ops, rec, met = decode_greedy_analytic(M, prims, "qk", thresholds, max_ops=64, device="cpu")
                entries.append({"layer": li, "head": hi, "delta": d, "target": M.cpu().float(), "recon": rec.cpu().float(), "residual": (M-rec).cpu().float(), "base_err": float(met["roundtrip_error"]), "base_coverage": float(met["typed_coverage"]), "ops_count": float(met["ops_count"])})

    train_entries, held_entries = entry_split(entries, train_heads, held_heads)
    if not train_entries or not held_entries:
        raise SystemExit("Not enough heads/deltas for fixed train/heldout split. Use at least 2 heads and max_delta>=1.")

    accepted_atoms: List[torch.Tensor] = []
    accepted_rows: List[Dict[str, Any]] = []
    min_held = float(thresholds.get("MIN_HELDOUT_GAIN", 1e-3))
    min_ext = float(thresholds.get("MIN_EXTENSION_GAIN", 1e-3))
    max_dup = float(thresholds.get("MAX_DUP_CORR", 0.985))

    current_train = train_entries
    current_held = held_entries
    for bi in range(args.qk_births):
        atoms = mine_pca_atoms(current_train, k=max(8, args.qk_births * 2))
        best = None
        for ai, atom in enumerate(atoms):
            tr = eval_atom(current_train, atom)
            he = eval_atom(current_held, atom)
            dc = dup_corr(atom, accepted_atoms)
            ok = tr["gain_mean"] >= min_ext and he["gain_mean"] >= min_held and dc <= max_dup
            score = (he["gain_mean"], tr["gain_mean"], -dc)
            if ok and (best is None or score > best[0]):
                best = (score, ai, atom, tr, he, dc)
        if best is None:
            break
        _, ai, atom, tr, he, dc = best
        accepted_atoms.append(atom.cpu())
        sig = type_qk_birth(atom)
        row = {"op_id": f"QK_BirthTypedRoute_L{layers}_b{bi}", "dictionary_level": "verified_extension_dictionary", "source": "target_mined_residual", "universal": False, "transferable": False, "eligible_for_exact_base_decode": False, "op_type": sig["op_type"], "birth_index": bi, "pca_index": ai, "train_gain": tr["gain_mean"], "heldout_gain": he["gain_mean"], "train_err_after": tr["err_after_mean"], "heldout_err_after": he["err_after_mean"], "dup_corr": dc, "typing_status": "typed", "signature": sig}
        accepted_rows.append(row)
        # Sequentially update residual/recon/base_err for next birth on both train and heldout sets.
        for e in entries:
            ev = apply_atom(e, atom)
            c = ev["coeff"]
            e["recon"] = e["recon"] + c * atom.cpu()
            e["residual"] = e["target"] - e["recon"]
            e["base_err"] = ev["err_after"]

    split_manifest = {"split_protocol_version": "split_v1_fixed_70_30_parity_delta", "qk_train_heads": train_heads, "qk_heldout_heads": held_heads, "qk_train_deltas_rule": "delta % 2 == 0", "qk_heldout_deltas_rule": "delta % 2 == 1", "vo_split_rule": "VO out of this script; requires prompt-heldout functional gate", "no_overlap_verified": True}
    report = {"version": VERSION, "mode": "qwen_level1_verified_extension", "model": args.model, "layers": layers, "heads": heads, "entries": len(entries), "train_entries": len(train_entries), "heldout_entries": len(held_entries), "accepted_qk_births": len(accepted_rows), "base_train_err_mean": sum(e["base_err"] for e in train_entries)/len(train_entries), "base_heldout_err_mean": sum(e["base_err"] for e in held_entries)/len(held_entries), "thresholds_loaded_from": args.thresholds, "extension_dictionary_used": True, "unpromoted_extension_used": True, "universal_claim_allowed": False, "raw_weight_passthrough_used": False, "kl_distillation_used_as_main_method": False, "alpha_sweep_used_as_main_method": False, "gradient_used_as_main_method": False, "level2_promotion_in_v1_scope": False, "vo_level1_in_this_script": False, "split": split_manifest}

    write_json(out / "manifest.json", report)
    write_json(out / "split_manifest.json", split_manifest)
    write_jsonl(out / "verified_extension_dictionary.jsonl", accepted_rows)
    write_jsonl(out / "per_matrix_level0_and_final.jsonl", [{"layer": e["layer"], "head": e["head"], "delta": e["delta"], "final_err": e["base_err"], "base_coverage": e["base_coverage"], "ops_count": e["ops_count"]} for e in entries])
    print("=== Qwen Level1 Verified Extension ===")
    print(json.dumps({k: report[k] for k in ["entries", "train_entries", "heldout_entries", "accepted_qk_births", "base_train_err_mean", "base_heldout_err_mean"]}, indent=2))
    print(f"out={out}")
    if len(accepted_rows) == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
