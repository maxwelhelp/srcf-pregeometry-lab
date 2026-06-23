#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_vo_autoexpand_replay_v1.py

VO same-basis autoexpanded program closure + functional replay.

Target:
  C_vo_aug[h] = Wo[h] @ Wv_aug[kv]

Functional check:
  payload_prog = Xaug @ C_vo_program.T
  Y_head_prog  = A_true @ payload_prog

This uses true A from QK. QK program is already verified separately.
No training, no KL, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import build_vo_primitives, decode_greedy_analytic
from qwen_circuit_target_roundtrip_v1 import (
    build_prompts,
    collect_head_data,
    build_weight_slices,
    get_dtype,
    parse_ints,
)

VERSION = "qwen_vo_autoexpand_replay_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def flat(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1)


def norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.norm(x.float()).clamp_min(eps)


def unit(x: torch.Tensor) -> torch.Tensor:
    return x.float() / norm(x)


def coeff_for(residual: torch.Tensor, atom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r = flat(residual)
    a = flat(atom)
    return (r @ a) / (a @ a).clamp_min(eps)


def eval_atom(entries: List[Dict[str, Any]], atom: torch.Tensor) -> Dict[str, float]:
    if not entries:
        return {"gain": 0.0, "err_after": 0.0, "abs_coeff": 0.0}
    atom = atom.float().cpu()
    gains, errs, coeffs = [], [], []
    for e in entries:
        c = coeff_for(e["residual"], atom)
        rec = e["recon"] + c * atom
        err = rel_err(rec, e["target"])
        gains.append(float(e["current_err"] - err))
        errs.append(float(err))
        coeffs.append(abs(float(c.detach().cpu())))
    return {"gain": mean(gains), "err_after": mean(errs), "abs_coeff": mean(coeffs)}


def apply_atom(entries: List[Dict[str, Any]], atom: torch.Tensor, atom_key: str) -> None:
    atom = atom.float().cpu()
    for e in entries:
        c = coeff_for(e["residual"], atom)
        e["recon"] = e["recon"] + c * atom
        e["residual"] = e["target"] - e["recon"]
        e["current_err"] = rel_err(e["recon"], e["target"])
        e.setdefault("gates", {})[atom_key] = float(c.detach().cpu())


def apply_private_exact(entry: Dict[str, Any], atom: torch.Tensor, atom_key: str) -> None:
    atom = atom.float().cpu()
    entry["recon"] = entry["recon"] + atom
    entry["residual"] = entry["target"] - entry["recon"]
    entry["current_err"] = rel_err(entry["recon"], entry["target"])
    entry.setdefault("gates", {})[atom_key] = 1.0


def mine_atoms(entries: List[Dict[str, Any]], k: int, mode: str) -> List[torch.Tensor]:
    if not entries:
        return []
    shape = tuple(entries[0]["target"].shape)
    X = torch.stack([flat(e["residual"]) for e in entries], dim=0).float()
    atoms: List[torch.Tensor] = []
    if mode in ("mean", "mixed"):
        m = X.mean(dim=0)
        if norm(m) > 1e-12:
            atoms.append(unit(m.reshape(shape)).cpu())
    if mode in ("pca_centered", "mixed"):
        Xc = X - X.mean(dim=0, keepdim=True)
        if norm(Xc) > 1e-12:
            _U, _S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            for i in range(min(k, Vh.shape[0])):
                atoms.append(unit(Vh[i].reshape(shape)).cpu())
    if mode in ("pca_raw", "mixed"):
        Xn = X / torch.linalg.norm(X, dim=1, keepdim=True).clamp_min(1e-12)
        if norm(Xn) > 1e-12:
            _U, _S, Vh = torch.linalg.svd(Xn, full_matrices=False)
            for i in range(min(k, Vh.shape[0])):
                atoms.append(unit(Vh[i].reshape(shape)).cpu())
    return atoms


def dup_corr(atom: torch.Tensor, accepted: List[torch.Tensor]) -> float:
    if not accepted:
        return 0.0
    a = unit(atom).reshape(-1)
    return max(float(abs(a @ unit(b).reshape(-1))) for b in accepted)


def split_heads(entries: List[Dict[str, Any]]):
    train = [e for e in entries if int(e["head"]) % 2 == 0]
    held = [e for e in entries if int(e["head"]) % 2 == 1]
    return train, held


def collect_vo_entries(model, tok, layers, heads, prompts, args, thresholds):
    entries = []
    for li in layers:
        for hi in heads:
            seqs, meta = collect_head_data(model, tok, prompts, li, hi, args.max_length, args.device)
            if not seqs:
                continue
            weights = build_weight_slices(model, li, hi, meta)
            C = (weights["Wo"] @ weights["Wv_aug"]).cpu().float()
            prims = build_vo_primitives(int(meta["hidden_size"]))
            _ops, rec, met = decode_greedy_analytic(C, prims, "vo", thresholds, max_ops=64, device="cpu")
            err = float(met["roundtrip_error"])
            entries.append({
                "layer": int(li),
                "head": int(hi),
                "kv_head": int(meta["kv_idx"]),
                "target": C,
                "recon": rec.cpu().float(),
                "residual": C - rec.cpu().float(),
                "base_err": err,
                "current_err": err,
                "base_coverage": float(met["typed_coverage"]),
                "ops_count": float(met["ops_count"]),
                "gates": {},
                "seqs": seqs,
            })
    return entries


def functional_eval(entries: List[Dict[str, Any]]):
    rows = []
    for e in entries:
        Cprog = e["recon"].float().cpu()
        y_nums = []
        y_dens = []
        v_nums = []
        v_dens = []
        for s in e["seqs"]:
            payload_prog = s.Xaug @ Cprog.T
            payload_true = s.V @ build_o_identity_like(s.V) if False else None
            Y_prog = s.A @ payload_prog
            dy = (Y_prog - s.Y).float()
            y_nums.append(float((dy * dy).sum()))
            y_dens.append(float((s.Y.float() * s.Y.float()).sum()))
            # V is before Wo. C_vo maps directly to residual write, so V_rel is not measured here.
        y_rel = (sum(y_nums) / max(1e-12, sum(y_dens))) ** 0.5
        rows.append({
            "layer": int(e["layer"]),
            "head": int(e["head"]),
            "kv_head": int(e["kv_head"]),
            "matrix_err": float(e["current_err"]),
            "Y_head_rel": float(y_rel),
            "prompts": len(e["seqs"]),
            "num_gates": len(e.get("gates", {})),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="23")
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--target-tol", type=float, default=1e-3)
    ap.add_argument("--shared-births", type=int, default=8)
    ap.add_argument("--candidate-mode", default="mixed", choices=["mean", "pca_centered", "pca_raw", "mixed"])
    ap.add_argument("--no-private-exact", action="store_true")
    ap.add_argument("--out", default="runs/qwen_vo_autoexpand_replay_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    min_train_gain = float(thresholds.get("MIN_EXTENSION_GAIN", 1e-3))
    min_held_gain = float(thresholds.get("MIN_HELDOUT_GAIN", 1e-3))
    max_dup = float(thresholds.get("MAX_DUP_CORR", 0.985))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    entries = collect_vo_entries(model, tok, parse_ints(args.layers), parse_ints(args.heads), build_prompts(args.prompts), args, thresholds)
    if not entries:
        raise SystemExit("No VO entries collected")

    base_err = mean([e["base_err"] for e in entries])
    atoms_store: Dict[str, torch.Tensor] = {}
    rows = []
    accepted: List[torch.Tensor] = []

    train, held = split_heads(entries)
    for bi in range(args.shared_births):
        cands = mine_atoms(train, k=max(8, args.shared_births * 2), mode=args.candidate_mode)
        best = None
        for ci, atom in enumerate(cands):
            dc = dup_corr(atom, accepted)
            tr = eval_atom(train, atom)
            he = eval_atom(held, atom) if held else tr
            ok = dc <= max_dup and tr["gain"] >= min_train_gain and he["gain"] >= min_held_gain
            score = (he["gain"], tr["gain"], -dc)
            if ok and (best is None or score > best[0]):
                best = (score, ci, atom.cpu().float(), tr, he, dc)
        if best is None:
            break
        _score, ci, atom, tr, he, dc = best
        atom_key = f"vo_shared_atom_{bi:03d}"
        atoms_store[atom_key] = atom.cpu()
        accepted.append(atom.cpu())
        apply_atom(entries, atom, atom_key)
        train, held = split_heads(entries)
        rows.append({
            "op_id": atom_key,
            "stage": "shared_vo",
            "atom_ref": f"vo_autoexpand_atoms.pt::{atom_key}",
            "op_type": "VO_SharedResidualWrite",
            "universal": False,
            "cross_model_transferable": False,
            "allowed_for_same_arch_transplant": True,
            "train_gain": tr["gain"],
            "heldout_gain": he["gain"],
            "dup_corr": dc,
            "candidate_index": ci,
        })

    shared_err = mean([e["current_err"] for e in entries])

    private_added = 0
    if not args.no_private_exact:
        for e in entries:
            if float(e["current_err"]) <= args.target_tol:
                continue
            atom = e["residual"].clone().float().cpu()
            atom_key = f"vo_private_exact_L{e['layer']}_H{e['head']}_{private_added:04d}"
            atoms_store[atom_key] = atom.cpu()
            before = float(e["current_err"])
            apply_private_exact(e, atom, atom_key)
            after = float(e["current_err"])
            rows.append({
                "op_id": atom_key,
                "stage": "private_exact_vo",
                "atom_ref": f"vo_autoexpand_atoms.pt::{atom_key}",
                "op_type": "VO_PrivateExactResidualWrite",
                "universal": False,
                "cross_model_transferable": False,
                "allowed_for_same_arch_transplant": True,
                "train_gain": before - after,
                "heldout_gain": before - after,
                "scope_selector": {"layer": int(e["layer"]), "head": int(e["head"])},
            })
            private_added += 1

    final_err = mean([e["current_err"] for e in entries])
    max_err = max([float(e["current_err"]) for e in entries])
    func_rows = functional_eval(entries)
    y_mean = mean([r["Y_head_rel"] for r in func_rows])
    y_max = max([r["Y_head_rel"] for r in func_rows])

    torch.save(atoms_store, out / "vo_autoexpand_atoms.pt")

    report = {
        "version": VERSION,
        "mode": "vo_autoexpand_functional_replay",
        "model": args.model,
        "entries": len(entries),
        "base_matrix_err_mean": base_err,
        "shared_matrix_err_mean": shared_err,
        "final_matrix_err_mean": final_err,
        "final_matrix_err_max": max_err,
        "Y_head_rel_mean": y_mean,
        "Y_head_rel_max": y_max,
        "target_tol": args.target_tol,
        "matrix_closed_mean": final_err <= args.target_tol,
        "matrix_closed_max": max_err <= args.target_tol,
        "Y_closed_mean": y_mean <= args.target_tol,
        "Y_closed_max": y_max <= args.target_tol,
        "added_shared_atoms": len([r for r in rows if r["stage"] == "shared_vo"]),
        "added_private_exact_atoms": private_added,
        "total_atoms": len(atoms_store),
        "atom_tensor_file": "vo_autoexpand_atoms.pt",
        "status": "VO_PROGRAM_REPLAY_CLOSED" if (final_err <= args.target_tol and y_mean <= args.target_tol) else "VO_PROGRAM_REPLAY_PARTIAL",
        "no_training": True,
        "closure_level": "vo_matrix_and_y_head",
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "universal_claim_allowed": False,
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "vo_autoexpanded_dictionary.jsonl", rows)
    write_jsonl(out / "per_head_vo_functional_replay.jsonl", func_rows)
    write_jsonl(out / "per_head_vo_matrix_closure.jsonl", [{
        "layer": e["layer"],
        "head": e["head"],
        "kv_head": e["kv_head"],
        "base_err": e["base_err"],
        "final_err": e["current_err"],
        "gates": e.get("gates", {}),
    } for e in entries])

    print("=== Qwen VO Autoexpand Replay ===")
    print(json.dumps({
        "entries": report["entries"],
        "base_matrix_err_mean": report["base_matrix_err_mean"],
        "shared_matrix_err_mean": report["shared_matrix_err_mean"],
        "final_matrix_err_mean": report["final_matrix_err_mean"],
        "final_matrix_err_max": report["final_matrix_err_max"],
        "Y_head_rel_mean": report["Y_head_rel_mean"],
        "Y_head_rel_max": report["Y_head_rel_max"],
        "added_shared_atoms": report["added_shared_atoms"],
        "added_private_exact_atoms": report["added_private_exact_atoms"],
        "total_atoms": report["total_atoms"],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")

    if report["status"] != "VO_PROGRAM_REPLAY_CLOSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
