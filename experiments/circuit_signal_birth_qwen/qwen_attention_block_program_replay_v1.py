#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_attention_block_program_replay_v1.py

Full attention sublayer replay from saved same-basis QK + VO program bundles.

Checks:
  QK_program -> score_program -> A_program
  VO_program -> payload_program = Xaug @ C_vo_program.T
  Y_all_program = sum_heads A_program @ payload_program

Compares against true summed head output Y_all_true and residual attention state
H_attn_after = H_before + Y_all.

No training, no KL distillation, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import build_vo_primitives, decode_greedy_analytic
from qwen_circuit_target_roundtrip_v1 import (
    build_prompts,
    collect_head_data,
    build_weight_slices,
    causal_softmax,
    get_dtype,
    parse_ints,
)
from qwen_qk_program_attention_replay_v1 import (
    load_saved_program as load_qk_program,
    build_program_mdelta,
)

VERSION = "qwen_attention_block_program_replay_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def load_vo_program(vo_program_run: Path):
    atoms_path = vo_program_run / "vo_autoexpand_atoms.pt"
    gates_path = vo_program_run / "per_head_vo_matrix_closure.jsonl"
    if not atoms_path.exists():
        raise FileNotFoundError(atoms_path)
    if not gates_path.exists():
        raise FileNotFoundError(gates_path)
    atoms = torch.load(atoms_path, map_location="cpu")
    gates_by_key: Dict[Tuple[int, int], Dict[str, float]] = {}
    for line in gates_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (int(row["layer"]), int(row["head"]))
        gates_by_key[key] = {str(k): float(v) for k, v in row.get("gates", {}).items()}
    return atoms, gates_by_key


def apply_vo_gates(recon: torch.Tensor, atoms: Dict[str, torch.Tensor], gates: Dict[str, float]):
    out = recon.clone().float().cpu()
    missing = []
    for atom_key, coeff in gates.items():
        if atom_key not in atoms:
            missing.append(atom_key)
            continue
        out = out + float(coeff) * atoms[atom_key].float().cpu()
    return out, missing


def build_vo_program_C(model, layer_idx: int, head_idx: int, meta: Dict[str, Any], thresholds, vo_atoms, vo_gates_by_key):
    weights = build_weight_slices(model, layer_idx, head_idx, meta)
    C_exact = (weights["Wo"] @ weights["Wv_aug"]).cpu().float()
    prims = build_vo_primitives(int(meta["hidden_size"]))
    _ops, rec, met = decode_greedy_analytic(C_exact, prims, "vo", thresholds, max_ops=64, device="cpu")
    gates = vo_gates_by_key.get((layer_idx, head_idx), {})
    C_prog, missing = apply_vo_gates(rec.cpu().float(), vo_atoms, gates)
    return C_prog, C_exact, {
        "base_err": float(met["roundtrip_error"]),
        "program_matrix_err": rel_err(C_prog, C_exact),
        "num_gates": len(gates),
        "missing_atoms": missing,
    }


def program_scores_A_for_seq(s, Mdelta_prog: Dict[int, torch.Tensor], max_delta: int):
    T = int(s.Xaug.shape[0])
    D = int(s.Q.shape[-1])
    scores_true = (s.Q @ s.K.T) / math.sqrt(D)
    scores_prog = torch.zeros_like(scores_true)
    mask_eval = torch.zeros_like(scores_true, dtype=torch.bool)
    for i in range(T):
        for j in range(i + 1):
            d = i - j
            if d <= max_delta and d in Mdelta_prog:
                scores_prog[i, j] = s.Xaug[i] @ Mdelta_prog[d] @ s.Xaug[j]
                mask_eval[i, j] = True
    row_covered = torch.tensor([
        all(((i - j) <= max_delta and (i - j) in Mdelta_prog) for j in range(i + 1))
        for i in range(T)
    ], dtype=torch.bool)
    A_prog = causal_softmax(scores_prog.masked_fill(~mask_eval, torch.finfo(scores_prog.dtype).min))
    return scores_prog, A_prog, scores_true, row_covered, mask_eval


@torch.no_grad()
def collect_h_before(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str):
    refs = {}
    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc.get("attention_mask")
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True, use_cache=False)
        refs[pi] = outputs.hidden_states[layer_idx][0].detach().float().cpu()
    return refs


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
    ap.add_argument("--qk-program-run", required=True)
    ap.add_argument("--vo-program-run", required=True)
    ap.add_argument("--y-tol", type=float, default=1e-3)
    ap.add_argument("--h-tol", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/qwen_attention_block_program_replay_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    qk_atoms, qk_gates_by_key = load_qk_program(Path(args.qk_program_run))
    vo_atoms, vo_gates_by_key = load_vo_program(Path(args.vo_program_run))

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    layers = parse_ints(args.layers)
    heads = parse_ints(args.heads)
    prompts = build_prompts(args.prompts)

    per_head = []
    per_prompt_acc: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    missing_all = []

    for li in layers:
        h_before = collect_h_before(model, tok, prompts, li, args.max_length, args.device)
        for pi, H0 in h_before.items():
            per_prompt_acc[(li, pi)] = {
                "Y_prog": torch.zeros_like(H0),
                "Y_true": torch.zeros_like(H0),
                "H_before": H0,
            }

        for hi in heads:
            seqs, meta = collect_head_data(model, tok, prompts, li, hi, args.max_length, args.device)
            if not seqs:
                continue

            Mprog, _Mexact, qk_delta_rows, qk_missing = build_program_mdelta(
                model, li, hi, meta, seqs, args, thresholds, qk_atoms, qk_gates_by_key
            )
            Cprog, _Cexact, vo_info = build_vo_program_C(model, li, hi, meta, thresholds, vo_atoms, vo_gates_by_key)
            missing_all.extend(qk_missing)
            missing_all.extend(vo_info["missing_atoms"])

            score_rels = []
            A_rels = []
            Y_rels = []
            top1s = []
            covs = []

            for s in seqs:
                scores_prog, A_prog, scores_true, row_covered, mask_eval = program_scores_A_for_seq(s, Mprog, args.max_delta)
                payload_prog = s.Xaug @ Cprog.T
                Y_head_prog = A_prog @ payload_prog
                Y_head_true = s.Y

                per_prompt_acc[(li, int(s.prompt_id))]["Y_prog"] += Y_head_prog.float().cpu()
                per_prompt_acc[(li, int(s.prompt_id))]["Y_true"] += Y_head_true.float().cpu()

                if int(mask_eval.sum()) > 0:
                    score_rels.append(rel_err(scores_prog[mask_eval], scores_true[mask_eval]))
                if int(row_covered.sum()) > 0:
                    A_rels.append(rel_err(A_prog[row_covered], s.A[row_covered]))
                    top_true = torch.argmax(s.A[row_covered], dim=-1)
                    top_prog = torch.argmax(A_prog[row_covered], dim=-1)
                    top1s.append(float((top_true == top_prog).float().mean()))
                Y_rels.append(rel_err(Y_head_prog, Y_head_true))
                covs.append(float(int(row_covered.sum()) / max(1, int(s.Xaug.shape[0]))))

            row = {
                "layer": li,
                "head": hi,
                "kv_head": int(meta["kv_idx"]),
                "score_rel": mean(score_rels),
                "A_rel": mean(A_rels),
                "Y_head_rel": mean(Y_rels),
                "top1": mean(top1s),
                "row_coverage": mean(covs),
                "vo_matrix_err": vo_info["program_matrix_err"],
                "vo_num_gates": vo_info["num_gates"],
            }
            per_head.append(row)
            print(f"L{li}H{hi}: score={row['score_rel']:.3e} A={row['A_rel']:.3e} Yh={row['Y_head_rel']:.3e} top1={row['top1']:.3f}")

    per_prompt = []
    for (li, pi), acc in sorted(per_prompt_acc.items()):
        Y_prog = acc["Y_prog"]
        Y_true = acc["Y_true"]
        H0 = acc["H_before"]
        H_prog = H0 + Y_prog
        H_true = H0 + Y_true
        per_prompt.append({
            "layer": li,
            "prompt_id": pi,
            "T": int(Y_true.shape[0]),
            "Y_all_rel": rel_err(Y_prog, Y_true),
            "H_attn_after_rel": rel_err(H_prog, H_true),
        })

    y_mean = mean([r["Y_all_rel"] for r in per_prompt])
    y_max = max([r["Y_all_rel"] for r in per_prompt]) if per_prompt else 999.0
    h_mean = mean([r["H_attn_after_rel"] for r in per_prompt])
    h_max = max([r["H_attn_after_rel"] for r in per_prompt]) if per_prompt else 999.0
    score_mean = mean([r["score_rel"] for r in per_head])
    A_mean = mean([r["A_rel"] for r in per_head])
    yh_mean = mean([r["Y_head_rel"] for r in per_head])

    report = {
        "version": VERSION,
        "mode": "attention_block_program_replay",
        "model": args.model,
        "qk_program_run": args.qk_program_run,
        "vo_program_run": args.vo_program_run,
        "qk_atom_count": len(qk_atoms),
        "vo_atom_count": len(vo_atoms),
        "heads": len(per_head),
        "prompts": len(per_prompt),
        "score_rel_mean": score_mean,
        "A_rel_mean": A_mean,
        "Y_head_rel_mean": yh_mean,
        "Y_all_rel_mean": y_mean,
        "Y_all_rel_max": y_max,
        "H_attn_after_rel_mean": h_mean,
        "H_attn_after_rel_max": h_max,
        "y_tol": args.y_tol,
        "h_tol": args.h_tol,
        "Y_all_closed_mean": y_mean <= args.y_tol,
        "Y_all_closed_max": y_max <= args.y_tol,
        "H_after_closed_mean": h_mean <= args.h_tol,
        "H_after_closed_max": h_max <= args.h_tol,
        "status": "ATTENTION_BLOCK_PROGRAM_REPLAY_CLOSED" if (y_mean <= args.y_tol and h_mean <= args.h_tol) else "ATTENTION_BLOCK_PROGRAM_REPLAY_PARTIAL",
        "no_training": True,
        "closure_level": "qk_vo_attention_sublayer",
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "universal_claim_allowed": False,
        "missing_atom_count": len(set(missing_all)),
        "missing_atoms": sorted(set(missing_all))[:50],
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_head_attention_block_replay.jsonl", per_head)
    write_jsonl(out / "per_prompt_attention_block_replay.jsonl", per_prompt)

    print("=== Qwen Attention Block Program Replay ===")
    print(json.dumps({
        "qk_atom_count": report["qk_atom_count"],
        "vo_atom_count": report["vo_atom_count"],
        "heads": report["heads"],
        "prompts": report["prompts"],
        "score_rel_mean": report["score_rel_mean"],
        "A_rel_mean": report["A_rel_mean"],
        "Y_head_rel_mean": report["Y_head_rel_mean"],
        "Y_all_rel_mean": report["Y_all_rel_mean"],
        "Y_all_rel_max": report["Y_all_rel_max"],
        "H_attn_after_rel_mean": report["H_attn_after_rel_mean"],
        "H_attn_after_rel_max": report["H_attn_after_rel_max"],
        "status": report["status"],
        "missing_atom_count": report["missing_atom_count"],
    }, indent=2))
    print(f"out={out}")

    if report["status"] != "ATTENTION_BLOCK_PROGRAM_REPLAY_CLOSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
