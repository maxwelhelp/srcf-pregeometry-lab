#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_qk_program_attention_replay_v1.py

Functional QK replay from a saved same-basis program bundle.

Checks:
  saved program matrices -> score matrix -> softmax attention A

This is the next gate after qwen_same_basis_program_replay_v1.py:
  matrix closure is not enough; the program must reproduce routing scores/A on prompts.

No training, no KL distillation, no LoRA, no alpha sweep.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import build_qk_primitives, decode_greedy_analytic
from qwen_circuit_target_roundtrip_v1 import (
    build_prompts,
    collect_head_data,
    build_weight_slices,
    build_rope_by_pos,
    qk_delta_matrices_affine,
    causal_softmax,
    get_dtype,
    parse_ints,
)

VERSION = "qwen_qk_program_attention_replay_v1.0"


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


def apply_saved_gates(recon: torch.Tensor, atoms: Dict[str, torch.Tensor], gates: Dict[str, float]):
    out = recon.clone().float().cpu()
    missing = []
    for atom_key, coeff in gates.items():
        if atom_key not in atoms:
            missing.append(atom_key)
            continue
        out = out + float(coeff) * atoms[atom_key].float().cpu()
    return out, missing


def build_program_mdelta(model, layer_idx: int, head_idx: int, meta: Dict[str, Any], seqs, args, thresholds, atoms, gates_by_key):
    weights = build_weight_slices(model, layer_idx, head_idx, meta)
    max_pos = min(args.max_delta, max(int(s.Xn.shape[0]) - 1 for s in seqs))
    Rpos = build_rope_by_pos(seqs, max_pos)
    Mdelta_exact = qk_delta_matrices_affine(weights, Rpos, max_pos, int(meta["head_dim"]))
    prims = build_qk_primitives(int(meta["hidden_size"]))

    Mdelta_prog = {}
    per_delta = []
    missing_all = []
    for d, M in Mdelta_exact.items():
        _ops, rec, met = decode_greedy_analytic(M, prims, "qk", thresholds, max_ops=64, device="cpu")
        gates = gates_by_key.get((layer_idx, head_idx, int(d)), {})
        prog, missing = apply_saved_gates(rec.cpu().float(), atoms, gates)
        missing_all.extend(missing)
        Mdelta_prog[int(d)] = prog
        per_delta.append({
            "layer": layer_idx,
            "head": head_idx,
            "delta": int(d),
            "base_err": float(met["roundtrip_error"]),
            "program_matrix_err": rel_err(prog, M.cpu().float()),
            "num_gates": len(gates),
            "missing_atoms": missing,
        })
    return Mdelta_prog, Mdelta_exact, per_delta, missing_all


def eval_qk_scores_for_head(seqs, Mdelta_prog: Dict[int, torch.Tensor], max_delta: int):
    rows = []
    score_num = score_den = 0.0
    a_num = a_den = 0.0
    top1_ok = top1_total = 0
    coverage_rows = 0
    total_rows = 0

    for s in seqs:
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

        if int(mask_eval.sum()) > 0:
            diff = (scores_prog[mask_eval] - scores_true[mask_eval]).float()
            score_num += float((diff * diff).sum())
            score_den += float((scores_true[mask_eval].float() ** 2).sum())

        row_covered = torch.tensor([
            all(((i - j) <= max_delta and (i - j) in Mdelta_prog) for j in range(i + 1))
            for i in range(T)
        ], dtype=torch.bool)
        total_rows += T
        coverage_rows += int(row_covered.sum())

        A_prog = causal_softmax(scores_prog.masked_fill(~mask_eval, torch.finfo(scores_prog.dtype).min))
        if int(row_covered.sum()) > 0:
            da = (A_prog[row_covered] - s.A[row_covered]).float()
            a_num += float((da * da).sum())
            a_den += float((s.A[row_covered].float() ** 2).sum())
            top_true = torch.argmax(s.A[row_covered], dim=-1)
            top_prog = torch.argmax(A_prog[row_covered], dim=-1)
            top1_ok += int((top_true == top_prog).sum())
            top1_total += int(top_true.numel())

        rows.append({
            "prompt_id": int(s.prompt_id),
            "T": T,
            "score_rel": rel_err(scores_prog[mask_eval], scores_true[mask_eval]) if int(mask_eval.sum()) else None,
            "A_rel": rel_err(A_prog[row_covered], s.A[row_covered]) if int(row_covered.sum()) else None,
            "top1": float((torch.argmax(A_prog[row_covered], dim=-1) == torch.argmax(s.A[row_covered], dim=-1)).float().mean()) if int(row_covered.sum()) else None,
            "row_coverage": float(int(row_covered.sum()) / max(1, T)),
        })

    return {
        "score_rel": math.sqrt(score_num / max(1e-12, score_den)),
        "A_rel": math.sqrt(a_num / max(1e-12, a_den)) if a_den > 0 else None,
        "top1": float(top1_ok / max(1, top1_total)),
        "row_coverage": float(coverage_rows / max(1, total_rows)),
    }, rows


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
    ap.add_argument("--score-tol", type=float, default=1e-3)
    ap.add_argument("--a-tol", type=float, default=1e-3)
    ap.add_argument("--out", default="runs/qwen_qk_program_attention_replay_v1")
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

    per_head = []
    per_prompt = []
    per_delta = []
    missing_all = []

    for li in parse_ints(args.layers):
        for hi in parse_ints(args.heads):
            seqs, meta = collect_head_data(model, tok, build_prompts(args.prompts), li, hi, args.max_length, args.device)
            if not seqs:
                continue
            Mprog, _Mexact, delta_rows, missing = build_program_mdelta(
                model, li, hi, meta, seqs, args, thresholds, atoms, gates_by_key
            )
            missing_all.extend(missing)
            per_delta.extend(delta_rows)
            summary, prompt_rows = eval_qk_scores_for_head(seqs, Mprog, args.max_delta)
            summary.update({"layer": li, "head": hi, "kv_head": int(meta["kv_idx"]), "seqs": len(seqs)})
            per_head.append(summary)
            for r in prompt_rows:
                r.update({"layer": li, "head": hi})
                per_prompt.append(r)
            print(f"L{li}H{hi}: score={summary['score_rel']:.3e} A={summary['A_rel']:.3e} top1={summary['top1']:.3f} cov={summary['row_coverage']:.2f}")

    report = {
        "version": VERSION,
        "mode": "qk_program_attention_replay",
        "model": args.model,
        "program_run": args.program_run,
        "atom_count": len(atoms),
        "heads": len(per_head),
        "score_rel_mean": mean([x["score_rel"] for x in per_head]),
        "A_rel_mean": mean([x["A_rel"] for x in per_head if x["A_rel"] is not None]),
        "top1_mean": mean([x["top1"] for x in per_head]),
        "row_coverage_mean": mean([x["row_coverage"] for x in per_head]),
        "score_tol": args.score_tol,
        "a_tol": args.a_tol,
        "score_closed": mean([x["score_rel"] for x in per_head]) <= args.score_tol,
        "A_closed": mean([x["A_rel"] for x in per_head if x["A_rel"] is not None]) <= args.a_tol,
        "status": "QK_PROGRAM_ATTENTION_REPLAY_CLOSED" if (mean([x["score_rel"] for x in per_head]) <= args.score_tol and mean([x["A_rel"] for x in per_head if x["A_rel"] is not None]) <= args.a_tol) else "QK_PROGRAM_ATTENTION_REPLAY_PARTIAL",
        "no_training": True,
        "closure_level": "score_and_attention",
        "same_basis_transplant_allowed": True,
        "cross_model_claim_allowed": False,
        "universal_claim_allowed": False,
        "missing_atom_count": len(set(missing_all)),
        "missing_atoms": sorted(set(missing_all))[:50],
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_head_attention_replay.jsonl", per_head)
    write_jsonl(out / "per_prompt_attention_replay.jsonl", per_prompt)
    write_jsonl(out / "per_delta_program_matrix_replay.jsonl", per_delta)

    print("=== Qwen QK Program Attention Replay ===")
    print(json.dumps({
        "atom_count": report["atom_count"],
        "heads": report["heads"],
        "score_rel_mean": report["score_rel_mean"],
        "A_rel_mean": report["A_rel_mean"],
        "top1_mean": report["top1_mean"],
        "row_coverage_mean": report["row_coverage_mean"],
        "status": report["status"],
        "missing_atom_count": report["missing_atom_count"],
    }, indent=2))
    print(f"out={out}")

    if report["status"] != "QK_PROGRAM_ATTENTION_REPLAY_CLOSED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
