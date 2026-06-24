#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_head_circuit_program_diff_v1.py

Correct abstraction level for Qwen Coder -> Instruct program diff.

This script DOES NOT compare raw Wq/Wk/Wv/Wo tensors and DOES NOT compare raw
private atoms across independently grown dictionaries.

It compares each attention head as a circuit/function:

  QK route target per head/delta:
    M_qk_aug[h,d] = Wq_aug[h].T @ R_delta @ Wk_aug[kv(h)] / sqrt(head_dim)

  VO write target per head:
    C_vo_aug[h] = Wo[h] @ Wv_aug[kv(h)]

For each model separately it can use its own saved autoexpanded program bundle
(QK atoms/gates + VO atoms/gates) to reconstruct the head-function matrices.
Then it diffs teacher vs student at the head-circuit matrix level:

  diff(decode_head_circuit(teacher), decode_head_circuit(student))

not:

  diff(Wq), diff(Wv), diff(raw private atom ids)

It also emits a lightweight MLP trace/function summary via the actual SwiGLU
pipeline input/output, not raw MLP weights. MLP program decoding is marked as
trace-only here; the attention head-circuit path is the closed exact path.

No training. No KL distillation. No LoRA. No alpha sweep.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import load_thresholds, write_json, write_jsonl
from analytic_primitives_v1_1 import build_qk_primitives, build_vo_primitives, decode_greedy_analytic
from qwen_circuit_target_roundtrip_v1 import (
    build_prompts,
    collect_head_data,
    build_weight_slices,
    build_rope_by_pos,
    qk_delta_matrices_affine,
    get_dtype,
    parse_ints,
)
from qwen_qk_program_attention_replay_v1 import (
    load_saved_program as load_qk_program,
    build_program_mdelta,
)
from qwen_attention_block_program_replay_v1 import (
    load_vo_program,
    build_vo_program_C,
    program_scores_A_for_seq,
)

VERSION = "qwen_head_circuit_program_diff_v1.0"


def mean(xs):
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return float("nan")
    af = a.float().reshape(-1)
    bf = b.float().reshape(-1)
    return float(torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf)).clamp_min(eps))


def tensor_rank(x: torch.Tensor, tol: float = 1e-5) -> int:
    try:
        s = torch.linalg.svdvals(x.float())
        return int((s > tol * s.max().clamp_min(1e-12)).sum().item())
    except Exception:
        return -1


def op_to_row(op: Any) -> Dict[str, Any]:
    params = dict(getattr(op, "params", {}) or {})
    return {
        "op_id": getattr(op, "op_id", None),
        "op_type": getattr(op, "op_type", None),
        "primitive_name": params.get("primitive_name"),
        "formula": params.get("formula"),
        "coeff": float(params.get("coeff", 0.0)),
        "condition_type": getattr(op, "condition_type", None),
        "condition": dict(getattr(op, "condition", {}) or {}),
        "read_fields": list(getattr(op, "read_fields", []) or []),
        "write_fields": list(getattr(op, "write_fields", []) or []),
        "marginal_error_drop": float(getattr(op, "marginal_error_drop", 0.0) or 0.0),
        "decode_error_after": float(getattr(op, "decode_error_after", 0.0) or 0.0),
        "energy": float(getattr(op, "energy", 0.0) or 0.0),
        "transferable": bool(getattr(op, "transferable", False)),
    }


def decode_level0_matrix(M: torch.Tensor, kind: str, hidden_size: int, thresholds: Dict[str, Any], max_ops: int) -> Tuple[List[Dict[str, Any]], torch.Tensor, Dict[str, Any]]:
    prims = build_qk_primitives(hidden_size) if kind == "qk" else build_vo_primitives(hidden_size)
    ops, rec, met = decode_greedy_analytic(M.float().cpu(), prims, kind, thresholds, max_ops=max_ops, device="cpu")
    return [op_to_row(o) for o in ops], rec.cpu().float(), {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in met.items()}


def exact_head_targets(model, tokenizer, prompts: List[str], layer_idx: int, head_idx: int, max_length: int, max_delta: int, device: str):
    seqs, meta = collect_head_data(model, tokenizer, prompts, layer_idx, head_idx, max_length, device)
    if not seqs:
        return None
    weights = build_weight_slices(model, layer_idx, head_idx, meta)
    max_pos = min(max_delta, max(int(s.Xn.shape[0]) - 1 for s in seqs))
    Rpos = build_rope_by_pos(seqs, max_pos)
    M_exact = qk_delta_matrices_affine(weights, Rpos, max_pos, int(meta["head_dim"]))
    C_exact = (weights["Wo"] @ weights["Wv_aug"]).cpu().float()
    return {
        "seqs": seqs,
        "meta": meta,
        "M_exact": {int(k): v.cpu().float() for k, v in M_exact.items()},
        "C_exact": C_exact,
    }


def compile_head_program(model, layer_idx: int, head_idx: int, target: Dict[str, Any], args, thresholds, qk_atoms, qk_gates, vo_atoms, vo_gates):
    seqs = target["seqs"]
    meta = target["meta"]

    if qk_atoms is not None and qk_gates is not None:
        M_prog, M_exact_again, qk_delta_rows, qk_missing = build_program_mdelta(
            model, layer_idx, head_idx, meta, seqs, args, thresholds, qk_atoms, qk_gates
        )
    else:
        M_prog = {}
        qk_delta_rows = []
        qk_missing = []
        H = int(meta["hidden_size"])
        for d, M in target["M_exact"].items():
            _ops, rec, met = decode_level0_matrix(M, "qk", H, thresholds, args.max_level0_ops)
            M_prog[int(d)] = rec.cpu().float()
            qk_delta_rows.append({
                "layer": layer_idx,
                "head": head_idx,
                "delta": int(d),
                "base_err": float(met["roundtrip_error"]),
                "program_matrix_err": rel_err(rec, M),
                "num_gates": 0,
                "missing_atoms": [],
                "program_source": "level0_only",
            })

    if vo_atoms is not None and vo_gates is not None:
        C_prog, C_exact_again, vo_info = build_vo_program_C(model, layer_idx, head_idx, meta, thresholds, vo_atoms, vo_gates)
    else:
        H = int(meta["hidden_size"])
        _ops, rec, met = decode_level0_matrix(target["C_exact"], "vo", H, thresholds, args.max_level0_ops)
        C_prog = rec.cpu().float()
        vo_info = {
            "base_err": float(met["roundtrip_error"]),
            "program_matrix_err": rel_err(C_prog, target["C_exact"]),
            "num_gates": 0,
            "missing_atoms": [],
            "program_source": "level0_only",
        }

    return {
        "M_prog": {int(k): v.cpu().float() for k, v in M_prog.items()},
        "C_prog": C_prog.cpu().float(),
        "qk_delta_rows": qk_delta_rows,
        "qk_missing": qk_missing,
        "vo_info": vo_info,
    }


def eval_compiled_head_function(target: Dict[str, Any], compiled: Dict[str, Any], max_delta: int):
    score_rels, A_rels, Y_rels, top1s, covs = [], [], [], [], []
    for s in target["seqs"]:
        scores_prog, A_prog, scores_true, row_covered, mask_eval = program_scores_A_for_seq(s, compiled["M_prog"], max_delta)
        payload_prog = s.Xaug @ compiled["C_prog"].T
        Y_prog = A_prog @ payload_prog
        Y_true = s.Y
        if int(mask_eval.sum()) > 0:
            score_rels.append(rel_err(scores_prog[mask_eval], scores_true[mask_eval]))
        if int(row_covered.sum()) > 0:
            A_rels.append(rel_err(A_prog[row_covered], s.A[row_covered]))
            top_true = torch.argmax(s.A[row_covered], dim=-1)
            top_prog = torch.argmax(A_prog[row_covered], dim=-1)
            top1s.append(float((top_true == top_prog).float().mean()))
        Y_rels.append(rel_err(Y_prog, Y_true))
        covs.append(float(int(row_covered.sum()) / max(1, int(s.Xaug.shape[0]))))
    return {
        "score_rel": mean(score_rels),
        "A_rel": mean(A_rels),
        "Y_head_rel": mean(Y_rels),
        "top1": mean(top1s),
        "row_coverage": mean(covs),
    }


def matrix_diff_row(kind: str, layer: int, head: int, subkey: Any, teacher_M: torch.Tensor, student_M: torch.Tensor, teacher_prog: torch.Tensor, student_prog: torch.Tensor, thresholds, hidden_size: int, max_ops: int):
    exact_delta = teacher_M.float() - student_M.float()
    prog_delta = teacher_prog.float() - student_prog.float()
    delta_ops, delta_rec, delta_met = decode_level0_matrix(prog_delta, kind, hidden_size, thresholds, max_ops)
    top_ops = sorted(delta_ops, key=lambda r: abs(float(r["coeff"])), reverse=True)[:8]
    return {
        "kind": kind,
        "layer": layer,
        "head": head,
        "subkey": subkey,
        "shape": list(teacher_M.shape),
        "exact_teacher_student_rel": rel_err(teacher_M, student_M),
        "program_teacher_student_rel": rel_err(teacher_prog, student_prog),
        "exact_cosine": cosine(teacher_M, student_M),
        "program_cosine": cosine(teacher_prog, student_prog),
        "exact_delta_norm": float(torch.linalg.norm(exact_delta.float()).item()),
        "program_delta_norm": float(torch.linalg.norm(prog_delta.float()).item()),
        "teacher_exact_norm": float(torch.linalg.norm(teacher_M.float()).item()),
        "student_exact_norm": float(torch.linalg.norm(student_M.float()).item()),
        "teacher_program_norm": float(torch.linalg.norm(teacher_prog.float()).item()),
        "student_program_norm": float(torch.linalg.norm(student_prog.float()).item()),
        "exact_delta_rank": tensor_rank(exact_delta),
        "program_delta_rank": tensor_rank(prog_delta),
        "delta_level0_roundtrip_error": float(delta_met["roundtrip_error"]),
        "delta_level0_typed_coverage": float(delta_met.get("typed_coverage", 0.0)),
        "delta_level0_ops_count": int(delta_met.get("ops_count", 0.0)),
        "delta_top_level0_ops": top_ops,
    }


def head_diff(layer: int, head: int, teacher_target, student_target, teacher_compiled, student_compiled, thresholds, max_ops: int):
    H = int(teacher_target["meta"]["hidden_size"])
    rows = []
    deltas = sorted(set(teacher_target["M_exact"].keys()) & set(student_target["M_exact"].keys()))
    for d in deltas:
        if d not in teacher_compiled["M_prog"] or d not in student_compiled["M_prog"]:
            continue
        rows.append(matrix_diff_row(
            "qk", layer, head, int(d),
            teacher_target["M_exact"][d], student_target["M_exact"][d],
            teacher_compiled["M_prog"][d], student_compiled["M_prog"][d],
            thresholds, H, max_ops,
        ))
    rows.append(matrix_diff_row(
        "vo", layer, head, "C_vo",
        teacher_target["C_exact"], student_target["C_exact"],
        teacher_compiled["C_prog"], student_compiled["C_prog"],
        thresholds, H, max_ops,
    ))
    qk_rows = [r for r in rows if r["kind"] == "qk"]
    vo_rows = [r for r in rows if r["kind"] == "vo"]
    return rows, {
        "layer": layer,
        "head": head,
        "kv_head_teacher": int(teacher_target["meta"]["kv_idx"]),
        "kv_head_student": int(student_target["meta"]["kv_idx"]),
        "qk_delta_count": len(qk_rows),
        "qk_program_rel_mean": mean([r["program_teacher_student_rel"] for r in qk_rows]),
        "qk_program_cos_mean": mean([r["program_cosine"] for r in qk_rows]),
        "qk_delta_norm_mean": mean([r["program_delta_norm"] for r in qk_rows]),
        "qk_delta_level0_coverage_mean": mean([r["delta_level0_typed_coverage"] for r in qk_rows]),
        "vo_program_rel": vo_rows[0]["program_teacher_student_rel"] if vo_rows else None,
        "vo_program_cos": vo_rows[0]["program_cosine"] if vo_rows else None,
        "vo_delta_norm": vo_rows[0]["program_delta_norm"] if vo_rows else None,
        "vo_delta_level0_coverage": vo_rows[0]["delta_level0_typed_coverage"] if vo_rows else None,
        "vo_delta_top_ops": vo_rows[0]["delta_top_level0_ops"] if vo_rows else [],
    }


@torch.no_grad()
def collect_mlp_trace(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str):
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    traces: Dict[int, Dict[str, torch.Tensor]] = {}
    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        cap: Dict[str, torch.Tensor] = {}

        def hook(_mod, inp, out):
            x = inp[0].detach()
            cap["mlp_in"] = x[0].float().cpu()
            cap["mlp_out"] = out.detach()[0].float().cpu()
            gate = mlp.gate_proj(x)
            up = mlp.up_proj(x)
            act = getattr(mlp, "act_fn", None)
            if act is None:
                gact = F.silu(gate)
            else:
                gact = act(gate)
            z = gact * up
            cap["gate"] = gate.detach()[0].float().cpu()
            cap["up"] = up.detach()[0].float().cpu()
            cap["gated"] = z.detach()[0].float().cpu()

        h = mlp.register_forward_hook(hook)
        try:
            model(**enc, use_cache=False)
        finally:
            h.remove()
        if "mlp_out" not in cap:
            raise RuntimeError("failed to capture MLP trace")
        traces[pi] = cap
    return traces


def mlp_trace_diff_rows(layer: int, teacher_trace: Dict[int, Dict[str, torch.Tensor]], student_trace: Dict[int, Dict[str, torch.Tensor]]):
    rows = []
    for pi in sorted(set(teacher_trace.keys()) & set(student_trace.keys())):
        tr = teacher_trace[pi]
        sr = student_trace[pi]
        row = {"layer": layer, "prompt_id": pi, "component": "mlp_trace"}
        for key in ["mlp_in", "gate", "up", "gated", "mlp_out"]:
            T = min(tr[key].shape[0], sr[key].shape[0])
            a = tr[key][-T:]
            b = sr[key][-T:]
            row[f"{key}_rel"] = rel_err(a, b)
            row[f"{key}_cos"] = cosine(a, b)
            row[f"{key}_teacher_norm"] = float(torch.linalg.norm(a.float()).item())
            row[f"{key}_student_norm"] = float(torch.linalg.norm(b.float()).item())
        rows.append(row)
    return rows


def load_optional_programs(qk_path: Optional[str], vo_path: Optional[str]):
    qk_atoms = qk_gates = vo_atoms = vo_gates = None
    if qk_path:
        qk_atoms, qk_gates = load_qk_program(Path(qk_path))
    if vo_path:
        vo_atoms, vo_gates = load_vo_program(Path(vo_path))
    return qk_atoms, qk_gates, vo_atoms, vo_gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layer", type=int, default=23)
    ap.add_argument("--heads", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13")
    ap.add_argument("--focus-heads", default="1")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--max-delta", type=int, default=16)
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--teacher-qk-program-run", default=None)
    ap.add_argument("--teacher-vo-program-run", default=None)
    ap.add_argument("--student-qk-program-run", default=None)
    ap.add_argument("--student-vo-program-run", default=None)
    ap.add_argument("--max-level0-ops", type=int, default=64)
    ap.add_argument("--include-mlp-trace", action="store_true")
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/head_circuit_program_diff_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    thresholds = load_thresholds(args.thresholds)
    heads = parse_ints(args.heads)
    focus_heads = set(parse_ints(args.focus_heads))
    prompts = build_prompts(args.prompts)

    tqk_atoms, tqk_gates, tvo_atoms, tvo_gates = load_optional_programs(args.teacher_qk_program_run, args.teacher_vo_program_run)
    sqk_atoms, sqk_gates, svo_atoms, svo_gates = load_optional_programs(args.student_qk_program_run, args.student_vo_program_run)

    tok_t = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    tok_s = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

    print("loading teacher...")
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    teacher.eval()

    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    student.eval()

    per_model_closure = []
    per_matrix_diff = []
    per_head_diff = []
    per_delta_ops = []
    missing_all = []

    for h in heads:
        print(f"\n--- head circuit L{args.layer}H{h} ---")
        tt = exact_head_targets(teacher, tok_t, prompts, args.layer, h, args.max_length, args.max_delta, args.device)
        st = exact_head_targets(student, tok_s, prompts, args.layer, h, args.max_length, args.max_delta, args.device)
        if tt is None or st is None:
            continue
        tc = compile_head_program(teacher, args.layer, h, tt, args, thresholds, tqk_atoms, tqk_gates, tvo_atoms, tvo_gates)
        sc = compile_head_program(student, args.layer, h, st, args, thresholds, sqk_atoms, sqk_gates, svo_atoms, svo_gates)

        t_eval = eval_compiled_head_function(tt, tc, args.max_delta)
        s_eval = eval_compiled_head_function(st, sc, args.max_delta)
        t_eval.update({"model_role": "teacher", "layer": args.layer, "head": h, "kv_head": int(tt["meta"]["kv_idx"]), "vo_matrix_err": float(tc["vo_info"].get("program_matrix_err", 0.0))})
        s_eval.update({"model_role": "student", "layer": args.layer, "head": h, "kv_head": int(st["meta"]["kv_idx"]), "vo_matrix_err": float(sc["vo_info"].get("program_matrix_err", 0.0))})
        per_model_closure.extend([t_eval, s_eval])
        missing_all.extend(tc.get("qk_missing", []))
        missing_all.extend(sc.get("qk_missing", []))
        missing_all.extend(tc.get("vo_info", {}).get("missing_atoms", []))
        missing_all.extend(sc.get("vo_info", {}).get("missing_atoms", []))

        rows, summary = head_diff(args.layer, h, tt, st, tc, sc, thresholds, args.max_level0_ops)
        per_matrix_diff.extend(rows)
        per_head_diff.append(summary)
        for r in rows:
            for op in r.get("delta_top_level0_ops", []):
                op_row = {
                    "kind": r["kind"],
                    "layer": args.layer,
                    "head": h,
                    "subkey": r["subkey"],
                    "program_delta_norm": r["program_delta_norm"],
                    "delta_level0_roundtrip_error": r["delta_level0_roundtrip_error"],
                    **op,
                }
                per_delta_ops.append(op_row)
        print(
            f"H{h}: teacher_close score={t_eval['score_rel']:.2e} A={t_eval['A_rel']:.2e} Y={t_eval['Y_head_rel']:.2e} | "
            f"student_close score={s_eval['score_rel']:.2e} A={s_eval['A_rel']:.2e} Y={s_eval['Y_head_rel']:.2e} | "
            f"diff qk_rel={summary['qk_program_rel_mean']:.3f} vo_rel={summary['vo_program_rel']:.3f}"
        )

    mlp_rows = []
    if args.include_mlp_trace:
        print("\ncollecting MLP SwiGLU function traces...")
        t_mlp = collect_mlp_trace(teacher, tok_t, prompts, args.layer, args.max_length, args.device)
        s_mlp = collect_mlp_trace(student, tok_s, prompts, args.layer, args.max_length, args.device)
        mlp_rows = mlp_trace_diff_rows(args.layer, t_mlp, s_mlp)

    del teacher, student
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    focus = [r for r in per_head_diff if int(r["head"]) in focus_heads]
    report = {
        "version": VERSION,
        "mode": "head_circuit_level_program_diff",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "heads": heads,
        "focus_heads": sorted(focus_heads),
        "prompts": args.prompts,
        "max_delta": args.max_delta,
        "teacher_qk_program_run": args.teacher_qk_program_run,
        "teacher_vo_program_run": args.teacher_vo_program_run,
        "student_qk_program_run": args.student_qk_program_run,
        "student_vo_program_run": args.student_vo_program_run,
        "uses_head_circuit_targets": True,
        "uses_raw_weight_diff": False,
        "uses_raw_atom_cosine_diff_as_primary": False,
        "qk_target_formula": "M_qk_aug[h,d] = Wq_aug[h].T @ R_delta @ Wk_aug[kv(h)] / sqrt(head_dim)",
        "vo_target_formula": "C_vo_aug[h] = Wo[h] @ Wv_aug[kv(h)]",
        "closure_summary": {
            "teacher_score_rel_mean": mean([r["score_rel"] for r in per_model_closure if r["model_role"] == "teacher"]),
            "teacher_A_rel_mean": mean([r["A_rel"] for r in per_model_closure if r["model_role"] == "teacher"]),
            "teacher_Y_head_rel_mean": mean([r["Y_head_rel"] for r in per_model_closure if r["model_role"] == "teacher"]),
            "student_score_rel_mean": mean([r["score_rel"] for r in per_model_closure if r["model_role"] == "student"]),
            "student_A_rel_mean": mean([r["A_rel"] for r in per_model_closure if r["model_role"] == "student"]),
            "student_Y_head_rel_mean": mean([r["Y_head_rel"] for r in per_model_closure if r["model_role"] == "student"]),
        },
        "diff_summary": {
            "qk_program_rel_mean": mean([r["qk_program_rel_mean"] for r in per_head_diff]),
            "qk_program_cos_mean": mean([r["qk_program_cos_mean"] for r in per_head_diff]),
            "vo_program_rel_mean": mean([r["vo_program_rel"] for r in per_head_diff if r["vo_program_rel"] is not None]),
            "vo_program_cos_mean": mean([r["vo_program_cos"] for r in per_head_diff if r["vo_program_cos"] is not None]),
        },
        "top_focus_heads": focus,
        "mlp_trace_included": bool(args.include_mlp_trace),
        "mlp_status": "trace_function_summary_only_not_mlp_program_transplant" if args.include_mlp_trace else "not_requested",
        "missing_atom_count": len(set(missing_all)),
        "missing_atoms": sorted(set(missing_all))[:50],
        "status": "HEAD_CIRCUIT_PROGRAM_DIFF_RAN",
        "no_training": True,
        "prompt_independent_targets": True,
        "note": "Attention comparison is at head-circuit function level M_qk/C_vo. MLP section is activation/function trace only in this v1.",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_model_head_program_closure.jsonl", per_model_closure)
    write_jsonl(out / "per_head_circuit_diff.jsonl", per_head_diff)
    write_jsonl(out / "per_matrix_circuit_diff.jsonl", per_matrix_diff)
    write_jsonl(out / "per_delta_level0_ops.jsonl", per_delta_ops)
    if mlp_rows:
        write_jsonl(out / "mlp_swiglu_trace_diff.jsonl", mlp_rows)

    print("=== Qwen Head-Circuit Program Diff v1 ===")
    print(json.dumps({
        "layer": args.layer,
        "heads": heads,
        "focus_heads": sorted(focus_heads),
        "closure_summary": report["closure_summary"],
        "diff_summary": report["diff_summary"],
        "top_focus_heads": report["top_focus_heads"],
        "mlp_status": report["mlp_status"],
        "missing_atom_count": report["missing_atom_count"],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
