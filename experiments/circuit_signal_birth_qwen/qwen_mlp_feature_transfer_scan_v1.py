#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_mlp_feature_transfer_scan_v1.py

MLP/FFN feature-level transfer scan for Qwen Coder -> Qwen Instruct/Base.

This is the MLP analogue of the attention term-level pseudocode scan.
It does NOT decode W_gate/W_up/W_down separately and does NOT replace whole MLP.
It treats the Qwen SwiGLU MLP as a function/program:

    gate = W_gate x
    up   = W_up x
    z    = silu(gate) * up
    y    = W_down z

Feature r:
    feature_r(x) = silu(w_gate_r · x) * (w_up_r · x)
    write_r      = W_down[:, r]

Selection:
    choose teacher features with high code activation, low retain activation,
    and meaningful teacher/student write or contribution difference.

Intervention:
    hook layer.mlp output in the STUDENT model and add:

    mode=feature_delta:
        ΔY = Σ_r [ z_teacher_r * Wdown_teacher[:,r]
                 - z_student_r * Wdown_student[:,r] ]

    mode=down_write:
        ΔY = Σ_r z_student_r * (Wdown_teacher[:,r] - Wdown_student[:,r])

    mode=activation:
        ΔY = Σ_r (z_teacher_r - z_student_r) * Wdown_student[:,r]

Then the full student model continues through later layers to lm_head.

Outputs train_code/train_retain/heldout_code/heldout_retain metrics.
No training. No gradient. No LoRA. Pure function-level MLP patch scan.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
import qwen_teacher_student_attention_transfer_v1 as ts

VERSION = "qwen_mlp_feature_transfer_scan_v1.0"


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def safe_json(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (list, tuple)):
        return [safe_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): safe_json(x) for k, x in v.items()}
    return v


def train_code_prompts() -> List[str]:
    return [
        "Write a Python function `is_prime(n)` and explain the edge cases.",
        "Complete this Python code:\n\ndef fibonacci(n):\n    # return the nth Fibonacci number\n",
        "Fix the bug in this code:\n\nfor i in range(len(items)):\n    print(items[i+1])\n",
        "Implement binary search in Python for a sorted list of integers.",
        "Given a list of strings, write Python code to group them by their first letter.",
        "Explain what this JavaScript code does:\n\nconst xs = arr.map(x => x * 2).filter(x => x > 10);",
        "Write a SQL query to find users who made more than 3 orders last month.",
        "Convert this recursive factorial function to an iterative version in Python.",
    ]


def train_retain_prompts() -> List[str]:
    return [
        "Explain why the sky looks blue in simple terms.",
        "Write a short summary of the causes of the French Revolution.",
        "What are three healthy habits for better sleep?",
        "Translate this sentence into Ukrainian: The weather is warm today.",
        "Give a simple explanation of photosynthesis.",
        "What is the difference between a planet and a star?",
        "Summarize the plot of a classic adventure story in one paragraph.",
        "Explain compound interest with a small numerical example.",
    ]


def heldout_code_prompts() -> List[str]:
    return [
        "Write a Python function that checks whether a string has balanced parentheses.",
        "Complete this code:\n\ndef merge_sorted(a, b):\n    # return one sorted list\n",
        "Find the bug:\n\nwhile i < len(nums):\n    total += nums[i]\n",
        "Explain what this Python list comprehension does: [x*x for x in xs if x % 2 == 0]",
        "Write a JavaScript function that removes duplicate values from an array.",
        "Create a SQL query that returns the top 5 products by total sales.",
        "Refactor this code into a helper function:\n\nprint(user.name)\nprint(user.email)\n",
        "Given a dictionary of counts, write Python code to return the key with the largest value.",
    ]


def heldout_retain_prompts() -> List[str]:
    return [
        "Explain how rainbows form in simple words.",
        "Give a short overview of why exercise is good for health.",
        "Describe the water cycle for a middle school student.",
        "Translate this sentence into Ukrainian: I will visit my friend tomorrow.",
        "What are the main differences between reptiles and mammals?",
        "Summarize why people use maps in geography.",
        "Explain what a budget is with a simple example.",
        "Write a short paragraph about why libraries are useful.",
    ]


def get_dataset(name: str) -> List[str]:
    if name == "train_code":
        return train_code_prompts()
    if name == "train_retain":
        return train_retain_prompts()
    if name == "heldout_code":
        return heldout_code_prompts()
    if name == "heldout_retain":
        return heldout_retain_prompts()
    raise KeyError(name)


@torch.no_grad()
def collect_logits(model, tokenizer, prompts: List[str], max_length: int, device: str) -> Dict[int, Dict[str, torch.Tensor]]:
    out = {}
    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        res = model(**enc, use_cache=False)
        out[pi] = {"logits_true": res.logits[0].detach().float().cpu()}
    return out


def mlp_act(mlp, gate: torch.Tensor) -> torch.Tensor:
    if hasattr(mlp, "act_fn"):
        return mlp.act_fn(gate)
    return F.silu(gate)


@torch.no_grad()
def collect_mlp_traces(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str, keep_z: bool = True) -> Dict[int, Dict[str, torch.Tensor]]:
    """Collect MLP input and SwiGLU hidden feature activations z for one layer."""
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    traces: Dict[int, Dict[str, torch.Tensor]] = {}

    for pi, text in enumerate(prompts):
        box: Dict[str, torch.Tensor] = {}

        def pre_hook(_module, inputs):
            box["mlp_in"] = inputs[0].detach()
            return None

        handle = mlp.register_forward_pre_hook(pre_hook)
        try:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            _ = model(**enc, use_cache=False)
        finally:
            handle.remove()

        X = box["mlp_in"]  # [1, T, H]
        G = mlp.gate_proj(X)
        U = mlp.up_proj(X)
        Z = mlp_act(mlp, G) * U
        row = {
            "mlp_in": X[0].detach().float().cpu(),
            "seq_len": torch.tensor([X.shape[1]], dtype=torch.long),
        }
        if keep_z:
            row["z"] = Z[0].detach().float().cpu()
        else:
            row["z_abs_mean"] = Z[0].detach().float().abs().mean(dim=0).cpu()
        traces[pi] = row
    return traces


def concat_z(traces: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
    return torch.cat([traces[k]["z"].float() for k in sorted(traces)], dim=0)


def down_weight(model, layer_idx: int) -> torch.Tensor:
    # Qwen down_proj weight shape: [hidden_size, intermediate_size]
    return model.model.layers[layer_idx].mlp.down_proj.weight.detach().float().cpu()


def compute_feature_stats(teacher, student, t_code, t_retain, s_code, s_retain, layer_idx: int, retain_ratio: float, eps: float = 1e-8) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    Ztc = concat_z(t_code).abs()
    Ztr = concat_z(t_retain).abs()
    Zsc = concat_z(s_code).abs()
    Zsr = concat_z(s_retain).abs()

    tc = Ztc.mean(dim=0)
    tr = Ztr.mean(dim=0)
    sc = Zsc.mean(dim=0)
    sr = Zsr.mean(dim=0)

    Wt = down_weight(teacher, layer_idx)
    Ws = down_weight(student, layer_idx)
    wt_norm = torch.linalg.norm(Wt, dim=0)
    ws_norm = torch.linalg.norm(Ws, dim=0)
    dw_norm = torch.linalg.norm(Wt - Ws, dim=0)

    specificity = tc / (tr + eps)
    code_minus_retain = tc - retain_ratio * tr
    student_code_minus_retain = sc - retain_ratio * sr
    # Score favors Coder code-specific activation and a real write difference.
    score = torch.relu(code_minus_retain) * (dw_norm + 0.1 * wt_norm) * torch.log1p(specificity)

    rows = []
    for i in range(int(score.numel())):
        rows.append({
            "feature": i,
            "score": float(score[i].item()),
            "teacher_code_abs": float(tc[i].item()),
            "teacher_retain_abs": float(tr[i].item()),
            "student_code_abs": float(sc[i].item()),
            "student_retain_abs": float(sr[i].item()),
            "teacher_specificity": float(specificity[i].item()),
            "teacher_code_minus_retain": float(code_minus_retain[i].item()),
            "student_code_minus_retain": float(student_code_minus_retain[i].item()),
            "teacher_write_norm": float(wt_norm[i].item()),
            "student_write_norm": float(ws_norm[i].item()),
            "delta_write_norm": float(dw_norm[i].item()),
        })
    return score, rows


def select_feature_sets(score: torch.Tensor, feature_counts: List[int]) -> Dict[int, torch.Tensor]:
    order = torch.argsort(score, descending=True)
    out = {}
    for k in feature_counts:
        kk = min(int(k), int(order.numel()))
        out[int(k)] = order[:kk].cpu()
    return out


def make_delta_y_for_prompt(t_trace: Dict[str, torch.Tensor], s_trace: Dict[str, torch.Tensor], Wt: torch.Tensor, Ws: torch.Tensor, features: torch.Tensor, mode: str) -> torch.Tensor:
    idx = features.long()
    Zt = t_trace["z"].float()[:, idx]
    Zs = s_trace["z"].float()[:, idx]
    Wt_sel = Wt[:, idx].float()  # [H, K]
    Ws_sel = Ws[:, idx].float()
    if mode == "feature_delta":
        return Zt @ Wt_sel.T - Zs @ Ws_sel.T
    if mode == "down_write":
        return Zs @ (Wt_sel - Ws_sel).T
    if mode == "activation":
        return (Zt - Zs) @ Ws_sel.T
    raise ValueError(f"unknown patch mode {mode}")


@torch.no_grad()
def run_student_with_mlp_delta_patch(student, tokenizer, prompts: List[str], delta_by_prompt: Dict[int, torch.Tensor], layer_idx: int, max_length: int, device: str, alpha: float) -> Dict[int, Dict[str, torch.Tensor]]:
    layer = student.model.layers[layer_idx]
    mlp = layer.mlp
    out: Dict[int, Dict[str, torch.Tensor]] = {}

    for pi, text in enumerate(prompts):
        delta_cpu = delta_by_prompt[pi].float().cpu()

        def hook(_module, _inputs, output):
            D = delta_cpu.to(device=output.device, dtype=output.dtype).unsqueeze(0)
            if tuple(D.shape) != tuple(output.shape):
                raise RuntimeError(f"delta shape {tuple(D.shape)} != mlp output shape {tuple(output.shape)}")
            return output + float(alpha) * D

        handle = mlp.register_forward_hook(hook)
        try:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            res = student(**enc, use_cache=False)
        finally:
            handle.remove()

        out[pi] = {
            "logits_prog": res.logits[0].detach().float().cpu(),
            "delta_y": delta_cpu,
        }
    return out


@torch.no_grad()
def evaluate_mlp_patch(student, tokenizer, prompts: List[str], teacher_truth, student_truth, delta_by_prompt: Dict[int, torch.Tensor], layer_idx: int, max_length: int, device: str, alpha: float) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    patched = run_student_with_mlp_delta_patch(student, tokenizer, prompts, delta_by_prompt, layer_idx, max_length, device, alpha)
    rows = []
    for pi in sorted(student_truth.keys()):
        teacher_logits = teacher_truth[pi]["logits_true"]
        student_logits = student_truth[pi]["logits_true"]
        prog_logits = patched[pi]["logits_prog"]
        before = ts.logits_metrics(teacher_logits, student_logits)
        after = ts.logits_metrics(teacher_logits, prog_logits)
        shift = ts.logits_metrics(student_logits, prog_logits)
        dy = patched[pi]["delta_y"]
        rows.append({
            "prompt_id": pi,
            "before_logits_rel": before["logits_rel"],
            "after_logits_rel": after["logits_rel"],
            "logits_improvement": before["logits_rel"] - after["logits_rel"],
            "before_KL": before["last_token_KL_a_to_b"],
            "after_KL": after["last_token_KL_a_to_b"],
            "KL_improvement": before["last_token_KL_a_to_b"] - after["last_token_KL_a_to_b"],
            "before_top1_match": before["last_top1_match"],
            "after_top1_match": after["last_top1_match"],
            "student_shift_logits_rel": shift["logits_rel"],
            "student_shift_KL": shift["last_token_KL_a_to_b"],
            "delta_y_norm": float(torch.linalg.norm(dy.float()).item()),
        })
    sums = {
        "logits_improvement": mean(r["logits_improvement"] for r in rows),
        "KL_improvement": mean(r["KL_improvement"] for r in rows),
        "student_shift_logits_rel": mean(r["student_shift_logits_rel"] for r in rows),
        "student_shift_KL": mean(r["student_shift_KL"] for r in rows),
        "after_top1_match_rate": mean(1.0 if r["after_top1_match"] else 0.0 for r in rows),
        "delta_y_norm_mean": mean(r["delta_y_norm"] for r in rows),
    }
    return sums, rows


def final_score(row: Dict[str, Any]) -> float:
    code_kl = float(row.get("heldout_code_KL_gain", 0.0))
    retain_kl = float(row.get("heldout_retain_KL_gain", 0.0))
    code_logit = float(row.get("heldout_code_logit_gain", 0.0))
    retain_logit = float(row.get("heldout_retain_logit_gain", 0.0))
    retain_damage = max(0.0, -retain_kl)
    return code_kl + 0.25 * code_logit - 2.0 * retain_damage - 0.25 * max(0.0, retain_logit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="12,14,16,18")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--feature-counts", default="16,64,256")
    ap.add_argument("--alphas", default="0.01,0.025,0.05")
    ap.add_argument("--patch-modes", default="feature_delta,down_write,activation")
    ap.add_argument("--retain-ratio", type=float, default=0.5)
    ap.add_argument("--top-feature-report", type=int, default=50)
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/mlp_feature_transfer_scan_v1")
    args = ap.parse_args()

    layers = parse_ints(args.layers)
    feature_counts = parse_ints(args.feature_counts)
    alphas = parse_floats(args.alphas)
    patch_modes = [x.strip() for x in args.patch_modes.split(",") if x.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print("loading teacher...")
    tok_teacher = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=get_dtype(args.dtype), device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True,
    ).to(args.device).eval()

    print("loading student...")
    tok_student = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model, torch_dtype=get_dtype(args.dtype), device_map=None,
        attn_implementation=args.attn_implementation, trust_remote_code=True,
    ).to(args.device).eval()

    datasets = {
        "train_code": train_code_prompts(),
        "train_retain": train_retain_prompts(),
        "heldout_code": heldout_code_prompts(),
        "heldout_retain": heldout_retain_prompts(),
    }

    # Truth logits do not depend on layer, so compute once.
    print("collecting truth logits...")
    teacher_truth = {name: collect_logits(teacher, tok_teacher, prompts, args.max_length, args.device) for name, prompts in datasets.items()}
    student_truth = {name: collect_logits(student, tok_student, prompts, args.max_length, args.device) for name, prompts in datasets.items()}

    all_results: List[Dict[str, Any]] = []
    per_prompt_rows: List[Dict[str, Any]] = []
    feature_report_rows: List[Dict[str, Any]] = []

    for L in layers:
        print(f"\n=== L{L} MLP feature scan ===")
        # Collect traces for selection.
        print("collecting train traces...")
        t_train_code = collect_mlp_traces(teacher, tok_teacher, datasets["train_code"], L, args.max_length, args.device, keep_z=True)
        t_train_retain = collect_mlp_traces(teacher, tok_teacher, datasets["train_retain"], L, args.max_length, args.device, keep_z=True)
        s_train_code = collect_mlp_traces(student, tok_student, datasets["train_code"], L, args.max_length, args.device, keep_z=True)
        s_train_retain = collect_mlp_traces(student, tok_student, datasets["train_retain"], L, args.max_length, args.device, keep_z=True)

        score, feat_rows = compute_feature_stats(teacher, student, t_train_code, t_train_retain, s_train_code, s_train_retain, L, args.retain_ratio)
        feat_rows.sort(key=lambda r: r["score"], reverse=True)
        for r in feat_rows[: args.top_feature_report]:
            rr = dict(r); rr["layer"] = L
            feature_report_rows.append(rr)
        selected_sets = select_feature_sets(score, feature_counts)

        # Collect heldout traces too, once per layer.
        print("collecting heldout traces...")
        traces = {
            "train_code": (t_train_code, s_train_code),
            "train_retain": (t_train_retain, s_train_retain),
            "heldout_code": (
                collect_mlp_traces(teacher, tok_teacher, datasets["heldout_code"], L, args.max_length, args.device, keep_z=True),
                collect_mlp_traces(student, tok_student, datasets["heldout_code"], L, args.max_length, args.device, keep_z=True),
            ),
            "heldout_retain": (
                collect_mlp_traces(teacher, tok_teacher, datasets["heldout_retain"], L, args.max_length, args.device, keep_z=True),
                collect_mlp_traces(student, tok_student, datasets["heldout_retain"], L, args.max_length, args.device, keep_z=True),
            ),
        }

        Wt = down_weight(teacher, L)
        Ws = down_weight(student, L)

        # Precompute delta_y for every dataset/mode/count.
        delta_cache: Dict[Tuple[str, str, int], Dict[int, torch.Tensor]] = {}
        for ds_name, (t_tr, s_tr) in traces.items():
            for count, feats in selected_sets.items():
                for mode in patch_modes:
                    dmap = {}
                    for pi in sorted(t_tr):
                        dmap[pi] = make_delta_y_for_prompt(t_tr[pi], s_tr[pi], Wt, Ws, feats, mode)
                    delta_cache[(ds_name, mode, int(count))] = dmap

        for mode in patch_modes:
            for count, feats in selected_sets.items():
                for alpha in alphas:
                    row: Dict[str, Any] = {
                        "candidate": f"L{L}_MLP_{mode}_top{count}_a{alpha:g}",
                        "layer": L,
                        "patch_mode": mode,
                        "feature_count": int(count),
                        "alpha": float(alpha),
                        "top_features": [int(x) for x in feats[: min(20, int(feats.numel()))].tolist()],
                    }
                    for ds_name, prompts in datasets.items():
                        sums, rows = evaluate_mlp_patch(
                            student, tok_student, prompts,
                            teacher_truth[ds_name], student_truth[ds_name],
                            delta_cache[(ds_name, mode, int(count))],
                            L, args.max_length, args.device, float(alpha),
                        )
                        prefix = ds_name
                        row[f"{prefix}_logit_gain"] = sums["logits_improvement"]
                        row[f"{prefix}_KL_gain"] = sums["KL_improvement"]
                        row[f"{prefix}_shift"] = sums["student_shift_logits_rel"]
                        row[f"{prefix}_shift_KL"] = sums["student_shift_KL"]
                        row[f"{prefix}_top1"] = sums["after_top1_match_rate"]
                        row[f"{prefix}_delta_y_norm"] = sums["delta_y_norm_mean"]
                        for pr in rows:
                            pr.update({"candidate": row["candidate"], "dataset": ds_name, "layer": L, "patch_mode": mode, "feature_count": int(count), "alpha": float(alpha)})
                            per_prompt_rows.append(pr)
                    row["heldout_score"] = final_score(row)
                    row["heldout_clean"] = bool(
                        row["heldout_code_KL_gain"] > 0
                        and row["heldout_code_logit_gain"] >= 0
                        and max(0.0, -row["heldout_retain_KL_gain"]) <= max(0.002, 0.25 * row["heldout_code_KL_gain"])
                    )
                    all_results.append(row)
                    print(
                        f"{row['candidate']}: "
                        f"train_code_KL={row['train_code_KL_gain']:+.5f} "
                        f"heldout_code_KL={row['heldout_code_KL_gain']:+.5f} "
                        f"heldout_retain_KL={row['heldout_retain_KL_gain']:+.5f} "
                        f"score={row['heldout_score']:+.5f} clean={row['heldout_clean']}"
                    )

        # Free per-layer traces.
        del t_train_code, t_train_retain, s_train_code, s_train_retain, traces, delta_cache
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    all_results.sort(key=lambda r: r["heldout_score"], reverse=True)
    clean = [r for r in all_results if r["heldout_clean"]]
    report = {
        "version": VERSION,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layers": layers,
        "feature_counts": feature_counts,
        "alphas": alphas,
        "patch_modes": patch_modes,
        "retain_ratio": args.retain_ratio,
        "result_count": len(all_results),
        "heldout_clean_count": len(clean),
        "top": all_results[:100],
        "top_clean": clean[:100],
        "status": "MLP_FEATURE_TRANSFER_SCAN_RAN",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_mlp_feature_scan.jsonl", all_results)
    write_jsonl(out / "per_prompt_mlp_feature_scan.jsonl", per_prompt_rows)
    write_jsonl(out / "top_feature_report.jsonl", feature_report_rows)

    print("\n=== Qwen MLP Feature Transfer Scan v1 ===")
    print(json.dumps({
        "result_count": len(all_results),
        "heldout_clean_count": len(clean),
        "top10": all_results[:10],
        "top_clean10": clean[:10],
        "out": str(out),
    }, indent=2))

    del teacher, student
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
