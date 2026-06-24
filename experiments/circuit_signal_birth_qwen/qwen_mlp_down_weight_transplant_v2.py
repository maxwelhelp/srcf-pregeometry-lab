#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_mlp_down_weight_transplant_v2.py

Static MLP/FFN down-column transplant for Qwen Coder -> Qwen Instruct/Base.

This is the next step after qwen_mlp_feature_transfer_scan_v1.
v1 used function-output hooks:
    feature_delta / down_write / activation
and showed that MLP, especially L16, generalizes better than attention.

v2 does the actual static weight-level patch, but at the correct MLP-program
abstraction:

    feature_r(x) = silu(w_gate_r · x) * (w_up_r · x)
    write_r      = W_down[:, r]
    MLP_r(x)     = feature_r(x) * write_r

A channel is considered a clean transplant candidate if:
    cos(w_gate_coder[r], w_gate_student[r]) >= gate_cos_min
    cos(w_up_coder[r],   w_up_student[r])   >= up_cos_min
    cos(Wdown_coder[:,r], Wdown_student[:,r]) <= down_cos_max
    code-specific activation is high enough

Then the static patch is:
    W_down_student[:, r] += alpha * (W_down_coder[:, r] - W_down_student[:, r])

This is NOT an activation hook and NOT a whole W_down copy.
It is a pseudocode-channel-filtered writer transplant.

Outputs:
    manifest.json
    per_candidate_down_transplant.jsonl
    per_prompt_down_transplant.jsonl
    selected_channels.jsonl
    mlp_channel_pseudocode.md
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
import qwen_teacher_student_attention_transfer_v1 as ts

VERSION = "qwen_mlp_down_weight_transplant_v2.0"


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


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


def datasets() -> Dict[str, List[str]]:
    return {
        "train_code": train_code_prompts(),
        "train_retain": train_retain_prompts(),
        "heldout_code": heldout_code_prompts(),
        "heldout_retain": heldout_retain_prompts(),
    }


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
def collect_mlp_z_abs_mean(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str) -> torch.Tensor:
    """Return mean abs SwiGLU hidden activation per intermediate channel."""
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp
    acc = None
    n = 0
    for text in prompts:
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

        X = box["mlp_in"]
        G = mlp.gate_proj(X)
        U = mlp.up_proj(X)
        Z = mlp_act(mlp, G) * U
        z_abs = Z[0].detach().float().abs().mean(dim=0).cpu()
        acc = z_abs if acc is None else acc + z_abs
        n += 1
    return acc / max(1, n)


def layer_weights(model, layer_idx: int) -> Dict[str, torch.Tensor]:
    mlp = model.model.layers[layer_idx].mlp
    return {
        "gate": mlp.gate_proj.weight.detach().float().cpu(),       # [I, H]
        "up": mlp.up_proj.weight.detach().float().cpu(),           # [I, H]
        "down": mlp.down_proj.weight.detach().float().cpu(),       # [H, I]
    }


def row_cos(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (A * B).sum(dim=1) / (torch.linalg.norm(A, dim=1) * torch.linalg.norm(B, dim=1)).clamp_min(eps)


def col_cos(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (A * B).sum(dim=0) / (torch.linalg.norm(A, dim=0) * torch.linalg.norm(B, dim=0)).clamp_min(eps)


def compute_channel_table(
    teacher, student, tok_teacher, tok_student, layer_idx: int, max_length: int, device: str,
    retain_ratio: float,
) -> List[Dict[str, Any]]:
    Wt = layer_weights(teacher, layer_idx)
    Ws = layer_weights(student, layer_idx)
    cos_gate = row_cos(Wt["gate"], Ws["gate"])
    cos_up = row_cos(Wt["up"], Ws["up"])
    cos_down = col_cos(Wt["down"], Ws["down"])
    delta_down_norm = torch.linalg.norm(Wt["down"] - Ws["down"], dim=0)
    teacher_down_norm = torch.linalg.norm(Wt["down"], dim=0)
    student_down_norm = torch.linalg.norm(Ws["down"], dim=0)

    t_code = collect_mlp_z_abs_mean(teacher, tok_teacher, train_code_prompts(), layer_idx, max_length, device)
    t_retain = collect_mlp_z_abs_mean(teacher, tok_teacher, train_retain_prompts(), layer_idx, max_length, device)
    s_code = collect_mlp_z_abs_mean(student, tok_student, train_code_prompts(), layer_idx, max_length, device)
    s_retain = collect_mlp_z_abs_mean(student, tok_student, train_retain_prompts(), layer_idx, max_length, device)

    # We use student activation for actual down-column transplant, because after
    # the patch the detector remains student gate/up. Teacher activation is only
    # auxiliary evidence.
    student_specific = s_code - retain_ratio * s_retain
    teacher_specific = t_code - retain_ratio * t_retain
    detector_cos = 0.5 * (cos_gate + cos_up)

    # Static weight patch score. Favors aligned detectors, different writers,
    # code-specific student activation, and nontrivial writer delta.
    score = (
        torch.relu(student_specific)
        * torch.relu(detector_cos)
        * torch.relu(1.0 - cos_down)
        * torch.log1p(delta_down_norm)
    )

    rows = []
    for r in range(int(score.numel())):
        rows.append({
            "layer": layer_idx,
            "feature": r,
            "score": float(score[r].item()),
            "cos_gate": float(cos_gate[r].item()),
            "cos_up": float(cos_up[r].item()),
            "detector_cos_mean": float(detector_cos[r].item()),
            "cos_down": float(cos_down[r].item()),
            "delta_down_norm": float(delta_down_norm[r].item()),
            "teacher_down_norm": float(teacher_down_norm[r].item()),
            "student_down_norm": float(student_down_norm[r].item()),
            "teacher_code_abs": float(t_code[r].item()),
            "teacher_retain_abs": float(t_retain[r].item()),
            "student_code_abs": float(s_code[r].item()),
            "student_retain_abs": float(s_retain[r].item()),
            "teacher_specific": float(teacher_specific[r].item()),
            "student_specific": float(student_specific[r].item()),
        })
    return rows


def select_channels(
    rows: List[Dict[str, Any]],
    gate_cos_min: float,
    up_cos_min: float,
    down_cos_max: float,
    top_k: int,
    allow_fallback: bool = True,
) -> List[Dict[str, Any]]:
    filt = [
        r for r in rows
        if r["cos_gate"] >= gate_cos_min
        and r["cos_up"] >= up_cos_min
        and r["cos_down"] <= down_cos_max
        and r["student_specific"] > 0
    ]
    filt.sort(key=lambda r: r["score"], reverse=True)
    if len(filt) >= top_k or not allow_fallback:
        return filt[:top_k]

    # Fallback is useful because cross-model same-index channels may be only
    # partially aligned. Keep the result marked by caller via selected_count.
    all_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    return all_sorted[:top_k]


@torch.no_grad()
def eval_model_against_truth(model, tokenizer, prompts: List[str], teacher_truth, student_truth, max_length: int, device: str) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows = []
    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        res = model(**enc, use_cache=False)
        prog_logits = res.logits[0].detach().float().cpu()
        teacher_logits = teacher_truth[pi]["logits_true"]
        student_logits = student_truth[pi]["logits_true"]
        before = ts.logits_metrics(teacher_logits, student_logits)
        after = ts.logits_metrics(teacher_logits, prog_logits)
        shift = ts.logits_metrics(student_logits, prog_logits)
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
        })
    sums = {
        "logits_improvement": mean(r["logits_improvement"] for r in rows),
        "KL_improvement": mean(r["KL_improvement"] for r in rows),
        "student_shift_logits_rel": mean(r["student_shift_logits_rel"] for r in rows),
        "student_shift_KL": mean(r["student_shift_KL"] for r in rows),
        "after_top1_match_rate": mean(1.0 if r["after_top1_match"] else 0.0 for r in rows),
    }
    return sums, rows


def patch_down_columns_inplace(student, teacher, layer_idx: int, selected: List[Dict[str, Any]], alpha: float) -> torch.Tensor:
    """Patch student W_down columns and return original copy for restore."""
    s_down = student.model.layers[layer_idx].mlp.down_proj.weight
    t_down = teacher.model.layers[layer_idx].mlp.down_proj.weight
    old = s_down.detach().clone()
    idx = torch.tensor([int(r["feature"]) for r in selected], device=s_down.device, dtype=torch.long)
    if idx.numel() == 0:
        return old
    with torch.no_grad():
        delta = t_down[:, idx].to(device=s_down.device, dtype=s_down.dtype) - s_down[:, idx]
        s_down[:, idx] += float(alpha) * delta
    return old


def restore_down_columns(student, layer_idx: int, old: torch.Tensor) -> None:
    s_down = student.model.layers[layer_idx].mlp.down_proj.weight
    with torch.no_grad():
        s_down.copy_(old)


def heldout_score(row: Dict[str, Any]) -> float:
    code_kl = float(row.get("heldout_code_KL_gain", 0.0))
    retain_kl = float(row.get("heldout_retain_KL_gain", 0.0))
    code_logit = float(row.get("heldout_code_logit_gain", 0.0))
    retain_logit = float(row.get("heldout_retain_logit_gain", 0.0))
    retain_damage = max(0.0, -retain_kl)
    return code_kl + 0.25 * code_logit - 2.0 * retain_damage - 0.25 * max(0.0, retain_logit)


def clean_flag(row: Dict[str, Any]) -> bool:
    return bool(
        row.get("heldout_code_KL_gain", 0.0) > 0
        and row.get("heldout_code_logit_gain", 0.0) >= 0
        and max(0.0, -row.get("heldout_retain_KL_gain", 0.0)) <= max(0.002, 0.25 * row.get("heldout_code_KL_gain", 0.0))
    )


def write_pseudocode(path: Path, report: Dict[str, Any]) -> None:
    lines = []
    lines.append("# MLP down-column transplant pseudocode\n")
    lines.append("```python")
    lines.append("# Channel r program")
    lines.append("detect_r(x) = silu(w_gate_r @ x) * (w_up_r @ x)")
    lines.append("write_r = W_down[:, r]")
    lines.append("MLP_r(x) = detect_r(x) * write_r")
    lines.append("")
    lines.append("# Static transplant for selected channels")
    lines.append("if cos_gate(r) >= gate_min and cos_up(r) >= up_min and cos_down(r) <= down_max:")
    lines.append("    W_down_student[:, r] += alpha * (W_down_coder[:, r] - W_down_student[:, r])")
    lines.append("````\n".replace("````", "```"))
    for item in report.get("top", [])[:20]:
        lines.append(f"## {item['candidate']}\n")
        lines.append(f"- heldout_code_KL: {item['heldout_code_KL_gain']:+.6f}")
        lines.append(f"- heldout_retain_KL: {item['heldout_retain_KL_gain']:+.6f}")
        lines.append(f"- heldout_code_logit: {item['heldout_code_logit_gain']:+.6f}")
        lines.append(f"- score: {item['heldout_score']:+.6f}")
        lines.append(f"- clean: {item['heldout_clean']}")
        lines.append(f"- selected_count: {item['selected_count']}")
        lines.append("\nTop channels:")
        lines.append("| r | score | cos_gate | cos_up | cos_down | student_code | student_retain | delta_down |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for ch in item.get("selected_preview", [])[:20]:
            lines.append(
                f"| {ch['feature']} | {ch['score']:.4g} | {ch['cos_gate']:.3f} | {ch['cos_up']:.3f} | "
                f"{ch['cos_down']:.3f} | {ch['student_code_abs']:.4g} | {ch['student_retain_abs']:.4g} | {ch['delta_down_norm']:.4g} |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="16")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--feature-counts", default="16,64,256")
    ap.add_argument("--alphas", default="0.01,0.025,0.05")
    ap.add_argument("--gate-cos-min", type=float, default=0.5)
    ap.add_argument("--up-cos-min", type=float, default=0.5)
    ap.add_argument("--down-cos-max", type=float, default=0.9)
    ap.add_argument("--retain-ratio", type=float, default=0.5)
    ap.add_argument("--no-fallback", action="store_true", help="Do not fallback to top score if strict cos filters select too few channels")
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/mlp_down_weight_transplant_v2")
    args = ap.parse_args()

    layers = parse_ints(args.layers)
    feature_counts = parse_ints(args.feature_counts)
    alphas = parse_floats(args.alphas)
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

    ds = datasets()
    print("collecting truth logits...")
    teacher_truth = {name: collect_logits(teacher, tok_teacher, prompts, args.max_length, args.device) for name, prompts in ds.items()}
    student_truth = {name: collect_logits(student, tok_student, prompts, args.max_length, args.device) for name, prompts in ds.items()}

    all_results: List[Dict[str, Any]] = []
    per_prompt_all: List[Dict[str, Any]] = []
    channel_rows_all: List[Dict[str, Any]] = []

    for L in layers:
        print(f"\n=== L{L} static W_down transplant ===")
        ch_rows = compute_channel_table(teacher, student, tok_teacher, tok_student, L, args.max_length, args.device, args.retain_ratio)
        ch_rows_sorted = sorted(ch_rows, key=lambda r: r["score"], reverse=True)
        for r in ch_rows_sorted[:500]:
            channel_rows_all.append(r)

        strict_count = sum(
            1 for r in ch_rows
            if r["cos_gate"] >= args.gate_cos_min
            and r["cos_up"] >= args.up_cos_min
            and r["cos_down"] <= args.down_cos_max
            and r["student_specific"] > 0
        )
        print(f"strict selected pool L{L}: {strict_count}")

        for k in feature_counts:
            selected = select_channels(
                ch_rows,
                args.gate_cos_min,
                args.up_cos_min,
                args.down_cos_max,
                int(k),
                allow_fallback=not args.no_fallback,
            )
            selected_features = [int(r["feature"]) for r in selected]
            strict_used = len(selected) <= strict_count if strict_count >= int(k) else (args.no_fallback)
            for alpha in alphas:
                old = patch_down_columns_inplace(student, teacher, L, selected, float(alpha))
                try:
                    row: Dict[str, Any] = {
                        "candidate": f"L{L}_MLP_static_down_top{k}_a{alpha:g}",
                        "layer": L,
                        "feature_count": int(k),
                        "alpha": float(alpha),
                        "gate_cos_min": args.gate_cos_min,
                        "up_cos_min": args.up_cos_min,
                        "down_cos_max": args.down_cos_max,
                        "strict_pool_count": strict_count,
                        "selected_count": len(selected),
                        "strict_filter_used": bool(strict_used),
                        "selected_features": selected_features,
                        "selected_preview": selected[:20],
                    }
                    for name, prompts in ds.items():
                        sums, rows = eval_model_against_truth(student, tok_student, prompts, teacher_truth[name], student_truth[name], args.max_length, args.device)
                        row[f"{name}_logit_gain"] = sums["logits_improvement"]
                        row[f"{name}_KL_gain"] = sums["KL_improvement"]
                        row[f"{name}_shift"] = sums["student_shift_logits_rel"]
                        row[f"{name}_shift_KL"] = sums["student_shift_KL"]
                        row[f"{name}_top1"] = sums["after_top1_match_rate"]
                        for pr in rows:
                            pr.update({"candidate": row["candidate"], "dataset": name, "layer": L, "feature_count": int(k), "alpha": float(alpha)})
                            per_prompt_all.append(pr)
                    row["heldout_score"] = heldout_score(row)
                    row["heldout_clean"] = clean_flag(row)
                    all_results.append(row)
                    print(
                        f"{row['candidate']}: "
                        f"train_code_KL={row['train_code_KL_gain']:+.5f} "
                        f"heldout_code_KL={row['heldout_code_KL_gain']:+.5f} "
                        f"heldout_retain_KL={row['heldout_retain_KL_gain']:+.5f} "
                        f"code_logit={row['heldout_code_logit_gain']:+.5f} "
                        f"score={row['heldout_score']:+.5f} clean={row['heldout_clean']}"
                    )
                finally:
                    restore_down_columns(student, L, old)

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
        "gate_cos_min": args.gate_cos_min,
        "up_cos_min": args.up_cos_min,
        "down_cos_max": args.down_cos_max,
        "retain_ratio": args.retain_ratio,
        "result_count": len(all_results),
        "heldout_clean_count": len(clean),
        "top": all_results[:100],
        "top_clean": clean[:100],
        "status": "MLP_STATIC_DOWN_WEIGHT_TRANSPLANT_SCAN_RAN",
        "patch_type": "static_weight_patch_W_down_columns_only",
    }
    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_down_transplant.jsonl", all_results)
    write_jsonl(out / "per_prompt_down_transplant.jsonl", per_prompt_all)
    write_jsonl(out / "selected_channels.jsonl", channel_rows_all)
    write_pseudocode(out / "mlp_channel_pseudocode.md", report)

    print("\n=== Qwen MLP Static Down Weight Transplant v2 ===")
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
