#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_mlp_semantic_down_subset_transplant_v3.py

Semantic subset static MLP down-column transplant for Qwen Coder -> Qwen Instruct/Base.

This is the next step after:
  - qwen_mlp_feature_transfer_scan_v1.py
  - qwen_mlp_down_weight_transplant_v2.py
  - qwen_mlp_channel_program_reader_v2_1.py

v2 tested top-k channel sets from a statistical/cosine score. It got positive
heldout KL but often slightly negative code-logit.

v3 tests SMALL SEMANTIC SUBSETS explicitly, for example:
  2785
  738
  2785,738
  2785,738,3004,3339

Patch is true static weight patch:
  W_down_student[:, r] += alpha * (W_down_coder[:, r] - W_down_student[:, r])

No training. No gradient. No activation hook. No generic whole-matrix copy.

Outputs train/heldout code/retain for each subset and alpha.
"""
from __future__ import annotations

import argparse
import json
import gc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
import qwen_teacher_student_attention_transfer_v1 as ts

VERSION = "qwen_mlp_semantic_down_subset_transplant_v3.0"


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_channel_sets(s: str) -> List[List[int]]:
    out: List[List[int]] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        chans = [int(x.strip()) for x in part.split(",") if x.strip()]
        # preserve order, remove duplicates
        seen = set(); clean = []
        for c in chans:
            if c not in seen:
                clean.append(c); seen.add(c)
        if clean:
            out.append(clean)
    return out


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
    out: Dict[int, Dict[str, torch.Tensor]] = {}
    for pi, text in enumerate(prompts):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        res = model(**enc, use_cache=False)
        out[pi] = {"logits_true": res.logits[0].detach().float().cpu()}
    return out


@torch.no_grad()
def eval_model_against_truth(model, tokenizer, prompts: List[str], teacher_truth, student_truth, max_length: int, device: str) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
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


def layer_weights(model, layer_idx: int) -> Dict[str, torch.Tensor]:
    mlp = model.model.layers[layer_idx].mlp
    return {
        "gate": mlp.gate_proj.weight.detach().float().cpu(),
        "up": mlp.up_proj.weight.detach().float().cpu(),
        "down": mlp.down_proj.weight.detach().float().cpu(),
    }


def row_cos(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (A * B).sum(dim=1) / (torch.linalg.norm(A, dim=1) * torch.linalg.norm(B, dim=1)).clamp_min(eps)


def col_cos(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (A * B).sum(dim=0) / (torch.linalg.norm(A, dim=0) * torch.linalg.norm(B, dim=0)).clamp_min(eps)


def load_semantic(path: str | None) -> Dict[int, Dict[str, Any]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    items = data.get("candidates", data) if isinstance(data, dict) else data
    out: Dict[int, Dict[str, Any]] = {}
    if isinstance(items, list):
        for r in items:
            if isinstance(r, dict) and "channel" in r:
                out[int(r["channel"])] = r
    return out


def channel_meta(teacher, student, layer_idx: int, channels: List[int], semantic_coder: Dict[int, Dict[str, Any]], semantic_student: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    Wt = layer_weights(teacher, layer_idx)
    Ws = layer_weights(student, layer_idx)
    cg = row_cos(Wt["gate"], Ws["gate"])
    cu = row_cos(Wt["up"], Ws["up"])
    cd = col_cos(Wt["down"], Ws["down"])
    dd = torch.linalg.norm(Wt["down"] - Ws["down"], dim=0)
    rows = []
    for ch in channels:
        row: Dict[str, Any] = {
            "channel": int(ch),
            "cos_gate": float(cg[ch].item()),
            "cos_up": float(cu[ch].item()),
            "detector_cos_mean": float(((cg[ch] + cu[ch]) * 0.5).item()),
            "cos_down": float(cd[ch].item()),
            "delta_down_norm": float(dd[ch].item()),
        }
        if ch in semantic_coder:
            sc = semantic_coder[ch].get("score", {})
            row["coder_semantic"] = {
                "ratio": sc.get("ratio"),
                "gap": sc.get("gap"),
                "stable_gap": sc.get("stable_gap"),
                "label": semantic_coder[ch].get("label") or semantic_coder[ch].get("semantic_label"),
            }
        if ch in semantic_student:
            ss = semantic_student[ch].get("score", {})
            row["student_semantic"] = {
                "ratio": ss.get("ratio"),
                "gap": ss.get("gap"),
                "stable_gap": ss.get("stable_gap"),
                "label": semantic_student[ch].get("label") or semantic_student[ch].get("semantic_label"),
            }
        rows.append(row)
    return rows


def patch_down_columns_inplace(student, teacher, layer_idx: int, channels: List[int], alpha: float) -> torch.Tensor:
    s_down = student.model.layers[layer_idx].mlp.down_proj.weight
    t_down = teacher.model.layers[layer_idx].mlp.down_proj.weight
    old = s_down.detach().clone()
    if not channels:
        return old
    idx = torch.tensor(channels, device=s_down.device, dtype=torch.long)
    with torch.no_grad():
        delta = t_down[:, idx].to(device=s_down.device, dtype=s_down.dtype) - s_down[:, idx]
        s_down[:, idx] += float(alpha) * delta
    return old


def restore_down(student, layer_idx: int, old: torch.Tensor) -> None:
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


def is_clean(row: Dict[str, Any]) -> bool:
    return bool(
        row.get("heldout_code_KL_gain", 0.0) > 0
        and row.get("heldout_code_logit_gain", 0.0) >= 0
        and max(0.0, -row.get("heldout_retain_KL_gain", 0.0)) <= max(0.002, 0.25 * row.get("heldout_code_KL_gain", 0.0))
    )


def subset_name(chs: List[int]) -> str:
    if len(chs) == 1:
        return f"ch{chs[0]}"
    if len(chs) <= 6:
        return "ch" + "_".join(map(str, chs))
    return "ch" + "_".join(map(str, chs[:6])) + f"_plus{len(chs)-6}"


def write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Semantic MLP down-column subset transplant v3\n")
    lines.append("```python")
    lines.append("# MLP channel program")
    lines.append("detect_r(x) = silu(w_gate_r @ x) * (w_up_r @ x)")
    lines.append("write_r = W_down[:, r]")
    lines.append("MLP_r(x) = detect_r(x) * write_r")
    lines.append("")
    lines.append("# Static semantic subset patch")
    lines.append("for r in semantic_subset:")
    lines.append("    W_down_student[:, r] += alpha * (W_down_coder[:, r] - W_down_student[:, r])")
    lines.append("```\n")
    lines.append(f"result_count: {report['result_count']}  ")
    lines.append(f"heldout_clean_count: {report['heldout_clean_count']}\n")
    lines.append("## Top candidates\n")
    for r in report.get("top", [])[:30]:
        lines.append(f"### {r['candidate']}\n")
        lines.append(f"- heldout_code_KL: {r['heldout_code_KL_gain']:+.6f}")
        lines.append(f"- heldout_retain_KL: {r['heldout_retain_KL_gain']:+.6f}")
        lines.append(f"- heldout_code_logit: {r['heldout_code_logit_gain']:+.6f}")
        lines.append(f"- score: {r['heldout_score']:+.6f}")
        lines.append(f"- clean: {r['heldout_clean']}")
        lines.append(f"- channels: {r['channels']}\n")
        lines.append("| ch | cos_gate | cos_up | cos_down | coder_ratio | student_ratio |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for cm in r.get("channel_meta", []):
            cr = cm.get("coder_semantic", {}).get("ratio")
            sr = cm.get("student_semantic", {}).get("ratio")
            lines.append(f"| {cm['channel']} | {cm['cos_gate']:.3f} | {cm['cos_up']:.3f} | {cm['cos_down']:.3f} | {cr} | {sr} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--channel-sets", default="2785;738;2785,738;2785,738,3004,3339;2785,738,3202,2680,2560,1969;326,1800,4600,1020,3339,2424,3004,1275,3452,738,3905,4852,4729,2785,4113,881")
    ap.add_argument("--alphas", default="0.01,0.025,0.05,0.075,0.1")
    ap.add_argument("--semantic-coder", default="")
    ap.add_argument("--semantic-student", default="")
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/mlp_semantic_down_subset_transplant_v3")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ch_sets = parse_channel_sets(args.channel_sets)
    alphas = parse_floats(args.alphas)
    ds = datasets()

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

    print("collecting truth logits...")
    teacher_truth = {name: collect_logits(teacher, tok_teacher, prompts, args.max_length, args.device) for name, prompts in ds.items()}
    student_truth = {name: collect_logits(student, tok_student, prompts, args.max_length, args.device) for name, prompts in ds.items()}

    semantic_coder = load_semantic(args.semantic_coder)
    semantic_student = load_semantic(args.semantic_student)

    all_results: List[Dict[str, Any]] = []
    per_prompt: List[Dict[str, Any]] = []

    for chs in ch_sets:
        cm = channel_meta(teacher, student, args.layer, chs, semantic_coder, semantic_student)
        for alpha in alphas:
            cand = f"L{args.layer}_MLP_semantic_down_{subset_name(chs)}_a{alpha:g}"
            old = patch_down_columns_inplace(student, teacher, args.layer, chs, float(alpha))
            try:
                row: Dict[str, Any] = {
                    "candidate": cand,
                    "layer": args.layer,
                    "channels": chs,
                    "channel_count": len(chs),
                    "alpha": float(alpha),
                    "channel_meta": cm,
                }
                for name, prompts in ds.items():
                    sums, rows = eval_model_against_truth(student, tok_student, prompts, teacher_truth[name], student_truth[name], args.max_length, args.device)
                    row[f"{name}_logit_gain"] = sums["logits_improvement"]
                    row[f"{name}_KL_gain"] = sums["KL_improvement"]
                    row[f"{name}_shift"] = sums["student_shift_logits_rel"]
                    row[f"{name}_shift_KL"] = sums["student_shift_KL"]
                    row[f"{name}_top1"] = sums["after_top1_match_rate"]
                    for pr in rows:
                        pr.update({"candidate": cand, "dataset": name, "layer": args.layer, "channels": chs, "alpha": float(alpha)})
                        per_prompt.append(pr)
                row["heldout_score"] = heldout_score(row)
                row["heldout_clean"] = is_clean(row)
                all_results.append(row)
                print(
                    f"{cand}: "
                    f"heldout_code_KL={row['heldout_code_KL_gain']:+.6f} "
                    f"heldout_retain_KL={row['heldout_retain_KL_gain']:+.6f} "
                    f"heldout_code_logit={row['heldout_code_logit_gain']:+.6f} "
                    f"score={row['heldout_score']:+.6f} clean={row['heldout_clean']}"
                )
            finally:
                restore_down(student, args.layer, old)

    all_results.sort(key=lambda r: r["heldout_score"], reverse=True)
    clean = [r for r in all_results if r["heldout_clean"]]
    report = {
        "version": VERSION,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layer": args.layer,
        "channel_sets": ch_sets,
        "alphas": alphas,
        "semantic_coder": args.semantic_coder,
        "semantic_student": args.semantic_student,
        "result_count": len(all_results),
        "heldout_clean_count": len(clean),
        "top": all_results[:100],
        "top_clean": clean[:100],
        "status": "MLP_SEMANTIC_DOWN_SUBSET_TRANSPLANT_RAN",
        "patch_type": "static_W_down_columns_manual_semantic_subset",
    }
    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_semantic_subset_transplant.jsonl", all_results)
    write_jsonl(out / "per_prompt_semantic_subset_transplant.jsonl", per_prompt)
    write_markdown(out / "semantic_subset_report.md", report)

    print("\n=== Qwen MLP Semantic Down Subset Transplant v3 ===")
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
