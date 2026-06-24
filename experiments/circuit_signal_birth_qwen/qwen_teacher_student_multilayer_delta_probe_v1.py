#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_teacher_student_multilayer_delta_probe_v1.py

True multi-layer no-training Coder -> Instruct delta probe.

Why this exists:
  Earlier L23 scans were valid because L23 is the final decoder layer. For L6/L12/L18,
  we must patch a hidden/component output and then let the student model continue
  through layers L+1..end. This script does that with forward hooks.

It probes output-level deltas, not final weight/program rewrite yet:
  attn:  layer_output_student + alpha * (attn_out_teacher - attn_out_student)
  mlp:   layer_output_student + alpha * (mlp_out_teacher  - mlp_out_student)
  layer: layer_output_student + alpha * (layer_out_teacher - layer_out_student)

Then the patched hidden is passed through the remaining student layers normally.

Goal:
  Find layers/components where code prompts improve while retain prompts are minimally damaged.
  This separates late style/write shifts from earlier code-structure knowledge.

No training, no distillation, no LoRA.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
from qwen_teacher_student_code_transfer_v2 import select_prompts

VERSION = "qwen_teacher_student_multilayer_delta_probe_v1.0"


def mean(xs):
    return sum(xs) / max(1, len(xs))


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def crop_logits(a: torch.Tensor, b: torch.Tensor):
    T = min(a.shape[0], b.shape[0])
    V = min(a.shape[-1], b.shape[-1])
    return a[-T:, :V].float(), b[-T:, :V].float()


def logits_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    aa, bb = crop_logits(a, b)
    lt_a = aa[-1]
    lt_b = bb[-1]
    logp_a = F.log_softmax(lt_a, dim=-1)
    logp_b = F.log_softmax(lt_b, dim=-1)
    p_a = torch.exp(logp_a)
    kl = float(torch.sum(p_a * (logp_a - logp_b)))
    top1_a = int(torch.argmax(lt_a).item())
    top1_b = int(torch.argmax(lt_b).item())
    return {
        "logits_rel": rel_err(aa, bb),
        "last_token_KL_a_to_b": kl,
        "last_top1_match": bool(top1_a == top1_b),
        "top1_a_id": top1_a,
        "top1_b_id": top1_b,
    }


@torch.no_grad()
def collect_layer_traces(model, tokenizer, prompts: List[str], layer_idx: int, max_length: int, device: str):
    layer = model.model.layers[layer_idx]
    traces: Dict[int, Dict[str, Any]] = {}

    for pi, text in enumerate(prompts):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        captured: Dict[str, torch.Tensor] = {}

        def attn_hook(_mod, _inp, out):
            y = out[0] if isinstance(out, (tuple, list)) else out
            captured["attn_out"] = y.detach()[0].float().cpu()

        def mlp_hook(_mod, _inp, out):
            captured["mlp_out"] = out.detach()[0].float().cpu()

        def layer_hook(_mod, _inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            captured["layer_out"] = hs.detach()[0].float().cpu()

        h_attn = layer.self_attn.register_forward_hook(attn_hook)
        h_mlp = layer.mlp.register_forward_hook(mlp_hook)
        h_layer = layer.register_forward_hook(layer_hook)
        try:
            outputs = model(**encoded, use_cache=False, output_hidden_states=False)
        finally:
            h_attn.remove()
            h_mlp.remove()
            h_layer.remove()

        for k in ("attn_out", "mlp_out", "layer_out"):
            if k not in captured:
                raise RuntimeError(f"failed to capture {k} at layer {layer_idx}")

        traces[pi] = {
            "text": text,
            "input_len": int(encoded["input_ids"].shape[1]),
            "logits": outputs.logits.detach()[0].float().cpu(),
            "attn_out": captured["attn_out"],
            "mlp_out": captured["mlp_out"],
            "layer_out": captured["layer_out"],
        }
    return traces


def component_delta(student_trace: Dict[str, Any], teacher_trace: Dict[str, Any], component: str):
    if component == "attn":
        key = "attn_out"
    elif component == "mlp":
        key = "mlp_out"
    elif component == "layer":
        key = "layer_out"
    else:
        raise ValueError(component)
    S = student_trace[key].float()
    T = teacher_trace[key].float()
    n = min(S.shape[0], T.shape[0])
    return (T[-n:] - S[-n:]).float(), key, n


@torch.no_grad()
def run_student_with_layer_delta(model, tokenizer, text: str, layer_idx: int, delta_cpu: torch.Tensor, alpha: float, max_length: int, device: str):
    layer = model.model.layers[layer_idx]
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    encoded = {k: v.to(device) for k, v in encoded.items()}

    def patch_layer_hook(_mod, _inp, out):
        if isinstance(out, (tuple, list)):
            hs = out[0]
            rest = tuple(out[1:])
        else:
            hs = out
            rest = None
        patched = hs.clone()
        d = delta_cpu.to(device=patched.device, dtype=patched.dtype)
        n = min(patched.shape[1], d.shape[0])
        patched[:, -n:, :] = patched[:, -n:, :] + float(alpha) * d[-n:].unsqueeze(0)
        if rest is None:
            return patched
        return (patched,) + rest

    handle = layer.register_forward_hook(patch_layer_hook)
    try:
        outputs = model(**encoded, use_cache=False, output_hidden_states=False)
    finally:
        handle.remove()
    return outputs.logits.detach()[0].float().cpu()


def summarize(rows: List[Dict[str, Any]]):
    before_rel = mean([r["teacher_student_before_logits_rel"] for r in rows])
    after_rel = mean([r["teacher_student_after_logits_rel"] for r in rows])
    before_kl = mean([r["teacher_student_before_KL"] for r in rows])
    after_kl = mean([r["teacher_student_after_KL"] for r in rows])
    return {
        "before_logits_rel": before_rel,
        "after_logits_rel": after_rel,
        "logits_improvement": before_rel - after_rel,
        "logits_improvement_pct_of_before": (before_rel - after_rel) / max(1e-12, before_rel),
        "before_KL": before_kl,
        "after_KL": after_kl,
        "KL_improvement": before_kl - after_kl,
        "KL_improvement_pct_of_before": (before_kl - after_kl) / max(1e-12, before_kl),
        "student_shift_logits_rel": mean([r["student_before_after_logits_rel"] for r in rows]),
        "student_shift_KL": mean([r["student_before_after_KL"] for r in rows]),
        "delta_rel_to_component_mean": mean([r["delta_rel_to_component"] for r in rows]),
        "applied_delta_rel_to_layer_out_mean": mean([r["applied_delta_rel_to_layer_out"] for r in rows]),
        "after_top1_match_rate": mean([1.0 if r["teacher_student_after_top1_match"] else 0.0 for r in rows]),
    }


def score_candidate(sums: Dict[str, float], kl_weight: float, shift_weight: float, delta_weight: float):
    kl_damage = max(0.0, -float(sums["KL_improvement"]))
    return (
        float(sums["logits_improvement"])
        - kl_weight * kl_damage
        - shift_weight * float(sums["student_shift_logits_rel"])
        - delta_weight * float(sums["applied_delta_rel_to_layer_out_mean"])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="6,12,18,23")
    ap.add_argument("--components", default="attn,mlp,layer")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--prompt-mode", default="code", choices=["code", "retain", "mixed"])
    ap.add_argument("--alphas", default="0.125,0.25,0.5,0.75,1.0")
    ap.add_argument("--kl-weight", type=float, default=0.25)
    ap.add_argument("--shift-weight", type=float, default=0.05)
    ap.add_argument("--delta-weight", type=float, default=0.02)
    ap.add_argument("--out", default="runs/qwen_teacher_student_multilayer_delta_probe_v1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = parse_ints(args.layers)
    components = parse_strs(args.components)
    alphas = parse_floats(args.alphas)
    prompts = select_prompts(args.prompt_mode, args.prompts)

    tok_teacher = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    tok_student = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)

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

    summary_all = []
    per_prompt_all = []

    for li in layers:
        print(f"\n--- layer L{li} traces ---")
        teacher_trace = collect_layer_traces(teacher, tok_teacher, prompts, li, args.max_length, args.device)
        student_trace = collect_layer_traces(student, tok_student, prompts, li, args.max_length, args.device)

        for comp in components:
            for alpha in alphas:
                rows = []
                for pi in sorted(student_trace.keys()):
                    delta, key, n = component_delta(student_trace[pi], teacher_trace[pi], comp)
                    logits_patch = run_student_with_layer_delta(
                        student,
                        tok_student,
                        student_trace[pi]["text"],
                        li,
                        delta,
                        alpha,
                        args.max_length,
                        args.device,
                    )
                    before = logits_metrics(teacher_trace[pi]["logits"], student_trace[pi]["logits"])
                    after = logits_metrics(teacher_trace[pi]["logits"], logits_patch)
                    shift = logits_metrics(student_trace[pi]["logits"], logits_patch)
                    layer_out = student_trace[pi]["layer_out"][-n:].float()
                    row = {
                        "prompt_id": pi,
                        "prompt_mode": args.prompt_mode,
                        "layer": li,
                        "component": comp,
                        "alpha": float(alpha),
                        "text": student_trace[pi]["text"][:180],
                        "teacher_student_before_logits_rel": before["logits_rel"],
                        "teacher_student_after_logits_rel": after["logits_rel"],
                        "teacher_student_logits_rel_improvement": before["logits_rel"] - after["logits_rel"],
                        "teacher_student_before_KL": before["last_token_KL_a_to_b"],
                        "teacher_student_after_KL": after["last_token_KL_a_to_b"],
                        "teacher_student_KL_improvement": before["last_token_KL_a_to_b"] - after["last_token_KL_a_to_b"],
                        "teacher_student_before_top1_match": before["last_top1_match"],
                        "teacher_student_after_top1_match": after["last_top1_match"],
                        "student_before_after_logits_rel": shift["logits_rel"],
                        "student_before_after_KL": shift["last_token_KL_a_to_b"],
                        "delta_key": key,
                        "delta_tokens": int(n),
                        "delta_rel_to_component": rel_err(teacher_trace[pi][key][-n:], student_trace[pi][key][-n:]),
                        "applied_delta_rel_to_layer_out": rel_err(float(alpha) * delta, layer_out),
                    }
                    rows.append(row)
                    per_prompt_all.append(row)
                sums = summarize(rows)
                score = score_candidate(sums, args.kl_weight, args.shift_weight, args.delta_weight)
                item = {
                    "candidate": f"L{li}_{comp}_a{alpha:g}",
                    "layer": li,
                    "component": comp,
                    "alpha": float(alpha),
                    "prompt_mode": args.prompt_mode,
                    **sums,
                    "selection_score": score,
                    "useful_logits": sums["logits_improvement"] > 0,
                    "useful_KL": sums["KL_improvement"] > 0,
                }
                summary_all.append(item)
                print(
                    f"L{li}_{comp}_a{alpha:g}: "
                    f"logit_gain={sums['logits_improvement']:+.5f} "
                    f"KL_gain={sums['KL_improvement']:+.5f} "
                    f"shift={sums['student_shift_logits_rel']:.5f} "
                    f"delta={sums['applied_delta_rel_to_layer_out_mean']:.5f} "
                    f"score={score:+.5f}"
                )

        del teacher_trace, student_trace
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    summary_all = sorted(summary_all, key=lambda r: r["selection_score"], reverse=True)

    report = {
        "version": VERSION,
        "mode": "true_multilayer_output_delta_probe",
        "prompt_mode": args.prompt_mode,
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layers": layers,
        "components": components,
        "prompts": len(prompts),
        "alphas": alphas,
        "best_candidates": summary_all[:30],
        "status": "MULTILAYER_DELTA_PROBE_RAN",
        "no_training": True,
        "closure_level": "forward_hook_layer_output_delta_then_remaining_layers",
        "note": "This is output-level causal localization, not final program/weight transplant. Use it to choose layers/components before exact decoding/weight rewrite.",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_candidate_multilayer_delta_probe.jsonl", summary_all)
    write_jsonl(out / "per_prompt_multilayer_delta_probe.jsonl", per_prompt_all)

    print("=== Qwen Teacher -> Student True Multi-Layer Delta Probe v1 ===")
    print(json.dumps({
        "prompt_mode": report["prompt_mode"],
        "layers": report["layers"],
        "components": report["components"],
        "prompts": report["prompts"],
        "top10": [{
            "candidate": r["candidate"],
            "logits_improvement": r["logits_improvement"],
            "KL_improvement": r["KL_improvement"],
            "student_shift_logits_rel": r["student_shift_logits_rel"],
            "applied_delta_rel_to_layer_out_mean": r["applied_delta_rel_to_layer_out_mean"],
            "selection_score": r["selection_score"],
        } for r in summary_all[:10]],
        "status": report["status"],
    }, indent=2))
    print(f"out={out}")


if __name__ == "__main__":
    main()
