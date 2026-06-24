#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_contrastive_gradient_weight_transfer_v1_1.py

Sign-aware contrastive gradient weight transfer.

v1 selected weights with high |grad_code| / |grad_retain|, then moved them toward
Coder weights. That found code-sensitive weights, but did not check whether the
Coder-Student delta was actually aligned with the anti-gradient of code KL.

v1.1 fixes the mask:

  delta = W_teacher - W_student
  code_descent = -grad_code * delta
  retain_effect = abs(grad_retain * delta)
  score = relu(code_descent) / (retain_effect + eps * mean(abs(delta)))
  mask = top(score) AND code_descent > 0 AND retain_effect <= retain_ratio * code_descent

Then applies:

  W_new = W_student + alpha * mask * delta

No training, no distillation, no LoRA. This is a weight-level selective transplant
probe. If it works, decode selected tensors/modules into exact mechanisms before
claiming reusable program transplant.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from program_dsl_v1 import write_json, write_jsonl
from qwen_circuit_target_roundtrip_v1 import get_dtype
from qwen_teacher_student_code_transfer_v2 import select_prompts

VERSION = "qwen_contrastive_gradient_weight_transfer_v1.1-sign-aware"


def mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / max(1, len(xs))


def parse_ints(s: str) -> List[int]:
    if not s or s.lower() in ("all", "none"):
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def last_metrics(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> Dict[str, Any]:
    v = min(teacher_logits.shape[-1], student_logits.shape[-1])
    lt = teacher_logits[0, -1, :v].float()
    ls = student_logits[0, -1, :v].float()
    logpt = F.log_softmax(lt, dim=-1)
    logps = F.log_softmax(ls, dim=-1)
    pt = torch.exp(logpt)
    top1_t = int(torch.argmax(lt).item())
    top1_s = int(torch.argmax(ls).item())
    top5_t = set(torch.topk(lt, k=5).indices.tolist())
    top5_s = set(torch.topk(ls, k=5).indices.tolist())
    return {
        "last_logits_rel": rel_err(lt, ls),
        "kl_teacher_to_student": float(torch.sum(pt * (logpt - logps))),
        "top1_match": bool(top1_t == top1_s),
        "top5_overlap": len(top5_t & top5_s),
    }


@torch.no_grad()
def teacher_logp(teacher, tokenizer, text: str, max_length: int, device: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = teacher(**enc, use_cache=False).logits
    return F.log_softmax(logits[0, -1].float(), dim=-1).detach()


def kl_to_teacher(student, tokenizer, text: str, teacher_logp_vec: torch.Tensor, max_length: int, device: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = student(**enc, use_cache=False).logits
    v = min(logits.shape[-1], teacher_logp_vec.shape[-1])
    logps = F.log_softmax(logits[0, -1, :v].float(), dim=-1)
    logpt = teacher_logp_vec[:v].to(device=device, dtype=torch.float32)
    pt = torch.exp(logpt)
    return torch.sum(pt * (logpt - logps))


def suffixes_for_components(components: List[str]) -> List[str]:
    out: List[str] = []
    for c in components:
        if c == "qk":
            out += ["self_attn.q_proj", "self_attn.k_proj"]
        elif c == "vo":
            out += ["self_attn.v_proj", "self_attn.o_proj"]
        elif c == "attn":
            out += ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        elif c == "mlp":
            out += ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
        elif c == "all":
            out += [
                "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
            ]
        else:
            raise ValueError(f"unknown component: {c}")
    uniq: List[str] = []
    seen = set()
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def select_parameter_names(model, layers: List[int], components: List[str]) -> List[str]:
    suffixes = suffixes_for_components(components)
    names: List[str] = []
    for name, _p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if layers:
            if not any(name.startswith(f"model.layers.{li}.") for li in layers):
                continue
        else:
            if not name.startswith("model.layers."):
                continue
        if any(f".{s}." in name for s in suffixes):
            names.append(name)
    return names


def set_trainable_only(model, names: List[str]) -> Dict[str, torch.nn.Parameter]:
    wanted = set(names)
    selected: Dict[str, torch.nn.Parameter] = {}
    for name, p in model.named_parameters():
        req = name in wanted
        p.requires_grad_(req)
        if req:
            selected[name] = p
    return selected


def zero_grads(params: Dict[str, torch.nn.Parameter]) -> None:
    for p in params.values():
        p.grad = None


def compute_grad_map(teacher, student, tok_t, tok_s, prompts: List[str], params: Dict[str, torch.nn.Parameter], max_length: int, device: str) -> Tuple[Dict[str, torch.Tensor], float]:
    zero_grads(params)
    losses: List[float] = []
    for text in prompts:
        logpt = teacher_logp(teacher, tok_t, text, max_length, device)
        loss = kl_to_teacher(student, tok_s, text, logpt, max_length, device)
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        del loss, logpt
    grads: Dict[str, torch.Tensor] = {}
    denom = float(max(1, len(prompts)))
    for name, p in params.items():
        if p.grad is None:
            grads[name] = torch.zeros_like(p.detach(), dtype=torch.float32, device="cpu")
        else:
            grads[name] = (p.grad.detach().float().cpu() / denom).contiguous()
    zero_grads(params)
    return grads, mean(losses)


def cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    af = a.reshape(-1).float()
    bf = b.reshape(-1).float()
    return float(torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf)).clamp_min(eps))


def make_directional_masks(
    teacher,
    student,
    names: List[str],
    grads_code: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
    top_frac: float,
    retain_ratio: float,
    eps_scale: float,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
    sp = dict(student.named_parameters())
    tp = dict(teacher.named_parameters())
    masks: Dict[str, torch.Tensor] = {}
    rows: List[Dict[str, Any]] = []
    for name in names:
        gc = grads_code[name]
        gr = grads_retain[name]
        delta = (tp[name].detach().float().cpu() - sp[name].detach().float().cpu()).contiguous()
        code_descent = -gc * delta
        retain_effect = (gr * delta).abs()
        eps = float(eps_scale) * float(delta.abs().mean().item() + 1e-12)
        score = torch.relu(code_descent) / (retain_effect + eps + 1e-12)
        good = (code_descent > 0) & (retain_effect <= float(retain_ratio) * code_descent)
        n = score.numel()
        k = max(1, int(float(top_frac) * n))
        flat = score.reshape(-1)
        if k >= n:
            thresh = flat.min()
        else:
            thresh = torch.topk(flat, k=k, largest=True).values[-1]
        mask = (score >= thresh) & good
        masks[name] = mask.cpu()
        selected = int(mask.sum().item())
        rows.append({
            "name": name,
            "numel": int(n),
            "selected": selected,
            "selected_frac": float(mask.float().mean().item()),
            "code_grad_norm": float(torch.linalg.norm(gc.float()).item()),
            "retain_grad_norm": float(torch.linalg.norm(gr.float()).item()),
            "delta_norm": float(torch.linalg.norm(delta.float()).item()),
            "grad_code_retain_cosine": cosine(gc, gr),
            "delta_code_grad_cosine": cosine(delta, -gc),
            "code_descent_mean": float(code_descent.float().mean().item()),
            "code_descent_positive_frac": float((code_descent > 0).float().mean().item()),
            "retain_effect_mean": float(retain_effect.float().mean().item()),
            "score_threshold": float(thresh.item()),
            "score_mean": float(score.float().mean().item()),
            "score_max": float(score.float().max().item()),
        })
    return masks, rows


@torch.no_grad()
def backup_params(model, names: List[str]) -> Dict[str, torch.Tensor]:
    mp = dict(model.named_parameters())
    return {n: mp[n].detach().clone() for n in names}


@torch.no_grad()
def restore_params(model, backup: Dict[str, torch.Tensor]) -> None:
    mp = dict(model.named_parameters())
    for n, t in backup.items():
        mp[n].data.copy_(t.to(device=mp[n].device, dtype=mp[n].dtype))


@torch.no_grad()
def apply_masked_delta(student, teacher, names: List[str], masks: Dict[str, torch.Tensor], alpha: float) -> List[Dict[str, Any]]:
    sp = dict(student.named_parameters())
    tp = dict(teacher.named_parameters())
    rows: List[Dict[str, Any]] = []
    for name in names:
        ws = sp[name]
        wt = tp[name].detach().to(device=ws.device, dtype=ws.dtype)
        mask = masks[name].to(device=ws.device)
        delta = wt - ws.detach()
        patch = delta * mask.to(dtype=delta.dtype)
        ws.data.add_(float(alpha) * patch)
        rows.append({
            "name": name,
            "alpha": float(alpha),
            "mask_frac": float(mask.float().mean().item()),
            "delta_norm": float(torch.linalg.norm(delta.float()).item()),
            "patch_norm": float(torch.linalg.norm(patch.float()).item()),
        })
    return rows


@torch.no_grad()
def eval_prompt_set(teacher, student, tok_t, tok_s, prompts: List[str], max_length: int, device: str, label: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for i, text in enumerate(prompts):
        et = tok_t(text, return_tensors="pt", truncation=True, max_length=max_length)
        es = tok_s(text, return_tensors="pt", truncation=True, max_length=max_length)
        et = {k: v.to(device) for k, v in et.items()}
        es = {k: v.to(device) for k, v in es.items()}
        lt = teacher(**et, use_cache=False).logits.detach()
        ls = student(**es, use_cache=False).logits.detach()
        m = last_metrics(lt, ls)
        rows.append({"set": label, "prompt_id": i, "text": text[:180], **m})
    summary = {
        f"{label}_logits_rel": mean([r["last_logits_rel"] for r in rows]),
        f"{label}_KL": mean([r["kl_teacher_to_student"] for r in rows]),
        f"{label}_top1": mean([1.0 if r["top1_match"] else 0.0 for r in rows]),
        f"{label}_top5_overlap": mean([float(r["top5_overlap"]) for r in rows]),
    }
    return summary, rows


def gain(before: Dict[str, Any], after: Dict[str, Any], label: str) -> Dict[str, float]:
    return {
        f"{label}_logits_gain": before[f"{label}_logits_rel"] - after[f"{label}_logits_rel"],
        f"{label}_KL_gain": before[f"{label}_KL"] - after[f"{label}_KL"],
        f"{label}_logits_before": before[f"{label}_logits_rel"],
        f"{label}_logits_after": after[f"{label}_logits_rel"],
        f"{label}_KL_before": before[f"{label}_KL"],
        f"{label}_KL_after": after[f"{label}_KL"],
    }


def candidate_score(row: Dict[str, float], retain_damage_weight: float, retain_shift_weight: float) -> float:
    retain_damage = max(0.0, -float(row["retain_KL_gain"]))
    retain_shift = abs(float(row["retain_logits_gain"]))
    return float(row["code_KL_gain"]) + float(row["code_logits_gain"]) - retain_damage_weight * retain_damage - retain_shift_weight * retain_shift


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    ap.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--attn-implementation", default="eager")
    ap.add_argument("--layers", default="18,23")
    ap.add_argument("--components", default="vo,mlp")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--alphas", default="0.05,0.1,0.25")
    ap.add_argument("--mask-top-frac", type=float, default=0.001)
    ap.add_argument("--retain-ratio", type=float, default=0.5)
    ap.add_argument("--eps-scale", type=float, default=0.05)
    ap.add_argument("--retain-damage-weight", type=float, default=2.0)
    ap.add_argument("--retain-shift-weight", type=float, default=0.25)
    ap.add_argument("--out", default="runs/exact_program_transplant_v1/contrastive_gradient_weight_transfer_v1_1")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    layers = parse_ints(args.layers)
    components = parse_strs(args.components)
    alphas = parse_floats(args.alphas)
    code_prompts = select_prompts("code", args.prompts)
    retain_prompts = select_prompts("retain", args.prompts)

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
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("loading student...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        torch_dtype=get_dtype(args.dtype),
        device_map=None,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).to(args.device)
    student.eval()

    names = select_parameter_names(student, layers, components)
    if not names:
        raise SystemExit("no selected parameters; check --layers/--components")
    print(f"selected params: {len(names)}")
    for n in names[:30]:
        print(" ", n)
    if len(names) > 30:
        print(" ...")

    params = set_trainable_only(student, names)

    print("baseline...")
    base_code, base_code_rows = eval_prompt_set(teacher, student, tok_t, tok_s, code_prompts, args.max_length, args.device, "code")
    base_retain, base_retain_rows = eval_prompt_set(teacher, student, tok_t, tok_s, retain_prompts, args.max_length, args.device, "retain")

    print("code gradients...")
    grads_code, code_loss = compute_grad_map(teacher, student, tok_t, tok_s, code_prompts, params, args.max_length, args.device)
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    print("retain gradients...")
    grads_retain, retain_loss = compute_grad_map(teacher, student, tok_t, tok_s, retain_prompts, params, args.max_length, args.device)

    print("directional masks...")
    masks, mask_rows = make_directional_masks(teacher, student, names, grads_code, grads_retain, args.mask_top_frac, args.retain_ratio, args.eps_scale)
    total = sum(r["numel"] for r in mask_rows)
    selected = sum(r["selected"] for r in mask_rows)
    print(f"selected weights: {selected}/{total} = {selected / max(1, total):.8f}")

    backup = backup_params(student, names)
    candidate_rows: List[Dict[str, Any]] = []
    prompt_rows: List[Dict[str, Any]] = []
    patch_rows_all: List[Dict[str, Any]] = []

    for alpha in alphas:
        restore_params(student, backup)
        patch_rows = apply_masked_delta(student, teacher, names, masks, alpha)
        patch_rows_all.extend(patch_rows)
        after_code, after_code_rows = eval_prompt_set(teacher, student, tok_t, tok_s, code_prompts, args.max_length, args.device, "code")
        after_retain, after_retain_rows = eval_prompt_set(teacher, student, tok_t, tok_s, retain_prompts, args.max_length, args.device, "retain")
        row: Dict[str, Any] = {
            "candidate": f"directional_contrastive_grad_a{alpha:g}",
            "alpha": float(alpha),
            **gain(base_code, after_code, "code"),
            **gain(base_retain, after_retain, "retain"),
            "selected_weights": selected,
            "selected_frac": selected / max(1, total),
        }
        row["selection_score"] = candidate_score(row, args.retain_damage_weight, args.retain_shift_weight)
        candidate_rows.append(row)
        for r in after_code_rows + after_retain_rows:
            r["alpha"] = float(alpha)
            r["candidate"] = row["candidate"]
            prompt_rows.append(r)
        print(
            f"a={alpha:g}: "
            f"code_logit={row['code_logits_gain']:+.5f} "
            f"code_KL={row['code_KL_gain']:+.5f} "
            f"retain_logit={row['retain_logits_gain']:+.5f} "
            f"retain_KL={row['retain_KL_gain']:+.5f} "
            f"score={row['selection_score']:+.5f}"
        )

    restore_params(student, backup)
    candidate_rows = sorted(candidate_rows, key=lambda x: x["selection_score"], reverse=True)

    report = {
        "version": VERSION,
        "mode": "directional_contrastive_gradient_weight_delta",
        "teacher_model": args.teacher_model,
        "student_model": args.student_model,
        "layers": layers if layers else "all",
        "components": components,
        "prompts": args.prompts,
        "alphas": alphas,
        "mask_top_frac": args.mask_top_frac,
        "retain_ratio": args.retain_ratio,
        "eps_scale": args.eps_scale,
        "selected_parameter_count": len(names),
        "selected_weights": selected,
        "selected_frac": selected / max(1, total),
        "gradient_losses": {"code_loss": code_loss, "retain_loss": retain_loss},
        "baseline": {**base_code, **base_retain},
        "best_candidates": candidate_rows,
        "status": "DIRECTIONAL_CONTRASTIVE_GRADIENT_WEIGHT_TRANSFER_RAN",
        "no_training": True,
        "closure_level": "weight_level_directional_masked_teacher_student_delta_probe",
        "note": "Mask uses score=relu(-grad_code*delta)/(abs(grad_retain*delta)+eps*mean(abs(delta))).",
    }

    write_json(out / "manifest.json", report)
    write_jsonl(out / "per_parameter_directional_mask_stats.jsonl", mask_rows)
    write_jsonl(out / "per_candidate_weight_transfer.jsonl", candidate_rows)
    write_jsonl(out / "per_prompt_after_weight_transfer.jsonl", prompt_rows)
    write_jsonl(out / "per_parameter_patch_stats.jsonl", patch_rows_all)
    write_jsonl(out / "baseline_code_prompts.jsonl", base_code_rows)
    write_jsonl(out / "baseline_retain_prompts.jsonl", base_retain_rows)

    print("=== Qwen Directional Contrastive Gradient Weight Transfer v1.1 ===")
    print(json.dumps({
        "layers": report["layers"],
        "components": report["components"],
        "selected_parameter_count": report["selected_parameter_count"],
        "selected_frac": report["selected_frac"],
        "gradient_losses": report["gradient_losses"],
        "baseline": report["baseline"],
        "top": [
            {
                "candidate": r["candidate"],
                "code_logits_gain": r["code_logits_gain"],
                "code_KL_gain": r["code_KL_gain"],
                "retain_logits_gain": r["retain_logits_gain"],
                "retain_KL_gain": r["retain_KL_gain"],
                "selection_score": r["selection_score"],
            }
            for r in candidate_rows[:10]
        ],
        "status": report["status"],
    }, indent=2))
    print("out=", out)


if __name__ == "__main__":
    main()
