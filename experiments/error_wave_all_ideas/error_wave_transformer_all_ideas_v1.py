#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error-Wave Transformer All-Ideas Benchmark v1

Проверяет все текущие идеи на реальной causal LM:

Методы:
  full_bp              обычный full backprop
  random_sparse        случайная sparse-маска
  magnitude_sparse     top-k по |g_patch|
  dare_sparse          random sparse + rescale 1/density
  error_wave_v4        score = |g_patch| - penalty*|g_retain|
  cga_soft             v4 mask + soft per-weight arbitration
  cga_hard             top-k только среди безопасных направлений
  drcf_v4              v4 + память конфликтов по весам
  drcf_cga             CGA hard + память конфликтов
  wave_orthogonal      v4 mask + ортогональная patch-компонента к retain-gradient
  orthogonal_full      full orthogonal gradient surgery по trainable-параметрам
  lora                 LoRA baseline, если установлен peft

Задача:
  retain = нормальный WikiText causal LM
  patch  = конфликтный objective на тех же input:
           random_vocab / reverse_labels / shuffle_labels / mismatched_labels

Метрики:
  patch_improve = patch_before - patch_after
  forget_loss   = retain_after - retain_before ; >0 значит retain ухудшился
  density_mean  = доля ненулевых градиентов среди trainable параметров

Главная цель:
  понять Pareto-фронт:
    кто лучше учит patch
    кто меньше ломает retain
    сколько градиента реально применяет
"""

import argparse
import csv
import gc
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-name", type=str, default="distilgpt2")
    p.add_argument("--retain-dataset", type=str, default="wikitext")
    p.add_argument("--retain-config", type=str, default="wikitext-2-raw-v1")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--hf-token", type=str, default=None)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="fp16", choices=["off", "fp16", "bf16"])
    p.add_argument("--seeds", type=str, default="0")
    p.add_argument(
        "--methods",
        type=str,
        default="full_bp,random_sparse,magnitude_sparse,dare_sparse,error_wave_v4,cga_soft,cga_hard,drcf_v4,drcf_cga,wave_orthogonal,orthogonal_full,lora",
    )

    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--mask-batch", type=int, default=1)
    p.add_argument("--eval-batch", type=int, default=2)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--diag-every", type=int, default=50)

    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--train-last-blocks", type=int, default=2)

    p.add_argument("--density-start", type=float, default=0.03)
    p.add_argument("--density-end", type=float, default=0.008)
    p.add_argument("--frac-decay", action="store_true")
    p.add_argument("--retain-penalty", type=float, default=0.5)
    p.add_argument("--global-topk", action="store_true")

    p.add_argument("--cga-beta", type=float, default=3.0)
    p.add_argument("--cga-eps", type=float, default=1e-12)
    p.add_argument("--drcf-decay", type=float, default=0.95)
    p.add_argument("--drcf-alpha", type=float, default=0.05)
    p.add_argument("--drcf-strength", type=float, default=1.0)
    p.add_argument("--orthogonal-eps", type=float, default=1e-12)

    p.add_argument("--max-texts", type=int, default=4000)
    p.add_argument(
        "--patch-mode",
        type=str,
        default="random_vocab",
        choices=["random_vocab", "reverse_labels", "shuffle_labels", "mismatched_labels"],
    )
    p.add_argument("--random-label-low", type=int, default=100)
    p.add_argument("--random-label-high", type=int, default=30000)

    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)

    p.add_argument("--results-csv", type=str, default="results/error_wave_all_ideas_v1.csv")
    p.add_argument("--diag-csv", type=str, default="results/error_wave_all_ideas_diag_v1.csv")
    p.add_argument("--summary-md", type=str, default="results/error_wave_all_ideas_summary_v1.md")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_dtype(args):
    if args.amp == "fp16":
        return torch.float16
    if args.amp == "bf16":
        return torch.bfloat16
    return None


def _load_dataset_try(names, config=None, split="train", args=None):
    from datasets import load_dataset
    kw = {}
    if args is not None and getattr(args, "hf_token", None):
        kw["token"] = args.hf_token
    if args is not None and getattr(args, "local_files_only", False):
        try:
            from datasets import DownloadConfig
            kw["download_config"] = DownloadConfig(local_files_only=True)
        except Exception:
            pass

    last_err = None
    for name in names:
        try:
            if config:
                print(f"  try HF dataset: {name}/{config} split={split}")
                return load_dataset(name, config, split=split, **kw)
            print(f"  try HF dataset: {name} split={split}")
            return load_dataset(name, split=split, **kw)
        except Exception as e:
            print(f"  failed: {name}: {type(e).__name__}: {str(e)[:240]}")
            last_err = e
    raise RuntimeError("Не смог загрузить датасет. Пробовал: " + ", ".join(names)) from last_err


def _texts_from_dataset(ds, max_texts: int) -> List[str]:
    keys = list(ds.column_names)
    text_key = "text" if "text" in keys else keys[0]
    out = []
    for x in ds.select(range(min(max_texts, len(ds)))):
        v = x.get(text_key, "")
        if isinstance(v, (list, tuple)):
            v = " ".join(map(str, v))
        v = str(v).strip()
        if len(v) > 20:
            out.append(v)
    return out


def load_retain_texts(args) -> List[str]:
    aliases = {
        "wikitext": ["Salesforce/wikitext", "wikitext"],
        "wiki_text": ["Salesforce/wikitext", "wikitext"],
    }
    names = aliases.get(args.retain_dataset, [args.retain_dataset])
    print(f"loading retain dataset: {args.retain_dataset}/{args.retain_config}")
    ds = _load_dataset_try(names, config=args.retain_config if args.retain_dataset in aliases else None, split="train", args=args)
    texts = _texts_from_dataset(ds, args.max_texts)
    if len(texts) < 64:
        raise RuntimeError(f"Мало текстов: {len(texts)}")
    print(f"loaded retain texts: {len(texts)}")
    return texts


class CleanSampler:
    def __init__(self, tokenizer, texts: List[str], seq_len: int, device: torch.device):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device
        self.examples = []
        self._build(texts)

    def _build(self, texts: List[str]):
        ids = []
        eos = self.tokenizer.eos_token_id
        for t in texts:
            enc = self.tokenizer.encode(t, add_special_tokens=False)
            if len(enc) < 8:
                continue
            ids.extend(enc + [eos])
            if len(ids) > 2_500_000:
                break
        step = self.seq_len + 1
        if len(ids) < step + 1:
            raise RuntimeError("Слишком мало токенов.")
        for i in range(0, len(ids) - step, step):
            chunk = ids[i:i + step]
            if len(chunk) == step:
                self.examples.append(torch.tensor(chunk, dtype=torch.long))
        if len(self.examples) < 32:
            raise RuntimeError(f"Слишком мало чанков: {len(self.examples)}")

    def sample_pair(self, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = random.choices(self.examples, k=batch)
        x = torch.stack(xs, dim=0).to(self.device)
        return x[:, :-1], x[:, 1:].contiguous()

    def fixed_pairs(self, batches: int, batch: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        py_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        out = [self.sample_pair(batch) for _ in range(batches)]
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        return out

    def __len__(self):
        return len(self.examples)


class ConflictSampler:
    def __init__(self, clean_sampler: CleanSampler, mode: str, vocab_size: int, low: int, high: int):
        self.clean = clean_sampler
        self.mode = mode
        self.vocab_size = vocab_size
        self.low = max(0, low)
        self.high = min(max(low + 1, high), vocab_size)

    def sample_pair(self, batch: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inp, labels = self.clean.sample_pair(batch)
        if self.mode == "mismatched_labels":
            _, wrong = self.clean.sample_pair(batch)
            labels = wrong
        elif self.mode == "shuffle_labels":
            labels = labels.clone()
            for b in range(labels.shape[0]):
                labels[b] = labels[b, torch.randperm(labels.shape[1], device=labels.device)]
        elif self.mode == "reverse_labels":
            labels = torch.flip(labels, dims=[1]).contiguous()
        elif self.mode == "random_vocab":
            labels = torch.randint(self.low, self.high, labels.shape, device=labels.device, dtype=torch.long)
        else:
            raise ValueError(self.mode)
        return inp, labels.contiguous()

    def fixed_pairs(self, batches: int, batch: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        py_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        out = [self.sample_pair(batch) for _ in range(batches)]
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        return out


def load_model_and_tokenizer(model_name: str, device: torch.device, args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=args.local_files_only)
    model.config.use_cache = False
    model.to(device)
    return model, tok


def mark_last_blocks_trainable(model: nn.Module, train_last_blocks: int):
    for p in model.parameters():
        p.requires_grad_(False)

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n = len(model.transformer.h)
        keep = list(range(max(0, n - train_last_blocks), n))
        prefixes = [f"transformer.h.{i}." for i in keep]
        print(f"trainable GPT blocks: {keep}")
        for name, p in model.named_parameters():
            if any(name.startswith(pref) for pref in prefixes) or name.startswith("transformer.ln_f."):
                p.requires_grad_(True)

    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        n = len(model.model.layers)
        keep = list(range(max(0, n - train_last_blocks), n))
        prefixes = [f"model.layers.{i}." for i in keep]
        print(f"trainable qwen/llama blocks: {keep}")
        for name, p in model.named_parameters():
            if any(name.startswith(pref) for pref in prefixes) or name.startswith("model.norm."):
                p.requires_grad_(True)
    else:
        raise RuntimeError("Не нашёл GPT/Qwen/Llama блоки.")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total:,} trainable={trainable:,} ({100*trainable/max(total,1):.3f}%)")
    return total, trainable


def setup_lora(model: nn.Module, args):
    try:
        from peft import LoraConfig, get_peft_model
    except Exception:
        print("[SKIP] peft не установлен. LoRA baseline пропущен. Установи: pip install peft")
        return None

    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        targets = ["c_attn", "c_proj", "c_fc"]
        fan = True
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        fan = False
    else:
        targets = None
        fan = False

    cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        fan_in_fan_out=fan,
    )
    return get_peft_model(model, cfg)


def lm_loss_pair(model: nn.Module, inp: torch.Tensor, labels: torch.Tensor, dtype=None):
    with torch.autocast(device_type=inp.device.type, dtype=dtype, enabled=(dtype is not None and inp.device.type == "cuda")):
        return model(inp, labels=labels).loss


@torch.no_grad()
def eval_fixed(model: nn.Module, fixed_batches: List[Tuple[torch.Tensor, torch.Tensor]], dtype=None) -> float:
    model.eval()
    vals = []
    for inp, labels in fixed_batches:
        vals.append(float(lm_loss_pair(model, inp, labels, dtype).detach().cpu()))
    model.train()
    return sum(vals) / max(1, len(vals))


def current_density(step: int, total: int, start: float, end: float, decay: bool) -> float:
    if not decay:
        return start
    if total <= 1:
        return end
    t = (step - 1) / (total - 1)
    return start * (1 - t) + end * t


def collect_grads(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


def grad_density(model: nn.Module) -> float:
    nz = total = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach()
            nz += int((g != 0).sum().item())
            total += g.numel()
    return nz / max(total, 1)


def topk_masks(scores: Dict[str, torch.Tensor], density: float, global_topk: bool):
    if not scores:
        return {}
    if global_topk:
        flat = torch.cat([s.reshape(-1) for s in scores.values()])
        k = max(1, int(density * flat.numel()))
        if k >= flat.numel():
            return {n: torch.ones_like(s, dtype=torch.bool) for n, s in scores.items()}
        thr = torch.topk(flat, k=k, largest=True).values[-1]
        return {n: (s >= thr) for n, s in scores.items()}

    masks = {}
    for n, s in scores.items():
        flat = s.reshape(-1)
        k = max(1, int(density * flat.numel()))
        if k >= flat.numel():
            masks[n] = torch.ones_like(s, dtype=torch.bool)
        else:
            thr = torch.topk(flat, k=k, largest=True).values[-1]
            masks[n] = (s >= thr)
    return masks


def apply_mask_and_scales(model: nn.Module, masks: Dict[str, torch.Tensor], scales: Optional[Dict[str, torch.Tensor]] = None, replacement_grads: Optional[Dict[str, torch.Tensor]] = None):
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if replacement_grads is not None and n in replacement_grads:
            if p.grad is None:
                p.grad = replacement_grads[n].clone()
            else:
                p.grad.copy_(replacement_grads[n])
        if p.grad is None:
            continue
        if n not in masks:
            p.grad.zero_()
            continue
        p.grad.mul_(masks[n].to(p.grad.dtype))
        if scales is not None and n in scales:
            p.grad.mul_(scales[n].to(p.grad.dtype))


def random_mask(model: nn.Module, density: float, rescale: bool = False):
    scale = 1.0 / max(density, 1e-8) if rescale else 1.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_((torch.rand_like(p.grad) < density).to(p.grad.dtype)).mul_(scale)


def compute_scores_masks(patch_grads: Dict[str, torch.Tensor], retain_grads: Dict[str, torch.Tensor], method: str, args, state: dict, density: float):
    mag_scores = {n: g.abs() for n, g in patch_grads.items()}

    v4_scores = {}
    for n, gp in patch_grads.items():
        gr = retain_grads.get(n)
        if gr is None:
            v4_scores[n] = gp.abs()
        else:
            v4_scores[n] = (gp.abs() - args.retain_penalty * gr.abs()).clamp_min(0.0)

    if method in ("drcf_v4", "drcf_cga"):
        hist = state.setdefault("conflict_history", {})
        for n, s in v4_scores.items():
            if n not in hist:
                hist[n] = torch.zeros_like(s)
            prior = (1.0 - args.drcf_strength * hist[n]).clamp(0.0, 1.0)
            v4_scores[n] = s * prior

    mag_masks = topk_masks(mag_scores, density, args.global_topk)
    v4_masks = topk_masks(v4_scores, density, args.global_topk)

    return mag_scores, v4_scores, mag_masks, v4_masks


def cga_scales_and_safe(patch_grads, retain_grads, args):
    scales = {}
    safe_masks = {}
    conflict_fracs = []
    for n, gp in patch_grads.items():
        gr = retain_grads.get(n)
        if gr is None:
            scales[n] = torch.ones_like(gp)
            safe_masks[n] = torch.ones_like(gp, dtype=torch.bool)
            conflict_fracs.append(0.0)
            continue

        prod = gp * gr
        # update = -gp, retain delta ≈ -lr * gr * gp.
        # prod > 0 => retain loss decreases/safe
        safe = (prod >= 0) | (gr.abs() < args.cga_eps)
        align = prod / (gp.abs() * gr.abs() + args.cga_eps)
        scale = torch.sigmoid(args.cga_beta * align)
        scale = torch.where(gr.abs() < args.cga_eps, torch.ones_like(scale), scale)

        scales[n] = scale
        safe_masks[n] = safe
        conflict_fracs.append(1.0 - float(safe.float().mean().detach().cpu()))
    return scales, safe_masks, sum(conflict_fracs) / max(len(conflict_fracs), 1)


def hard_safe_refill(v4_scores, safe_masks, density, global_topk):
    safe_scores = {}
    for n, s in v4_scores.items():
        safe = safe_masks.get(n)
        if safe is None:
            safe_scores[n] = s
        else:
            safe_scores[n] = s * safe.to(s.dtype)
    return topk_masks(safe_scores, density, global_topk)


def orthogonal_grads(patch_grads, retain_grads, eps):
    out = {}
    coeffs = []
    for n, gp in patch_grads.items():
        gr = retain_grads.get(n)
        if gr is None:
            out[n] = gp
            continue
        denom = (gr * gr).sum() + eps
        coeff = (gp * gr).sum() / denom
        out[n] = gp - coeff * gr
        coeffs.append(float(coeff.detach().cpu()))
    return out, (sum(coeffs) / max(len(coeffs), 1) if coeffs else 0.0)


def mask_jaccard(a, b):
    inter = union = 0
    for n in a:
        if n not in b:
            continue
        inter += int((a[n] & b[n]).sum().item())
        union += int((a[n] | b[n]).sum().item())
    return inter / max(union, 1)


def selected_risk(mask, patch_grads, retain_grads):
    ps, rs = [], []
    for n, m in mask.items():
        if n not in patch_grads:
            continue
        gp = patch_grads[n].abs()
        gr = retain_grads.get(n, torch.zeros_like(gp)).abs()
        if int(m.sum().item()) > 0:
            ps.append(gp[m].mean())
            rs.append(gr[m].mean())
    if not ps:
        return 0.0, 0.0, 0.0
    p = float(torch.stack(ps).mean().detach().cpu())
    r = float(torch.stack(rs).mean().detach().cpu())
    return p, r, r / max(p, 1e-12)


def update_drcf_state(state, patch_grads, retain_grads, mask, args):
    hist = state.setdefault("conflict_history", {})
    for n, gp in patch_grads.items():
        gr = retain_grads.get(n)
        if gr is None or n not in mask:
            continue
        conflict = ((gp * gr) < 0).to(gp.dtype) * mask[n].to(gp.dtype)
        if n not in hist:
            hist[n] = torch.zeros_like(conflict)
        hist[n].mul_(args.drcf_decay).add_(args.drcf_alpha * conflict)


def train_one_step(model, opt, method, clean_sampler, conflict_sampler, args, dtype, step, state):
    density = current_density(step, args.steps, args.density_start, args.density_end, args.frac_decay)

    if method in ("full_bp", "lora", "random_sparse", "dare_sparse", "magnitude_sparse"):
        opt.zero_grad(set_to_none=True)
        inp, labels = conflict_sampler.sample_pair(args.batch)
        loss = lm_loss_pair(model, inp, labels, dtype)
        loss.backward()

        if method == "random_sparse":
            random_mask(model, density, rescale=False)
        elif method == "dare_sparse":
            random_mask(model, density, rescale=True)
        elif method == "magnitude_sparse":
            patch_grads = collect_grads(model)
            masks = topk_masks({n: g.abs() for n, g in patch_grads.items()}, density, args.global_topk)
            apply_mask_and_scales(model, masks)

        dens = grad_density(model)
        opt.step()
        return float(loss.detach().cpu()), dens, {}

    # retain grads
    opt.zero_grad(set_to_none=True)
    inp_r, labels_r = clean_sampler.sample_pair(args.mask_batch)
    loss_r = lm_loss_pair(model, inp_r, labels_r, dtype)
    loss_r.backward()
    retain_grads = collect_grads(model)

    # patch grads
    opt.zero_grad(set_to_none=True)
    inp_p, labels_p = conflict_sampler.sample_pair(args.batch)
    loss = lm_loss_pair(model, inp_p, labels_p, dtype)
    loss.backward()
    patch_grads = collect_grads(model)

    mag_scores, v4_scores, mag_masks, v4_masks = compute_scores_masks(patch_grads, retain_grads, method, args, state, density)

    diag = {
        "density_target": density,
        "jacc_v4_mag": mask_jaccard(v4_masks, mag_masks),
    }

    if method == "error_wave_v4":
        final_masks = v4_masks
        apply_mask_and_scales(model, final_masks)

    elif method == "cga_soft":
        scales, safe_masks, conflict_frac = cga_scales_and_safe(patch_grads, retain_grads, args)
        final_masks = v4_masks
        apply_mask_and_scales(model, final_masks, scales=scales)
        diag["conflict_frac_all"] = conflict_frac

    elif method == "cga_hard":
        scales, safe_masks, conflict_frac = cga_scales_and_safe(patch_grads, retain_grads, args)
        final_masks = hard_safe_refill(v4_scores, safe_masks, density, args.global_topk)
        apply_mask_and_scales(model, final_masks)
        diag["conflict_frac_all"] = conflict_frac

    elif method == "drcf_v4":
        final_masks = v4_masks
        apply_mask_and_scales(model, final_masks)
        update_drcf_state(state, patch_grads, retain_grads, final_masks, args)

    elif method == "drcf_cga":
        scales, safe_masks, conflict_frac = cga_scales_and_safe(patch_grads, retain_grads, args)
        final_masks = hard_safe_refill(v4_scores, safe_masks, density, args.global_topk)
        apply_mask_and_scales(model, final_masks)
        update_drcf_state(state, patch_grads, retain_grads, final_masks, args)
        diag["conflict_frac_all"] = conflict_frac

    elif method == "wave_orthogonal":
        final_masks = v4_masks
        ortho, coeff = orthogonal_grads(patch_grads, retain_grads, args.orthogonal_eps)
        apply_mask_and_scales(model, final_masks, replacement_grads=ortho)
        diag["orthogonal_coeff_mean"] = coeff

    elif method == "orthogonal_full":
        final_masks = {n: torch.ones_like(g, dtype=torch.bool) for n, g in patch_grads.items()}
        ortho, coeff = orthogonal_grads(patch_grads, retain_grads, args.orthogonal_eps)
        apply_mask_and_scales(model, final_masks, replacement_grads=ortho)
        diag["orthogonal_coeff_mean"] = coeff

    else:
        raise ValueError(method)

    dens = grad_density(model)

    _, _, risk_final = selected_risk(final_masks, patch_grads, retain_grads)
    _, _, risk_v4 = selected_risk(v4_masks, patch_grads, retain_grads)
    _, _, risk_mag = selected_risk(mag_masks, patch_grads, retain_grads)
    diag.update({
        "risk_final": risk_final,
        "risk_v4": risk_v4,
        "risk_mag": risk_mag,
        "jacc_final_v4": mask_jaccard(final_masks, v4_masks),
        "jacc_final_mag": mask_jaccard(final_masks, mag_masks),
        "density_actual": dens,
    })

    opt.step()
    return float(loss.detach().cpu()), dens, diag


def run_method(method: str, seed: int, args, texts: List[str], fixed_seed_offset=12345):
    set_seed(seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = amp_dtype(args)

    model, tok = load_model_and_tokenizer(args.model_name, device, args)

    if method == "lora":
        maybe = setup_lora(model, args)
        if maybe is None:
            return {"skip": 1, "method": method, "seed": seed}, []
        model = maybe
    else:
        mark_last_blocks_trainable(model, args.train_last_blocks)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params total={total:,} trainable={trainable:,} ({100*trainable/max(total,1):.3f}%)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt_lr = args.lora_lr if method == "lora" else args.lr
    opt = torch.optim.AdamW(params, lr=opt_lr, weight_decay=args.weight_decay)

    clean_sampler = CleanSampler(tok, texts, args.seq_len, device)
    conflict_sampler = ConflictSampler(clean_sampler, args.patch_mode, len(tok), args.random_label_low, args.random_label_high)
    print(f"chunks clean={len(clean_sampler)} patch_mode={args.patch_mode}")

    # fixed eval identical across methods and repeatable
    set_seed(seed + fixed_seed_offset)
    fixed_patch = conflict_sampler.fixed_pairs(args.eval_batches, args.eval_batch)
    set_seed(seed + fixed_seed_offset + 1)
    fixed_retain = clean_sampler.fixed_pairs(args.eval_batches, args.eval_batch)

    before_patch = eval_fixed(model, fixed_patch, dtype)
    before_retain = eval_fixed(model, fixed_retain, dtype)
    print(f"BEFORE {method}: patch_loss={before_patch:.4f} retain_loss={before_retain:.4f}")

    state = {}
    dens_sum = 0.0
    diag_rows = []
    t0 = time.time()

    for step in range(1, args.steps + 1):
        loss, dens, diag = train_one_step(model, opt, method, clean_sampler, conflict_sampler, args, dtype, step, state)
        dens_sum += dens

        if diag and (step == 1 or step % args.diag_every == 0 or step == args.steps):
            diag_rows.append({
                "seed": seed,
                "method": method,
                "step": step,
                **diag,
            })

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            pl = eval_fixed(model, fixed_patch[:max(2, args.eval_batches // 2)], dtype)
            rl = eval_fixed(model, fixed_retain[:max(2, args.eval_batches // 2)], dtype)
            print(
                f"  {method:18s} step {step:04d}/{args.steps} "
                f"loss={loss:.4f} dens={dens:.4f} "
                f"patch={pl:.4f} retain={rl:.4f} t={time.time()-t0:.1f}s"
            )
            if diag:
                print(
                    f"    diag risk final/v4/mag="
                    f"{diag.get('risk_final', 0):.3f}/"
                    f"{diag.get('risk_v4', 0):.3f}/"
                    f"{diag.get('risk_mag', 0):.3f} "
                    f"jacc_final_v4={diag.get('jacc_final_v4', 0):.3f}"
                )

    after_patch = eval_fixed(model, fixed_patch, dtype)
    after_retain = eval_fixed(model, fixed_retain, dtype)

    row = {
        "seed": seed,
        "method": method,
        "model": args.model_name,
        "patch_mode": args.patch_mode,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_frac_total": trainable / max(total, 1),
        "patch_before": before_patch,
        "patch_after": after_patch,
        "retain_before": before_retain,
        "retain_after": after_retain,
        "patch_improve": before_patch - after_patch,
        "forget_loss": after_retain - before_retain,
        "density_mean": dens_sum / args.steps,
        "skip": 0,
    }

    print(
        f"FINAL {method}: patch {before_patch:.4f}->{after_patch:.4f} "
        f"improve={row['patch_improve']:+.4f} | "
        f"retain {before_retain:.4f}->{after_retain:.4f} "
        f"forget={row['forget_loss']:+.4f} | "
        f"dens={row['density_mean']:.4f}"
    )

    del model, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return row, diag_rows


def write_csv(path: str, rows: List[dict], fields: List[str]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_summary(path: str, rows: List[dict], diag_rows: List[dict]):
    valid = [r for r in rows if not int(r.get("skip", 0))]
    by = {}
    for r in valid:
        by.setdefault(r["method"], []).append(r)

    def mean(rs, k):
        return sum(float(x[k]) for x in rs) / max(len(rs), 1)

    lines = []
    lines.append("# Error-Wave All Ideas Benchmark Summary\n")
    lines.append("## Main metrics\n")
    lines.append("| method | seeds | patch_improve | forget_loss | density | trainable_frac |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for method, rs in by.items():
        lines.append(
            f"| {method} | {len(rs)} | "
            f"{mean(rs, 'patch_improve'):+.4f} | "
            f"{mean(rs, 'forget_loss'):+.4f} | "
            f"{mean(rs, 'density_mean'):.4f} | "
            f"{mean(rs, 'trainable_frac_total'):.4f} |"
        )

    if diag_rows:
        lines.append("\n## Diagnostics\n")
        bym = {}
        for d in diag_rows:
            bym.setdefault(d["method"], []).append(d)
        lines.append("| method | risk_final | risk_v4 | risk_mag | jacc_final_v4 | jacc_final_mag |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for method, ds in bym.items():
            def dmean(k):
                vals = [float(x[k]) for x in ds if x.get(k, "") != ""]
                return sum(vals) / max(len(vals), 1)
            lines.append(
                f"| {method} | "
                f"{dmean('risk_final'):.4f} | "
                f"{dmean('risk_v4'):.4f} | "
                f"{dmean('risk_mag'):.4f} | "
                f"{dmean('jacc_final_v4'):.4f} | "
                f"{dmean('jacc_final_mag'):.4f} |"
            )

    lines.append("\nСмысл: `patch_improve` больше — лучше выучил конфликтный patch. `forget_loss > 0` — retain ухудшился.")
    lines.append("Для публикации ищем Pareto: высокий `patch_improve`, низкий `forget_loss`, низкая `density`.\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()

    print("Error-Wave Transformer All-Ideas Benchmark v1")
    print(f"model={args.model_name}")
    print(f"methods={args.methods}")
    print(f"patch_mode={args.patch_mode} steps={args.steps} seeds={args.seeds}")
    print(f"density={args.density_start}->{args.density_end} retain_penalty={args.retain_penalty}")
    print(f"cga_beta={args.cga_beta} drcf_decay={args.drcf_decay} drcf_alpha={args.drcf_alpha} drcf_strength={args.drcf_strength}")

    texts = load_retain_texts(args)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    rows = []
    diag_rows = []

    for seed in seeds:
        print("\n" + "=" * 100)
        print(f"SEED {seed}")
        for method in methods:
            print("\n" + "-" * 80)
            print(f"RUN METHOD: {method}")
            row, diag = run_method(method, seed, args, texts)
            rows.append(row)
            diag_rows.extend(diag)

    main_fields = [
        "seed", "method", "model", "patch_mode",
        "total_params", "trainable_params", "trainable_frac_total",
        "patch_before", "patch_after", "retain_before", "retain_after",
        "patch_improve", "forget_loss", "density_mean", "skip",
    ]
    diag_fields = [
        "seed", "method", "step",
        "density_target", "density_actual",
        "risk_final", "risk_v4", "risk_mag",
        "jacc_v4_mag", "jacc_final_v4", "jacc_final_mag",
        "conflict_frac_all", "orthogonal_coeff_mean",
    ]

    write_csv(args.results_csv, rows, main_fields)
    write_csv(args.diag_csv, diag_rows, diag_fields)
    write_summary(args.summary_md, rows, diag_rows)

    print(f"saved CSV: {args.results_csv}")
    print(f"saved diag: {args.diag_csv}")
    print(f"saved summary: {args.summary_md}")

    print("\nSUMMARY mean over seeds:")
    valid = [r for r in rows if not int(r.get("skip", 0))]
    by = {}
    for r in valid:
        by.setdefault(r["method"], []).append(r)
    for method, rs in by.items():
        def mean(k):
            return sum(float(x[k]) for x in rs) / max(len(rs), 1)
        print(
            f"  {method:18s} patch_improve={mean('patch_improve'):+.4f} "
            f"forget_loss={mean('forget_loss'):+.4f} "
            f"density={mean('density_mean'):.4f}"
        )


if __name__ == "__main__":
    main()
