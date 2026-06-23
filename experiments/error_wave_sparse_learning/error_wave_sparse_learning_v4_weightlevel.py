#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error-Wave Sparse Learning v4 — activity-gated selectivity + first-layer weight masks

Идея теста:
  Full BP       : обычный backprop обновляет все веса.
  RandomSparse  : такая же плотность обновления, но случайные веса.
  ErrorWave     : loss считается обычный, но после backward градиент обрезается
                  только по "виновным" путям, найденным волной blame от ошибки.

Задача:
  1) pretrain: сеть учит несколько context-specific правил.
  2) patch: одно правило меняется только для context=0.
  3) проверяем:
     - как быстро модель выучила новое правило;
     - сколько забыла по старым context;
     - какая доля весов реально обновлялась.

Это НЕ доказывает биологическое обучение. Это минимальный честный тест:
  sparse targeted update vs full gradient vs random sparse gradient.
"""

import argparse
import copy
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- utils -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def amp_context(device: torch.device, amp: str):
    enabled = (amp == "fp16" and device.type == "cuda")
    return torch.cuda.amp.autocast(enabled=enabled)


def topk_mask_1d(score: torch.Tensor, frac: float, min_k: int = 1) -> torch.Tensor:
    """score [D] -> bool mask [D] selecting top frac."""
    D = score.numel()
    k = max(min_k, int(math.ceil(D * frac)))
    k = min(k, D)
    if k >= D:
        return torch.ones_like(score, dtype=torch.bool)
    idx = torch.topk(score, k=k, largest=True).indices
    m = torch.zeros_like(score, dtype=torch.bool)
    m[idx] = True
    return m


def grad_density(model: nn.Module) -> float:
    total = 0
    nz = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += g.numel()
        nz += (g.abs() > 0).sum().item()
    return nz / max(1, total)


# ----------------------------- data -----------------------------

@dataclass
class RuleSet:
    orig_w: torch.Tensor      # [C, block_dim]
    orig_b: torch.Tensor      # [C]
    patch_w: torch.Tensor     # [block_dim]
    patch_b: torch.Tensor


def make_rules(contexts: int, block_dim: int, seed: int, device: torch.device) -> RuleSet:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + 12345)
    orig_w = torch.randn(contexts, block_dim, generator=g)
    orig_w = orig_w / (orig_w.norm(dim=-1, keepdim=True) + 1e-6)
    orig_b = torch.randn(contexts, generator=g) * 0.05

    # patched rule for context 0: related but not identical; harder than pure flip.
    patch_w = -0.65 * orig_w[0] + 0.75 * torch.randn(block_dim, generator=g)
    patch_w = patch_w / (patch_w.norm() + 1e-6)
    patch_b = torch.randn((), generator=g) * 0.05
    return RuleSet(orig_w.to(device), orig_b.to(device), patch_w.to(device), patch_b.to(device))


def sample_batch(
    batch: int,
    contexts: int,
    block_dim: int,
    rules: RuleSet,
    device: torch.device,
    mode: str,
    only_context: int = -1,
    noise: float = 0.10,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    x = [context one-hot | all context blocks]
    label depends on active context's block.

    mode:
      orig  : all contexts use original rule
      patch : context 0 uses patched rule, others original
    """
    if only_context >= 0:
        c = torch.full((batch,), only_context, device=device, dtype=torch.long)
    else:
        c = torch.randint(0, contexts, (batch,), device=device)

    blocks = torch.randn(batch, contexts, block_dim, device=device)
    ctx_oh = F.one_hot(c, num_classes=contexts).float()

    active = blocks[torch.arange(batch, device=device), c]  # [B, block_dim]
    logits = torch.empty(batch, device=device)
    for ci in range(contexts):
        m = (c == ci)
        if not m.any():
            continue
        if mode == "patch" and ci == 0:
            logits[m] = active[m].matmul(rules.patch_w) + rules.patch_b
        else:
            logits[m] = (active[m] * rules.orig_w[ci]).sum(dim=-1) + rules.orig_b[ci]
    logits = logits + noise * torch.randn_like(logits)
    y = (logits > 0).long()
    x = torch.cat([ctx_oh, blocks.reshape(batch, contexts * block_dim)], dim=-1)
    return x, y, c


# ----------------------------- model -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, out_dim: int = 2):
        super().__init__()
        assert layers >= 2
        dims = [in_dim] + [hidden] * (layers - 1) + [out_dim]
        self.linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])

    def forward(self, x: torch.Tensor, return_acts: bool = False):
        acts = [x]
        h = x
        for i, lin in enumerate(self.linears):
            h = lin(h)
            if i < len(self.linears) - 1:
                h = F.relu(h)
            acts.append(h)
        if return_acts:
            return h, acts
        return h



def update_fisher_ema(model: MLP, decay: float = 0.98):
    """Online diagonal-Fisher-like input importance per layer: EMA of grad_w^2 summed over outputs."""
    if not hasattr(model, "_fisher_in_ema"):
        model._fisher_in_ema = [None for _ in model.linears]
    if not hasattr(model, "_fisher_weight_ema"):
        model._fisher_weight_ema = {}
    with torch.no_grad():
        for li, lin in enumerate(model.linears):
            if lin.weight.grad is None:
                continue
            gw2 = lin.weight.grad.detach().float().pow(2)
            fs = gw2.mean(dim=0)  # [in]
            old = model._fisher_in_ema[li]
            if old is None or old.shape != fs.shape:
                model._fisher_in_ema[li] = fs.clone()
            else:
                model._fisher_in_ema[li] = decay * old + (1.0 - decay) * fs
            oldw = model._fisher_weight_ema.get(li)
            model._fisher_weight_ema[li] = gw2.clone() if oldw is None or oldw.shape != gw2.shape else decay * oldw + (1.0 - decay) * gw2

def make_error_wave_masks(
    model: MLP,
    acts: List[torch.Tensor],
    logits: torch.Tensor,
    y: torch.Tensor,
    c: torch.Tensor,
    frac: float,
    input_frac: float,
    target_context: int = 0,
    retain_penalty: float = 0.5,
    ema_decay: float = 0.0,
    selectivity_power: float = 0.0,
    fisher_guard: float = 0.0,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Context-aware error-wave masks.

    Old v1 used score = mean(blame) over the whole batch, so it lost the
    difference between the patched context and the contexts we want to retain.

    New score:
      score = blame(context==target) - retain_penalty * blame(context!=target)

    Optional EMA stabilizes the mask over patch steps.
    """
    with torch.no_grad():
        probs = F.softmax(logits.float(), dim=-1)
        target = F.one_hot(y, num_classes=probs.shape[-1]).float()
        blame = (probs - target).abs()  # [B, out]
        c = c.detach()

        if ema_decay > 0 and not hasattr(model, "_wave_score_ema"):
            model._wave_score_ema = [None for _ in model.linears]

        masks_rev = []
        out_mask = torch.ones(model.linears[-1].out_features, device=logits.device, dtype=torch.bool)

        for li in reversed(range(len(model.linears))):
            lin = model.linears[li]
            Wabs = lin.weight.detach().abs()  # [out, in]
            inp = acts[li].detach().abs()     # [B, in]

            in_blame = blame.matmul(Wabs) * (inp + 0.05)  # [B, in]

            m_t = (c == target_context)
            m_r = ~m_t
            if m_t.any():
                score_target = in_blame[m_t].mean(dim=0)
            else:
                score_target = in_blame.mean(dim=0)
            if m_r.any():
                score_retain = in_blame[m_r].mean(dim=0)
            else:
                score_retain = torch.zeros_like(score_target)

            score = score_target - retain_penalty * score_retain
            # topk works with negative scores, but shifting improves numerical safety/logical clarity
            score = score - score.min() + 1e-8

            # v3/v4: context selectivity. Generalist neurons are bad patch candidates;
            # neurons mostly active on target context are safer to update.
            # v4 fixes dead-neuron issue: inactive neurons get no selectivity bonus.
            if selectivity_power > 0:
                act_abs = inp  # already abs
                if m_t.any():
                    act_t = act_abs[m_t].mean(dim=0)
                else:
                    act_t = act_abs.mean(dim=0)
                act_all = act_abs.mean(dim=0)
                sel_raw = act_t / (act_all + 1e-6)
                sel_raw = torch.clamp(sel_raw, 0.05, 5.0)
                sel_raw = sel_raw / (sel_raw.mean() + 1e-6)

                # Dead / nearly-dead neurons should NOT look specialized just because act_t≈act_all≈0.
                # active_gate in [0,1], near 0 below threshold, near 1 for clearly active neurons.
                thr = float(getattr(model, "_activity_threshold", 0.02))
                scale = act_all.mean().clamp_min(1e-6)
                active_gate = torch.clamp((act_all - thr * scale) / (scale + 1e-6), 0.0, 1.0)
                sel_bonus = 1.0 + active_gate * ((sel_raw ** selectivity_power) - 1.0)
                if getattr(model, "_dead_zero", False):
                    sel_bonus = sel_bonus * (active_gate > 0).float()
                score = score * sel_bonus
                sel = sel_bonus

                # lightweight diagnostic: overlap between top blame and top selectivity
                use_frac_tmp = input_frac if li == 0 else frac
                mb = topk_mask_1d(score, use_frac_tmp, min_k=1)
                ms = topk_mask_1d(sel, use_frac_tmp, min_k=1)
                denom = max(1, int(ms.sum().item()))
                if not hasattr(model, "_last_sel_overlap"):
                    model._last_sel_overlap = []
                model._last_sel_overlap.append(float((mb & ms).sum().item() / denom))

            # v3 optional Fisher guard: avoid weights/inputs that were important during pretrain.
            if fisher_guard > 0 and hasattr(model, "_fisher_in_ema"):
                fs = model._fisher_in_ema[li]
                if fs is not None and fs.shape == score.shape:
                    f = fs.to(score.device).float()
                    f = f / (f.mean() + 1e-8)
                    score = score / (1.0 + fisher_guard * f)

            if ema_decay > 0:
                old = model._wave_score_ema[li]
                if old is None or old.shape != score.shape:
                    ema = score.detach().clone()
                else:
                    ema = ema_decay * old + (1.0 - ema_decay) * score.detach()
                model._wave_score_ema[li] = ema
                score_for_mask = ema
            else:
                score_for_mask = score

            use_frac = input_frac if li == 0 else frac
            in_mask = topk_mask_1d(score_for_mask, use_frac, min_k=1)
            masks_rev.append((out_mask.clone(), in_mask.clone()))

            # propagate only through selected inputs to previous layer
            blame = in_blame * in_mask.float().unsqueeze(0)
            out_mask = in_mask.clone()

        return list(reversed(masks_rev))


def make_first_layer_weight_mask_v4(model: MLP, rules: RuleSet, args, device: torch.device, frac: float) -> torch.Tensor:
    """
    Weight-level mask for the first layer only.
    Uses two separate gradient probes:
      score = |grad_c0| * context_input_activity - retain_penalty * |grad_retain|
    This is more precise than neuron outer-product masks for context-specific inputs.
    """
    was_training = model.training
    model.eval()
    lin0 = model.linears[0]
    old_grads = [None if p.grad is None else p.grad.detach().clone() for p in model.parameters()]
    for p in model.parameters():
        p.grad = None

    # Target context=0 patched loss
    x0, y0, _ = sample_batch(args.mask_batch, args.contexts, args.block_dim, rules, device, "patch", only_context=args.patch_context)
    with amp_context(device, args.amp):
        l0 = F.cross_entropy(model(x0), y0)
    l0.backward()
    g0 = lin0.weight.grad.detach().abs().float().clone()
    act0 = x0.detach().abs().mean(dim=0).float()

    for p in model.parameters():
        p.grad = None

    # Retain contexts original loss
    xr, yr, _ = sample_batch(args.mask_batch, args.contexts, args.block_dim, rules, device, "orig", only_context=-1)
    # remove target context samples if any; if all target by chance, keep mixed fallback
    # sample_batch mixed has random contexts, so enough at mask_batch=128.
    with amp_context(device, args.amp):
        lr = F.cross_entropy(model(xr), yr)
    lr.backward()
    gr = lin0.weight.grad.detach().abs().float().clone()

    score = g0 * (act0.unsqueeze(0) + 0.05) - args.retain_penalty * gr
    score = score - score.min() + 1e-8

    # Optional Fisher guard: suppress first-layer weights important during pretrain.
    if args.fisher_guard > 0 and hasattr(model, "_fisher_weight_ema"):
        fw = getattr(model, "_fisher_weight_ema", {}).get(0, None)
        if fw is not None and fw.shape == score.shape:
            f = fw.to(score.device).float()
            f = f / (f.mean() + 1e-8)
            score = score / (1.0 + args.fisher_guard * f)

    total = score.numel()
    k = max(1, int(math.ceil(total * frac)))
    idx = torch.topk(score.flatten(), k=k, largest=True).indices
    mask = torch.zeros_like(score, dtype=torch.bool).flatten()
    mask[idx] = True
    mask = mask.view_as(score)

    # restore grads
    for p, g in zip(model.parameters(), old_grads):
        p.grad = g
    if was_training:
        model.train()
    return mask

def apply_wave_masks(model: MLP, masks: List[Tuple[torch.Tensor, torch.Tensor]], first_weight_mask: torch.Tensor = None) -> float:
    with torch.no_grad():
        for lin, (out_m, in_m) in zip(model.linears, masks):
            if lin.weight.grad is not None:
                wm = out_m.float().unsqueeze(1) * in_m.float().unsqueeze(0)
                if first_weight_mask is not None and lin is model.linears[0]:
                    wm = first_weight_mask.to(wm.device, dtype=wm.dtype)
                lin.weight.grad.mul_(wm)
            if lin.bias is not None and lin.bias.grad is not None:
                lin.bias.grad.mul_(out_m.float())
    return grad_density(model)


def apply_random_sparse(model: MLP, density_target: float) -> float:
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is None:
                continue
            m = (torch.rand_like(p.grad.float()) < density_target).to(p.grad.dtype)
            p.grad.mul_(m)
    return grad_density(model)


# ----------------------------- train/eval -----------------------------

@torch.no_grad()
def eval_contexts(model: MLP, rules: RuleSet, args, device: torch.device) -> Dict[str, float]:
    model.eval()
    out = {}
    # orig accuracy for every context under original rules
    accs_orig = []
    for c in range(args.contexts):
        xs, ys, _ = sample_batch(args.eval_batch, args.contexts, args.block_dim, rules, device, "orig", only_context=c)
        pred = model(xs).argmax(dim=-1)
        accs_orig.append((pred == ys).float().mean().item())
        out[f"orig_c{c}"] = accs_orig[-1]

    # patched accuracy for context 0
    xs, ys, _ = sample_batch(args.eval_batch, args.contexts, args.block_dim, rules, device, "patch", only_context=0)
    pred = model(xs).argmax(dim=-1)
    out["patch_c0"] = (pred == ys).float().mean().item()
    out["retain_c1plus"] = float(np.mean(accs_orig[1:])) if args.contexts > 1 else accs_orig[0]
    out["orig_all"] = float(np.mean(accs_orig))
    return out


def train_step(model: MLP, opt, rules: RuleSet, args, device, mode: str, update: str, scaler=None, patch_step: int = 0) -> Tuple[float, float]:
    model.train()
    opt.zero_grad(set_to_none=True)
    data_mode = "orig" if mode == "pretrain" else "patch"
    only_context = -1 if mode == "pretrain" else 0
    x, y, c = sample_batch(args.batch, args.contexts, args.block_dim, rules, device, data_mode, only_context=only_context)

    with amp_context(device, args.amp):
        logits, acts = model(x, return_acts=True)
        loss = F.cross_entropy(logits, y)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
    else:
        loss.backward()

    if mode == "pretrain" and args.fisher_guard > 0:
        update_fisher_ema(model, args.fisher_ema)

    if update == "full":
        dens = grad_density(model)
    elif update in ("wave", "wave_sel", "wave_v4"):
        # During patching, compute the mask from a mixed-context batch so the wave can
        # prefer context-0 blame while explicitly avoiding neurons important for contexts 1..N.
        use_frac = args.patch_wave_frac if mode == "patch" else args.wave_frac
        if mode == "patch" and args.frac_decay and args.patch_steps > 1:
            # start larger, decay to patch_wave_frac by the end
            prog = max(0.0, min(1.0, (patch_step - 1) / max(1, args.patch_steps - 1)))
            use_frac = args.patch_wave_frac_end + (args.patch_wave_frac - args.patch_wave_frac_end) * (1.0 - prog)
        if mode == "patch" and args.context_blame:
            with torch.no_grad():
                xm, ym, cm = sample_batch(args.mask_batch, args.contexts, args.block_dim, rules, device, "patch", only_context=-1)
                lm, actsm = model(xm, return_acts=True)
            masks = make_error_wave_masks(
                model, actsm, lm.detach(), ym, cm,
                use_frac, args.input_frac,
                target_context=args.patch_context,
                retain_penalty=args.retain_penalty,
                ema_decay=args.wave_ema,
                selectivity_power=(args.selectivity_power if update in ("wave_sel", "wave_v4") else 0.0),
                fisher_guard=(args.fisher_guard if update in ("wave_sel", "wave_v4") else 0.0),
            )
        else:
            masks = make_error_wave_masks(
                model, acts, logits.detach(), y, c,
                use_frac, args.input_frac,
                target_context=args.patch_context,
                retain_penalty=args.retain_penalty,
                ema_decay=args.wave_ema,
                selectivity_power=(args.selectivity_power if update in ("wave_sel", "wave_v4") else 0.0),
                fisher_guard=(args.fisher_guard if update in ("wave_sel", "wave_v4") else 0.0),
            )
        first_wmask = None
        if update == "wave_v4" and mode == "patch" and args.weight_level_first:
            first_wmask = make_first_layer_weight_mask_v4(model, rules, args, device, max(0.001, use_frac * args.input_frac))
        dens = apply_wave_masks(model, masks, first_wmask)
    elif update == "random":
        # approximate density comparable to wave in the current phase.
        use_frac = args.patch_wave_frac if mode == "patch" else args.wave_frac
        if mode == "patch" and args.frac_decay and args.patch_steps > 1:
            # start larger, decay to patch_wave_frac by the end
            prog = max(0.0, min(1.0, (patch_step - 1) / max(1, args.patch_steps - 1)))
            use_frac = args.patch_wave_frac_end + (args.patch_wave_frac - args.patch_wave_frac_end) * (1.0 - prog)
        if mode == "patch" and args.random_patch_density > 0:
            density_target = args.random_patch_density
        else:
            density_target = max(0.001, min(1.0, use_frac * use_frac))
        dens = apply_random_sparse(model, density_target)
    else:
        raise ValueError(update)

    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    return loss.item(), dens



@torch.no_grad()
def context_specialization_score(model: MLP, rules: RuleSet, args, device: torch.device) -> Dict[str, float]:
    model.eval()
    per_ctx_acts = []
    for ci in range(args.contexts):
        x, _, _ = sample_batch(args.eval_batch, args.contexts, args.block_dim, rules, device, "orig", only_context=ci)
        _, acts = model(x, return_acts=True)
        per_ctx_acts.append([a.detach().abs().mean(dim=0).float() for a in acts[:-1]])
    out = {}
    for li in range(len(model.linears)):
        A = torch.stack([per_ctx_acts[ci][li] for ci in range(args.contexts)], dim=0)  # [C,D]
        all_mean = A.mean(dim=0)
        max_ratio = (A.max(dim=0).values / (all_mean + 1e-6))
        active = all_mean > (args.activity_threshold * all_mean.mean().clamp_min(1e-6))
        if active.any():
            out[f"spec_l{li}"] = max_ratio[active].mean().item()
            out[f"dead_l{li}"] = 1.0 - active.float().mean().item()
        else:
            out[f"spec_l{li}"] = 0.0
            out[f"dead_l{li}"] = 1.0
    out["spec_mean"] = float(np.mean([v for k,v in out.items() if k.startswith("spec_l")]))
    out["dead_mean"] = float(np.mean([v for k,v in out.items() if k.startswith("dead_l")]))
    return out

def fmt_metrics(m: Dict[str, float]) -> str:
    return f"orig={m['orig_all']:.3f} patch={m['patch_c0']:.3f} retain={m['retain_c1plus']:.3f}"


def run(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    rows = []

    for seed in args.seeds:
        set_seed(seed)
        rules = make_rules(args.contexts, args.block_dim, seed, device)
        in_dim = args.contexts + args.contexts * args.block_dim
        base = MLP(in_dim, args.hidden, args.layers).to(device)

        models = {
            "full_bp": copy.deepcopy(base),
            "random_sparse": copy.deepcopy(base),
            "error_wave_v2": copy.deepcopy(base),
            "error_wave_sel": copy.deepcopy(base),
            "error_wave_v4": copy.deepcopy(base),
        }
        for m in models.values():
            m._activity_threshold = args.activity_threshold
            m._dead_zero = args.dead_zero
        opts = {k: torch.optim.AdamW(v.parameters(), lr=args.lr, weight_decay=args.weight_decay) for k, v in models.items()}
        scalers = {k: torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16" and device.type == "cuda")) for k in models}

        n_params = sum(p.numel() for p in base.parameters())
        print(f"\n=== seed={seed} params={n_params:,} device={device} ===")
        print(f"task: pretrain original rules, then patch only context=0; measure retention on contexts 1..{args.contexts-1}")

        t0 = time.time()
        # pretrain
        for step in range(1, args.pretrain_steps + 1):
            dens_log = {}
            loss_log = {}
            for name, model in models.items():
                loss, dens = train_step(model, opts[name], rules, args, device, "pretrain", "full", scalers[name])
                dens_log[name] = dens
                loss_log[name] = loss
            if step == 1 or step % args.eval_every == 0 or step == args.pretrain_steps:
                print(f"pre {step:04d}/{args.pretrain_steps} t={time.time()-t0:.1f}s")
                for name, model in models.items():
                    m = eval_contexts(model, rules, args, device)
                    print(f"  {name:13s} loss={loss_log[name]:.4f} dens={dens_log[name]:.3f} {fmt_metrics(m)}")

        pre_metrics = {name: eval_contexts(model, rules, args, device) for name, model in models.items()}
        print("pretrain specialization diagnostics:")
        for name, model in models.items():
            sp = context_specialization_score(model, rules, args, device)
            print(f"  {name:13s} spec_mean={sp['spec_mean']:.3f} dead_mean={sp['dead_mean']:.3f}")

        # patch train only c0 new rule
        for step in range(1, args.patch_steps + 1):
            dens_log = {}
            loss_log = {}
            for name, model in models.items():
                upd = "full" if name == "full_bp" else ("random" if name == "random_sparse" else ("wave_v4" if name == "error_wave_v4" else ("wave_sel" if name == "error_wave_sel" else "wave")))
                loss, dens = train_step(model, opts[name], rules, args, device, "patch", upd, scalers[name], patch_step=step)
                dens_log[name] = dens
                loss_log[name] = loss
            if step == 1 or step % args.eval_every == 0 or step == args.patch_steps:
                print(f"patch {step:04d}/{args.patch_steps} t={time.time()-t0:.1f}s")
                for name, model in models.items():
                    m = eval_contexts(model, rules, args, device)
                    print(f"  {name:13s} loss={loss_log[name]:.4f} dens={dens_log[name]:.3f} {fmt_metrics(m)}")

        for name, model in models.items():
            post = eval_contexts(model, rules, args, device)
            pre = pre_metrics[name]
            row = {
                "seed": seed,
                "model": name,
                "pre_orig_all": pre["orig_all"],
                "post_orig_all": post["orig_all"],
                "post_patch_c0": post["patch_c0"],
                "post_retain_c1plus": post["retain_c1plus"],
                "forget_c1plus": pre["retain_c1plus"] - post["retain_c1plus"],
            }
            rows.append(row)

        print("\nFINAL seed", seed)
        for r in rows[-5:]:
            print(
                f"  {r['model']:13s} patch={r['post_patch_c0']:.3f} retain={r['post_retain_c1plus']:.3f} "
                f"forget={r['forget_c1plus']:+.3f} orig={r['post_orig_all']:.3f}"
            )

    # summary
    keys = ["post_patch_c0", "post_retain_c1plus", "forget_c1plus", "post_orig_all"]
    print("\n=== SUMMARY mean±std ===")
    for model in ["full_bp", "random_sparse", "error_wave_v2", "error_wave_sel", "error_wave_v4"]:
        rs = [r for r in rows if r["model"] == model]
        parts = []
        for k in keys:
            vals = np.array([r[k] for r in rs], dtype=np.float32)
            parts.append(f"{k}={vals.mean():.3f}±{vals.std():.3f}")
        print(f"{model:13s} " + " | ".join(parts))

    with open(args.results_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"saved {args.results_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", default="fp16", choices=["off", "fp16"])
    p.add_argument("--seeds", default="0", type=str)
    p.add_argument("--pretrain-steps", type=int, default=400)
    p.add_argument("--patch-steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--contexts", type=int, default=4)
    p.add_argument("--block-dim", type=int, default=12)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--wave-frac", type=float, default=0.25)
    p.add_argument("--patch-wave-frac", type=float, default=0.15)
    p.add_argument("--input-frac", type=float, default=0.35)
    p.add_argument("--context-blame", action="store_true", help="use mixed-context target-minus-retain blame for wave masks during patch")
    p.add_argument("--patch-context", type=int, default=0)
    p.add_argument("--retain-penalty", type=float, default=0.5)
    p.add_argument("--wave-ema", type=float, default=0.90)
    p.add_argument("--selectivity-power", type=float, default=1.0)
    p.add_argument("--fisher-guard", type=float, default=0.0)
    p.add_argument("--activity-threshold", type=float, default=0.05)
    p.add_argument("--dead-zero", action="store_true")
    p.add_argument("--weight-level-first", action="store_true")
    p.add_argument("--frac-decay", action="store_true")
    p.add_argument("--patch-wave-frac-end", type=float, default=0.04)
    p.add_argument("--fisher-ema", type=float, default=0.98)
    p.add_argument("--random-patch-density", type=float, default=-1.0)
    p.add_argument("--mask-batch", type=int, default=128)
    p.add_argument("--results-csv", default="results/error_wave_sparse_learning_v3.csv")
    a = p.parse_args()
    a.seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    return a


if __name__ == "__main__":
    run(parse_args())
