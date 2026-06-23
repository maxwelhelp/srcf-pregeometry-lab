#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Error-Wave Sparse Learning v8 — CGA vs v4

CGA = Causal Gradient Arbitration.

Проверяем идею:
  v4 выбирает веса по score = |g_patch| - penalty*|g_retain|
  CGA делает ещё один шаг:
    перед применением обновления проверяет направление:
      retain_change ≈ -lr * g_patch * g_retain

    если g_patch * g_retain > 0:
      update -g_patch уменьшает retain-loss или не конфликтует -> безопасно
    если g_patch * g_retain < 0:
      update -g_patch повышает retain-loss -> конфликт

Методы:
  full_bp
  random_sparse
  magnitude_sparse
  error_wave_v4
  cga_soft
  cga_hard_refill

Главные метрики:
  patch_gain  — насколько хорошо выучили новый context=0
  retain_drop — насколько сломали старые contexts 1..C-1; меньше лучше
  density     — доля реально обновлённых весов
  conflict_fraction — сколько v4-кандидатов CGA считает конфликтными
"""

import argparse
import csv
import copy
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="fp16", choices=["off", "fp16", "bf16"])
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--pretrain-steps", type=int, default=400)
    p.add_argument("--patch-steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--mask-batch", type=int, default=128)
    p.add_argument("--eval-batch", type=int, default=512)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=50)

    p.add_argument("--contexts", type=int, default=4)
    p.add_argument("--block-dim", type=int, default=12)
    p.add_argument("--hidden", type=int, default=96)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)

    p.add_argument("--patch-wave-frac", type=float, default=0.08)
    p.add_argument("--patch-wave-frac-end", type=float, default=0.04)
    p.add_argument("--random-patch-density", type=float, default=0.014)
    p.add_argument("--frac-decay", action="store_true")
    p.add_argument("--retain-penalty", type=float, default=0.5)
    p.add_argument("--global-topk", action="store_true")

    p.add_argument("--cga-beta", type=float, default=3.0)
    p.add_argument("--cga-zero-retain-safe", action="store_true", default=True)
    p.add_argument("--cga-eps", type=float, default=1e-12)

    p.add_argument("--results-csv", type=str, default="results/error_wave_v8_cga.csv")
    p.add_argument("--diag-csv", type=str, default="results/error_wave_v8_cga_diag.csv")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(args):
    if args.amp == "fp16":
        return torch.float16
    if args.amp == "bf16":
        return torch.bfloat16
    return None


class ContextRuleTask:
    def __init__(self, contexts: int, block_dim: int, device: torch.device, seed: int):
        assert contexts >= 2
        g = torch.Generator(device="cpu")
        g.manual_seed(1000 + seed)

        self.contexts = contexts
        self.block_dim = block_dim
        self.device = device
        self.input_dim = contexts + contexts * block_dim

        self.rules_old = torch.randn(contexts, block_dim, generator=g)
        self.rules_old = self.rules_old / (self.rules_old.norm(dim=1, keepdim=True) + 1e-9)

        # новое правило для context=0 ортогонализуем относительно старого
        r = torch.randn(block_dim, generator=g)
        r = r - (r @ self.rules_old[0]) * self.rules_old[0]
        r = r / (r.norm() + 1e-9)
        self.rule_new0 = r

        self.rules_old = self.rules_old.to(device)
        self.rule_new0 = self.rule_new0.to(device)

    def _sample_contexts(self, batch: int, kind: str):
        if kind == "all":
            return torch.randint(0, self.contexts, (batch,), device=self.device)
        if kind == "patch0":
            return torch.zeros(batch, device=self.device, dtype=torch.long)
        if kind == "retain":
            return torch.randint(1, self.contexts, (batch,), device=self.device)
        raise ValueError(kind)

    def sample(self, batch: int, label_mode: str, context_kind: str):
        ctx = self._sample_contexts(batch, context_kind)
        feats = torch.randn(batch, self.contexts, self.block_dim, device=self.device)

        x = torch.zeros(batch, self.input_dim, device=self.device)
        x[torch.arange(batch, device=self.device), ctx] = 1.0
        x[:, self.contexts:] = feats.reshape(batch, -1)

        active = feats[torch.arange(batch, device=self.device), ctx]

        if label_mode == "old":
            w = self.rules_old[ctx]
            y = ((active * w).sum(dim=1) > 0).long()

        elif label_mode == "patch":
            y = torch.empty(batch, device=self.device, dtype=torch.long)
            is0 = (ctx == 0)
            if is0.any():
                y[is0] = ((active[is0] * self.rule_new0).sum(dim=1) > 0).long()
            if (~is0).any():
                w = self.rules_old[ctx[~is0]]
                y[~is0] = ((active[~is0] * w).sum(dim=1) > 0).long()
        else:
            raise ValueError(label_mode)

        return x, y, ctx


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, layers: int):
        super().__init__()
        dims = [input_dim] + [hidden] * (layers - 1) + [2]
        self.net = nn.ModuleList()
        for a, b in zip(dims[:-1], dims[1:]):
            lin = nn.Linear(a, b)
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
            self.net.append(lin)

    def forward(self, x):
        h = x
        for i, lin in enumerate(self.net):
            h = lin(h)
            if i < len(self.net) - 1:
                h = F.gelu(h)
        return h


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def autocast_ctx(device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype is not None and device.type == "cuda"))


def loss_backward(model, x, y, dtype):
    with autocast_ctx(x.device, dtype):
        logits = model(x)
        loss = F.cross_entropy(logits, y)
    loss.backward()
    return loss


@torch.no_grad()
def eval_model(model, task, batch: int, batches: int):
    model.eval()

    def acc(label_mode, context_kind):
        vals = []
        for _ in range(batches):
            x, y, _ = task.sample(batch, label_mode, context_kind)
            pred = model(x).argmax(dim=-1)
            vals.append((pred == y).float().mean().item())
        return sum(vals) / len(vals)

    out = {
        "orig": acc("old", "all"),
        "patch": acc("patch", "patch0"),
        "retain": acc("old", "retain"),
    }
    model.train()
    return out


def get_grads(model) -> Dict[str, torch.Tensor]:
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def current_frac(step, total, start, end, decay):
    if not decay:
        return start
    if total <= 1:
        return end
    t = (step - 1) / (total - 1)
    return start * (1 - t) + end * t


def make_topk_masks(scores: Dict[str, torch.Tensor], frac: float, global_topk: bool):
    if global_topk:
        flat = torch.cat([s.reshape(-1) for s in scores.values()])
        k = max(1, int(frac * flat.numel()))
        if k >= flat.numel():
            return {n: torch.ones_like(s, dtype=torch.bool) for n, s in scores.items()}
        thr = torch.topk(flat, k=k, largest=True).values[-1]
        return {n: (s >= thr) for n, s in scores.items()}

    masks = {}
    for n, s in scores.items():
        flat = s.reshape(-1)
        k = max(1, int(frac * flat.numel()))
        if k >= flat.numel():
            masks[n] = torch.ones_like(s, dtype=torch.bool)
        else:
            thr = torch.topk(flat, k=k, largest=True).values[-1]
            masks[n] = (s >= thr)
    return masks


def apply_masks(model, masks, scales=None):
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        if n not in masks:
            p.grad.zero_()
            continue
        m = masks[n].to(p.grad.dtype)
        p.grad.mul_(m)
        if scales is not None and n in scales:
            p.grad.mul_(scales[n].to(p.grad.dtype))


def random_mask(model, frac):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_((torch.rand_like(p.grad) < frac).to(p.grad.dtype))


def grad_density(model):
    nz = total = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            nz += int((g != 0).sum().item())
            total += g.numel()
    return nz / max(total, 1)


def mask_density(masks):
    nz = sum(int(m.sum().item()) for m in masks.values())
    total = sum(m.numel() for m in masks.values())
    return nz / max(total, 1)


def jaccard(a, b):
    inter = union = 0
    for n in a:
        if n not in b:
            continue
        inter += int((a[n] & b[n]).sum().item())
        union += int((a[n] | b[n]).sum().item())
    return inter / max(union, 1)


def selected_risk(mask, gp, gr):
    ps, rs = [], []
    for n, m in mask.items():
        if n not in gp:
            continue
        p = gp[n].detach().abs()
        r = gr.get(n, torch.zeros_like(p)).detach().abs()
        if int(m.sum().item()) > 0:
            ps.append(p[m].mean())
            rs.append(r[m].mean())
    if not ps:
        return 0.0, 0.0, 0.0
    pmean = float(torch.stack(ps).mean().cpu())
    rmean = float(torch.stack(rs).mean().cpu())
    return pmean, rmean, rmean / max(pmean, 1e-12)


def cga_scales(patch_grads, retain_grads, args):
    scales = {}
    safe_masks = {}
    conflict_fracs = {}
    for n, gp in patch_grads.items():
        gr = retain_grads.get(n)
        if gr is None:
            scales[n] = torch.ones_like(gp)
            safe_masks[n] = torch.ones_like(gp, dtype=torch.bool)
            conflict_fracs[n] = 0.0
            continue

        prod = gp * gr
        # Для gradient descent update -gp:
        # retain loss delta ≈ -lr * gr * gp.
        # prod > 0 безопасно, prod < 0 конфликт.
        safe = (prod >= 0)
        if args.cga_zero_retain_safe:
            safe = safe | (gr.abs() < args.cga_eps)

        # scalar per-weight "cos": sign-like normalized product
        align = prod / (gp.abs() * gr.abs() + args.cga_eps)
        soft = torch.sigmoid(args.cga_beta * align)
        if args.cga_zero_retain_safe:
            soft = torch.where(gr.abs() < args.cga_eps, torch.ones_like(soft), soft)

        scales[n] = soft
        safe_masks[n] = safe
        conflict_fracs[n] = 1.0 - float(safe.float().mean().detach().cpu())
    return scales, safe_masks, conflict_fracs


def hard_refill_masks(v4_scores, safe_masks, frac, global_topk):
    safe_scores = {}
    for n, s in v4_scores.items():
        safe = safe_masks.get(n)
        if safe is None:
            safe_scores[n] = s
        else:
            safe_scores[n] = s * safe.to(s.dtype)
    return make_topk_masks(safe_scores, frac, global_topk)


def train_pretrain_step(model, opt, task, args, dtype):
    opt.zero_grad(set_to_none=True)
    x, y, _ = task.sample(args.batch, "old", "all")
    loss = loss_backward(model, x, y, dtype)
    opt.step()
    return float(loss.detach().cpu())


def train_patch_step(model, opt, method, task, args, dtype, step):
    frac = current_frac(step, args.patch_steps, args.patch_wave_frac, args.patch_wave_frac_end, args.frac_decay)

    if method == "full_bp":
        opt.zero_grad(set_to_none=True)
        x, y, _ = task.sample(args.batch, "patch", "patch0")
        loss = loss_backward(model, x, y, dtype)
        dens = grad_density(model)
        opt.step()
        return float(loss.detach().cpu()), dens, {}

    if method == "random_sparse":
        opt.zero_grad(set_to_none=True)
        x, y, _ = task.sample(args.batch, "patch", "patch0")
        loss = loss_backward(model, x, y, dtype)
        random_mask(model, args.random_patch_density)
        dens = grad_density(model)
        opt.step()
        return float(loss.detach().cpu()), dens, {}

    # retain gradients for v4/CGA
    opt.zero_grad(set_to_none=True)
    xr, yr, _ = task.sample(args.mask_batch, "old", "retain")
    loss_r = loss_backward(model, xr, yr, dtype)
    retain_grads = get_grads(model)

    # patch gradients
    opt.zero_grad(set_to_none=True)
    xp, yp, _ = task.sample(args.batch, "patch", "patch0")
    loss = loss_backward(model, xp, yp, dtype)
    patch_grads = get_grads(model)

    mag_scores = {n: g.abs() for n, g in patch_grads.items()}
    mag_masks = make_topk_masks(mag_scores, frac, args.global_topk)

    v4_scores = {}
    for n, gp in patch_grads.items():
        gr = retain_grads.get(n)
        if gr is None:
            v4_scores[n] = gp.abs()
        else:
            v4_scores[n] = (gp.abs() - args.retain_penalty * gr.abs()).clamp_min(0.0)
    v4_masks = make_topk_masks(v4_scores, frac, args.global_topk)

    diag = {}

    if method == "magnitude_sparse":
        apply_masks(model, mag_masks)

    elif method == "error_wave_v4":
        apply_masks(model, v4_masks)

    elif method == "cga_soft":
        scales, safe_masks, conflict_fracs = cga_scales(patch_grads, retain_grads, args)
        apply_masks(model, v4_masks, scales=scales)
        diag = make_diag(v4_masks, v4_masks, mag_masks, patch_grads, retain_grads, safe_masks, conflict_fracs, "soft")

    elif method == "cga_hard_refill":
        scales, safe_masks, conflict_fracs = cga_scales(patch_grads, retain_grads, args)
        cga_masks = hard_refill_masks(v4_scores, safe_masks, frac, args.global_topk)
        apply_masks(model, cga_masks)
        diag = make_diag(cga_masks, v4_masks, mag_masks, patch_grads, retain_grads, safe_masks, conflict_fracs, "hard_refill")

    else:
        raise ValueError(method)

    dens = grad_density(model)
    opt.step()
    return float(loss.detach().cpu()), dens, diag


def make_diag(mask, v4_mask, mag_mask, patch_grads, retain_grads, safe_masks, conflict_fracs, mode):
    p, r, risk = selected_risk(mask, patch_grads, retain_grads)
    vp, vr, vrisk = selected_risk(v4_mask, patch_grads, retain_grads)
    mp, mr, mrisk = selected_risk(mag_mask, patch_grads, retain_grads)

    # conflict among selected
    sel_conf = []
    for n, m in mask.items():
        safe = safe_masks.get(n)
        if safe is not None and int(m.sum().item()) > 0:
            sel_conf.append((~safe[m]).float().mean())
    selected_conflict = float(torch.stack(sel_conf).mean().detach().cpu()) if sel_conf else 0.0

    return {
        "cga_mode": mode,
        "jacc_cga_v4": jaccard(mask, v4_mask),
        "jacc_cga_mag": jaccard(mask, mag_mask),
        "cga_patch_abs": p,
        "cga_retain_abs": r,
        "cga_risk": risk,
        "v4_risk": vrisk,
        "mag_risk": mrisk,
        "selected_conflict_frac": selected_conflict,
        "all_conflict_frac": sum(conflict_fracs.values()) / max(len(conflict_fracs), 1),
        "cga_density": mask_density(mask),
    }


def run_seed(seed, args):
    set_seed(seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    dtype = get_dtype(args)

    task = ContextRuleTask(args.contexts, args.block_dim, device, seed)
    base = MLP(task.input_dim, args.hidden, args.layers).to(device)
    opt = torch.optim.AdamW(base.parameters(), lr=args.lr)

    print(f"seed={seed} params={count_params(base):,} device={device}")

    t0 = time.time()
    print("pretraining base full_bp on old rules...")
    for step in range(1, args.pretrain_steps + 1):
        loss = train_pretrain_step(base, opt, task, args, dtype)
        if step == 1 or step % args.eval_every == 0 or step == args.pretrain_steps:
            ev = eval_model(base, task, args.eval_batch, args.eval_batches)
            print(f"pre {step:04d}/{args.pretrain_steps} loss={loss:.4f} orig={ev['orig']:.3f} patch={ev['patch']:.3f} retain={ev['retain']:.3f} t={time.time()-t0:.1f}s")

    methods = ["full_bp", "random_sparse", "magnitude_sparse", "error_wave_v4", "cga_soft", "cga_hard_refill"]
    models = {m: copy.deepcopy(base) for m in methods}
    opts = {m: torch.optim.AdamW(models[m].parameters(), lr=args.lr) for m in methods}
    before = {m: eval_model(models[m], task, args.eval_batch, args.eval_batches) for m in methods}
    dens_sum = {m: 0.0 for m in methods}
    diag_rows = []

    print("patching context=0...")
    for step in range(1, args.patch_steps + 1):
        for m in methods:
            loss, dens, diag = train_patch_step(models[m], opts[m], m, task, args, dtype, step)
            dens_sum[m] += dens
            if diag and (step == 1 or step % args.eval_every == 0 or step == args.patch_steps):
                diag_rows.append({"seed": seed, "step": step, "method": m, **diag})

        if step == 1 or step % args.eval_every == 0 or step == args.patch_steps:
            print(f"patch {step:04d}/{args.patch_steps} t={time.time()-t0:.1f}s")
            for m in methods:
                ev = eval_model(models[m], task, args.eval_batch, args.eval_batches)
                print(f"  {m:18s} patch={ev['patch']:.3f} retain={ev['retain']:.3f} orig={ev['orig']:.3f} dens={dens_sum[m]/step:.4f}")
            if diag_rows:
                latest = [d for d in diag_rows if d["step"] == step]
                for d in latest:
                    print(f"  DIAG {d['method']}: jacc_v4={d['jacc_cga_v4']:.3f} risk cga/v4/mag={d['cga_risk']:.3f}/{d['v4_risk']:.3f}/{d['mag_risk']:.3f} sel_conf={d['selected_conflict_frac']:.3f}")

    rows = []
    print("FINAL")
    for m in methods:
        after = eval_model(models[m], task, args.eval_batch, args.eval_batches)
        row = {
            "seed": seed,
            "method": m,
            "patch_before": before[m]["patch"],
            "patch_after": after["patch"],
            "retain_before": before[m]["retain"],
            "retain_after": after["retain"],
            "orig_after": after["orig"],
            "patch_gain": after["patch"] - before[m]["patch"],
            "retain_drop": before[m]["retain"] - after["retain"],
            "density": dens_sum[m] / args.patch_steps,
        }
        rows.append(row)
        print(f"  {m:18s} patch {row['patch_before']:.3f}->{row['patch_after']:.3f} gain={row['patch_gain']:+.3f} | retain {row['retain_before']:.3f}->{row['retain_after']:.3f} drop={row['retain_drop']:+.3f} | dens={row['density']:.4f}")

    return rows, diag_rows


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    args = parse_args()
    print("Error-Wave v8 CGA vs v4")
    print(f"seeds={args.seeds} contexts={args.contexts} block_dim={args.block_dim} hidden={args.hidden} layers={args.layers}")
    print(f"patch frac={args.patch_wave_frac}->{args.patch_wave_frac_end} retain_penalty={args.retain_penalty} cga_beta={args.cga_beta}")

    all_rows = []
    all_diag = []
    for seed in [int(x.strip()) for x in args.seeds.split(",") if x.strip()]:
        print("\n" + "=" * 90)
        rows, diag = run_seed(seed, args)
        all_rows.extend(rows)
        all_diag.extend(diag)

    result_fields = [
        "seed", "method",
        "patch_before", "patch_after",
        "retain_before", "retain_after", "orig_after",
        "patch_gain", "retain_drop", "density",
    ]
    diag_fields = [
        "seed", "step", "method", "cga_mode",
        "jacc_cga_v4", "jacc_cga_mag",
        "cga_patch_abs", "cga_retain_abs",
        "cga_risk", "v4_risk", "mag_risk",
        "selected_conflict_frac", "all_conflict_frac", "cga_density",
    ]

    write_csv(args.results_csv, all_rows, result_fields)
    write_csv(args.diag_csv, all_diag, diag_fields)
    print(f"saved {args.results_csv}")
    print(f"saved {args.diag_csv}")

    print("\nSUMMARY mean over seeds")
    by = {}
    for r in all_rows:
        by.setdefault(r["method"], []).append(r)
    for m, rs in by.items():
        def mean(k):
            return sum(float(x[k]) for x in rs) / len(rs)
        print(f"  {m:18s} patch_gain={mean('patch_gain'):+.4f} retain_drop={mean('retain_drop'):+.4f} density={mean('density'):.4f}")

    if all_diag:
        print("\nDIAG mean")
        bym = {}
        for d in all_diag:
            bym.setdefault(d["method"], []).append(d)
        for m, ds in bym.items():
            def mean(k):
                return sum(float(x[k]) for x in ds) / len(ds)
            print(f"  {m:18s} jacc_v4={mean('jacc_cga_v4'):.4f} risk={mean('cga_risk'):.4f} sel_conf={mean('selected_conflict_frac'):.4f}")


if __name__ == "__main__":
    main()
