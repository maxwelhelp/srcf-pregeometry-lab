#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRCF Benchmark v4 hard/calibrated
=================================

Purpose:
  Test whether Self-Referential Closure Field (SRCF) gives a useful signal beyond
  ordinary embedding distance.

Key change vs v2/v3:
  Anomaly score is calibrated against normal closure metric distributions.
  It detects values that are too high OR too low, because a collapsed anomaly can
  look suspiciously stable rather than unstable.

Tasks:
  anomaly  - hard synthetic relation anomalies close to normal data
  dna      - real E. coli k-mer relation states, normal vs shuffled/chimera states

Requires only torch. sklearn is optional; a local AUROC fallback is included.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from types import SimpleNamespace
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from self_referential_closure_field_v5_basin_dna import (
    SelfReferentialClosureField,
    RelationSampler,
    random_relation_batch_hard,
    normalize_rel,
    perturb_relation,
    intrinsic_losses,
    autocast_context,
    set_seed,
)

try:
    from sklearn.metrics import roc_auc_score as _sk_auc
except Exception:
    _sk_auc = None


def auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    y = labels.detach().cpu().float().view(-1)
    s = scores.detach().cpu().float().view(-1)
    if _sk_auc is not None:
        return float(_sk_auc(y.numpy(), s.numpy()))
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")
    # Mann-Whitney AUC with tie handling approximation.
    cmp = (pos[:, None] > neg[None, :]).float() + 0.5 * (pos[:, None] == neg[None, :]).float()
    return float(cmp.mean().item())


def raw_relation_summary(r: torch.Tensor) -> torch.Tensor:
    sym = 0.5 * (r + r.transpose(1, 2))
    anti = 0.5 * (r - r.transpose(1, 2))
    row = r.mean(dim=2)
    col = r.mean(dim=1)
    parts = [
        r.mean(dim=(1, 2)),
        r.std(dim=(1, 2), unbiased=False),
        sym.pow(2).mean(dim=(1, 2)).sqrt(),
        anti.pow(2).mean(dim=(1, 2)).sqrt(),
        row.std(dim=1, unbiased=False),
        col.std(dim=1, unbiased=False),
        (row - col).abs().mean(dim=1),
    ]
    return torch.cat(parts, dim=-1).float()


def descriptor_distance_to_train(x: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    center = train.mean(dim=0, keepdim=True)
    sd = train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    z = (x - center) / sd
    return z.pow(2).mean(dim=-1).sqrt()


def train_cfg(args) -> SimpleNamespace:
    return SimpleNamespace(
        iters=args.iters,
        detach_pairs=args.detach_pairs,
        fixed_iters=args.fixed_iters,
        recovery_iters=args.recovery_iters,
        state_noise=args.state_noise,
        margin=args.margin,
        batch_margin=args.batch_margin,
        same_contract=args.same_contract,
        same_abs_target=args.same_abs_target,
        contract_eps=args.contract_eps,
        hidden_same_contract=args.hidden_same_contract,
        far_keep=args.far_keep,
        min_move=args.min_move,
        min_early_delta=args.min_early_delta,
        delta_decay=args.delta_decay,
        max_late_delta=args.max_late_delta,
        same_w=args.same_w,
        contract_w=args.contract_w,
        contract_ratio_w=args.contract_ratio_w,
        hidden_same_w=args.hidden_same_w,
        hidden_contract_w=args.hidden_contract_w,
        sep_w=args.sep_w,
        far_preserve_w=args.far_preserve_w,
        fixed_w=args.fixed_w,
        recovery_w=args.recovery_w,
        move_w=args.move_w,
        batch_sep_w=args.batch_sep_w,
        motion_w=args.motion_w,
        converge_w=args.converge_w,
        late_w=args.late_w,
        op_usage_w=args.op_usage_w,
        op_sparse_w=args.op_sparse_w,
        state_var_floor=args.state_var_floor,
        desc_var_floor=args.desc_var_floor,
        var_floor_w=args.var_floor_w,
    )


def make_model(args) -> SelfReferentialClosureField:
    return SelfReferentialClosureField(
        rel_dim=args.rel_dim,
        dim=args.dim,
        n_ops=args.ops,
        iters=args.iters,
        controller_temp=args.controller_temp,
        checkpoint_ops=args.checkpoint_ops,
    ).to(args.device)


def pretrain(model, sampler: RelationSampler, args) -> None:
    cfg = train_cfg(args)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    t0 = time.time()
    print(f"pretrain: steps={args.pretrain_steps} batch={args.batch} data={args.data} no labels")
    for step in range(1, args.pretrain_steps + 1):
        model.train()
        triplet = sampler.sample_triplet(args.batch, n_override=args.n)
        opt.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            loss, metrics = intrinsic_losses(model, triplet, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if step == 1 or step % max(1, args.pretrain_steps // 5) == 0:
            curve = ",".join(f"{x:.3f}" for x in metrics["curve"][:4])
            print(
                f"  step {step:4d}/{args.pretrain_steps} loss={metrics['loss']:.4f} "
                f"contract={metrics['same_contract']:.2f} h_contract={metrics['h_contract']:.2f} "
                f"far_keep={metrics['far_keep']:.2f} move={metrics['move']:.3f} "
                f"state_var={metrics['state_var']:.3f} curve={curve} t={time.time()-t0:.1f}s"
            )


@torch.no_grad()
def closure_feature_matrix(model, data: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    cfg = train_cfg(args)
    feats: List[torch.Tensor] = []
    raw: Dict[str, List[torch.Tensor]] = {k: [] for k in [
        "instability", "recovery", "fixed", "move", "state_var", "desc_var",
        "curve_start", "curve_end", "curve_decay", "contract", "h_contract", "far_keep",
    ]}
    descs: List[torch.Tensor] = []
    raws: List[torch.Tensor] = []
    for i in range(0, data.shape[0], args.eval_batch):
        r = data[i:i+args.eval_batch]
        rn = perturb_relation(r, args.input_noise, renorm=True)
        rf = random_relation_batch_hard(r.shape[0], r.shape[1], args.rel_dim, args.device, args.structure_mix)
        _, m = intrinsic_losses(model, (r, rn, rf), cfg)
        h0, h, _, dn = model(r, args.iters)
        h2, _, _ = model.continue_from_hidden(h, iters=args.extra_iters)
        rec_pert = h + args.state_noise * torch.randn_like(h)
        hrec, _, _ = model.continue_from_hidden(rec_pert, iters=args.recovery_iters)
        desc = model.descriptor(h)
        descs.append(desc.float())
        raws.append(raw_relation_summary(r))

        inst = (h2 - h).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
        rec = (hrec - h).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
        fixed = inst
        move = torch.full_like(inst, float(m["move"]))
        state_var = h.float().var(dim=(1, 2)).mean(dim=-1)
        desc_raw = model.invariant_summary(h).float()
        desc_var = desc_raw.var(dim=-1, unbiased=False)
        curve_start = dn[0].float()
        curve_end = dn[-1].float()
        curve_decay = curve_end / curve_start.clamp_min(1e-6)
        # scalar train-time metrics repeated per sample; still useful for calibrated distribution.
        contract = torch.full_like(inst, float(m["same_contract"]))
        h_contract = torch.full_like(inst, float(m["h_contract"]))
        far_keep = torch.full_like(inst, float(m["far_keep"]))
        vals = dict(
            instability=inst, recovery=rec, fixed=fixed, move=move, state_var=state_var,
            desc_var=desc_var, curve_start=curve_start, curve_end=curve_end,
            curve_decay=curve_decay, contract=contract, h_contract=h_contract, far_keep=far_keep,
        )
        for k, v in vals.items():
            raw[k].append(v.detach().float())
        feat = torch.stack([vals[k] for k in raw.keys()], dim=-1)
        feats.append(feat)
    feats_t = torch.cat(feats, dim=0)
    raw_t = {k: torch.cat(v, dim=0) for k, v in raw.items()}
    raw_t["descriptor"] = torch.cat(descs, dim=0)
    raw_t["raw_summary"] = torch.cat(raws, dim=0)
    return feats_t, raw_t


def robust_calibrated_score(normal_feats: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
    med = normal_feats.median(dim=0).values
    mad = (normal_feats - med).abs().median(dim=0).values.clamp_min(1e-5)
    z = (feats - med).abs() / (1.4826 * mad)
    return z.mean(dim=-1)


def corrupt_sparse(r: torch.Tensor, frac: float, std: float) -> torch.Tensor:
    x = r.clone()
    b, n, _, c = x.shape
    mask = (torch.rand(b, n, n, c, device=x.device) < frac).float()
    x = x + mask * std * torch.randn_like(x)
    return normalize_rel(x)


def rowcol_chimera(a: torch.Tensor, b: torch.Tensor, frac: float) -> torch.Tensor:
    x = a.clone()
    B, N, _, C = x.shape
    k = max(1, int(N * frac))
    for i in range(B):
        idx = torch.randperm(N, device=x.device)[:k]
        x[i, idx, :, :] = b[i, idx, :, :]
        x[i, :, idx, :] = b[i, :, idx, :]
    return normalize_rel(x)


def channel_mix(r: torch.Tensor, strength: float) -> torch.Tensor:
    B, N, _, C = r.shape
    m = torch.eye(C, device=r.device)[None].repeat(B, 1, 1)
    m = m + strength * torch.randn_like(m)
    return normalize_rel(torch.einsum("bnmc,bcd->bnmd", r, m))


def broken_direction(r: torch.Tensor, strength: float) -> torch.Tensor:
    # Blend selected channels with their transpose: keeps marginal stats, weakens directed closure.
    x = r.clone()
    ch = torch.arange(0, r.shape[-1], 2, device=r.device)
    x[..., ch] = (1.0 - strength) * x[..., ch] + strength * x[..., ch].transpose(1, 2)
    return normalize_rel(x)


def make_hard_anomaly_dataset(args) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    normal = random_relation_batch_hard(args.n_normal, args.n, args.rel_dim, args.device, args.structure_mix)
    base = random_relation_batch_hard(args.n_anom_per_type, args.n, args.rel_dim, args.device, args.structure_mix)
    other = random_relation_batch_hard(args.n_anom_per_type, args.n, args.rel_dim, args.device, args.structure_mix)
    types = []
    chunks = [normal]
    labels = [torch.zeros(normal.shape[0], device=args.device)]

    anomaly_builders = [
        ("subtle_1pct", lambda: corrupt_sparse(base, 0.01, 1.0)),
        ("subtle_3pct", lambda: corrupt_sparse(base, 0.03, 1.0)),
        ("rowcol_chimera_10pct", lambda: rowcol_chimera(base, other, 0.10)),
        ("rowcol_chimera_20pct", lambda: rowcol_chimera(base, other, 0.20)),
        ("channel_mix_weak", lambda: channel_mix(base, 0.25)),
        ("direction_broken", lambda: broken_direction(base, 0.80)),
    ]
    for name, fn in anomaly_builders:
        x = fn()
        chunks.append(x)
        labels.append(torch.ones(x.shape[0], device=args.device))
        types.extend([name] * x.shape[0])
    types = ["normal"] * normal.shape[0] + types
    data = torch.cat(chunks, dim=0)
    y = torch.cat(labels, dim=0).long()
    p = torch.randperm(data.shape[0], device=args.device)
    data = data[p]
    y = y[p]
    types = [types[int(i)] for i in p.detach().cpu().tolist()]
    return data, y, types


def task_anomaly(args):
    print("\n" + "="*72)
    print("TASK anomaly — hard calibrated closure scores")
    print("="*72)
    model = make_model(args)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")
    sampler = RelationSampler(args, args.device)
    pretrain(model, sampler, args)

    print("building hard anomaly dataset...")
    calib = random_relation_batch_hard(args.n_calib, args.n, args.rel_dim, args.device, args.structure_mix)
    data, labels, types = make_hard_anomaly_dataset(args)
    calib_feats, calib_raw = closure_feature_matrix(model, calib, args)
    feats, raw = closure_feature_matrix(model, data, args)

    cal_score = robust_calibrated_score(calib_feats, feats)
    desc_score = descriptor_distance_to_train(raw["descriptor"], calib_raw["descriptor"])
    raw_score = descriptor_distance_to_train(raw["raw_summary"], calib_raw["raw_summary"])
    inst_score = raw["instability"]
    rec_score = raw["recovery"]
    rng_score = torch.rand_like(cal_score)

    rows = [
        ("calibrated_closure", auroc(labels, cal_score)),
        ("instability", auroc(labels, inst_score)),
        ("recovery", auroc(labels, rec_score)),
        ("embedding_dist", auroc(labels, desc_score)),
        ("raw_summary_dist", auroc(labels, raw_score)),
        ("random", auroc(labels, rng_score)),
    ]
    print("\nAUROC overall:")
    for k, v in rows:
        print(f"  {k:20s}: {v:.4f}")

    print("\nPer-type AUROC:")
    type_names = sorted(set(t for t in types if t != "normal"))
    for t in type_names:
        idx = torch.tensor([i for i, name in enumerate(types) if name == "normal" or name == t], device=args.device)
        yy = labels[idx]
        print(
            f"  {t:22s}: calib={auroc(yy, cal_score[idx]):.4f} "
            f"instab={auroc(yy, inst_score[idx]):.4f} "
            f"embed={auroc(yy, desc_score[idx]):.4f} raw={auroc(yy, raw_score[idx]):.4f}"
        )
    write_results(args, "anomaly", rows)


def task_dna(args):
    print("\n" + "="*72)
    print("TASK dna — real k-mer relation closure calibration")
    print("="*72)
    args.data = "dna"
    model = make_model(args)
    sampler = RelationSampler(args, args.device)
    pretrain(model, sampler, args)

    # Calibration normals: original windows. Evaluation: normal vs far windows from sampler.
    normals = []
    fars = []
    nears = []
    for _ in range(math.ceil(args.n_calib / args.eval_batch)):
        r0, rn, rf = sampler.sample_triplet(args.eval_batch, n_override=args.n)
        normals.append(r0); nears.append(rn); fars.append(rf)
    normal = torch.cat(normals, dim=0)[:args.n_calib]
    near = torch.cat(nears, dim=0)[:args.n_eval]
    far = torch.cat(fars, dim=0)[:args.n_eval]
    test_normal = normal[:args.n_eval]
    data = torch.cat([test_normal, far], dim=0)
    labels = torch.cat([torch.zeros(test_normal.shape[0], device=args.device), torch.ones(far.shape[0], device=args.device)]).long()

    calib_feats, calib_raw = closure_feature_matrix(model, normal, args)
    feats, raw = closure_feature_matrix(model, data, args)
    cal_score = robust_calibrated_score(calib_feats, feats)
    desc_score = descriptor_distance_to_train(raw["descriptor"], calib_raw["descriptor"])
    raw_score = descriptor_distance_to_train(raw["raw_summary"], calib_raw["raw_summary"])

    # Same-basin sanity: normal vs near should be close after SRCF.
    near_data = torch.cat([test_normal[:near.shape[0]], near], dim=0)
    near_labels = torch.cat([torch.zeros(near.shape[0], device=args.device), torch.ones(near.shape[0], device=args.device)]).long()
    near_feats, near_raw = closure_feature_matrix(model, near_data, args)
    near_score = robust_calibrated_score(calib_feats, near_feats)

    rows = [
        ("calibrated_closure_far", auroc(labels, cal_score)),
        ("embedding_dist_far", auroc(labels, desc_score)),
        ("raw_summary_dist_far", auroc(labels, raw_score)),
        ("near_should_be_low_auc", auroc(near_labels, near_score)),
    ]
    print("\nDNA AUROC:")
    for k, v in rows:
        print(f"  {k:24s}: {v:.4f}")
    print("  note: near_should_be_low_auc should NOT be high; near mutations should remain normal-like.")
    write_results(args, "dna", rows)


def write_results(args, task: str, rows: List[Tuple[str, float]]):
    if not args.results_csv:
        return
    os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
    new = not os.path.exists(args.results_csv)
    with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task", "metric", "value", "n", "dim", "ops", "iters", "pretrain_steps", "data"])
        for metric, val in rows:
            w.writerow([task, metric, f"{val:.6f}", args.n, args.dim, args.ops, args.iters, args.pretrain_steps, args.data])
    print(f"appended results: {args.results_csv}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    p.add_argument("--task", choices=["anomaly", "dna", "all"], default="anomaly")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--data", choices=["synthetic", "dna", "mixed"], default="synthetic")
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--rel-dim", type=int, default=8)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--ops", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--fixed-iters", type=int, default=2)
    p.add_argument("--recovery-iters", type=int, default=4)
    p.add_argument("--extra-iters", type=int, default=2)
    p.add_argument("--controller-temp", type=float, default=1.0)
    p.add_argument("--checkpoint-ops", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--detach-pairs", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--pretrain-steps", type=int, default=150)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--n-calib", type=int, default=128)
    p.add_argument("--n-normal", type=int, default=128)
    p.add_argument("--n-anom-per-type", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--structure-mix", type=float, default=0.70)
    p.add_argument("--input-noise", type=float, default=0.18)

    # DNA args used by RelationSampler.
    p.add_argument("--dna-accession", type=str, default="NC_000913.3")
    p.add_argument("--dna-bases", type=int, default=200000)
    p.add_argument("--dna-window", type=int, default=2048)
    p.add_argument("--dna-cache", type=str, default="./dna_NC_000913_200k.fasta")
    p.add_argument("--download-dna", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kmer", type=int, default=3)
    p.add_argument("--dna-mut-rate", type=float, default=0.040)
    p.add_argument("--mixed-dna-prob", type=float, default=0.5)

    # intrinsic loss hyperparams matched to v5 defaults.
    p.add_argument("--state-noise", type=float, default=0.08)
    p.add_argument("--margin", type=float, default=0.65)
    p.add_argument("--batch-margin", type=float, default=0.45)
    p.add_argument("--same-contract", type=float, default=0.70)
    p.add_argument("--same-abs-target", type=float, default=0.020)
    p.add_argument("--contract-eps", type=float, default=0.010)
    p.add_argument("--hidden-same-contract", type=float, default=0.75)
    p.add_argument("--far-keep", type=float, default=0.80)
    p.add_argument("--min-move", type=float, default=0.14)
    p.add_argument("--min-early-delta", type=float, default=0.020)
    p.add_argument("--delta-decay", type=float, default=0.65)
    p.add_argument("--max-late-delta", type=float, default=0.25)
    p.add_argument("--same-w", type=float, default=2.0)
    p.add_argument("--contract-w", type=float, default=8.0)
    p.add_argument("--contract-ratio-w", type=float, default=1.5)
    p.add_argument("--hidden-same-w", type=float, default=1.0)
    p.add_argument("--hidden-contract-w", type=float, default=2.0)
    p.add_argument("--sep-w", type=float, default=1.0)
    p.add_argument("--far-preserve-w", type=float, default=0.9)
    p.add_argument("--fixed-w", type=float, default=0.25)
    p.add_argument("--recovery-w", type=float, default=0.35)
    p.add_argument("--move-w", type=float, default=0.25)
    p.add_argument("--batch-sep-w", type=float, default=0.35)
    p.add_argument("--motion-w", type=float, default=0.15)
    p.add_argument("--converge-w", type=float, default=0.25)
    p.add_argument("--late-w", type=float, default=0.10)
    p.add_argument("--op-usage-w", type=float, default=0.004)
    p.add_argument("--op-sparse-w", type=float, default=0.001)
    p.add_argument("--state-var-floor", type=float, default=0.20)
    p.add_argument("--desc-var-floor", type=float, default=1e-4)
    p.add_argument("--var-floor-w", type=float, default=0.15)
    p.add_argument("--results-csv", default="results/summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(max(1, args.threads))
    set_seed(args.seed)
    print("SRCF Benchmark v4 hard/calibrated")
    print(f"device={args.device} amp={args.amp} task={args.task} n={args.n} dim={args.dim} ops={args.ops} iters={args.iters}")
    if args.task in ("anomaly", "all"):
        args.data = "synthetic"
        task_anomaly(args)
    if args.task in ("dna", "all"):
        task_dna(args)
    print("Done.")


if __name__ == "__main__":
    main()
