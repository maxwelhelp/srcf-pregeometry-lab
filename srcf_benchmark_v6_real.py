#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRCF Benchmark v6 real/two-sided
================================

Русская версия benchmark для SRCF.

Что проверяет:
1) synthetic-hard anomaly:
   - искусственные relation-тензоры без координат;
   - аномалии сделаны тонкими: слабое повреждение, сломанная направленность,
     химеры строк/столбцов, слабое смешивание каналов;
   - главный score теперь НЕ "чем выше instability, тем аномальнее", а
     two-sided typicality: аномалия может быть слишком нестабильной ИЛИ
     подозрительно сверхстабильной / слишком типичной / over-closed.

2) real DNA benchmark:
   - нормальные окна берутся из реальной E. coli последовательности;
   - relation matrix строится по 3-mer переходам: N=64 узла;
   - сравниваем SRCF closure-score против простых baseline:
       raw_summary_dist, embedding_dist, raw_flat_dist, kmer_freq_dist;
   - аномалии: shuffle, motif injection, chimera, reverse-complement,
     phage/cross sequence fallback.

Важно:
- SRCF pretrain без labels.
- Labels используются только для оценки AUROC.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from self_referential_closure_field_v6_basin_dna import (
    SelfReferentialClosureField,
    RelationSampler,
    intrinsic_losses,
    random_relation_batch_hard,
    perturb_relation,
    normalize_rel,
    autocast_context,
    set_seed,
    download_dna_sequence,
    dna_relation_from_seq,
    mutate_seq,
    shuffle_seq,
    revcomp,
)


def auroc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """AUROC без sklearn. labels: 0 normal, 1 anomaly."""
    y = labels.detach().cpu().float().view(-1)
    s = scores.detach().cpu().float().view(-1)
    pos = y == 1
    neg = y == 0
    n_pos = int(pos.sum().item())
    n_neg = int(neg.sum().item())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(s)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float32)
    rank_sum_pos = ranks[pos].sum().item()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def auc_triplet(labels: torch.Tensor, scores: torch.Tensor) -> Tuple[float, float, float]:
    hi = auroc(labels, scores)
    lo = 1.0 - hi
    return hi, lo, max(hi, lo)


def print_auc(name: str, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[str, float, float, float]:
    hi, lo, best = auc_triplet(labels, scores)
    print(f"  {name:26s}: high={hi:.4f} low={lo:.4f} best={best:.4f}")
    return name, hi, lo, best


def make_model(args):
    return SelfReferentialClosureField(
        rel_dim=args.rel_dim,
        dim=args.dim,
        n_ops=args.ops,
        iters=args.iters,
        controller_temp=args.controller_temp,
        checkpoint_ops=args.checkpoint_ops,
    ).to(args.device)


def pretrain(model, sampler: RelationSampler, args):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device.startswith("cuda") and args.amp == "fp16"))
    t0 = time.time()
    print(f"pretrain: steps={args.pretrain_steps} batch={args.batch} data={args.data} no labels")
    for step in range(1, args.pretrain_steps + 1):
        triplet = sampler.sample_triplet(args.batch)
        opt.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            loss, met = intrinsic_losses(model, triplet, args)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        if step == 1 or step % max(1, args.pretrain_steps // 5) == 0:
            print(
                f"  step {step:4d}/{args.pretrain_steps} "
                f"loss={met['loss']:.4f} contract={met['same_contract']:.2f} "
                f"h_contract={met['h_contract']:.2f} far_keep={met['far_keep']:.2f} "
                f"move={met['move']:.3f} state_var={met['state_var']:.3f} "
                f"curve={met['early_delta']:.3f}->{met['late_delta']:.3f} "
                f"t={time.time()-t0:.1f}s"
            )


@torch.no_grad()
def raw_relation_summary(r: torch.Tensor) -> torch.Tensor:
    mean = r.mean(dim=(1, 2))
    std = r.std(dim=(1, 2), unbiased=False)
    mx = r.amax(dim=(1, 2))
    mn = r.amin(dim=(1, 2))
    sym = 0.5 * (r + r.transpose(1, 2))
    anti = 0.5 * (r - r.transpose(1, 2))
    sym_e = sym.pow(2).mean(dim=(1, 2)).sqrt()
    anti_e = anti.pow(2).mean(dim=(1, 2)).sqrt()
    row = r.mean(dim=2)
    col = r.mean(dim=1)
    closure_res = (row - col).abs().mean(dim=1)
    return torch.cat([mean, std, mx, mn, sym_e, anti_e, closure_res], dim=-1).float()


@torch.no_grad()
def kmer_freq_proxy(r: torch.Tensor) -> torch.Tensor:
    # Для DNA relation матриц первый канал близок к transition i->j.
    # Для synthetic это просто baseline-статистика строки.
    x = r[..., 0].clamp_min(0)
    f = x.sum(dim=2)
    f = f / f.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return f.float()


@torch.no_grad()
def raw_flat(r: torch.Tensor) -> torch.Tensor:
    return r.float().flatten(1)


@torch.no_grad()
def descriptor_distance_to_calib(x: torch.Tensor, calib_x: torch.Tensor) -> torch.Tensor:
    center = calib_x.float().mean(dim=0, keepdim=True)
    return (x.float() - center).pow(2).sum(dim=-1).sqrt()


@torch.no_grad()
def closure_feature_matrix(model, data: torch.Tensor, args) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    model.eval()
    feats, descs, raw_sums, raw_flats, freqs = [], [], [], [], []
    raw: Dict[str, List[torch.Tensor]] = {}
    for i in range(0, data.shape[0], args.eval_batch):
        r = data[i:i + args.eval_batch]
        h0, h, w, dn = model(r, args.iters)

        # self-stability
        h2, _, dn2 = model.continue_from_hidden(h, iters=args.extra_iters)
        instability = (h2 - h).pow(2).mean(dim=(1, 2, 3)).sqrt()

        # recovery from state perturbation
        hp = h + args.state_noise * torch.randn_like(h)
        hr, _, _ = model.continue_from_hidden(hp, iters=args.recovery_iters)
        recovery = (hr - h).pow(2).mean(dim=(1, 2, 3)).sqrt()

        # near perturbation contraction, per sample
        rn = perturb_relation(r, args.input_noise)
        h0n, hn, _, _ = model(rn, args.iters)
        desc0 = model.descriptor(h0)
        desc0n = model.descriptor(h0n)
        desc = model.descriptor(h)
        descn = model.descriptor(hn)
        same_in = (desc0 - desc0n).pow(2).sum(dim=-1).sqrt()
        same_out = (desc - descn).pow(2).sum(dim=-1).sqrt()
        contract = same_out / same_in.clamp_min(args.contract_eps)

        h_in = (h0 - h0n).pow(2).mean(dim=(1, 2, 3)).sqrt()
        h_out = (h - hn).pow(2).mean(dim=(1, 2, 3)).sqrt()
        h_contract = h_out / h_in.clamp_min(args.contract_eps)

        move = (h - h0).pow(2).mean(dim=(1, 2, 3)).sqrt()
        state_var = h.float().var(dim=(1, 2, 3), unbiased=False)
        curve_start = dn[0].float()
        curve_end = dn[-1].float()
        curve_decay = curve_end / curve_start.clamp_min(1e-6)

        vals = dict(
            instability=instability,
            recovery=recovery,
            contract=contract,
            h_contract=h_contract,
            move=move,
            state_var=state_var,
            curve_start=curve_start,
            curve_end=curve_end,
            curve_decay=curve_decay,
            same_in=same_in,
            same_out=same_out,
            h_same_in=h_in,
            h_same_out=h_out,
        )
        for k, v in vals.items():
            raw.setdefault(k, []).append(v.detach().float())

        order = list(vals.keys())
        feats.append(torch.stack([vals[k].float() for k in order], dim=-1))
        descs.append(desc.detach().float())
        raw_sums.append(raw_relation_summary(r))
        raw_flats.append(raw_flat(r))
        freqs.append(kmer_freq_proxy(r))

    feats_t = torch.cat(feats, dim=0)
    raw_t = {k: torch.cat(v, dim=0) for k, v in raw.items()}
    raw_t["descriptor"] = torch.cat(descs, dim=0)
    raw_t["raw_summary"] = torch.cat(raw_sums, dim=0)
    raw_t["raw_flat"] = torch.cat(raw_flats, dim=0)
    raw_t["kmer_freq"] = torch.cat(freqs, dim=0)
    return feats_t, raw_t


def closure_typicality_score(normal_feats: torch.Tensor, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Robust scoring.

    distance_score:
      обычное отклонение от median normal.
    typicality_score:
      ловит и "слишком далеко", и "слишком идеально/слишком центрально".
      Это важно для SRCF, потому что часть аномалий over-closed:
      они имеют слишком низкую instability/recovery.
    """
    med = normal_feats.median(dim=0).values
    mad = (normal_feats - med).abs().median(dim=0).values.clamp_min(1e-5)
    z_norm = (normal_feats - med) / (1.4826 * mad)
    z = (feats - med) / (1.4826 * mad)

    abs_norm = z_norm.abs()
    abs_z = z.abs()

    distance_score = abs_z.mean(dim=-1)

    typical_abs = abs_norm.median(dim=0).values
    typical_mad = (abs_norm - typical_abs).abs().median(dim=0).values.clamp_min(1e-5)
    typicality_score = ((abs_z - typical_abs).abs() / (1.4826 * typical_mad)).mean(dim=-1)

    energy_norm = abs_norm.mean(dim=-1)
    energy = abs_z.mean(dim=-1)
    energy_med = energy_norm.median()
    energy_mad = (energy_norm - energy_med).abs().median().clamp_min(1e-5)
    energy_typicality = (energy - energy_med).abs() / (1.4826 * energy_mad)

    # Направленные сигналы: underclosed/overclosed.
    underclosed = distance_score
    overclosed = -energy  # high means suspiciously too central/closed

    return dict(
        closure_distance=distance_score,
        closure_typicality=typicality_score,
        closure_energy_typicality=energy_typicality,
        closure_underclosed=underclosed,
        closure_overclosed=overclosed,
    )


def corrupt_sparse(r: torch.Tensor, frac: float, std: float) -> torch.Tensor:
    x = r.clone()
    mask = (torch.rand_like(x) < frac).float()
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
    x = r.clone()
    ch = torch.arange(0, r.shape[-1], 2, device=r.device)
    x[..., ch] = (1.0 - strength) * x[..., ch] + strength * x[..., ch].transpose(1, 2)
    return normalize_rel(x)


def make_hard_anomaly_dataset(args) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    normal = random_relation_batch_hard(args.n_normal, args.n, args.rel_dim, args.device, args.structure_mix)
    base = random_relation_batch_hard(args.n_anom_per_type, args.n, args.rel_dim, args.device, args.structure_mix)
    other = random_relation_batch_hard(args.n_anom_per_type, args.n, args.rel_dim, args.device, args.structure_mix)

    chunks = [normal]
    labels = [torch.zeros(normal.shape[0], device=args.device)]
    types = ["normal"] * normal.shape[0]

    builders = [
        ("subtle_1pct", lambda: corrupt_sparse(base, 0.01, 1.0)),
        ("subtle_3pct", lambda: corrupt_sparse(base, 0.03, 1.0)),
        ("rowcol_chimera_10pct", lambda: rowcol_chimera(base, other, 0.10)),
        ("rowcol_chimera_20pct", lambda: rowcol_chimera(base, other, 0.20)),
        ("channel_mix_weak", lambda: channel_mix(base, 0.25)),
        ("direction_broken", lambda: broken_direction(base, 0.80)),
    ]
    for name, fn in builders:
        x = fn()
        chunks.append(x)
        labels.append(torch.ones(x.shape[0], device=args.device))
        types.extend([name] * x.shape[0])

    data = torch.cat(chunks, dim=0)
    y = torch.cat(labels, dim=0).long()
    p = torch.randperm(data.shape[0], device=args.device)
    data = data[p]
    y = y[p]
    types = [types[int(i)] for i in p.detach().cpu().tolist()]
    return data, y, types


def make_random_gc_matched(seq: str) -> str:
    from collections import Counter
    c = Counter(seq)
    bases = list("ACGT")
    total = max(1, sum(c.get(b, 0) for b in bases))
    probs = [c.get(b, 0) / total for b in bases]
    out = []
    for _ in range(len(seq)):
        r = random.random()
        acc = 0.0
        for b, p in zip(bases, probs):
            acc += p
            if r <= acc:
                out.append(b)
                break
    return "".join(out)


def inject_motif(seq: str, motif: str = "TATAAT", rate: float = 0.015) -> str:
    s = list(seq)
    step = max(1, len(motif))
    i = 0
    while i + len(motif) < len(s):
        if random.random() < rate:
            s[i:i+len(motif)] = list(motif)
            i += len(motif)
        else:
            i += step
    return "".join(s)


def sample_windows(seq: str, n: int, window: int) -> List[str]:
    out = []
    max_start = max(0, len(seq) - window - 1)
    for _ in range(n):
        st = random.randint(0, max_start)
        out.append(seq[st:st+window])
    return out


def seqs_to_relations(seqs: List[str], args) -> torch.Tensor:
    arr = [dna_relation_from_seq(s, args.kmer, args.rel_dim) for s in seqs]
    return torch.stack(arr, dim=0).to(args.device)


def load_phage_or_fallback(args) -> str:
    # Не обязательно нужен интернет: если NCBI не отвечает, fallback создаёт synthetic GC-matched.
    class A: pass
    a = A()
    a.dna_cache = f"./dna_NC_001416_{max(50000, args.dna_bases//2)//1000}k.fasta"
    a.dna_bases = max(50000, args.dna_bases // 2)
    a.dna_window = args.dna_window
    a.dna_accession = "NC_001416.1"  # bacteriophage lambda
    a.download_dna = args.download_dna
    a.seed = args.seed + 17
    try:
        return download_dna_sequence(a)
    except Exception:
        return ""


def make_real_dna_dataset(args) -> Tuple[torch.Tensor, torch.Tensor, List[str], torch.Tensor]:
    # Возвращает calib_normal, data, labels, types.
    args.data = "dna"
    seq = download_dna_sequence(args)
    phage = load_phage_or_fallback(args)

    calib_seqs = sample_windows(seq, args.n_calib, args.dna_window)
    normal_seqs = sample_windows(seq, args.n_eval, args.dna_window)

    types: List[str] = ["normal"] * len(normal_seqs)
    chunks = [seqs_to_relations(normal_seqs, args)]
    labels = [torch.zeros(len(normal_seqs), device=args.device)]

    # near sanity не в anomaly set; отдельно считаем.
    anom_builders = []

    base = sample_windows(seq, args.n_eval_per_type, args.dna_window)
    other = sample_windows(seq, args.n_eval_per_type, args.dna_window)

    anom_builders.append(("mono_shuffle", [shuffle_seq(s) for s in base]))
    anom_builders.append(("gc_matched_random", [make_random_gc_matched(s) for s in base]))
    anom_builders.append(("motif_inject", [inject_motif(s, rate=0.025) for s in base]))
    anom_builders.append(("reverse_complement", [revcomp(s) for s in base]))
    anom_builders.append(("chimera_50pct", [s[:len(s)//2] + t[len(t)//2:] for s, t in zip(base, other)]))

    if phage and len(phage) >= args.dna_window:
        phage_windows = sample_windows(phage, args.n_eval_per_type, args.dna_window)
    else:
        phage_windows = [make_random_gc_matched(s) for s in base]
    anom_builders.append(("phage_or_crossseq", phage_windows))

    for name, seqs in anom_builders:
        x = seqs_to_relations(seqs, args)
        chunks.append(x)
        labels.append(torch.ones(x.shape[0], device=args.device))
        types.extend([name] * x.shape[0])

    calib = seqs_to_relations(calib_seqs, args)
    data = torch.cat(chunks, dim=0)
    y = torch.cat(labels, dim=0).long()
    p = torch.randperm(data.shape[0], device=args.device)
    data = data[p]
    y = y[p]
    types = [types[int(i)] for i in p.detach().cpu().tolist()]
    return calib, data, y, types


def evaluate_dataset(model, calib, data, labels, types, args, task_name: str):
    print("extracting closure features...")
    calib_feats, calib_raw = closure_feature_matrix(model, calib, args)
    feats, raw = closure_feature_matrix(model, data, args)

    scores = closure_typicality_score(calib_feats, feats)
    scores["embedding_dist"] = descriptor_distance_to_calib(raw["descriptor"], calib_raw["descriptor"])
    scores["raw_summary_dist"] = descriptor_distance_to_calib(raw["raw_summary"], calib_raw["raw_summary"])
    scores["raw_flat_dist"] = descriptor_distance_to_calib(raw["raw_flat"], calib_raw["raw_flat"])
    scores["kmer_freq_dist"] = descriptor_distance_to_calib(raw["kmer_freq"], calib_raw["kmer_freq"])
    scores["random"] = torch.rand_like(labels.float())

    print("\nAUROC overall:")
    rows = []
    for name in [
        "closure_typicality",
        "closure_energy_typicality",
        "closure_distance",
        "closure_underclosed",
        "closure_overclosed",
        "instability",
        "recovery",
        "embedding_dist",
        "raw_summary_dist",
        "raw_flat_dist",
        "kmer_freq_dist",
        "random",
    ]:
        sc = scores[name] if name in scores else raw[name]
        metric_name, hi, lo, best = print_auc(name, labels, sc)
        rows.append((metric_name + "_high", hi))
        rows.append((metric_name + "_low", lo))
        rows.append((metric_name + "_best", best))

    print("\nPer-type AUROC high/low/best:")
    for t in sorted(set(x for x in types if x != "normal")):
        idx = torch.tensor([i for i, name in enumerate(types) if name == "normal" or name == t], device=args.device)
        yy = labels[idx]
        print(f"  {t}")
        for name in ["closure_typicality", "closure_energy_typicality", "embedding_dist", "raw_summary_dist", "raw_flat_dist", "kmer_freq_dist"]:
            sc = scores[name][idx]
            hi, lo, best = auc_triplet(yy, sc)
            print(f"    {name:26s}: {hi:.3f}/{lo:.3f}/{best:.3f}")

    print("\nСмысл:")
    print("  high — обычное направление score: выше = аномальнее")
    print("  low  — обратное направление: ниже = аномальнее")
    print("  best — разделимость вообще; если best высокий, сигнал есть, но может быть инвертирован")
    write_results(args, task_name, rows)


def task_anomaly(args):
    print("\n" + "="*72)
    print("TASK anomaly — synthetic hard, two-sided closure typicality")
    print("="*72)
    args.data = "synthetic"
    model = make_model(args)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")
    sampler = RelationSampler(args, args.device)
    pretrain(model, sampler, args)

    print("building synthetic hard anomaly dataset...")
    calib = random_relation_batch_hard(args.n_calib, args.n, args.rel_dim, args.device, args.structure_mix)
    data, labels, types = make_hard_anomaly_dataset(args)
    evaluate_dataset(model, calib, data, labels, types, args, "anomaly")


def task_dna(args):
    print("\n" + "="*72)
    print("TASK dna-real — real E.coli 3-mer relation anomalies")
    print("="*72)
    args.data = "dna"
    args.n = 4 ** args.kmer
    model = make_model(args)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")
    sampler = RelationSampler(args, args.device)
    pretrain(model, sampler, args)

    print("building real DNA benchmark dataset...")
    calib, data, labels, types = make_real_dna_dataset(args)
    evaluate_dataset(model, calib, data, labels, types, args, "dna_real")

    # near sanity: small mutations should look normal-like.
    seq = download_dna_sequence(args)
    normal = sample_windows(seq, args.n_eval, args.dna_window)
    near = [mutate_seq(s, args.dna_mut_rate) for s in normal]
    near_data = torch.cat([seqs_to_relations(normal, args), seqs_to_relations(near, args)], dim=0)
    near_labels = torch.cat([torch.zeros(len(normal), device=args.device), torch.ones(len(near), device=args.device)]).long()
    calib = seqs_to_relations(sample_windows(seq, args.n_calib, args.dna_window), args)
    cf, _ = closure_feature_matrix(model, calib, args)
    nf, _ = closure_feature_matrix(model, near_data, args)
    ns = closure_typicality_score(cf, nf)["closure_typicality"]
    print(f"\nNear-mutation sanity AUROC closure_typicality={auroc(near_labels, ns):.4f}")
    print("  Хорошо, если это НЕ высоко: малая мутация должна оставаться в нормальном basin.")


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

    p.add_argument("--data", choices=["synthetic", "dna"], default="synthetic")
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

    p.add_argument("--n-normal", type=int, default=256)
    p.add_argument("--n-anom-per-type", type=int, default=128)
    p.add_argument("--n-calib", type=int, default=192)
    p.add_argument("--n-eval", type=int, default=128)
    p.add_argument("--n-eval-per-type", type=int, default=96)

    p.add_argument("--structure-mix", type=float, default=0.70)
    p.add_argument("--input-noise", type=float, default=0.18)
    p.add_argument("--state-noise", type=float, default=0.08)

    # DNA
    p.add_argument("--dna-accession", type=str, default="NC_000913.3")
    p.add_argument("--dna-bases", type=int, default=200000)
    p.add_argument("--dna-window", type=int, default=2048)
    p.add_argument("--dna-cache", type=str, default="./dna_NC_000913_200k.fasta")
    p.add_argument("--download-dna", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--kmer", type=int, default=3)
    p.add_argument("--dna-mut-rate", type=float, default=0.040)
    p.add_argument("--mixed-dna-prob", type=float, default=0.5)

    # intrinsic loss hyperparams; must match model trainer.
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
    args = p.parse_args()
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return args


def main():
    args = parse_args()
    print(f"SRCF Benchmark v6 real/two-sided")
    print(f"device={args.device} amp={args.amp} task={args.task} n={args.n} dim={args.dim} ops={args.ops} iters={args.iters}")
    if args.task in ("anomaly", "all"):
        task_anomaly(args)
    if args.task in ("dna", "all"):
        task_dna(args)
    print("Done.")


if __name__ == "__main__":
    main()
