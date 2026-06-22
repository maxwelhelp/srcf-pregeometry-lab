#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SRCF Real Data Benchmark v1
===========================

Реальная проверка SRCF без синтетических данных.

Что проверяем:
  1) Можно ли обучить SRCF только на нормальном классе без labels аномалий.
  2) Даёт ли closure-dynamics anomaly score пользу против обычных baseline:
     raw distance, PCA reconstruction, IsolationForest, SRCF embedding distance.

Датасеты встроены в scikit-learn, интернет не нужен:
  - breast_cancer: benign = normal, malignant = anomaly
  - digits: выбранная цифра = normal, остальные = anomaly
  - wine: выбранный класс = normal, остальные = anomaly

Данные превращаются в relation tensor R[i,j,c] между признаками/пикселями:
  канал 0: x_i * x_j
  канал 1: x_i - x_j
  канал 2: |x_i - x_j|
  канал 3: (x_i + x_j)/2
  канал 4: tanh(x_i * x_j)
  канал 5: corr_train[i,j] * (1 + 0.25*(x_i+x_j))
  канал 6: centered outer относительно normal mean
  канал 7: identity / diagonal anchor

Запуск:
  python -u srcf_realdata_benchmark_v1.py --device cuda --amp fp16 --dataset all \
    --pretrain-steps 200 --batch 16 --dim 48 --ops 8 --iters 6 \
    --results-csv results/realdata_summary.csv | tee srcf_realdata_v1.log
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(device: str, amp: str):
    if device.startswith("cuda") and amp != "none":
        dtype = torch.float16 if amp == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", enabled=False)


def normalize_rel(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu = x.mean(dim=(1, 2), keepdim=True)
    sd = x.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(eps)
    return ((x - mu) / sd).clamp(-6.0, 6.0)


def auroc_safe(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def aupr_safe(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(average_precision_score(y, s))
    except Exception:
        return float("nan")


class FactorizedClosureQuestionOp(nn.Module):
    """Memory-safe relation operator.

    Вместо огромного cat([rij,rji,row,col,...]) считает отдельные проекции и суммирует.
    """
    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        self.p_rij = nn.Linear(dim, dim, bias=False)
        self.p_rji = nn.Linear(dim, dim, bias=False)
        self.p_row = nn.Linear(dim, dim, bias=False)
        self.p_col = nn.Linear(dim, dim, bias=False)
        self.p_glob = nn.Linear(dim, dim, bias=False)
        self.p_comp = nn.Linear(dim, dim, bias=False)
        self.p_misc = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, dim * hidden_mult), nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
        )
        self.left = nn.Linear(dim, dim, bias=False)
        self.right = nn.Linear(dim, dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def _forward_impl(self, r: torch.Tensor) -> torch.Tensor:
        b, n, _, d = r.shape
        rij = r
        rji = r.transpose(1, 2)
        row = r.mean(dim=2, keepdim=True).expand(b, n, n, d)
        col = r.mean(dim=1, keepdim=True).expand(b, n, n, d)
        glob = r.mean(dim=(1, 2), keepdim=True).expand(b, n, n, d)
        comp = torch.tanh(self.left(row) * self.right(col))
        recip = rij * rji
        asym = rij - rji
        misc = torch.tanh(0.5 * recip + 0.25 * asym)
        x = (self.p_rij(rij) + self.p_rji(rji) + self.p_row(row) + self.p_col(col)
             + self.p_glob(glob) + self.p_comp(comp) + self.p_misc(misc)) / math.sqrt(7.0)
        return torch.sigmoid(self.gate) * self.ff(x)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(r)


class SRCF(nn.Module):
    def __init__(self, rel_dim: int = 8, dim: int = 48, n_ops: int = 8,
                 iters: int = 6, controller_temp: float = 1.0, checkpoint_ops: bool = True):
        super().__init__()
        self.rel_dim = rel_dim
        self.dim = dim
        self.n_ops = n_ops
        self.iters = iters
        self.controller_temp = controller_temp
        self.checkpoint_ops = checkpoint_ops
        self.encoder = nn.Sequential(nn.Linear(rel_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ops = nn.ModuleList([FactorizedClosureQuestionOp(dim) for _ in range(n_ops)])
        self.norm = nn.LayerNorm(dim)
        sd = dim * 10
        self.controller = nn.Sequential(nn.Linear(sd, dim * 2), nn.GELU(), nn.Linear(dim * 2, n_ops))

    def summary(self, r: torch.Tensor) -> torch.Tensor:
        mean = r.mean(dim=(1, 2))
        std = r.std(dim=(1, 2), unbiased=False)
        mx = r.amax(dim=(1, 2))
        mn = r.amin(dim=(1, 2))
        row_mean = r.mean(dim=2)
        col_mean = r.mean(dim=1)
        row_disp = row_mean.std(dim=1, unbiased=False)
        col_disp = col_mean.std(dim=1, unbiased=False)
        closure_r = (row_mean - col_mean).abs().mean(dim=1)
        sym = (r - r.transpose(1, 2)).abs().mean(dim=(1, 2))
        recip = (r * r.transpose(1, 2)).mean(dim=(1, 2))
        diag = torch.diagonal(r, dim1=1, dim2=2).mean(dim=-1)
        return torch.cat([mean, std, mx, mn, row_disp, col_disp, closure_r, sym, recip, diag], dim=-1)

    def descriptor(self, r: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.summary(r).float(), dim=-1)

    def step_once(self, r: torch.Tensor):
        logits = self.controller(self.summary(r)) / max(1e-4, self.controller_temp)
        w = F.softmax(logits, dim=-1)
        delta = torch.zeros_like(r)
        for k, op in enumerate(self.ops):
            if self.checkpoint_ops and self.training and r.requires_grad:
                dk = checkpoint(op, r, use_reentrant=False)
            else:
                dk = op(r)
            delta = delta + w[:, k].view(-1, 1, 1, 1) * dk
        out = self.norm(r + delta)
        dn = delta.float().pow(2).mean(dim=(1, 2, 3)).sqrt()
        return out, w, dn

    def forward(self, r_in: torch.Tensor, iters: Optional[int] = None):
        r = self.encoder(r_in)
        h0 = r
        steps = self.iters if iters is None else iters
        ws, dns = [], []
        for _ in range(steps):
            r, w, dn = self.step_once(r)
            ws.append(w)
            dns.append(dn)
        return h0, r, torch.stack(ws, dim=0), torch.stack(dns, dim=0)

    def continue_from_hidden(self, h: torch.Tensor, iters: int = 1):
        r = h
        ws, dns = [], []
        for _ in range(iters):
            r, w, dn = self.step_once(r)
            ws.append(w); dns.append(dn)
        return r, torch.stack(ws, dim=0), torch.stack(dns, dim=0)


@dataclass
class RealDatasetPack:
    name: str
    X_train_norm: np.ndarray
    X_cal_norm: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    corr: np.ndarray
    normal_mean: np.ndarray


def load_real_pack(name: str, seed: int, normal_class: int = 0, max_anom: int = 800) -> RealDatasetPack:
    if name == "breast_cancer":
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        # sklearn: 0 malignant, 1 benign. benign normal, malignant anomaly.
        y_anom = (ds.target == 0).astype(np.int64)
        normal_mask = y_anom == 0
    elif name == "digits":
        ds = load_digits()
        X = ds.data.astype(np.float32) / 16.0
        y_anom = (ds.target != normal_class).astype(np.int64)
        normal_mask = y_anom == 0
    elif name == "wine":
        ds = load_wine()
        X = ds.data.astype(np.float32)
        y_anom = (ds.target != normal_class).astype(np.int64)
        normal_mask = y_anom == 0
    else:
        raise ValueError(f"unknown dataset: {name}")

    rng = np.random.RandomState(seed)
    Xn = X[normal_mask]
    Xa = X[~normal_mask]
    if len(Xa) > max_anom:
        idx = rng.choice(len(Xa), size=max_anom, replace=False)
        Xa = Xa[idx]
    Xn_tr, Xn_te = train_test_split(Xn, test_size=0.35, random_state=seed)
    Xn_fit, Xn_cal = train_test_split(Xn_tr, test_size=0.35, random_state=seed + 1)
    scaler = StandardScaler().fit(Xn_fit)
    Xn_fit_s = scaler.transform(Xn_fit).astype(np.float32)
    Xn_cal_s = scaler.transform(Xn_cal).astype(np.float32)
    Xn_te_s = scaler.transform(Xn_te).astype(np.float32)
    Xa_s = scaler.transform(Xa).astype(np.float32)
    X_test = np.concatenate([Xn_te_s, Xa_s], axis=0)
    y_test = np.concatenate([np.zeros(len(Xn_te_s), dtype=np.int64), np.ones(len(Xa_s), dtype=np.int64)])
    p = rng.permutation(len(y_test))
    X_test, y_test = X_test[p], y_test[p]
    corr = np.corrcoef(Xn_fit_s, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    normal_mean = Xn_fit_s.mean(axis=0).astype(np.float32)
    return RealDatasetPack(name, Xn_fit_s, Xn_cal_s, X_test.astype(np.float32), y_test, scaler, corr, normal_mean)


def vectors_to_relations(X: np.ndarray, corr: np.ndarray, normal_mean: np.ndarray,
                         device: str, rel_dim: int = 8) -> torch.Tensor:
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    b, n = x.shape
    xi = x[:, :, None]
    xj = x[:, None, :]
    corr_t = torch.as_tensor(corr, dtype=torch.float32, device=device).unsqueeze(0).expand(b, -1, -1)
    nm = torch.as_tensor(normal_mean, dtype=torch.float32, device=device)
    ci = (x - nm)[..., None]
    cj = (x - nm)[:, None, :]
    eye = torch.eye(n, device=device).unsqueeze(0).expand(b, -1, -1)
    channels = [
        xi * xj,
        xi - xj,
        (xi - xj).abs(),
        0.5 * (xi + xj),
        torch.tanh(xi * xj),
        corr_t * (1.0 + 0.25 * torch.tanh(xi + xj)),
        ci * cj,
        eye,
    ]
    R = torch.stack(channels[:rel_dim], dim=-1)
    return normalize_rel(R)


class RelationSampler:
    def __init__(self, pack: RealDatasetPack, device: str, rel_dim: int):
        self.pack = pack
        self.device = device
        self.rel_dim = rel_dim
        self.n = pack.X_train_norm.shape[1]

    def sample_normal_vectors(self, batch: int) -> np.ndarray:
        idx = np.random.randint(0, len(self.pack.X_train_norm), size=batch)
        return self.pack.X_train_norm[idx]

    def relations_from_vectors(self, X: np.ndarray) -> torch.Tensor:
        return vectors_to_relations(X, self.pack.corr, self.pack.normal_mean, self.device, self.rel_dim)

    def sample_triplet(self, batch: int, noise_std: float):
        x = self.sample_normal_vectors(batch)
        x_near = x + noise_std * np.random.randn(*x.shape).astype(np.float32)
        x_far = self.sample_normal_vectors(batch)
        # ensure far not same draw, add tiny shuffle of batch order
        np.random.shuffle(x_far)
        return self.relations_from_vectors(x), self.relations_from_vectors(x_near), self.relations_from_vectors(x_far)


def pair_dist_desc(model: SRCF, h1: torch.Tensor, h2: torch.Tensor) -> torch.Tensor:
    return (model.descriptor(h1) - model.descriptor(h2)).pow(2).sum(dim=-1).sqrt()


def intrinsic_losses(model: SRCF, triplet, args):
    r0, rnear, rfar = triplet
    h0, h, w, dns = model(r0, args.iters)
    h0n, hn, _, _ = model(rnear, args.iters)
    h0f, hf, _, _ = model(rfar, args.iters)
    same_d = pair_dist_desc(model, h, hn)
    far_d = pair_dist_desc(model, h, hf)
    id_same = pair_dist_desc(model, h0, h0n).detach()
    id_far = pair_dist_desc(model, h0, h0f).detach()
    h_same = (h - hn).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    h_id_same = (h0 - h0n).float().pow(2).mean(dim=(1, 2, 3)).sqrt().detach()
    same_loss = same_d.pow(2).mean()
    contract_loss = F.relu(same_d - args.same_contract * id_same - args.contract_eps).pow(2).mean()
    hidden_contract_loss = F.relu(h_same - args.hidden_same_contract * h_id_same - args.contract_eps).pow(2).mean()
    sep_loss = F.relu(args.margin - far_d).pow(2).mean()
    far_preserve = F.relu(args.far_keep * id_far - far_d).pow(2).mean()
    h_next, _, _ = model.continue_from_hidden(h, iters=args.fixed_iters)
    fixed_loss = (h_next - h).float().pow(2).mean()
    h_pert = h.detach() + args.state_noise * torch.randn_like(h)
    h_rec, _, _ = model.continue_from_hidden(h_pert, iters=args.recovery_iters)
    rec_loss = (h_rec - h.detach()).float().pow(2).mean()
    move = (h - h0).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    move_loss = F.relu(args.min_move - move).pow(2).mean()
    desc = model.descriptor(h)
    dist_mat = torch.cdist(desc.float(), desc.float())
    mask = ~torch.eye(desc.shape[0], dtype=torch.bool, device=desc.device)
    batch_sep = F.relu(args.batch_margin - dist_mat[mask]).pow(2).mean() if desc.shape[0] > 1 else torch.zeros((), device=desc.device)
    curve = dns.mean(dim=1)
    early_loss = F.relu(args.min_early_delta - curve[0]).pow(2)
    decay_loss = F.relu(curve[-1] - args.delta_decay * curve[0]).pow(2)
    late_loss = F.relu(curve[-1] - args.max_late_delta).pow(2)
    state_var = h.float().var(dim=0, unbiased=False).mean()
    desc_var = desc.float().var(dim=0, unbiased=False).mean()
    var_loss = F.relu(args.state_var_floor - state_var).pow(2) + F.relu(args.desc_var_floor - desc_var).pow(2)
    w_mean = w.mean(dim=(0, 1))
    usage_loss = ((w_mean - 1.0 / model.n_ops) ** 2).mean()
    loss = (args.same_w * same_loss + args.contract_w * contract_loss
            + args.hidden_contract_w * hidden_contract_loss + args.sep_w * sep_loss
            + args.far_preserve_w * far_preserve + args.fixed_w * fixed_loss
            + args.recovery_w * rec_loss + args.move_w * move_loss
            + args.batch_sep_w * batch_sep + args.motion_w * early_loss
            + args.converge_w * decay_loss + args.late_w * late_loss
            + args.var_floor_w * var_loss + args.op_usage_w * usage_loss)
    stats = dict(
        loss=float(loss.detach().cpu()),
        contract=float((same_d / (id_same + 1e-8)).mean().detach().cpu()),
        h_contract=float((h_same / (h_id_same + 1e-8)).mean().detach().cpu()),
        far_keep=float((far_d / (id_far + 1e-8)).mean().detach().cpu()),
        move=float(move.mean().detach().cpu()),
        state_var=float(state_var.detach().cpu()),
        desc_var=float(desc_var.detach().cpu()),
        curve_start=float(curve[0].detach().cpu()),
        curve_end=float(curve[-1].detach().cpu()),
    )
    return loss, stats


def pretrain(model: SRCF, sampler: RelationSampler, args):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.device.startswith("cuda") and args.amp == "fp16"))
    t0 = time.time()
    last = {}
    for step in range(1, args.pretrain_steps + 1):
        triplet = sampler.sample_triplet(args.batch, args.input_noise)
        opt.zero_grad(set_to_none=True)
        with autocast_context(args.device, args.amp):
            loss, stats = intrinsic_losses(model, triplet, args)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        last = stats
        if step == 1 or step % max(1, args.pretrain_steps // 5) == 0:
            print(f"  step {step:4d}/{args.pretrain_steps} loss={stats['loss']:.4f} "
                  f"contract={stats['contract']:.2f} h_contract={stats['h_contract']:.2f} "
                  f"far_keep={stats['far_keep']:.2f} move={stats['move']:.3f} "
                  f"state_var={stats['state_var']:.3f} curve={stats['curve_start']:.3f}->{stats['curve_end']:.3f} "
                  f"t={time.time()-t0:.1f}s")
    return last


@torch.no_grad()
def closure_features(model: SRCF, R: torch.Tensor, args) -> np.ndarray:
    model.eval()
    h0, h, _, dns = model(R, args.iters)
    h_next, _, _ = model.continue_from_hidden(h, iters=args.fixed_iters)
    h_pert = h + args.state_noise * torch.randn_like(h)
    h_rec, _, _ = model.continue_from_hidden(h_pert, iters=args.recovery_iters)
    move = (h - h0).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    fixed = (h_next - h).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    recovery = (h_rec - h).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    state_var_per = h.float().var(dim=(1, 2, 3), unbiased=False).sqrt()
    desc = model.descriptor(h)
    desc_norm = desc.float().norm(dim=-1)
    curve_start = dns[0].float()
    curve_end = dns[-1].float()
    curve_decay = curve_end / (curve_start + 1e-8)
    # local perturbation contraction per sample
    Rn = R + args.input_noise * torch.randn_like(R)
    Rn = normalize_rel(Rn)
    h0n, hn, _, _ = model(Rn, args.iters)
    id_same = pair_dist_desc(model, h0, h0n)
    out_same = pair_dist_desc(model, h, hn)
    contract = out_same / (id_same + 1e-8)
    h_id_same = (h0 - h0n).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    h_same = (h - hn).float().pow(2).mean(dim=(1, 2, 3)).sqrt()
    h_contract = h_same / (h_id_same + 1e-8)
    feats = torch.stack([move, fixed, recovery, state_var_per, desc_norm, curve_start, curve_end, curve_decay, contract, h_contract], dim=1)
    return feats.cpu().numpy().astype(np.float32)


@torch.no_grad()
def srcf_descriptors(model: SRCF, R: torch.Tensor, args, batch: int = 64) -> np.ndarray:
    model.eval(); outs = []
    for i in range(0, R.shape[0], batch):
        _, h, _, _ = model(R[i:i+batch], args.iters)
        outs.append(model.descriptor(h).cpu().numpy())
    return np.concatenate(outs, axis=0)


def mad_calibrated_scores(feat_cal: np.ndarray, feat_eval: np.ndarray) -> np.ndarray:
    med = np.median(feat_cal, axis=0)
    mad = np.median(np.abs(feat_cal - med), axis=0) + 1e-6
    z = np.abs((feat_eval - med) / mad)
    return z.mean(axis=1)


def eval_scores(args, pack: RealDatasetPack, model: SRCF) -> Dict[str, Tuple[float, float]]:
    sampler = RelationSampler(pack, args.device, args.rel_dim)
    R_cal = sampler.relations_from_vectors(pack.X_cal_norm)
    R_test = sampler.relations_from_vectors(pack.X_test)
    # batching for features
    def feat_batch(R):
        fs = []
        for i in range(0, R.shape[0], args.eval_batch):
            fs.append(closure_features(model, R[i:i+args.eval_batch], args))
        return np.concatenate(fs, axis=0)
    feat_cal = feat_batch(R_cal)
    feat_test = feat_batch(R_test)
    closure_typ = mad_calibrated_scores(feat_cal, feat_test)
    closure_instab = feat_test[:, 2] + feat_test[:, 6]  # recovery + curve_end
    # embedding dist
    emb_cal = srcf_descriptors(model, R_cal, args, args.eval_batch)
    emb_test = srcf_descriptors(model, R_test, args, args.eval_batch)
    center = emb_cal.mean(axis=0)
    emb_dist = np.linalg.norm(emb_test - center, axis=1)
    # raw baseline
    raw_center = pack.X_train_norm.mean(axis=0)
    raw_dist = np.linalg.norm(pack.X_test - raw_center, axis=1)
    # PCA recon baseline
    n_comp = max(2, min(args.pca_components, pack.X_train_norm.shape[1] - 1, len(pack.X_train_norm) - 1))
    pca = PCA(n_components=n_comp, random_state=args.seed).fit(pack.X_train_norm)
    rec = pca.inverse_transform(pca.transform(pack.X_test))
    pca_err = ((pack.X_test - rec) ** 2).mean(axis=1)
    # IsolationForest baseline
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=args.seed).fit(pack.X_train_norm)
    iso_score = -iso.score_samples(pack.X_test)
    y = pack.y_test
    scores = {
        "closure_typicality": closure_typ,
        "closure_instability": closure_instab,
        "embedding_dist": emb_dist,
        "raw_dist": raw_dist,
        "pca_recon": pca_err,
        "isolation_forest": iso_score,
        "random": np.random.RandomState(args.seed).rand(len(y)),
    }
    out = {}
    for k, s in scores.items():
        auc_high = auroc_safe(y, s)
        auc_low = auroc_safe(y, -s)
        auc_best = max(auc_high, auc_low)
        ap_high = aupr_safe(y, s)
        out[k] = (auc_high, auc_low, auc_best, ap_high)
    return out


def run_dataset(args, name: str) -> Dict[str, Tuple[float, float]]:
    print("\n" + "=" * 72)
    print(f"REAL DATASET: {name}")
    print("=" * 72)
    pack = load_real_pack(name, args.seed, normal_class=args.normal_class, max_anom=args.max_anom)
    print(f"normal train={len(pack.X_train_norm)} cal={len(pack.X_cal_norm)} test={len(pack.X_test)} anomalies={int(pack.y_test.sum())} N_features={pack.X_train_norm.shape[1]}")
    sampler = RelationSampler(pack, args.device, args.rel_dim)
    model = SRCF(rel_dim=args.rel_dim, dim=args.dim, n_ops=args.ops, iters=args.iters,
                 controller_temp=args.controller_temp, checkpoint_ops=args.checkpoint_ops).to(args.device)
    print(f"model params={sum(p.numel() for p in model.parameters()):,}")
    print("pretrain SRCF only on normal class, no anomaly labels")
    train_stats = pretrain(model, sampler, args)
    print("extract/evaluate scores...")
    res = eval_scores(args, pack, model)
    print("\nAUROC high/low/best, AP(high):")
    for k, (hi, lo, best, ap) in res.items():
        print(f"  {k:20s}: high={hi:.4f} low={lo:.4f} best={best:.4f} AP={ap:.4f}")
    if args.results_csv:
        os.makedirs(os.path.dirname(args.results_csv) or ".", exist_ok=True)
        exists = os.path.exists(args.results_csv)
        with open(args.results_csv, "a", newline="", encoding="utf-8") as f:
            fieldnames = ["dataset", "score", "auroc_high", "auroc_low", "auroc_best", "ap_high", "pretrain_steps", "dim", "ops", "iters", "contract", "h_contract", "far_keep", "move", "state_var"]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                w.writeheader()
            for k, (hi, lo, best, ap) in res.items():
                w.writerow({"dataset": name, "score": k, "auroc_high": hi, "auroc_low": lo, "auroc_best": best, "ap_high": ap,
                            "pretrain_steps": args.pretrain_steps, "dim": args.dim, "ops": args.ops, "iters": args.iters,
                            "contract": train_stats.get("contract", float("nan")), "h_contract": train_stats.get("h_contract", float("nan")),
                            "far_keep": train_stats.get("far_keep", float("nan")), "move": train_stats.get("move", float("nan")),
                            "state_var": train_stats.get("state_var", float("nan"))})
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", choices=["none", "fp16", "bf16"], default="fp16")
    p.add_argument("--dataset", choices=["all", "breast_cancer", "digits", "wine"], default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normal-class", type=int, default=0, help="for digits/wine: class treated as normal")
    p.add_argument("--max-anom", type=int, default=800)
    p.add_argument("--rel-dim", type=int, default=8)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--ops", type=int, default=8)
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--fixed-iters", type=int, default=2)
    p.add_argument("--recovery-iters", type=int, default=4)
    p.add_argument("--controller-temp", type=float, default=1.0)
    p.add_argument("--checkpoint-ops", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pretrain-steps", type=int, default=200)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--eval-batch", type=int, default=32)
    p.add_argument("--input-noise", type=float, default=0.08)
    p.add_argument("--state-noise", type=float, default=0.08)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.65)
    p.add_argument("--batch-margin", type=float, default=0.35)
    p.add_argument("--same-contract", type=float, default=0.70)
    p.add_argument("--hidden-same-contract", type=float, default=0.75)
    p.add_argument("--contract-eps", type=float, default=0.010)
    p.add_argument("--far-keep", type=float, default=0.75)
    p.add_argument("--min-move", type=float, default=0.10)
    p.add_argument("--min-early-delta", type=float, default=0.015)
    p.add_argument("--delta-decay", type=float, default=0.70)
    p.add_argument("--max-late-delta", type=float, default=0.25)
    p.add_argument("--state-var-floor", type=float, default=0.10)
    p.add_argument("--desc-var-floor", type=float, default=1e-4)
    p.add_argument("--same-w", type=float, default=1.5)
    p.add_argument("--contract-w", type=float, default=5.0)
    p.add_argument("--hidden-contract-w", type=float, default=1.5)
    p.add_argument("--sep-w", type=float, default=1.0)
    p.add_argument("--far-preserve-w", type=float, default=0.7)
    p.add_argument("--fixed-w", type=float, default=0.2)
    p.add_argument("--recovery-w", type=float, default=0.3)
    p.add_argument("--move-w", type=float, default=0.2)
    p.add_argument("--batch-sep-w", type=float, default=0.3)
    p.add_argument("--motion-w", type=float, default=0.1)
    p.add_argument("--converge-w", type=float, default=0.2)
    p.add_argument("--late-w", type=float, default=0.05)
    p.add_argument("--var-floor-w", type=float, default=0.1)
    p.add_argument("--op-usage-w", type=float, default=0.003)
    p.add_argument("--pca-components", type=int, default=8)
    p.add_argument("--results-csv", type=str, default="results/realdata_summary.csv")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    datasets = ["breast_cancer", "digits", "wine"] if args.dataset == "all" else [args.dataset]
    print("SRCF Real Data Benchmark v1")
    print(f"device={args.device} amp={args.amp} datasets={datasets} dim={args.dim} ops={args.ops} iters={args.iters} pretrain={args.pretrain_steps}")
    all_res = {}
    for ds in datasets:
        all_res[ds] = run_dataset(args, ds)
    print("\n" + "=" * 72)
    print("ИТОГ: лучший AUROC по датасетам")
    print("=" * 72)
    for ds, res in all_res.items():
        best_name, best_val = max(((k, v[2]) for k, v in res.items()), key=lambda x: x[1])
        clo = res.get("closure_typicality", (float('nan'), float('nan'), float('nan'), float('nan')))[2]
        emb = res.get("embedding_dist", (float('nan'), float('nan'), float('nan'), float('nan')))[2]
        raw = res.get("raw_dist", (float('nan'), float('nan'), float('nan'), float('nan')))[2]
        print(f"{ds:14s}: best={best_name}:{best_val:.4f} | closure={clo:.4f} embedding={emb:.4f} raw={raw:.4f}")
    print("\nГотово.")


if __name__ == "__main__":
    main()
