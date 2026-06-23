#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytic_primitives_v1_1.py
Small doctrine-safe fix over analytic_primitives_v1:
OMP candidate discovery over analytic primitives + closed-form LS refit over accepted typed ops only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence
import torch

from analytic_primitives_v1 import (
    Primitive, build_qk_primitives, build_vo_primitives, primitive_manifest,
    op_from_primitive, make_program, rel_err, energy
)

VERSION = "analytic_primitives_v1.1-omp-refit"


def _stack(prims: Sequence[Primitive], device="cpu") -> torch.Tensor:
    return torch.stack([p.matrix.to(device).float().reshape(-1) for p in prims], dim=1)


def _refit(target: torch.Tensor, prims: Sequence[Primitive], device="cpu"):
    target = target.float().to(device)
    if not prims:
        return torch.zeros_like(target), torch.empty(0, dtype=torch.float32, device=device)
    A = _stack(prims, device=device)
    y = target.reshape(-1, 1)
    sol = torch.linalg.lstsq(A, y).solution.squeeze(1).float()
    recon = (A @ sol).reshape_as(target).float()
    return recon, sol


def decode_omp_refit_analytic(target: torch.Tensor, primitives: Sequence[Primitive], block: str,
                              thresholds: Dict[str, Any], max_ops: int = 64, device="cpu"):
    target = target.float().to(device)
    total_energy = max(energy(target), 1e-12)
    min_coeff = float(thresholds.get("MIN_COEFF_ABS", 1e-6))
    min_drop = float(thresholds.get("MIN_MARGINAL_REL_DROP", 1e-4))
    min_frac = float(thresholds.get("MIN_EXPLAINED_ENERGY_FRAC", 1e-4))
    exact_tol = float(thresholds.get("SYNTHETIC_ROUNDTRIP_TOL", 1e-7))

    accepted: List[Primitive] = []
    used = set()
    recon = torch.zeros_like(target)
    residual = target - recon
    last_err = rel_err(recon, target)
    history = []

    for _ in range(max_ops):
        if last_err <= exact_tol:
            break
        best = None
        for p in primitives:
            if p.name in used:
                continue
            trial = accepted + [p]
            trial_recon, coeffs = _refit(target, trial, device=device)
            err = rel_err(trial_recon, target)
            drop = last_err - err
            explained_frac = (energy(residual) - energy(target - trial_recon)) / total_energy
            coeff = float(coeffs[-1].detach().cpu())
            if abs(coeff) < min_coeff:
                continue
            if (drop >= min_drop and explained_frac >= min_frac) or err <= exact_tol:
                score = (drop, -err, explained_frac, abs(coeff))
                if best is None or score > best[0]:
                    best = (score, p, trial_recon, coeffs, err, drop, explained_frac)
        if best is None:
            break
        _, p, recon, coeffs, last_err, drop, explained_frac = best
        accepted.append(p)
        used.add(p.name)
        residual = target - recon
        history.append({"primitive": p.name, "err": float(last_err), "drop": float(drop), "explained_frac": float(explained_frac)})

    recon, coeffs = _refit(target, accepted, device=device)
    kept = [(p, c) for p, c in zip(accepted, coeffs) if abs(float(c.detach().cpu())) >= min_coeff]
    if len(kept) != len(accepted):
        accepted = [p for p, c in kept]
        recon, coeffs = _refit(target, accepted, device=device)

    residual = target - recon
    ops = []
    prev_recon = torch.zeros_like(target)
    prev_err = rel_err(prev_recon, target)
    prev_res_energy = energy(target - prev_recon)
    for i, (p, c) in enumerate(zip(accepted, coeffs)):
        partial_recon, _ = _refit(target, accepted[:i+1], device=device)
        err_after = rel_err(partial_recon, target)
        res_energy_after = energy(target - partial_recon)
        ops.append(op_from_primitive(
            p, float(c.detach().cpu()), f"{block}_op_{i:03d}_{p.name}",
            max(0.0, prev_err - err_after), err_after, max(0.0, prev_res_energy - res_energy_after)
        ))
        prev_err = err_after
        prev_res_energy = res_energy_after

    metrics = {
        "target_energy": total_energy,
        "residual_energy": energy(residual),
        "residual_energy_ratio": energy(residual) / total_energy,
        "roundtrip_error": rel_err(recon, target),
        "typed_coverage": 1.0 - energy(residual) / total_energy,
        "ops_count": float(len(ops)),
        "global_refit_used": True,
        "global_refit_scope": "accepted_typed_ops_only",
        "candidate_history": history,
    }
    return ops, recon, metrics


# Backward-compatible name used by synthetic scripts.
decode_greedy_analytic = decode_omp_refit_analytic
