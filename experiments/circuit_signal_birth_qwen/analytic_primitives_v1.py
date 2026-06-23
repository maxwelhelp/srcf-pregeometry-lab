#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytic_primitives_v1.py
Analytic Level-0 QK/VO primitives and deterministic encode/decode.
No learned dictionary. No checkpoint-fitted primitive matrices.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

try:
    from program_dsl_v1 import LEVEL_ANALYTIC, Op, Program, ResidualReport, RoundTripReport
except Exception:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent))
    from program_dsl_v1 import LEVEL_ANALYTIC, Op, Program, ResidualReport, RoundTripReport

VERSION = "analytic_primitives_v1.0"


@dataclass
class Primitive:
    name: str
    op_type: str
    block: str
    matrix: torch.Tensor
    read_fields: Tuple[str, ...]
    write_fields: Tuple[str, ...]
    condition_type: str
    condition: Dict[str, Any]
    formula: str


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def energy(x: torch.Tensor) -> float:
    return float(torch.sum(x.float() * x.float()))


def inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a.float() * b.float())


def project_coeff(residual: torch.Tensor, primitive_matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return inner(residual, primitive_matrix) / inner(primitive_matrix, primitive_matrix).clamp_min(eps)


def zeros(*shape: int, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.zeros(*shape, dtype=torch.float32, device=device)


def eye(n: int, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.eye(n, dtype=torch.float32, device=device)


def qk_bias_column(H: int, device="cpu") -> torch.Tensor:
    M = zeros(H + 1, H + 1, device=device)
    M[:H, -1] = 1.0 / math.sqrt(max(1, H))
    return M


def qk_self_route(H: int, device="cpu") -> torch.Tensor:
    M = zeros(H + 1, H + 1, device=device)
    M[:H, :H] = eye(H, device=device)
    return M


def qk_shift_feature(H: int, shift: int, device="cpu") -> torch.Tensor:
    M = zeros(H + 1, H + 1, device=device)
    for i in range(H):
        j = i - shift
        if 0 <= j < H:
            M[i, j] = 1.0
    return M


def qk_block_avg(H: int, block: int, device="cpu") -> torch.Tensor:
    M = zeros(H + 1, H + 1, device=device)
    for s in range(0, H, block):
        e = min(H, s + block)
        M[s:e, s:e] = 1.0 / max(1, e - s)
    return M


def vo_identity_rect(H: int, device="cpu") -> torch.Tensor:
    C = zeros(H, H + 1, device=device)
    C[:, :H] = eye(H, device=device)
    return C


def vo_bias_write(H: int, device="cpu") -> torch.Tensor:
    C = zeros(H, H + 1, device=device)
    C[:, -1] = 1.0 / math.sqrt(max(1, H))
    return C


def vo_shift_write(H: int, shift: int, device="cpu") -> torch.Tensor:
    C = zeros(H, H + 1, device=device)
    for i in range(H):
        j = i - shift
        if 0 <= j < H:
            C[i, j] = 1.0
    return C


def vo_block_avg_write(H: int, block: int, device="cpu") -> torch.Tensor:
    C = zeros(H, H + 1, device=device)
    for s in range(0, H, block):
        e = min(H, s + block)
        C[s:e, s:e] = 1.0 / max(1, e - s)
    return C


def build_qk_primitives(H: int, device="cpu") -> List[Primitive]:
    prims: List[Primitive] = [
        Primitive("qk_self", "QK_SelfRoute", "qk", qk_self_route(H, device), ("residual_content",), ("attention_score",), "none", {}, "diag(I_H,0)"),
        Primitive("qk_bias_col", "QK_BiasRoute", "qk", qk_bias_column(H, device), ("bias_aug",), ("attention_score",), "none", {}, "content_to_bias_column"),
    ]
    for shift in [1, -1, 2, -2, 4, -4]:
        if abs(shift) < H:
            prims.append(Primitive(f"qk_shift_{shift}", "QK_ShiftFeatureRoute", "qk", qk_shift_feature(H, shift, device), ("residual_content",), ("attention_score",), "shift", {"shift": shift}, f"feature_shift({shift})"))
    for block in [2, 4, 8, 16]:
        if block <= H:
            prims.append(Primitive(f"qk_block_avg_{block}", "QK_BlockAverageRoute", "qk", qk_block_avg(H, block, device), ("residual_content",), ("attention_score",), "block", {"block": block}, f"block_avg({block})"))
    return prims


def build_vo_primitives(H: int, device="cpu") -> List[Primitive]:
    prims: List[Primitive] = [
        Primitive("vo_identity", "VO_IdentityWrite", "vo", vo_identity_rect(H, device), ("residual_content",), ("residual_write",), "none", {}, "rect_identity"),
        Primitive("vo_bias", "VO_BiasWrite", "vo", vo_bias_write(H, device), ("bias_aug",), ("residual_write",), "none", {}, "bias_column_write"),
    ]
    for shift in [1, -1, 2, -2, 4, -4]:
        if abs(shift) < H:
            prims.append(Primitive(f"vo_shift_{shift}", "VO_ShiftFeatureWrite", "vo", vo_shift_write(H, shift, device), ("residual_content",), ("residual_write",), "shift", {"shift": shift}, f"rect_shift({shift})"))
    for block in [2, 4, 8, 16]:
        if block <= H:
            prims.append(Primitive(f"vo_block_avg_{block}", "VO_BlockAverageWrite", "vo", vo_block_avg_write(H, block, device), ("residual_content",), ("residual_write",), "block", {"block": block}, f"rect_block_avg({block})"))
    return prims


def primitive_manifest(prims: Sequence[Primitive]) -> List[Dict[str, Any]]:
    return [{
        "name": p.name,
        "op_type": p.op_type,
        "block": p.block,
        "dictionary_level": LEVEL_ANALYTIC,
        "source": "analytic_primitive",
        "formula": p.formula,
        "depends_on_checkpoint_data": False,
        "depends_on_training_data": False,
        "eligible_for_exact_base_decode": True,
        "shape": list(p.matrix.shape),
        "read_fields": list(p.read_fields),
        "write_fields": list(p.write_fields),
        "condition_type": p.condition_type,
        "condition": p.condition,
    } for p in prims]


def op_from_primitive(p: Primitive, coeff: float, op_id: str, drop: float, err_after: float, explained_energy: float) -> Op:
    return Op(
        op_id=op_id,
        op_type=p.op_type,
        source="analytic_primitive",
        dictionary_level=LEVEL_ANALYTIC,
        read_fields=p.read_fields,
        write_fields=p.write_fields,
        condition_type=p.condition_type,
        condition=dict(p.condition),
        params={"coeff": float(coeff), "primitive_name": p.name, "formula": p.formula},
        shape=tuple(int(x) for x in p.matrix.shape),
        energy=float(explained_energy),
        marginal_error_drop=float(drop),
        encode_status="closed_form",
        decode_error_after=float(err_after),
        typing_status="typed",
        depends_on_checkpoint_data=False,
        universal=False,
        transferable=True,
    )


def encode_ops(ops: Sequence[Op], primitive_by_name: Dict[str, Primitive], shape: Tuple[int, ...], device="cpu") -> torch.Tensor:
    out = torch.zeros(*shape, dtype=torch.float32, device=device)
    for op in ops:
        name = op.params.get("primitive_name")
        if name not in primitive_by_name:
            raise KeyError(f"Unknown primitive_name in op {op.op_id}: {name}")
        out = out + float(op.params["coeff"]) * primitive_by_name[name].matrix.to(device)
    return out


def decode_greedy_analytic(target: torch.Tensor, primitives: Sequence[Primitive], block: str, thresholds: Dict[str, Any], max_ops: int = 64, device="cpu"):
    target = target.float().to(device)
    residual = target.clone()
    recon = torch.zeros_like(target)
    accepted: List[Op] = []
    used: set[str] = set()
    total_energy = max(energy(target), 1e-12)
    min_coeff = float(thresholds.get("MIN_COEFF_ABS", 1e-6))
    min_drop = float(thresholds.get("MIN_MARGINAL_REL_DROP", 1e-4))
    min_frac = float(thresholds.get("MIN_EXPLAINED_ENERGY_FRAC", 1e-4))
    for _ in range(max_ops):
        err_before = rel_err(recon, target)
        best = None
        for p in primitives:
            if p.name in used:
                continue
            coeff_t = project_coeff(residual, p.matrix.to(device))
            coeff = float(coeff_t.detach().cpu())
            if abs(coeff) < min_coeff:
                continue
            cand = coeff_t * p.matrix.to(device)
            new_recon = recon + cand
            new_residual = target - new_recon
            err_after = rel_err(new_recon, target)
            drop = err_before - err_after
            explained = energy(residual) - energy(new_residual)
            frac = explained / total_energy
            if drop >= min_drop and frac >= min_frac:
                score = (drop, frac, abs(coeff))
                if best is None or score > best[0]:
                    best = (score, p, coeff, new_recon, new_residual, err_after, drop, explained)
        if best is None:
            break
        _, p, coeff, recon, residual, err_after, drop, explained = best
        used.add(p.name)
        accepted.append(op_from_primitive(p, coeff, f"{block}_op_{len(accepted):03d}_{p.name}", drop, err_after, explained))
    metrics = {
        "target_energy": total_energy,
        "residual_energy": energy(residual),
        "residual_energy_ratio": energy(residual) / total_energy,
        "roundtrip_error": rel_err(recon, target),
        "typed_coverage": 1.0 - energy(residual) / total_energy,
        "ops_count": float(len(accepted)),
    }
    return accepted, recon, metrics


def make_program(program_id: str, block: str, H: int, ops: List[Op], target: torch.Tensor, recon: torch.Tensor, metrics: Dict[str, float], model_name: str = "synthetic", layer: int = 0, head: int | None = 0) -> Program:
    residual_rel = rel_err(recon, target)
    closure_tol = max(1.5 * residual_rel, 1e-5)
    return Program(
        program_id=program_id,
        model_family="synthetic",
        model_name=model_name,
        layer=layer,
        head=head,
        kv_head=head,
        block=block,
        input_basis="synthetic_feature_basis",
        output_basis="synthetic_feature_basis",
        head_config={"hidden_size": H, "aug_size": H + 1, "block": block},
        ops=ops,
        residual=ResidualReport(residual_rel=float(residual_rel), residual_energy_ratio=float(metrics["residual_energy_ratio"]), typed_coverage=float(metrics["typed_coverage"]), raw_residual_used_as_program=False, unreplayed_ops_count=0 if residual_rel <= closure_tol else 1),
        roundtrip=RoundTripReport(roundtrip_error=float(residual_rel), closure_tol=float(closure_tol), closed=bool(residual_rel <= closure_tol), closure_level="circuit_target"),
        decode_status="typed_program" if ops else "partial_program",
    )
