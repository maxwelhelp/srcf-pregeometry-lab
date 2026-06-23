#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
program_dsl_v1.py
Typed program DSL for Exact Program Transplant v1.
Scope v1: QK/VO circuit targets only. MLP is explicitly out of scope.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = "program_dsl_v1.0"

LEVEL_ANALYTIC = "analytic_base_dictionary"
LEVEL_EXTENSION = "verified_extension_dictionary"
LEVEL_PROMOTED = "universal_promoted_dictionary"

CLOSURE_CIRCUIT_TARGET = "circuit_target"
CLOSURE_WEIGHTS = "weights"
CLOSURE_FULL_FORWARD = "full_forward_logits"

STATUS_SYNTHETIC_CLOSED = "SYNTHETIC_PROGRAM_CLOSED"
STATUS_ANALYTIC_CLOSED = "ANALYTIC_PROGRAM_CLOSED"
STATUS_EXTENDED_CLOSED = "EXTENDED_TARGET_SPECIFIC_PROGRAM_CLOSED"
STATUS_PARTIAL = "PARTIAL_PROGRAM_CLOSED"
STATUS_REJECTED_RAW = "TRIVIAL_DECODE_REJECTED"
STATUS_MLP_OUT_OF_SCOPE = "MLP_OUT_OF_SCOPE_V1_REJECTED"


@dataclass
class Op:
    op_id: str
    op_type: str
    source: str
    dictionary_level: str
    read_fields: Tuple[str, ...]
    write_fields: Tuple[str, ...]
    condition_type: str
    condition: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    shape: Tuple[int, ...] = field(default_factory=tuple)
    energy: float = 0.0
    marginal_error_drop: float = 0.0
    encode_status: str = "closed_form"
    decode_error_after: float = 0.0
    typing_status: str = "typed"
    depends_on_checkpoint_data: bool = False
    universal: bool = False
    transferable: bool = False


@dataclass
class ResidualReport:
    residual_rel: float
    residual_energy_ratio: float
    typed_coverage: float
    raw_residual_used_as_program: bool
    unreplayed_ops_count: int


@dataclass
class RoundTripReport:
    roundtrip_error: float
    closure_tol: float
    closed: bool
    closure_level: str = CLOSURE_CIRCUIT_TARGET


@dataclass
class Program:
    program_id: str
    model_family: str
    model_name: str
    layer: int
    head: Optional[int]
    kv_head: Optional[int]
    block: str
    input_basis: str
    output_basis: str
    head_config: Dict[str, Any]
    ops: List[Op]
    residual: ResidualReport
    roundtrip: RoundTripReport
    decode_status: str


@dataclass
class Manifest:
    mode: str
    status: str
    closure_level: str
    dictionary_source: str
    base_dictionary_level: str
    extension_dictionary_used: bool
    raw_weight_passthrough_used: bool
    alpha_sweep_used_as_main_method: bool
    kl_distillation_used_as_main_method: bool
    coefficient_l2_used_as_main_method: bool
    gradient_used_as_main_method: bool
    mlp_in_v1_scope: bool
    mlp_decode_attempted: bool
    level2_promotion_in_v1_scope: bool
    level2_promotion_attempted: bool
    thresholds_loaded_from: str
    thresholds_recomputed_in_real_run: bool
    thresholds_frozen: bool
    head_pass_rate: float
    max_head_error: float
    program_closure_rate: float
    op_closure_rate: float
    structural_match_rate: float
    closure_report_exists: bool
    repair_used: bool
    repair_steps_needed: int
    extra: Dict[str, Any] = field(default_factory=dict)


DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "version": "calibrated_thresholds_v1",
    "source": "synthetic_program_roundtrip_v1",
    "frozen": True,
    "MIN_COEFF_ABS": 1e-6,
    "MIN_MARGINAL_REL_DROP": 1e-4,
    "MIN_EXPLAINED_ENERGY_FRAC": 1e-4,
    "MIN_EXTENSION_GAIN": 1e-3,
    "MIN_EXTENSION_ENERGY_FRAC": 1e-4,
    "MIN_HELDOUT_GAIN": 1e-3,
    "MAX_DUP_CORR": 0.985,
    "MAX_RETAIN_DELTA": 0.003,
    "closure_tol_multiplier": 1.5,
    "locked_before_real_runs": True,
}


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    return obj


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def write_jsonl(path: str | Path, rows: List[Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def load_thresholds(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        return dict(DEFAULT_THRESHOLDS)
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("frozen", False):
        raise ValueError(f"Threshold file is not frozen: {p}")
    return data


def ensure_v1_scope(blocks: List[str]) -> None:
    if any(b.lower() == "mlp" for b in blocks):
        raise ValueError("MLP_OUT_OF_SCOPE_V1_REJECTED")
