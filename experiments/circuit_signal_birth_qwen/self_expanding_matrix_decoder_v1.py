#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
self_expanding_matrix_decoder_v1.py

Universal matrix decoder with a self-expanding verified dictionary.

Purpose:
  Take ANY 2D matrix from a neural net and produce a readable matrix program:

    M ≈ Σ coeff_k * Primitive_k

  Start with analytic primitives, then mine residual primitives and accept them
  only if they improve both train and heldout matrix-entry reconstruction.

This is intentionally model-agnostic. For Qwen head circuits you should feed it
M_qk_aug[h,d], C_vo_aug[h], or MLP effective matrices exported by the head-circuit
scripts — not raw Wq/Wk/Wv/Wo unless you explicitly want raw-weight analysis.

Inputs:
  --matrix-file path.pt/.pth/.safetensors/.npy/.npz
  --key optional tensor key inside checkpoint
  OR --synthetic for a known toy matrix

Outputs:
  manifest.json
  accepted_atoms.jsonl
  candidates.jsonl
  reconstruction.pt
  matrix_pseudocode.md

No training. No LoRA. No model download. Pure matrix program extraction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

VERSION = "self_expanding_matrix_decoder_v1.0"


@dataclass
class Atom:
    name: str
    stage: str
    kind: str
    formula: str
    tensor: torch.Tensor
    meta: Dict[str, Any]


@dataclass
class AcceptedAtom:
    idx: int
    name: str
    stage: str
    kind: str
    formula: str
    coeff: float
    train_gain: float
    heldout_gain: float
    train_err_after: float
    heldout_err_after: float
    abs_cos_to_prev_max: float
    norm: float
    meta: Dict[str, Any]


def to_float_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu().contiguous()


def norm(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-12) -> torch.Tensor:
    if mask is None:
        return torch.linalg.norm(x.float()).clamp_min(eps)
    return torch.linalg.norm(x.float()[mask]).clamp_min(eps)


def dot(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return torch.sum(a.float() * b.float())
    return torch.sum(a.float()[mask] * b.float()[mask])


def rel_err(recon: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    return float(norm(recon - target, mask) / norm(target, mask))


def unit_flat(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    f = x.float().reshape(-1)
    return f / torch.linalg.norm(f).clamp_min(eps)


def tensor_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if tuple(a.shape) != tuple(b.shape):
        return 0.0
    return float(torch.dot(unit_flat(a), unit_flat(b)))


def safe_json(v: Any) -> Any:
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, dict):
        return {str(k): safe_json(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [safe_json(x) for x in v]
    return v


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(safe_json(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(safe_json(r), ensure_ascii=False) + "\n")


def pick_first_matrix(obj: Any, prefix: str = "") -> Tuple[str, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return prefix or "tensor", obj
    if isinstance(obj, np.ndarray):
        return prefix or "array", torch.from_numpy(obj)
    if isinstance(obj, dict):
        for k, v in obj.items():
            try:
                name, t = pick_first_matrix(v, f"{prefix}.{k}" if prefix else str(k))
                if t.ndim >= 2:
                    return name, t
            except Exception:
                pass
    raise ValueError("Could not find a tensor/array matrix in object")


def load_matrix(path: Optional[str], key: Optional[str], flatten: bool, synthetic: bool, n: int, m: int, seed: int) -> Tuple[str, torch.Tensor]:
    if synthetic:
        g = torch.Generator().manual_seed(seed)
        r = torch.arange(n).float().view(n, 1)
        c = torch.arange(m).float().view(1, m)
        M = 0.9 * torch.eye(n, m)
        M += 0.35 * torch.exp(-torch.abs(r - c) / 4.0)
        M += 0.18 * torch.cos(2 * math.pi * r / max(1, n)) @ torch.cos(2 * math.pi * c / max(1, m))
        u = torch.randn(n, 1, generator=g); v = torch.randn(1, m, generator=g)
        M += 0.12 * (u @ v) / math.sqrt(max(n, m))
        M += 0.03 * torch.randn(n, m, generator=g)
        return "synthetic_identity_decay_dct_lowrank_noise", M.float()

    if path is None:
        raise SystemExit("Provide --matrix-file or --synthetic")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()

    if suffix in [".npy"]:
        arr = np.load(p, allow_pickle=False)
        name, t = p.name, torch.from_numpy(arr)
    elif suffix in [".npz"]:
        z = np.load(p, allow_pickle=False)
        if key is None:
            key = list(z.keys())[0]
        name, t = key, torch.from_numpy(z[key])
    elif suffix in [".safetensors"]:
        try:
            from safetensors.torch import safe_open
        except Exception as e:
            raise SystemExit("safetensors is not installed. pip install safetensors") from e
        with safe_open(str(p), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            if key is None:
                candidates = [k for k in keys if f.get_tensor(k).ndim >= 2]
                if not candidates:
                    raise SystemExit("No >=2D tensors in safetensors file")
                key = candidates[0]
            name, t = key, f.get_tensor(key)
    else:
        obj = torch.load(p, map_location="cpu")
        if key is not None:
            cur = obj
            for part in key.split("."):
                if isinstance(cur, dict):
                    cur = cur[part]
                else:
                    cur = getattr(cur, part)
            name, t = key, cur
        else:
            name, t = pick_first_matrix(obj)

    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t)
    t = to_float_cpu(t)
    if t.ndim > 2:
        if not flatten:
            raise SystemExit(f"Tensor {name} has shape {tuple(t.shape)}. Use --flatten to make it 2D.")
        t = t.reshape(t.shape[0], -1).contiguous()
    if t.ndim != 2:
        raise SystemExit(f"Tensor {name} must be 2D after optional flatten, got {tuple(t.shape)}")
    return name, t


def make_split_mask(shape: Tuple[int, int], holdout_frac: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n, m = shape
    g = torch.Generator().manual_seed(seed)
    rnd = torch.rand(n, m, generator=g)
    held = rnd < holdout_frac
    train = ~held
    if int(held.sum()) == 0:
        held = torch.zeros(n, m, dtype=torch.bool)
        held.view(-1)[0] = True
        train = ~held
    return train, held


def dct_basis(n: int, k: int) -> torch.Tensor:
    x = torch.arange(n).float()
    if k == 0:
        v = torch.ones(n)
    else:
        v = torch.cos(math.pi * (x + 0.5) * k / n)
    return v / torch.linalg.norm(v).clamp_min(1e-12)


def atom(name: str, stage: str, kind: str, formula: str, T: torch.Tensor, **meta) -> Atom:
    return Atom(name=name, stage=stage, kind=kind, formula=formula, tensor=to_float_cpu(T), meta=meta)


def analytic_dictionary(n: int, m: int, dct_k: int, diag_radius: int, block_grids: List[int]) -> List[Atom]:
    atoms: List[Atom] = []
    atoms.append(atom("ones", "analytic", "global", "Ones(n,m)", torch.ones(n, m)))

    # diagonal / local route-style atoms
    for off in range(-diag_radius, diag_radius + 1):
        T = torch.zeros(n, m)
        for i in range(n):
            j = i + off
            if 0 <= j < m:
                T[i, j] = 1.0
        if torch.count_nonzero(T) > 0:
            atoms.append(atom(f"diag_offset_{off:+d}", "analytic", "diagonal", f"DiagOffset(offset={off})", T, offset=off))

    # exponential/local windows
    rr = torch.arange(n).float().view(n, 1)
    cc = torch.arange(m).float().view(1, m)
    for tau in [2.0, 4.0, 8.0, 16.0]:
        T = torch.exp(-torch.abs(rr - cc) / tau)
        atoms.append(atom(f"toeplitz_decay_tau_{int(tau)}", "analytic", "toeplitz", f"exp(-|i-j|/{tau:g})", T, tau=tau))

    # ramps / coordinate structure
    row = torch.linspace(-1, 1, n).view(n, 1).expand(n, m)
    col = torch.linspace(-1, 1, m).view(1, m).expand(n, m)
    atoms.append(atom("row_ramp", "analytic", "coordinate", "row_coordinate_ramp(i)", row))
    atoms.append(atom("col_ramp", "analytic", "coordinate", "col_coordinate_ramp(j)", col))
    atoms.append(atom("row_col_product", "analytic", "coordinate", "row_ramp(i)*col_ramp(j)", row * col))

    # DCT outer dictionary, compact smooth basis
    for kr in range(dct_k + 1):
        br = dct_basis(n, kr).view(n, 1)
        for kc in range(dct_k + 1):
            if kr == 0 and kc == 0:
                continue
            bc = dct_basis(m, kc).view(1, m)
            atoms.append(atom(f"dct_outer_r{kr}_c{kc}", "analytic", "dct_outer", f"DCT_row({kr}) ⊗ DCT_col({kc})", br @ bc, kr=kr, kc=kc))

    # block constants
    for g in block_grids:
        if g <= 1:
            continue
        for bi in range(g):
            i0 = int(round(bi * n / g)); i1 = int(round((bi + 1) * n / g))
            for bj in range(g):
                j0 = int(round(bj * m / g)); j1 = int(round((bj + 1) * m / g))
                if i1 <= i0 or j1 <= j0:
                    continue
                T = torch.zeros(n, m)
                T[i0:i1, j0:j1] = 1.0
                atoms.append(atom(f"block_g{g}_{bi}_{bj}", "analytic", "block", f"BlockConst(grid={g}, block=({bi},{bj}))", T, grid=g, bi=bi, bj=bj))
    return atoms


def project_candidate(R: torch.Tensor, P: torch.Tensor, target: torch.Tensor, train: torch.Tensor, held: torch.Tensor) -> Dict[str, float]:
    denom = dot(P, P, train)
    if float(denom) <= 1e-12:
        return {"coeff": 0.0, "train_gain": -1.0, "heldout_gain": -1.0, "train_after": 999.0, "held_after": 999.0}
    coeff = float(dot(R, P, train) / denom)
    before_train = float(norm(R, train) / norm(target, train))
    before_held = float(norm(R, held) / norm(target, held))
    R2 = R - coeff * P
    after_train = float(norm(R2, train) / norm(target, train))
    after_held = float(norm(R2, held) / norm(target, held))
    return {
        "coeff": coeff,
        "train_gain": before_train - after_train,
        "heldout_gain": before_held - after_held,
        "train_after": after_train,
        "held_after": after_held,
    }


def max_abs_cos_to_atoms(P: torch.Tensor, accepted_tensors: List[torch.Tensor]) -> float:
    if not accepted_tensors:
        return 0.0
    u = unit_flat(P)
    vals = [abs(float(torch.dot(u, unit_flat(a)))) for a in accepted_tensors if tuple(a.shape) == tuple(P.shape)]
    return max(vals) if vals else 0.0


def masked_residual_for_mining(R: torch.Tensor, train: torch.Tensor) -> torch.Tensor:
    X = torch.zeros_like(R.float())
    X[train] = R.float()[train]
    return X


def mine_svd_atoms(R: torch.Tensor, train: torch.Tensor, count: int) -> List[Atom]:
    X = masked_residual_for_mining(R, train)
    n, m = X.shape
    k = min(count, min(n, m))
    out: List[Atom] = []
    try:
        U, S, Vh = torch.linalg.svd(X.float(), full_matrices=False)
        for i in range(k):
            P = torch.outer(U[:, i], Vh[i, :])
            out.append(atom(f"birth_svd_rank1_{i}", "birth", "low_rank", f"ResidualRank1(u{i}, v{i})", P, sv=float(S[i].item()), rank1_index=i))
    except Exception:
        pass
    return out


def mine_profile_atoms(R: torch.Tensor, train: torch.Tensor) -> List[Atom]:
    n, m = R.shape
    X = masked_residual_for_mining(R, train)
    cnt_row = train.float().sum(dim=1).clamp_min(1.0)
    row_mean = X.sum(dim=1) / cnt_row
    P_row = row_mean.view(n, 1).expand(n, m).clone()

    cnt_col = train.float().sum(dim=0).clamp_min(1.0)
    col_mean = X.sum(dim=0) / cnt_col
    P_col = col_mean.view(1, m).expand(n, m).clone()

    return [
        atom("birth_row_profile", "birth", "profile", "ResidualRowProfile(i)", P_row),
        atom("birth_col_profile", "birth", "profile", "ResidualColProfile(j)", P_col),
    ]


def mine_diag_atoms(R: torch.Tensor, train: torch.Tensor, max_offsets: int) -> List[Atom]:
    n, m = R.shape
    X = masked_residual_for_mining(R, train)
    stats = []
    for off in range(-(n - 1), m):
        vals = []
        idxs = []
        for i in range(n):
            j = i + off
            if 0 <= j < m and train[i, j]:
                vals.append(float(X[i, j].item()))
                idxs.append((i, j))
        if not vals:
            continue
        score = float(np.sqrt(np.mean(np.square(vals)))) * math.sqrt(len(vals))
        stats.append((score, off, vals, idxs))
    stats.sort(reverse=True, key=lambda x: x[0])
    out = []
    for rank_i, (_score, off, vals, idxs) in enumerate(stats[:max_offsets]):
        T = torch.zeros(n, m)
        val = float(np.mean(vals))
        for i, j in idxs:
            T[i, j] = val
        out.append(atom(f"birth_diag_offset_{off:+d}", "birth", "diagonal_residual", f"ResidualDiagMean(offset={off})", T, offset=off, diag_score=_score, diag_rank=rank_i))
    return out


def mine_block_atoms(R: torch.Tensor, train: torch.Tensor, grids: List[int], top_per_grid: int) -> List[Atom]:
    n, m = R.shape
    X = masked_residual_for_mining(R, train)
    out: List[Atom] = []
    for g in grids:
        cand = []
        for bi in range(g):
            i0 = int(round(bi * n / g)); i1 = int(round((bi + 1) * n / g))
            for bj in range(g):
                j0 = int(round(bj * m / g)); j1 = int(round((bj + 1) * m / g))
                if i1 <= i0 or j1 <= j0:
                    continue
                mask = train[i0:i1, j0:j1]
                if int(mask.sum()) == 0:
                    continue
                val = float(X[i0:i1, j0:j1][mask].mean().item())
                score = abs(val) * math.sqrt(int(mask.sum()))
                cand.append((score, bi, bj, i0, i1, j0, j1, val))
        cand.sort(reverse=True, key=lambda x: x[0])
        for rank_i, (score, bi, bj, i0, i1, j0, j1, val) in enumerate(cand[:top_per_grid]):
            T = torch.zeros(n, m)
            T[i0:i1, j0:j1] = val
            out.append(atom(f"birth_block_g{g}_{bi}_{bj}", "birth", "block_residual", f"ResidualBlockMean(grid={g}, block=({bi},{bj}))", T, grid=g, bi=bi, bj=bj, value=val, block_score=score, block_rank=rank_i))
    return out


def mine_candidates(R: torch.Tensor, train: torch.Tensor, args) -> List[Atom]:
    out: List[Atom] = []
    out.extend(mine_svd_atoms(R, train, args.birth_svd))
    out.extend(mine_profile_atoms(R, train))
    out.extend(mine_diag_atoms(R, train, args.birth_diags))
    out.extend(mine_block_atoms(R, train, args.birth_block_grids, args.birth_blocks_per_grid))
    # remove empty tensors
    out = [a for a in out if float(torch.linalg.norm(a.tensor.float())) > 1e-12]
    return out


def greedy_decode(target: torch.Tensor, dictionary: List[Atom], train: torch.Tensor, held: torch.Tensor, args) -> Tuple[torch.Tensor, List[AcceptedAtom], List[Dict[str, Any]]]:
    target = target.float().cpu()
    recon = torch.zeros_like(target)
    accepted: List[AcceptedAtom] = []
    accepted_tensors: List[torch.Tensor] = []
    candidate_log: List[Dict[str, Any]] = []

    def try_accept_from_pool(pool: List[Atom], round_idx: int, stage: str) -> bool:
        nonlocal recon, accepted, accepted_tensors, candidate_log
        R = target - recon
        scored = []
        for a in pool:
            P = a.tensor.float().cpu()
            met = project_candidate(R, P, target, train, held)
            maxcos = max_abs_cos_to_atoms(P, accepted_tensors)
            row = {
                "round": round_idx,
                "name": a.name,
                "stage": a.stage,
                "kind": a.kind,
                "formula": a.formula,
                "coeff": met["coeff"],
                "train_gain": met["train_gain"],
                "heldout_gain": met["heldout_gain"],
                "train_err_after": met["train_after"],
                "heldout_err_after": met["held_after"],
                "abs_cos_to_prev_max": maxcos,
                "norm": float(torch.linalg.norm(P).item()),
                "accepted": False,
                "meta": a.meta,
            }
            candidate_log.append(row)
            scored.append((met["train_gain"] + args.heldout_weight * met["heldout_gain"], a, met, maxcos, row))
        scored.sort(reverse=True, key=lambda x: x[0])
        for _score, a, met, maxcos, row in scored:
            if len(accepted) >= args.max_atoms:
                return False
            if abs(float(met["coeff"])) < args.min_abs_coeff:
                continue
            if float(met["train_gain"]) < args.min_train_gain:
                continue
            if float(met["heldout_gain"]) < args.min_heldout_gain:
                continue
            if maxcos > args.dup_cos:
                continue
            P = a.tensor.float().cpu()
            recon = recon + float(met["coeff"]) * P
            acc = AcceptedAtom(
                idx=len(accepted),
                name=a.name,
                stage=stage,
                kind=a.kind,
                formula=a.formula,
                coeff=float(met["coeff"]),
                train_gain=float(met["train_gain"]),
                heldout_gain=float(met["heldout_gain"]),
                train_err_after=float(met["train_after"]),
                heldout_err_after=float(met["held_after"]),
                abs_cos_to_prev_max=float(maxcos),
                norm=float(torch.linalg.norm(P).item()),
                meta=a.meta,
            )
            accepted.append(acc)
            accepted_tensors.append(P)
            row["accepted"] = True
            row["accepted_idx"] = acc.idx
            return True
        return False

    # First pass: analytic dictionary only.
    for step in range(args.analytic_steps):
        ok = try_accept_from_pool(dictionary, step, "analytic")
        if not ok:
            break
        if rel_err(recon, target, train) <= args.target_tol and rel_err(recon, target, held) <= args.target_tol:
            return recon, accepted, candidate_log

    # Self-expanding residual births.
    for birth_round in range(args.birth_rounds):
        R = target - recon
        pool = mine_candidates(R, train, args)
        ok = try_accept_from_pool(pool, args.analytic_steps + birth_round, "birth")
        if not ok:
            break
        if rel_err(recon, target, train) <= args.target_tol and rel_err(recon, target, held) <= args.target_tol:
            break

    return recon, accepted, candidate_log


def pseudocode_lines(accepted: List[AcceptedAtom], top: int) -> List[str]:
    lines = []
    lines.append("# Matrix program")
    lines.append("M_hat = zeros_like(M)")
    for a in accepted[:top]:
        lines.append(f"M_hat += ({a.coeff:+.6g}) * {a.formula}  # {a.stage}/{a.kind}, train_gain={a.train_gain:.3g}, heldout_gain={a.heldout_gain:.3g}")
    if len(accepted) > top:
        lines.append(f"# ... {len(accepted)-top} more atoms omitted; see accepted_atoms.jsonl")
    return lines


def make_markdown(matrix_name: str, target: torch.Tensor, recon: torch.Tensor, accepted: List[AcceptedAtom], manifest: Dict[str, Any], top: int) -> str:
    lines = []
    lines.append(f"# Self-expanding matrix decode: `{matrix_name}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("```json")
    lines.append(json.dumps(safe_json(manifest["summary"]), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    lines.append("## Pseudocode")
    lines.append("```python")
    lines.extend(pseudocode_lines(accepted, top))
    lines.append("```")
    lines.append("")
    lines.append("## Accepted atoms")
    lines.append("| idx | coeff | stage | kind | formula | train gain | heldout gain | train err | heldout err |")
    lines.append("|---:|---:|---|---|---|---:|---:|---:|---:|")
    for a in accepted[:top]:
        formula = a.formula.replace("|", "\\|")
        lines.append(f"| {a.idx} | {a.coeff:+.4g} | {a.stage} | {a.kind} | `{formula}` | {a.train_gain:.4g} | {a.heldout_gain:.4g} | {a.train_err_after:.4g} | {a.heldout_err_after:.4g} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- `analytic` atoms are fixed primitives: diagonals, DCT outer products, blocks, coordinate ramps.")
    lines.append("- `birth` atoms are mined from residual and accepted only after train+heldout improvement.")
    lines.append("- For Qwen attention, feed exact head-circuit matrices (`M_qk_aug[h,d]`, `C_vo_aug[h]`) rather than raw Wq/Wk/Wv/Wo.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-file", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--flatten", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--dct-k", type=int, default=4)
    ap.add_argument("--diag-radius", type=int, default=16)
    ap.add_argument("--block-grids", default="2,4,8")
    ap.add_argument("--analytic-steps", type=int, default=64)
    ap.add_argument("--birth-rounds", type=int, default=64)
    ap.add_argument("--birth-svd", type=int, default=4)
    ap.add_argument("--birth-diags", type=int, default=8)
    ap.add_argument("--birth-block-grids", default="4,8")
    ap.add_argument("--birth-blocks-per-grid", type=int, default=6)
    ap.add_argument("--max-atoms", type=int, default=96)
    ap.add_argument("--target-tol", type=float, default=1e-3)
    ap.add_argument("--min-train-gain", type=float, default=1e-5)
    ap.add_argument("--min-heldout-gain", type=float, default=-1e-6)
    ap.add_argument("--min-abs-coeff", type=float, default=1e-8)
    ap.add_argument("--heldout-weight", type=float, default=1.0)
    ap.add_argument("--dup-cos", type=float, default=0.985)
    ap.add_argument("--print-top", type=int, default=40)
    ap.add_argument("--out", default="runs/self_expanding_matrix_decoder_v1")
    args = ap.parse_args()

    args.block_grids = [int(x) for x in str(args.block_grids).split(",") if x.strip()]
    args.birth_block_grids = [int(x) for x in str(args.birth_block_grids).split(",") if x.strip()]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    matrix_name, M = load_matrix(args.matrix_file, args.key, args.flatten, args.synthetic, args.n, args.m, args.seed)
    n, m = M.shape
    train, held = make_split_mask((n, m), args.holdout_frac, args.seed + 1)

    dictionary = analytic_dictionary(n, m, args.dct_k, args.diag_radius, args.block_grids)
    recon, accepted, candidate_log = greedy_decode(M, dictionary, train, held, args)

    full_err = rel_err(recon, M, None)
    train_err = rel_err(recon, M, train)
    held_err = rel_err(recon, M, held)
    stage_counts: Dict[str, int] = {}
    kind_counts: Dict[str, int] = {}
    for a in accepted:
        stage_counts[a.stage] = stage_counts.get(a.stage, 0) + 1
        kind_counts[a.kind] = kind_counts.get(a.kind, 0) + 1

    manifest = {
        "version": VERSION,
        "matrix_name": matrix_name,
        "shape": [n, m],
        "args": vars(args),
        "summary": {
            "atoms_accepted": len(accepted),
            "analytic_dictionary_size": len(dictionary),
            "full_rel_err": full_err,
            "train_rel_err": train_err,
            "heldout_rel_err": held_err,
            "target_tol": args.target_tol,
            "closed_full": full_err <= args.target_tol,
            "closed_train": train_err <= args.target_tol,
            "closed_heldout": held_err <= args.target_tol,
            "stage_counts": stage_counts,
            "kind_counts": kind_counts,
        },
    }

    acc_rows = [asdict(a) for a in accepted]
    # tensor is not stored in accepted rows; reconstruction.pt stores enough for replay.
    write_json(out / "manifest.json", manifest)
    write_jsonl(out / "accepted_atoms.jsonl", acc_rows)
    write_jsonl(out / "candidates.jsonl", candidate_log)
    torch.save({
        "matrix_name": matrix_name,
        "target": M,
        "reconstruction": recon,
        "residual": M - recon,
        "train_mask": train,
        "heldout_mask": held,
        "accepted": acc_rows,
    }, out / "reconstruction.pt")
    md = make_markdown(matrix_name, M, recon, accepted, manifest, args.print_top)
    (out / "matrix_pseudocode.md").write_text(md, encoding="utf-8")

    print("=== Self-Expanding Matrix Decoder v1 ===")
    print(json.dumps(safe_json(manifest["summary"]), indent=2, ensure_ascii=False))
    print("\n--- PSEUDOCODE TOP ---")
    for line in pseudocode_lines(accepted, min(args.print_top, len(accepted))):
        print(line)
    print(f"\nout={out}")
    print(f"markdown={out / 'matrix_pseudocode.md'}")


if __name__ == "__main__":
    main()
