#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_level1_verified_extension_v1_2.py

Wrapper-fix for v1_1:
v1_1 imports apply_atom/eval_atom from v1, but v1 expects entry["base_err"].
v1_1 entries use entry["current_err"].
This wrapper overrides those two functions and then calls v1_1.main().
"""
from __future__ import annotations

import torch
import qwen_level1_verified_extension_v1_1 as v11

VERSION = "qwen_level1_verified_extension_v1.2-current-err-wrapper"


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.float().reshape(-1)


def _rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return float(torch.linalg.norm((a - b).float()) / torch.linalg.norm(b.float()).clamp_min(eps))


def _coeff_for(residual: torch.Tensor, atom: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = _flat(atom)
    r = _flat(residual)
    return (r @ a) / (a @ a).clamp_min(eps)


def apply_atom_current(entry, atom):
    atom = atom.float().cpu()
    c = _coeff_for(entry["residual"], atom)
    rec1 = entry["recon"] + c * atom
    err1 = _rel_err(rec1, entry["target"])
    gain = float(entry["current_err"] - err1)
    return {"coeff": float(c.detach().cpu()), "err_after": float(err1), "gain": gain}


def eval_atom_current(entries, atom):
    vals = [apply_atom_current(e, atom) for e in entries]
    if not vals:
        return {"gain_mean": 0.0, "err_after_mean": 0.0, "abs_coeff_mean": 0.0}
    return {
        "gain_mean": sum(v["gain"] for v in vals) / len(vals),
        "err_after_mean": sum(v["err_after"] for v in vals) / len(vals),
        "abs_coeff_mean": sum(abs(v["coeff"]) for v in vals) / len(vals),
    }


v11.apply_atom = apply_atom_current
v11.eval_atom = eval_atom_current
v11.VERSION = VERSION

if __name__ == "__main__":
    v11.main()
