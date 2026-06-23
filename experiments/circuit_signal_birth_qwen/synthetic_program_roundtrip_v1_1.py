#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic_program_roundtrip_v1_1.py
Same synthetic Level-0 roundtrip, but uses analytic_primitives_v1_1 OMP-refit decoder.
"""
from __future__ import annotations

import synthetic_program_roundtrip_v1 as base
import analytic_primitives_v1_1 as fixed

base.VERSION = "synthetic_program_roundtrip_v1.1-omp-refit"
base.decode_greedy_analytic = fixed.decode_greedy_analytic
base.build_qk_primitives = fixed.build_qk_primitives
base.build_vo_primitives = fixed.build_vo_primitives
base.primitive_manifest = fixed.primitive_manifest
base.rel_err = fixed.rel_err

if __name__ == "__main__":
    base.main()
