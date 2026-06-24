#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qwen_contrastive_gradient_weight_transfer_heldout_v1_3.py

Small wrapper over v1_2 that expands the built-in prompt pool.

Reason:
  qwen_teacher_student_code_transfer_v2.select_prompts("code", 16) currently
  returns only 8 prompts. v1_2 needs train+heldout prompts, so v1_3 monkey-patches
  the select_prompts symbol inside v1_2 before calling v1_2.main().

The transfer logic is unchanged from v1_2:
  - build directional contrastive gradient mask on train code/retain prompts;
  - apply masked teacher-student delta;
  - evaluate on both train and heldout code/retain prompts.
"""
from __future__ import annotations

from qwen_teacher_student_code_transfer_v2 import select_prompts as _orig_select_prompts
import qwen_contrastive_gradient_weight_transfer_heldout_v1_2 as _run


EXTRA_CODE_PROMPTS = [
    "Write a Python function that returns the factorial of n using recursion.\n```python\n",
    "Complete the function:\n```python\ndef is_prime(n):\n    if n < 2:\n        return False\n",
    "Fix this JavaScript snippet and explain the bug:\n```js\nfunction sum(arr) {\n  let s = 0;\n",
    "Implement binary search in Python for a sorted list.\n```python\ndef binary_search(a, x):\n",
    "What is the output of this code?\n```python\nx = [1, 2, 3]\ny = x\ny.append(4)\n",
    "Complete this SQL query to select users created in the last 7 days:\n```sql\nSELECT * FROM users\n",
    "Write a regex that matches a simple email address and show one example.\n",
    "Explain what this Python decorator does:\n```python\ndef deco(fn):\n    def wrapper(*args, **kwargs):\n",
    "Complete the class method:\n```python\nclass Counter:\n    def __init__(self):\n        self.n = 0\n",
    "Convert this loop into a list comprehension:\n```python\nres = []\nfor x in nums:\n    if x % 2 == 0:\n",
    "Find the bug in this C code:\n```c\nint *p;\n*p = 5;\n",
    "Write a minimal HTTP server in Python using the standard library.\n```python\n",
    "Complete this TypeScript type guard:\n```ts\nfunction isString(x: unknown): x is string {\n",
    "Implement a stack with push and pop operations in Python.\n```python\nclass Stack:\n",
    "Explain the difference between deep copy and shallow copy with Python examples.\n",
    "Given a JSON object, write Python code to extract the field user.name safely.\n```python\n",
]

EXTRA_RETAIN_PROMPTS = [
    "Explain why the sky appears blue during the day.",
    "Write a short paragraph about the importance of sleep for teenagers.",
    "Summarize the main causes of the seasons on Earth.",
    "Give three tips for organizing a small study schedule.",
    "Describe how rain forms in simple terms.",
    "What are the main differences between a city and a village?",
    "Explain why exercise can improve mood.",
    "Write a short neutral description of a mountain landscape.",
    "What is photosynthesis? Explain it for a beginner.",
    "Give a brief overview of how a bicycle works.",
    "Explain the role of bees in pollination.",
    "Write a short note about keeping a room clean.",
    "Describe the difference between weather and climate.",
    "Explain what a library is and why people use it.",
    "Give a simple explanation of how a camera takes a picture.",
    "Write a short paragraph about learning a new language.",
]


def _dedupe_keep_order(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def extended_select_prompts(mode: str, n: int) -> list[str]:
    base = list(_orig_select_prompts(mode, n))
    if mode == "code":
        pool = _dedupe_keep_order(base + EXTRA_CODE_PROMPTS)
    elif mode == "retain":
        pool = _dedupe_keep_order(base + EXTRA_RETAIN_PROMPTS)
    elif mode == "mixed":
        code = extended_select_prompts("code", (n + 1) // 2)
        retain = extended_select_prompts("retain", n // 2)
        pool = _dedupe_keep_order(code + retain)
    else:
        pool = base
    return pool[:n]


# v1_2 imported select_prompts into its module globals, so replace that symbol.
_run.select_prompts = extended_select_prompts


if __name__ == "__main__":
    _run.main()
