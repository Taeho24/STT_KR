#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick syntax check for project Python files.
- Compiles all .py under the workspace excluding common non-source folders.
- Exits non-zero on first failure.
"""
import os
import sys
import py_compile
from pathlib import Path

EXCLUDES = {'.git', '.venv', 'venv', '__pycache__', 'env', 'build', 'dist', '.mypy_cache'}
EXCLUDE_PATH_PARTS = {"[ProtoType]Website", "test_models"}

ROOT = Path(__file__).resolve().parents[1]

failed = False
for path in ROOT.rglob("*.py"):
    rel = path.relative_to(ROOT)
    parts = set(rel.parts)
    if parts & EXCLUDES:
        continue
    if parts & EXCLUDE_PATH_PARTS:
        continue
    try:
        py_compile.compile(str(path), doraise=True)
        print(f"OK  {rel}")
    except Exception as e:
        print(f"ERR {rel}: {e}")
        failed = True
        break

sys.exit(1 if failed else 0)
