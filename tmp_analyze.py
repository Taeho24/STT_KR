#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deprecated wrapper for quick analysis. Use:
  python tools/analyzers/audio_top_scores.py --segments result/<video>_segments_for_labeling.jsonl

This wrapper forwards to the new tool to avoid hard-coded paths in core.
"""

import runpy
import sys
from pathlib import Path

TOOL_PATH = Path(__file__).parent / "tools" / "analyzers" / "audio_top_scores.py"

if __name__ == "__main__":
    sys.argv = [str(TOOL_PATH), *sys.argv[1:]]
    if not TOOL_PATH.exists():
        raise SystemExit("Missing tool: tools/analyzers/audio_top_scores.py")
    runpy.run_path(str(TOOL_PATH), run_name="__main__")
