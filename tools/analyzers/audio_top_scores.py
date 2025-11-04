#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quick analysis helper: summarize per-segment top audio emotion scores from a JSONL
produced by main.py (result/<video>_segments_for_labeling.jsonl).

Usage (PowerShell):
  python tools/analyzers/audio_top_scores.py --segments result/truman_segments_for_labeling.jsonl

If --segments is omitted, the script will search for the most recent
result/*_segments_for_labeling.jsonl file and use that.
"""

import argparse
import glob
import json
from pathlib import Path
from collections import Counter


def find_latest_segments_file() -> Path | None:
    candidates = sorted(glob.glob("result/*_segments_for_labeling.jsonl"))
    return Path(candidates[-1]) if candidates else None


def main():
    parser = argparse.ArgumentParser(description="Summarize top audio emotion scores from JSONL segments")
    parser.add_argument("--segments", type=str, default=None, help="Path to *_segments_for_labeling.jsonl")
    args = parser.parse_args()

    path: Path | None
    if args.segments:
        path = Path(args.segments)
    else:
        path = find_latest_segments_file()
        if path:
            print(f"Auto-selected latest segments file: {path}")

    if not path or not path.exists():
        raise SystemExit("No segments file found. Provide --segments pointing to result/<video>_segments_for_labeling.jsonl")

    stats = Counter()
    audio_tops: list[tuple[float, str, float]] = []
    max_audio = 0.0
    min_audio = 1.0

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            stats[data.get('predicted_emotion', 'unknown')] += 1
            audio_scores = data.get('audio_scores') or {}
            if not audio_scores:
                continue
            top_emotion = max(audio_scores, key=audio_scores.get)
            top_score = float(audio_scores[top_emotion])
            max_audio = max(max_audio, top_score)
            min_audio = min(min_audio, top_score)
            audio_tops.append((round(float(data.get('start', 0.0)), 3), top_emotion, round(top_score, 3)))

    print('emotion_counts =', stats)
    if audio_tops:
        print(f'audio top-score range = [{min_audio:.3f}, {max_audio:.3f}]')
        print('first 5 audio tops:', audio_tops[:5])
    else:
        print('No audio scores found in the file.')


if __name__ == "__main__":
    main()
