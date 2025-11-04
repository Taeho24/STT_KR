#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""라벨된 세그먼트를 사용해 감정 모델 조합을 평가하는 스크립트.

예시:
    python tools/model_evaluator.py \
        --video assets/simpson.mp4 \
        --labels labelled_simpson.jsonl \
        --audio-models ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition m3hrdadfi/wav2vec2-base-100k-emo \
        --text-models j-hartmann/emotion-english-distilroberta-base \
        --device auto

`--device auto`는 GPU가 가능하면 cuda, 아니면 cpu를 사용합니다.
`--audio-models` 혹은 `--text-models`를 생략하면 config에 정의된 기본 모델만 사용합니다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import io
from collections import defaultdict

# UTF-8 인코딩 강제
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

import whisperx

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config as subtitle_config
from emotion_classifier import EmotionClassifier, DEFAULT_AUDIO_MODEL, DEFAULT_TEXT_MODEL

EMOTIONS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
)


def auto_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def load_labels(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 파싱 실패 (line {line_no}): {exc}") from exc

            label = record.get("label")
            if label is None:
                continue
            label = str(label).strip().lower()
            if not label:
                continue
            if label not in EMOTIONS:
                # 지원하지 않는 감정은 건너뜀
                continue
            record["label"] = label
            record["text"] = str(record.get("text", "")).strip()
            records.append(record)

    if not records:
        raise ValueError("라벨 데이터가 비어 있거나 유효한 레이블이 없습니다.")
    return records


def build_segments(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for record in records:
        segments.append(
            {
                "start": float(record.get("start", 0.0)),
                "end": float(record.get("end", 0.0)),
                "text": record.get("text", ""),
                "speaker": record.get("speaker", "Unknown"),
                "voice_type": record.get("voice_type", "normal"),
            }
        )
    return segments


def metric_key(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        metrics["accuracy"],
        metrics["macro_f1"],
        -metrics["cross_entropy"],
    )


def summarize_confusion(cm: Dict[str, Dict[str, int]]) -> str:
    lines = ["Confusion Matrix (true -> pred):"]
    header = "        " + " ".join(f"{emotion[:3]:>6}" for emotion in EMOTIONS)
    lines.append(header)
    for true_emotion in EMOTIONS:
        row = [f"{true_emotion[:3]:>6}"]
        for pred_emotion in EMOTIONS:
            row.append(f"{cm[true_emotion][pred_emotion]:>6}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def compute_metrics(records: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, float]:
    total = len(records)
    correct = 0
    ce_sum = 0.0
    cm: Dict[str, Dict[str, int]] = {emotion: defaultdict(int) for emotion in EMOTIONS}

    for record, pred in zip(records, predictions):
        true_label = record["label"]
        cm[true_label][pred] += 1
        if pred == true_label:
            correct += 1
        # cross entropy (예측 확률이 없으므로 0/1 확률 기반)
        ce_sum -= math.log(1e-9 if pred != true_label else 1.0 - 1e-9)

    accuracy = correct / total
    macro_f1 = compute_macro_f1(cm)
    cross_entropy = ce_sum / total
    neutral_rate = sum(cm[label]["neutral"] for label in EMOTIONS) / total

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "cross_entropy": cross_entropy,
        "neutral_rate": neutral_rate,
        "confusion": cm,
    }


def compute_macro_f1(cm: Dict[str, Dict[str, int]]) -> float:
    f1_scores: List[float] = []
    for emotion in EMOTIONS:
        tp = cm[emotion][emotion]
        fp = sum(cm[other][emotion] for other in EMOTIONS if other != emotion)
        fn = sum(cm[emotion][other] for other in EMOTIONS if other != emotion)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
    return sum(f1_scores) / len(f1_scores)


def evaluate_combination(
    video_path: Path,
    records: List[Dict[str, Any]],
    audio_model: str,
    text_model: str | None,
    device: str,
    batch_size: int,
    cache_dir: Path,
) -> Dict[str, Any]:
    print("\n==============================")
    print(f"Evaluating combination")
    text_model_display = text_model if text_model is not None else "<disabled>"
    print(f"  audio_model: {audio_model}")
    print(f"  text_model : {text_model_display}")

    full_audio = whisperx.load_audio(str(video_path))
    segments = build_segments(records)

    classifier = EmotionClassifier(
        device=device,
        batch_size=batch_size,
        cache_dir=str(cache_dir),
        audio_model_name=audio_model,
        text_model_name=text_model,
        enable_text=text_model is not None,
    )

    # EmotionClassifier는 입력 세그먼트를 수정하므로 복사본 사용
    import copy

    classified_segments = classifier.classify_emotions(copy.deepcopy(segments), full_audio)
    predictions = [segment.get("emotion", "neutral") for segment in classified_segments]

    metrics = compute_metrics(records, predictions)
    print(
        "  accuracy={accuracy:.3f} | macro_f1={macro_f1:.3f} | neutral_rate={neutral_rate:.3f} | cross_entropy={cross_entropy:.3f}".format(
            **metrics
        )
    )
    print("  samples:")
    for record, pred in list(zip(records, predictions))[:5]:
        print(f"    - text='{record['text'][:40]}' | label={record['label']} | pred={pred}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "audio_model": audio_model,
        "text_model": text_model,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="감정 모델 조합 평가")
    parser.add_argument("--video", type=Path, required=True, help="평가에 사용할 비디오/오디오 파일 경로")
    parser.add_argument("--labels", type=Path, required=True, help="라벨 JSONL 경로")
    parser.add_argument(
        "--audio-models",
        nargs="*",
        default=None,
        help="평가할 오디오 모델 목록"
    )
    parser.add_argument(
        "--text-models",
        nargs="*",
        default=None,
        help="평가할 텍스트 모델 목록"
    )
    parser.add_argument("--device", default="auto", help="cuda/cpu/auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument(
        "--disable-text",
        action="store_true",
        help="텍스트 감정 모델을 완전히 비활성화하고 오디오 모델만 평가"
    )

    args = parser.parse_args()

    device = auto_device(args.device)
    cache_dir = args.cache_dir
    cache_dir.mkdir(exist_ok=True, parents=True)

    records = load_labels(args.labels)

    if args.audio_models:
        audio_models = args.audio_models
    else:
        audio_models = subtitle_config.get("models", "audio_candidates", default=None)
        if not audio_models:
            audio_models = [DEFAULT_AUDIO_MODEL]

    if args.disable_text:
        text_models = [None]
    elif args.text_models:
        text_models = args.text_models
    else:
        text_models = subtitle_config.get("models", "text_candidates", default=None)
        if not text_models:
            text_models = [DEFAULT_TEXT_MODEL]

    combinations: List[Dict[str, Any]] = []
    for audio_model in audio_models:
        for text_model in text_models:
            try:
                result = evaluate_combination(
                    video_path=args.video,
                    records=records,
                    audio_model=audio_model,
                    text_model=text_model,
                    device=device,
                    batch_size=args.batch_size,
                    cache_dir=cache_dir,
                )
                combinations.append(result)
            except Exception as exc:
                print(f"[WARN] 모델 조합 평가 실패: audio={audio_model}, text={text_model}, error={exc}")

    if not combinations:
        print("평가 가능한 모델 조합이 없습니다.")
        return

    combinations.sort(key=lambda item: metric_key(item["metrics"]), reverse=True)

    print("\n===== Evaluation Summary =====")
    for rank, combo in enumerate(combinations, start=1):
        metrics = combo["metrics"]
        print(
            f"[{rank}] accuracy={metrics['accuracy']:.3f} | macro_f1={metrics['macro_f1']:.3f} | neutral_rate={metrics['neutral_rate']:.3f}"
        )
        print(f"     audio={combo['audio_model']}")
        text_label = combo['text_model'] if combo['text_model'] is not None else "<disabled>"
        print(f"     text ={text_label}")
        if rank == 1:
            print(summarize_confusion(metrics["confusion"]))
    print("==============================")


if __name__ == "__main__":
    main()
