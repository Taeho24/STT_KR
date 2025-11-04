"""멀티모달 감정 분류기의 자기 점검을 위한 진단 스크립트.

라벨 데이터가 없어도 사용할 수 있도록 다음 지표를 계산합니다.

* 예측 감정 분포: 각 감정이 얼마나 자주 최종 결과로 선택됐는지.
* 모달리티 일치도: 오디오/텍스트 분포 간 Jensen-Shannon Divergence(JS Divergence).
* 모달리티 지배도: 각 세그먼트에서 어떤 모달리티가 앙상블을 주도했는지.
* 이견 구간 탐지: 오디오와 텍스트가 가장 크게 갈리는 구간 상위 N개.

예시:
    python tools/ensemble_diagnostics.py \
        --video assets/조커.mp4 \
        --segments result/조커_segments_for_labeling.jsonl \
        --device auto
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import whisperx

import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from emotion_classifier import EmotionClassifier  # noqa: E402

EMOTIONS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
)


@dataclass
class DiagnosticsResult:
    predictions: Counter
    kl_stats: Dict[str, float]
    dominance_stats: Dict[str, float]
    disagreement_segments: List[Dict[str, object]]
    high_confidence_segments: List[Dict[str, object]]
    confidence_settings: Dict[str, float]


def auto_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def load_segments(path: Path) -> List[Dict[str, object]]:
    segments: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL 파싱 실패 (line {line_no}): {exc}") from exc

            segment = {
                "start": float(record.get("start", 0.0)),
                "end": float(record.get("end", 0.0)),
                "text": str(record.get("text", "")),
                "speaker": record.get("speaker", "Unknown"),
                "voice_type": record.get("voice_type", "normal"),
            }
            segments.append(segment)

    if not segments:
        raise ValueError("세그먼트가 비어 있습니다. 파이프라인을 먼저 실행해 주세요.")
    return segments


def safe_distribution(raw: Dict[str, float]) -> np.ndarray:
    arr = np.array([max(float(raw.get(emotion, 0.0)), 1e-9) for emotion in EMOTIONS], dtype=np.float64)
    total = float(arr.sum())
    if total <= 0:
        return np.full(len(EMOTIONS), 1.0 / len(EMOTIONS), dtype=np.float64)
    return arr / total


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def modality_dominance(text_dist: np.ndarray, audio_dist: np.ndarray, combined_dist: np.ndarray) -> Tuple[float, float]:
    """각 모달리티가 최종 분포에 얼마나 기여했는지 측정한다.

    반환값은 (text_shift, audio_shift)로, combined_dist와의 평균 L1 차이로 정의한다.
    """

    text_shift = float(np.mean(np.abs(combined_dist - text_dist)))
    audio_shift = float(np.mean(np.abs(combined_dist - audio_dist)))
    return text_shift, audio_shift


def build_disagreement_records(segments: Iterable[Dict[str, object]],
                               text_dists: List[Dict[str, float]],
                               audio_dists: List[Dict[str, float]],
                               combined_dists: List[Dict[str, float]],
                               top_n: int) -> List[Dict[str, object]]:
    records: List[Tuple[float, Dict[str, object]]] = []
    for segment, text_dist, audio_dist, combined_dist in zip(segments, text_dists, audio_dists, combined_dists):
        text_np = safe_distribution(text_dist)
        audio_np = safe_distribution(audio_dist)
        combined_np = safe_distribution(combined_dist)

        # 텍스트와 오디오 간 JS divergence로 이견 정도 측정
        disagreement = js_divergence(text_np, audio_np)

        records.append(
            (
                disagreement,
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "disagreement": disagreement,
                    "text_top": top_labels(text_dist),
                    "audio_top": top_labels(audio_dist),
                    "combined_top": top_labels(combined_dist),
                },
            )
        )

    records.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in records[:top_n]]


def top_labels(dist: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
    return sorted(dist.items(), key=lambda item: item[1], reverse=True)[:k]


def extract_high_confidence_segments(
        segments: Iterable[Dict[str, object]],
        text_dists: List[Dict[str, float]],
        audio_dists: List[Dict[str, float]],
        combined_dists: List[Dict[str, float]],
        min_confidence: float,
        max_js: float,
        max_neutral: float,
        require_modal_agreement: bool) -> List[Dict[str, object]]:
    high_confidence: List[Dict[str, object]] = []

    for segment, text_dist, audio_dist, combined_dist in zip(segments, text_dists, audio_dists, combined_dists):
        if not combined_dist:
            continue

        combined_items = list(combined_dist.items())
        top_emotion, top_score = max(combined_items, key=lambda item: item[1])
        neutral_score = float(combined_dist.get("neutral", 0.0))

        if top_score < min_confidence:
            continue

        if top_emotion == "neutral" and neutral_score > max_neutral:
            continue

        text_top = top_labels(text_dist)
        audio_top = top_labels(audio_dist)
        combined_top = top_labels(combined_dist)

        modal_agreement = bool(text_top and audio_top and text_top[0][0] == top_emotion and audio_top[0][0] == top_emotion)
        if require_modal_agreement and not modal_agreement:
            continue

        text_np = safe_distribution(text_dist)
        audio_np = safe_distribution(audio_dist)
        js_value = js_divergence(text_np, audio_np)
        if js_value > max_js:
            continue

        text_snippet = str(segment.get("text", "")).strip()
        if not text_snippet:
            continue

        high_confidence.append({
            "start": float(segment.get("start", 0.0)),
            "end": float(segment.get("end", 0.0)),
            "text": text_snippet,
            "pseudo_label": top_emotion,
            "confidence": float(top_score),
            "neutral_score": neutral_score,
            "js_divergence": float(js_value),
            "modal_agreement": modal_agreement,
            "text_top": [{"emotion": emo, "score": float(score)} for emo, score in text_top],
            "audio_top": [{"emotion": emo, "score": float(score)} for emo, score in audio_top],
            "combined_top": [{"emotion": emo, "score": float(score)} for emo, score in combined_top],
        })

    return high_confidence


def run_diagnostics(video_path: Path,
                    segment_path: Path,
                    device: str,
                    batch_size: int,
                    cache_dir: Path,
                    top_n: int,
                    min_confidence: float,
                    max_js: float,
                    max_neutral: float,
                    require_modal_agreement: bool) -> DiagnosticsResult:
    segments = load_segments(segment_path)
    audio = whisperx.load_audio(str(video_path))

    classifier = EmotionClassifier(
        device=device,
        batch_size=batch_size,
        cache_dir=str(cache_dir),
    )

    classified = classifier.classify_emotions([dict(seg) for seg in segments], audio)

    predictions = Counter(seg.get("emotion", "neutral") for seg in classified)

    text_dists = [seg.get("text_scores", {}) for seg in classified]
    audio_dists = [seg.get("audio_scores", {}) for seg in classified]
    combined_dists = [seg.get("combined_scores", {}) for seg in classified]

    js_values: List[float] = []
    text_shifts: List[float] = []
    audio_shifts: List[float] = []

    for text_dist, audio_dist, combined_dist in zip(text_dists, audio_dists, combined_dists):
        text_np = safe_distribution(text_dist)
        audio_np = safe_distribution(audio_dist)
        combined_np = safe_distribution(combined_dist)

        js_values.append(js_divergence(text_np, audio_np))
        text_shift, audio_shift = modality_dominance(text_np, audio_np, combined_np)
        text_shifts.append(text_shift)
        audio_shifts.append(audio_shift)

    kl_stats = {
        "js_mean": float(np.mean(js_values)),
        "js_std": float(np.std(js_values)),
        "js_p90": float(np.percentile(js_values, 90)),
    }

    dominance_stats = {
        "text_shift_mean": float(np.mean(text_shifts)),
        "audio_shift_mean": float(np.mean(audio_shifts)),
        "dominant_modality": "audio" if np.mean(audio_shifts) < np.mean(text_shifts) else "text",
    }

    disagreement_segments = build_disagreement_records(
        segments,
        text_dists,
        audio_dists,
        combined_dists,
        top_n,
    )

    high_confidence_segments = extract_high_confidence_segments(
        segments,
        text_dists,
        audio_dists,
        combined_dists,
        min_confidence=min_confidence,
        max_js=max_js,
        max_neutral=max_neutral,
        require_modal_agreement=require_modal_agreement,
    )

    confidence_settings = {
        "min_confidence": float(min_confidence),
        "max_js": float(max_js),
        "max_neutral": float(max_neutral),
        "require_modal_agreement": bool(require_modal_agreement),
    }

    return DiagnosticsResult(
        predictions=predictions,
        kl_stats=kl_stats,
        dominance_stats=dominance_stats,
        disagreement_segments=disagreement_segments,
        high_confidence_segments=high_confidence_segments,
        confidence_settings=confidence_settings,
    )


def format_counter(counter: Counter) -> str:
    total = sum(counter.values())
    lines = []
    for emotion in EMOTIONS:
        count = counter.get(emotion, 0)
        ratio = count / total if total else 0.0
        lines.append(f"  - {emotion:8s}: {count:5d} ({ratio:.2%})")
    return "\n".join(lines)


def print_diagnostics(result: DiagnosticsResult) -> None:
    print("\n===== Ensemble Diagnostics =====")
    print("Prediction distribution:")
    print(format_counter(result.predictions))

    print("\nModality alignment (Jensen-Shannon Divergence between text/audio):")
    print(f"  mean={result.kl_stats['js_mean']:.4f} | std={result.kl_stats['js_std']:.4f} | p90={result.kl_stats['js_p90']:.4f}")

    print("\nModality dominance (average shift from combined distribution):")
    print("  text_shift_mean = {:.4f}".format(result.dominance_stats["text_shift_mean"]))
    print("  audio_shift_mean = {:.4f}".format(result.dominance_stats["audio_shift_mean"]))
    print("  dominant modality ≈ {}".format(result.dominance_stats["dominant_modality"]))

    if result.disagreement_segments:
        print("\nTop disagreement segments (text vs audio):")
        for idx, segment in enumerate(result.disagreement_segments, start=1):
            text_top = ", ".join(f"{emo}:{score:.2f}" for emo, score in segment["text_top"])
            audio_top = ", ".join(f"{emo}:{score:.2f}" for emo, score in segment["audio_top"])
            combined_top = ", ".join(f"{emo}:{score:.2f}" for emo, score in segment["combined_top"])

            print(f"  [{idx}] {segment['start']:.2f}s → {segment['end']:.2f}s | disagreement={segment['disagreement']:.4f}")
            print(f"      text    : {text_top}")
            print(f"      audio   : {audio_top}")
            print(f"      combined: {combined_top}")
            snippet = str(segment["text"]).strip()
            if snippet:
                print(f"      text snippet: {snippet[:80]}{'...' if len(snippet) > 80 else ''}")
    else:
        print("\nNo high-disagreement segments found.")

    hc_count = len(result.high_confidence_segments)
    settings = result.confidence_settings
    print("\nHigh-confidence consensus segments:")
    print("  extracted = {} (min_conf={:.2f}, max_js={:.2f}, max_neutral={:.2f}, require_modal_agreement={})".format(
        hc_count,
        settings.get("min_confidence", 0.0),
        settings.get("max_js", 0.0),
        settings.get("max_neutral", 0.0),
        settings.get("require_modal_agreement", True),
    ))
    if hc_count:
        sample = result.high_confidence_segments[:3]
        for idx, segment in enumerate(sample, start=1):
            combined_top = ", ".join(f"{item['emotion']}:{item['score']:.2f}" for item in segment["combined_top"][:2])
            print(f"    [{idx}] {segment['start']:.2f}s → {segment['end']:.2f}s | {segment['pseudo_label']} ({segment['confidence']:.2f}) | JS={segment['js_divergence']:.3f}")
            print(f"        combined top: {combined_top}")
            snippet = segment["text"]
            if snippet:
                print(f"        text: {snippet[:80]}{'...' if len(snippet) > 80 else ''}")
    else:
        print("  (none)")

    print("================================\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="멀티모달 감정 분류기 진단 도구")
    parser.add_argument("--video", type=Path, required=True, help="분석에 사용할 비디오/오디오 파일")
    parser.add_argument("--segments", type=Path, required=True, help="세그먼트 JSONL 경로")
    parser.add_argument("--device", default="auto", help="cuda/cpu/auto")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    parser.add_argument("--top-n", type=int, default=10, help="이견 구간 상위 N개 출력")
    parser.add_argument("--output", type=Path, default=None, help="진단 결과를 JSON으로 저장할 경로")
    parser.add_argument("--high-confidence-output", type=Path, default=None,
                        help="고신뢰 합의 세그먼트를 JSON/JSONL로 저장할 경로")
    parser.add_argument("--min-confidence", type=float, default=0.70,
                        help="합의 세그먼트로 간주할 최소 결합 확률")
    parser.add_argument("--max-js", type=float, default=0.12,
                        help="허용되는 최대 텍스트-오디오 JS divergence")
    parser.add_argument("--max-neutral", type=float, default=0.55,
                        help="중립 감정이 선택되었을 때 허용되는 최대 비율")
    parser.add_argument("--no-modal-agreement", action="store_true",
                        help="오디오/텍스트 상위 감정이 일치하지 않아도 포함")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = auto_device(args.device)

    result = run_diagnostics(
        video_path=args.video,
        segment_path=args.segments,
        device=device,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
        top_n=args.top_n,
        min_confidence=args.min_confidence,
        max_js=args.max_js,
        max_neutral=args.max_neutral,
        require_modal_agreement=not args.no_modal_agreement,
    )

    print_diagnostics(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "predictions": dict(result.predictions),
            "kl_stats": result.kl_stats,
            "dominance_stats": result.dominance_stats,
            "disagreement_segments": result.disagreement_segments,
            "high_confidence_segments": result.high_confidence_segments,
            "confidence_settings": result.confidence_settings,
        }
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"결과를 저장했습니다: {args.output}")

    if args.high_confidence_output:
        args.high_confidence_output.parent.mkdir(parents=True, exist_ok=True)
        records = result.high_confidence_segments
        if args.high_confidence_output.suffix.lower() == ".jsonl":
            with args.high_confidence_output.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            with args.high_confidence_output.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"고신뢰 세그먼트 {len(records)}개를 저장했습니다: {args.high_confidence_output}")


if __name__ == "__main__":
    main()
