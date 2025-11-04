"""세그먼트 감정 라벨링을 위한 간단한 CLI 도구.

사용 방법
---------
1. 먼저 `python main.py <video>`를 실행해 `result/<video_stem>_segments_for_labeling.jsonl`
   파일을 생성합니다.
2. 아래 예시처럼 이 스크립트를 실행하여 각 세그먼트에 대한 정답 감정을 입력합니다.

    python tools/interactive_labeler.py \
        --segments result/simpson_segments_for_labeling.jsonl \
        --output labelled_simpson.jsonl

3. 출력된 JSONL 파일을 `tools/weight_tuner.py`의 입력으로 사용할 수 있습니다.

라벨 입력 시 기본값은 빈 문자열입니다. 아무 것도 입력하지 않고 Enter를 누르면
모델이 예측한 감정을 그대로 사용합니다. 'q'를 입력하면 중간에 종료하며,
이미 라벨링한 결과는 `--output` 파일에 저장됩니다. 재실행 시 `--resume` 옵션을
주면 기존 출력 내용을 불러와 이어서 진행합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

EMOTIONS = ["neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"]
EMOTION_SHORTCUTS = {str(idx): emotion for idx, emotion in enumerate(EMOTIONS, start=1)}
SKIP_TOKENS = {"s", "skip"}


def load_segments(path: Path) -> List[Dict]:
    segments: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 파싱 실패 (line {line_no}): {exc}") from exc
            segments.append(record)
    if not segments:
        raise ValueError("세그먼트 파일이 비어 있습니다.")
    return segments


def load_existing_labels(path: Path) -> Dict[int, Dict]:
    if not path.exists():
        return {}
    labelled: Dict[int, Dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            idx = record.get("index")
            if idx is None:
                continue
            labelled[idx] = record
    return labelled


def show_segment(record: Dict, index: int) -> None:
    start = record.get("start", 0.0)
    end = record.get("end", 0.0)
    text = record.get("text", "").strip()
    predicted = record.get("predicted_emotion", "unknown")
    confidence = record.get("confidence", 0.0)
    text_scores = record.get("text_scores", {})
    audio_scores = record.get("audio_scores", {})
    combined_scores = record.get("combined_scores", {})

    def _top_k(scores: Dict[str, float], k: int = 3) -> str:
        items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return " | ".join(f"{emotion}:{value:.2f}" for emotion, value in items)

    print("\n====================================")
    print(f"[{index}] {start:.2f}s → {end:.2f}s")
    print(f"문장: {text}")
    print(f"예측 감정: {predicted} (confidence {confidence:.2f})")
    if text_scores:
        print(f"텍스트 상위: {_top_k(text_scores)}")
    if audio_scores:
        print(f"오디오 상위: {_top_k(audio_scores)}")
    if combined_scores:
        print(f"결합 상위: {_top_k(combined_scores)}")
    shortcuts = " | ".join(f"{num}:{emotion}" for num, emotion in EMOTION_SHORTCUTS.items())
    print(f"허용 감정: {', '.join(EMOTIONS)}")
    print(f"숫자 입력: {shortcuts}")
    print("입력 도움: 숫자/영문, 's' 건너뛰기, 'q' 종료")


def prompt_label(default: str) -> str:
    while True:
        user_input = input(f"정답 감정 입력(기본값 {default}): ").strip().lower()
        if user_input == "":
            return default
        if user_input in ("q", "quit"):
            return "__quit__"
        if user_input in SKIP_TOKENS:
            return "__skip__"
        if user_input in EMOTIONS:
            return user_input
        if user_input in EMOTION_SHORTCUTS:
            return EMOTION_SHORTCUTS[user_input]
        print(f"잘못된 감정입니다. 허용 값: {', '.join(EMOTIONS)}")
        print("숫자 입력도 가능합니다: " + ", ".join(f"{k}->{v}" for k, v in EMOTION_SHORTCUTS.items()))


def annotate(segments: List[Dict], output_path: Path, resume: bool = False) -> None:
    labelled = load_existing_labels(output_path) if resume else {}
    start_index = max(labelled.keys(), default=-1) + 1

    with output_path.open("a", encoding="utf-8") as out_f:
        for idx, segment in enumerate(segments):
            if idx < start_index:
                continue
            show_segment(segment, idx)
            default = segment.get("predicted_emotion", "neutral")
            label = prompt_label(default)
            if label == "__quit__":
                print("라벨링을 중단합니다. 나머지는 나중에 이어서 진행하세요.")
                break
            if label == "__skip__":
                record = {
                    "index": idx,
                    "video": segment.get("video"),
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text"),
                    "speaker": segment.get("speaker"),
                    "voice_type": segment.get("voice_type"),
                    "predicted_emotion": segment.get("predicted_emotion"),
                    "confidence": segment.get("confidence"),
                    "text_scores": segment.get("text_scores"),
                    "audio_scores": segment.get("audio_scores"),
                    "combined_scores": segment.get("combined_scores"),
                    "label": None,
                    "skipped": True,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                print("→ 건너뜀")
                continue
            record = {
                "index": idx,
                "video": segment.get("video"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text"),
                "speaker": segment.get("speaker"),
                "voice_type": segment.get("voice_type"),
                "predicted_emotion": segment.get("predicted_emotion"),
                "confidence": segment.get("confidence"),
                "text_scores": segment.get("text_scores"),
                "audio_scores": segment.get("audio_scores"),
                "combined_scores": segment.get("combined_scores"),
                "label": label,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
    print("라벨링이 종료되었습니다.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="세그먼트 감정 라벨링 CLI")
    parser.add_argument("--segments", type=Path, required=True, help="라벨링할 세그먼트 JSONL 경로")
    parser.add_argument("--output", type=Path, required=True, help="라벨링 결과 JSONL 경로")
    parser.add_argument("--resume", action="store_true", help="기존 출력 파일에서 이어서 진행")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    segments = load_segments(args.segments)
    if args.output.exists() and not args.resume:
        confirm = input("출력 파일이 이미 존재합니다. 덮어쓰려면 'y' 입력: ").strip().lower()
        if confirm != "y":
            print("작업을 취소합니다.")
            return
        args.output.unlink()

    annotate(segments, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
