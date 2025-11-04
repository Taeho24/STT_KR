"""간단한 감정 앙상블 하이퍼파라미터 탐색 스크립트.

입력 데이터 형식(JSON Lines):
{
  "segment_id": "optional",
  "label": "sad",
  "text_scores": {"neutral": 0.3, "sad": 0.5, ...},
  "audio_scores": {"neutral": 0.4, "sad": 0.4, ...}
}

- text_scores / audio_scores 는 EmotionClassifier 내부 로그 혹은 별도 덤프에서 추출한 확률 값.
- label 은 사람이 지정한 정답 감정.
- 모든 확률은 0~1 사이 값이며, 누락된 감정은 0으로 처리됨.

사용 예:
python tools/weight_tuner.py labelled_segments.jsonl --audio-weights 0.6 0.7 0.8 --emotion-grid neutral=0.8,0.9,1.0 sad=1.0,1.1,1.2
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

EMOTIONS: Tuple[str, ...] = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "surprise",
    "disgust",
)

DEFAULT_AUTO_AUDIO_CANDIDATES = [0.4, 0.5, 0.6, 0.7, 0.8]
DEFAULT_AUTO_EMOTION_CANDIDATES = [0.8, 0.9, 1.0, 1.1, 1.2]


def load_dataset(path: Path) -> List[Dict[str, Dict[str, float]]]:
    """JSON Lines 형식 데이터 로드."""
    samples: List[Dict[str, Dict[str, float]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON 파싱 실패 (line {line_no}): {exc}") from exc

            label = record.get("label")
            if label is None or (isinstance(label, str) and not label.strip()):
                # 건너뛴 세그먼트는 무시
                continue

            samples.append(record)
    if not samples:
        raise ValueError("데이터가 비어 있습니다.")
    return samples


def parse_emotion_grid(specs: Iterable[str]) -> Dict[str, List[float]]:
    """emotion=0.8,1.0 형식 문자열을 해석해 탐색 후보 사전을 반환."""
    grid: Dict[str, List[float]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"emotion-grid 형식 오류: {spec}")
        emotion, values = spec.split("=", 1)
        emotion = emotion.strip().lower()
        if emotion not in EMOTIONS:
            raise ValueError(f"알 수 없는 감정 이름: {emotion}")
        try:
            candidates = [float(v) for v in values.split(",")]
        except ValueError as exc:
            raise ValueError(f"숫자 파싱 실패: {spec}") from exc
        if not candidates:
            raise ValueError(f"조정 후보가 비어 있습니다: {spec}")
        grid[emotion] = candidates
    return grid


def combine_scores(
    text_scores: Dict[str, float],
    audio_scores: Dict[str, float],
    audio_weight: float,
    emotion_weights: Dict[str, float],
) -> Dict[str, float]:
    """텍스트/오디오 점수와 가중치를 결합해 최종 확률을 반환."""
    text_weight = 1.0 - audio_weight
    combined = {}
    for emotion in EMOTIONS:
        ts = float(text_scores.get(emotion, 0.0))
        ascore = float(audio_scores.get(emotion, 0.0))
        weighted = ts * text_weight + ascore * audio_weight
        weighted *= emotion_weights.get(emotion, 1.0)
        combined[emotion] = max(weighted, 0.0)

    total = sum(combined.values())
    if total <= 0:
        # 모든 값이 0인 경우 중립을 기본값으로 둠
        combined = {emotion: (1.0 if emotion == "neutral" else 0.0) for emotion in EMOTIONS}
        total = 1.0

    return {emotion: value / total for emotion, value in combined.items()}


def evaluate_dataset(
    dataset: List[Dict[str, Dict[str, float]]],
    audio_weight: float,
    emotion_weights: Dict[str, float],
) -> Dict[str, float]:
    """주어진 설정에서 accuracy, cross entropy, macro F1 등을 계산."""
    total = len(dataset)
    if not 0.0 <= audio_weight <= 1.0:
        raise ValueError("audio_weight 는 0~1 사이여야 합니다.")

    ce_sum = 0.0
    correct = 0
    cm = {emotion: defaultdict(int) for emotion in EMOTIONS}  # true -> pred

    for sample in dataset:
        label = sample["label"].lower()
        if label not in EMOTIONS:
            continue  # 미지정 레이블은 평가에서 제외
        text_scores = sample.get("text_scores", {})
        audio_scores = sample.get("audio_scores", {})
        probs = combine_scores(text_scores, audio_scores, audio_weight, emotion_weights)

        pred = max(probs.items(), key=lambda item: item[1])[0]
        cm[label][pred] += 1

        ce_sum -= math.log(max(probs.get(label, 1e-9), 1e-9))
        if pred == label:
            correct += 1

    accuracy = correct / total
    cross_entropy = ce_sum / total
    macro_f1 = compute_macro_f1(cm)
    neutral_rate = sum(cm[emotion]["neutral"] for emotion in EMOTIONS) / total

    return {
        "accuracy": accuracy,
        "cross_entropy": cross_entropy,
        "macro_f1": macro_f1,
        "neutral_rate": neutral_rate,
        "confusion": cm,
    }


def compute_macro_f1(cm: Dict[str, Dict[str, int]]) -> float:
    """혼동 행렬에서 macro F1 점수 계산."""
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


def summarize_confusion(cm: Dict[str, Dict[str, int]]) -> str:
    lines = ["Confusion Matrix (true -> pred):"]
    emotions = list(EMOTIONS)
    header = "        " + " ".join(f"{e[:3]:>6}" for e in emotions)
    lines.append(header)
    for true_emotion in emotions:
        row = [f"{true_emotion[:3]:>6}"]
        for pred_emotion in emotions:
            row.append(f"{cm[true_emotion][pred_emotion]:>6}")
        lines.append(" ".join(row))
    return "\n".join(lines)


def metric_key(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (
        metrics["accuracy"],
        metrics["macro_f1"],
        -metrics["cross_entropy"],
    )


def run_search(
    dataset_path: Path,
    audio_weights: List[float],
    emotion_grid: Dict[str, List[float]],
    top_k: int,
) -> None:
    dataset = load_dataset(dataset_path)
    base_emotion_weights = {emotion: 1.0 for emotion in EMOTIONS}

    best_results: List[Tuple[float, Dict[str, float], Dict[str, float]]] = []

    # emotion_grid 가 비어 있으면 기본 조합 하나만 평가
    if emotion_grid:
        emotion_combos = []
        emotions = list(emotion_grid.keys())
        value_lists = [emotion_grid[e] for e in emotions]
        for values in itertools.product(*value_lists):
            combo = base_emotion_weights.copy()
            combo.update({emotion: value for emotion, value in zip(emotions, values)})
            emotion_combos.append(combo)
    else:
        emotion_combos = [base_emotion_weights]

    for audio_weight in audio_weights:
        for combo in emotion_combos:
            metrics = evaluate_dataset(dataset, audio_weight, combo)
            best_results.append((audio_weight, combo, metrics))

    # 정확도, 매크로 F1, 크로스 엔트로피 기준 정렬
    best_results.sort(key=lambda item: metric_key(item[2]), reverse=True)

    print(f"총 조합 수: {len(best_results)}")
    print("")

    for idx, (audio_weight, combo, metrics) in enumerate(best_results[:top_k], start=1):
        print(f"[{idx}] audio_weight={audio_weight:.2f}")
        print(
            "    emotion_weights="
            + ", ".join(f"{k}:{v:.2f}" for k, v in combo.items() if v != 1.0)
        )
        print(
            "    accuracy={accuracy:.3f} | macro_f1={macro_f1:.3f} | neutral_rate={neutral_rate:.3f} | cross_entropy={cross_entropy:.3f}".format(
                **metrics
            )
        )
        if idx == 1:
            print(summarize_confusion(metrics["confusion"]))
        print("")


def auto_optimize(
    dataset: List[Dict[str, Dict[str, float]]],
    audio_candidates: List[float],
    emotion_candidate_map: Dict[str, List[float]],
    focus_emotions: List[str],
    iterations: int,
) -> Tuple[float, Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    if not audio_candidates:
        audio_candidates = DEFAULT_AUTO_AUDIO_CANDIDATES

    initial_audio = min(audio_candidates, key=lambda v: abs(v - 0.7)) if audio_candidates else 0.7
    best_audio = max(0.0, min(1.0, initial_audio))
    best_emotions = {emotion: 1.0 for emotion in EMOTIONS}
    history: List[Dict[str, Any]] = []

    best_metrics = evaluate_dataset(dataset, best_audio, best_emotions)

    def update_if_better(new_audio: float, new_weights: Dict[str, float], metrics: Dict[str, float], tag: str) -> None:
        nonlocal best_audio, best_emotions, best_metrics
        if metric_key(metrics) > metric_key(best_metrics):
            best_audio = new_audio
            best_emotions = new_weights
            best_metrics = metrics
            history.append(
                {
                    "step": tag,
                    "audio": best_audio,
                    "emotion_weights": best_emotions.copy(),
                    "metrics": metrics,
                }
            )

    for iteration in range(1, max(1, iterations) + 1):
        # 오디오 가중치 탐색
        for candidate in audio_candidates:
            candidate = max(0.0, min(1.0, candidate))
            metrics = evaluate_dataset(dataset, candidate, best_emotions)
            update_if_better(candidate, best_emotions.copy(), metrics, f"iter{iteration}-audio")

        # 감정별 보정 탐색
        for emotion in focus_emotions:
            candidates = list(emotion_candidate_map.get(emotion, DEFAULT_AUTO_EMOTION_CANDIDATES))
            if 1.0 not in candidates:
                candidates.append(1.0)

            for candidate in candidates:
                candidate = max(0.2, min(2.0, candidate))
                new_weights = best_emotions.copy()
                new_weights[emotion] = candidate
                metrics = evaluate_dataset(dataset, best_audio, new_weights)
                update_if_better(best_audio, new_weights, metrics, f"iter{iteration}-{emotion}")

    return best_audio, best_emotions, best_metrics, history


def main() -> None:
    parser = argparse.ArgumentParser(description="감정 앙상블 하이퍼파라미터 탐색")
    parser.add_argument("dataset", type=Path, help="라벨링된 JSONL 파일 경로")
    parser.add_argument(
        "--audio-weights",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8],
        help="탐색할 오디오 가중치 후보 (텍스트 가중치는 1-audio)",
    )
    parser.add_argument(
        "--emotion-grid",
        type=str,
        nargs="+",
        default=["neutral=0.8,0.9,1.0", "sad=1.0,1.1,1.2"],
        help="emotion=값1,값2 형식으로 감정별 가중치 후보 지정",
    )
    parser.add_argument("--top-k", type=int, default=5, help="상위 몇 개 조합을 출력할지")
    parser.add_argument("--auto", action="store_true", help=" 자동 조정 모드를 활성화")
    parser.add_argument(
        "--auto-iterations",
        type=int,
        default=2,
        help="자동 모드에서 좌표 탐색 반복 횟수",
    )
    parser.add_argument(
        "--auto-emotions",
        type=str,
        default=None,
        help="자동 모드에서 조정할 감정 목록(콤마 구분). 미지정 시 라벨에 등장한 감정 사용",
    )

    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    if args.auto:
        if args.auto_emotions:
            focus_emotions = [e.strip().lower() for e in args.auto_emotions.split(",") if e.strip()]
        else:
            focus_emotions = sorted({sample["label"] for sample in dataset if sample.get("label") in EMOTIONS})
        if not focus_emotions:
            focus_emotions = list(EMOTIONS)

        audio_candidates = [w for w in args.audio_weights if 0.0 <= w <= 1.0]
        if not audio_candidates:
            audio_candidates = DEFAULT_AUTO_AUDIO_CANDIDATES

        if args.emotion_grid:
            emotion_candidate_map = parse_emotion_grid(args.emotion_grid)
        else:
            emotion_candidate_map = {emotion: DEFAULT_AUTO_EMOTION_CANDIDATES for emotion in focus_emotions}

        best_audio, best_emotions, best_metrics, history = auto_optimize(
            dataset,
            audio_candidates,
            emotion_candidate_map,
            focus_emotions,
            args.auto_iterations,
        )

        print("자동 조정 결과")
        print(f"  audio_weight: {best_audio:.3f}")
        changed_weights = {
            emotion: weight for emotion, weight in best_emotions.items() if abs(weight - 1.0) > 1e-6
        }
        if changed_weights:
            print(
                "  emotion_weights: "
                + ", ".join(f"{emotion}:{weight:.3f}" for emotion, weight in changed_weights.items())
            )
        else:
            print("  emotion_weights: (변경 없음)")
        print(
            "  accuracy={accuracy:.3f} | macro_f1={macro_f1:.3f} | neutral_rate={neutral_rate:.3f} | cross_entropy={cross_entropy:.3f}".format(
                **best_metrics
            )
        )
        if history:
            print("  개선 단계:")
            for item in history:
                delta_weights = {
                    emotion: weight
                    for emotion, weight in item["emotion_weights"].items()
                    if abs(weight - 1.0) > 1e-6
                }
                if delta_weights:
                    weight_str = ", ".join(f"{k}:{v:.3f}" for k, v in delta_weights.items())
                else:
                    weight_str = "(변경 없음)"
                print(
                    f"    - {item['step']}: audio={item['audio']:.3f}, weights={weight_str} | accuracy={item['metrics']['accuracy']:.3f}, macro_f1={item['metrics']['macro_f1']:.3f}"
                )
        return

    audio_weights = [w for w in args.audio_weights if 0.0 <= w <= 1.0]
    if not audio_weights:
        raise ValueError("유효한 audio 가중치 후보가 없습니다.")

    emotion_grid = parse_emotion_grid(args.emotion_grid) if args.emotion_grid else {}
    run_search(args.dataset, audio_weights, emotion_grid, args.top_k)


if __name__ == "__main__":
    main()
