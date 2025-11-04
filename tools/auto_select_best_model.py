#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
자동 후보 탐색기: 라벨이 있는 여러 영상에 대해 다수의 오디오 감정 모델을 평가하고,
평균 정확도가 가장 높은 모델을 자동으로 추천/선정합니다.

사용 예시 (PowerShell):
  .\venv\Scripts\python.exe tools\auto_select_best_model.py --device auto

옵션:
  --models ...         : 특정 모델만 지정해서 평가 (생략 시 config의 audio_candidates + 일부 추가 후보)
  --disable-text       : 텍스트 감정 모델 비활성화 (기본값: 사용)  ← 오디오 모델 고유 성능을 보려면 켜세요
  --batch-size N       : 배치 크기 (기본 4)
  --cache-dir .cache   : 캐시 디렉토리
  --device auto|cuda|cpu
  --apply-best         : 최고 모델을 config.py의 models.audio에 적용

출력:
  result/auto_select_results.json
  result/auto_select_results.csv
  터미널 로그에 순위표 및 추천 모델 출력
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config as app_config  # type: ignore

# model_evaluator의 내부 함수를 재사용 (정확도/메트릭 계산, 조합 평가 등)
from tools import model_evaluator as ME  # type: ignore


@dataclass
class Dataset:
    name: str
    video: Path
    labels: Path


def discover_datasets() -> List[Dataset]:
    """assets/*.mp4와 labels/*_labels.jsonl, labelled_simpson.jsonl를 매칭해 평가 세트를 구성"""
    datasets: List[Dataset] = []

    assets = ROOT / "assets"
    labels_dir = ROOT / "labels"

    # 이름 규칙: labels/<name>_labels.jsonl ↔ assets/<name>.mp4
    for lp in labels_dir.glob("*_labels.jsonl"):
        name = lp.name.replace("_labels.jsonl", "")
        vp = assets / f"{name}.mp4"
        if vp.exists():
            datasets.append(Dataset(name=name, video=vp, labels=lp))

    # Simpson 특례 (루트에 존재)
    simpson_labels = ROOT / "labelled_simpson.jsonl"
    simpson_video = assets / "simpson.mp4"
    if simpson_labels.exists() and simpson_video.exists():
        datasets.append(Dataset(name="simpson", video=simpson_video, labels=simpson_labels))

    # 중복 제거 (이름 기준)
    uniq: Dict[str, Dataset] = {d.name: d for d in datasets}
    return list(uniq.values())


def load_candidates(extra: Optional[List[str]] = None) -> List[str]:
    """config의 audio_candidates에 기본 후보 + 추가 후보를 병합 후 중복 제거"""
    cfg_list: List[str] = app_config.get("models", "audio_candidates", default=[]) or []

    # 과거/요청 기반 추가 후보 (중복 자동 제거)
    extras = [
        # 사용자 요구 재확인 후보들
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "superb/wav2vec2-large-superb-er",
    ]
    if extra:
        extras.extend(extra)

    merged = []
    seen = set()
    for m in cfg_list + extras:
        if not m:
            continue
        if m not in seen:
            merged.append(m)
            seen.add(m)
    return merged


def evaluate_audio_model_on_dataset(audio_model: str, ds: Dataset, device: str, batch_size: int, cache_dir: Path) -> Dict[str, Any]:
    records = ME.load_labels(ds.labels)
    result = ME.evaluate_combination(
        video_path=ds.video,
        records=records,
        audio_model=audio_model,
        text_model=None,  # 오디오 단독 정확도
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )
    metrics = result["metrics"]

    # 비중립 정확도 계산 (라벨이 non-neutral인 샘플에 대해서만 정확도)
    non_neutral_total = 0
    non_neutral_correct = 0
    cm = metrics.get("confusion", {})
    # ME.EMOTIONS 순회
    for true_label in getattr(ME, "EMOTIONS", ("neutral","happy","sad","angry","fear","surprise","disgust")):
        if true_label == "neutral":
            continue
        row = cm.get(true_label, {})
        row_total = sum(row.values()) if isinstance(row, dict) else 0
        non_neutral_total += row_total
        non_neutral_correct += row.get(true_label, 0) if isinstance(row, dict) else 0
    non_neutral_acc = (non_neutral_correct / non_neutral_total) if non_neutral_total > 0 else None

    return {
        "dataset": ds.name,
        "samples": len(records),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "neutral_rate": metrics["neutral_rate"],
        "non_neutral_accuracy": non_neutral_acc,
    }


def aggregate_scores(per_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
    """데이터셋 크기(샘플 수)로 가중 평균을 계산"""
    total = sum(item["samples"] for item in per_dataset) or 1
    def wavg(key: str) -> float:
        return sum(item[key] * item["samples"] for item in per_dataset) / total
    # non-neutral은 None일 수 있으므로 유효값만 가중 평균
    nn_vals = [(item["non_neutral_accuracy"], item["samples"]) for item in per_dataset if item.get("non_neutral_accuracy") is not None]
    if nn_vals:
        nn_total = sum(n for _, n in nn_vals)
        nn_avg = sum(v * n for v, n in nn_vals) / nn_total
    else:
        nn_avg = float("nan")
    return {
        "avg_accuracy": wavg("accuracy"),
        "avg_macro_f1": wavg("macro_f1"),
        "avg_neutral_rate": wavg("neutral_rate"),
        "avg_non_neutral_accuracy": nn_avg,
        "total_samples": total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="여러 라벨된 영상에 대해 오디오 감정 모델 자동 벤치마크")
    ap.add_argument("--models", nargs="*", default=None, help="평가할 오디오 모델들 (생략 시 자동)")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache"))
    ap.add_argument("--apply-best", action="store_true", help="최고 모델을 config.py에 적용")
    args = ap.parse_args()

    # 장치 자동 결정
    device = ME.auto_device(args.device)
    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 데이터셋 검색
    datasets = discover_datasets()
    if not datasets:
        print("[ERR] 평가 가능한 데이터셋을 찾지 못했습니다. labels/*_labels.jsonl 또는 labelled_simpson.jsonl을 확인하세요.")
        sys.exit(2)

    print(f"\n📀 Datasets: {len(datasets)}개")
    for d in datasets:
        print(f" - {d.name}: video={d.video.name}, labels={d.labels.name}")

    # 후보 모델 수집
    candidates = args.models or load_candidates()
    print(f"\n🧪 Candidates: {len(candidates)}개")
    for m in candidates:
        print(f" - {m}")

    # 평가 루프
    all_results: List[Dict[str, Any]] = []
    ranking: List[Tuple[str, Dict[str, float]]] = []

    for idx, model in enumerate(candidates, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(candidates)}] Evaluating model: {model}")
        per_dataset: List[Dict[str, Any]] = []
        for ds in datasets:
            try:
                r = evaluate_audio_model_on_dataset(
                    audio_model=model,
                    ds=ds,
                    device=device,
                    batch_size=args.batch_size,
                    cache_dir=cache_dir,
                )
                per_dataset.append(r)
                nn = r.get("non_neutral_accuracy")
                nn_str = f"{nn:.3f}" if nn is not None else "n/a"
                print(" - {name:<8} | acc={acc:.3f} | f1={f1:.3f} | neu={neu:.3f} | nn_acc={nn_acc} | n={n}".format(
                    name=ds.name,
                    acc=r["accuracy"],
                    f1=r["macro_f1"],
                    neu=r["neutral_rate"],
                    nn_acc=nn_str,
                    n=r["samples"],
                ))
            except Exception as exc:
                print(f"   [WARN] {ds.name} 실패: {exc}")
        if not per_dataset:
            print("   [SKIP] 모든 데이터셋에서 실패")
            continue

        agg = aggregate_scores(per_dataset)
        nn_avg = agg.get("avg_non_neutral_accuracy")
        nn_avg_str = f"{nn_avg:.3f}" if nn_avg == nn_avg else "n/a"  # NaN 체크
        print(" -> AVG | acc={acc:.3f} | f1={f1:.3f} | neu={neu:.3f} | nn_acc={nn} | total={n}".format(
            acc=agg["avg_accuracy"], f1=agg["avg_macro_f1"], neu=agg["avg_neutral_rate"], nn=nn_avg_str, n=agg["total_samples"],
        ))

        all_results.append({
            "model": model,
            "per_dataset": per_dataset,
            "aggregate": agg,
        })
        ranking.append((model, agg))

    if not ranking:
        print("[ERR] 유효한 평가 결과가 없습니다.")
        sys.exit(3)

    # 정렬: 비중립 정확도 최우선(유효값만), 그 다음 전체 정확도, 그 다음 Macro F1
    def rank_key(item: Tuple[str, Dict[str, float]]):
        agg = item[1]
        nn = agg.get("avg_non_neutral_accuracy")
        nn_val = (-1.0 if nn != nn else nn)  # NaN이면 -1로 취급 (꼴찌)
        return (nn_val, agg.get("avg_accuracy", 0.0), agg.get("avg_macro_f1", 0.0))
    ranking.sort(key=rank_key, reverse=True)

    print("\n" + "=" * 80)
    print("🏆 Overall Ranking (weighted by samples)")
    print("{:<3} {:<55} {:>8} {:>8} {:>8} {:>8}".format("#", "model", "acc", "f1", "neu", "nn_acc"))
    print("-" * 84)
    for i, (m, agg) in enumerate(ranking, start=1):
        nn_avg = agg.get("avg_non_neutral_accuracy")
        nn_avg_str = f"{nn_avg:.3f}" if nn_avg == nn_avg else "n/a"
        print("{:<3} {:<55} {:>8.3f} {:>8.3f} {:>8.3f} {:>8}".format(
            i, m[:55], agg["avg_accuracy"], agg["avg_macro_f1"], agg["avg_neutral_rate"], nn_avg_str
        ))

    best_model, best_scores = ranking[0]
    print("\nBest model:")
    print(json.dumps({"model": best_model, **best_scores}, indent=2, ensure_ascii=False))

    # 결과 저장
    out_dir = ROOT / "result"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "auto_select_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # 간단 CSV도 저장
    try:
        import csv
        with (out_dir / "auto_select_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["model", "avg_acc", "avg_f1", "avg_neu", "total_samples"])
            for m, agg in ranking:
                w.writerow([m, agg["avg_accuracy"], agg["avg_macro_f1"], agg["avg_neutral_rate"], agg["total_samples"]])
    except Exception:
        pass

    # 선택 적용 (옵션)
    if args.apply_best:
        # config.py를 직접 수정하지 않고 사용자에게 추천만 할 수도 있지만, 플래그가 있으면 적용
        try:
            cfg_path = ROOT / "config.py"
            txt = cfg_path.read_text(encoding="utf-8")
            # 'audio': '<...>' 값을 치환 (간단한 방법)
            import re
            new_txt, n = re.subn(r"('audio'\s*:\s*)'[^']+'", rf"\1'{best_model}'", txt, count=1)
            if n == 0:
                print("[WARN] config.py에서 'audio' 항목을 찾지 못해 추가는 생략합니다.")
            else:
                cfg_path.write_text(new_txt, encoding="utf-8")
                print(f"[APPLIED] config.models.audio = {best_model}")
        except Exception as exc:
            print(f"[WARN] config 적용 실패: {exc}")


if __name__ == "__main__":
    main()
