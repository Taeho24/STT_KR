#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
심슨 영상의 고정 5개 세그먼트에 대해, 모든 후보 오디오 감정 모델의 예측을 터미널에 출력하는 스팟체크 도구.

사용 예시 (PowerShell):
  $env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe .\tools\spotcheck_models.py --device auto

옵션:
    --models ...      : 특정 모델만 지정 (생략 시 config의 audio_candidates + 핵심 후보 포함)
    --device ...      : auto|cuda|cpu (기본 auto)
    --batch-size N    : 배치 크기 (기본 4)
    --video PATH      : 기본 assets/simpson.mp4, 변경 가능
    --preset {set1|set2|set3} : 심슨 전용 프리셋 선택
        --auto             : 프리셋 대신 입력 영상 전체에서 자동으로 세그먼트 N개 추출
    --num-segments N   : --auto 사용 시 세그먼트 개수 (기본 15)
    --seg-len SEC      : --auto 사용 시 세그먼트 길이 초 단위 (기본 1.6)
        --asr              : --auto와 함께 사용 시 WhisperX로 텍스트를 인식해 각 세그먼트에 텍스트를 채움
        --whisper-model ID : WhisperX 모델 크기(small|medium|large-v2 등, 기본 small)
        --asr-lang CODE    : ASR 강제 언어 코드(예: ko, en). 생략 시 자동 감지

출력 형태:
  모델별로 5개 세그먼트에 대한 오디오 예측 Top-1 및 상위 분포를 보기 좋게 출력
  (텍스트 모델은 비활성화: 순수 오디오 모델 출력만 확인)
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import config as app_config  # type: ignore
from emotion_classifier import EmotionClassifier  # type: ignore
from tools.non_hf_adapters import AudeeringDimAdapter  # type: ignore
import whisperx  # type: ignore
import numpy as np
from utils import split_segment_by_max_words  # type: ignore


def auto_device(pref: str) -> str:
    if pref and pref.lower() in {"cpu", "cuda"}:
        return pref.lower()
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_candidates(extra: List[str] | None = None) -> List[str]:
    cfg_list: List[str] = app_config.get("models", "audio_candidates", default=[]) or []
    base = [
        # 핵심 비교 후보 (요청 반영)
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        # 4클래스(SUPERB)는 제외, 한국어 특화 후보는 유지
        "jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition",
        # 추가 검증 후보 (5개 이상 감정)
        "Aniemore/wavlm-emotion-russian-resd",
        "DunnBC22/wav2vec2-base-Speech_Emotion_Recognition",
        "harshit345/xlsr-wav2vec-speech-emotion-recognition",
        "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned",
        # 차원형(비범주) 후보: audeering (어댑터로 범주화)
        "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        # 트렌딩 HF 후보들(필요 시 확장)
        "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    ]
    merged: List[str] = []
    seen = set()
    for m in cfg_list + base + (extra or []):
        if not m:
            continue
        if m not in seen:
            merged.append(m)
            seen.add(m)
    return merged


def fixed_simpson_segments_set1() -> List[Dict[str, Any]]:
    """세그먼트 프리셋 1: 기존 5개."""
    return [
        # 혐오(격한 톤) 의심 구간
        {"start": 4.312, "end": 5.672, "text": "I never heard of a poison pork chop."},
        # 놀람 의심 구간
        {"start": 26.097, "end": 26.717, "text": "24 hours?"},
        # 슬픔 의심 구간
        {"start": 28.098, "end": 29.478, "text": "I'm sorry I kept you waiting so long."},
        # 분노 의심 구간
        {"start": 31.998, "end": 33.439, "text": "I'm gonna die!"},
        # 중립(설명) 의심 구간
        {"start": 21.756, "end": 25.797, "text": "probable, you have 24 hours to live."},
    ]

def fixed_simpson_segments_set2() -> List[Dict[str, Any]]:
    """세그먼트 프리셋 2: 다른 5개 구간(라벨 파일/원본 로그를 참고하여 다양성 확보)."""
    return [
        {"start": 14.154, "end": 15.655, "text": "Ooh, it's good news, isn't it?"},   # happy 의심
        {"start": 46.943, "end": 47.443, "text": "No way!"},                           # angry/shout 의심
        {"start": 49.563, "end": 53.064, "text": "Why, you little... After that comes fear."}, # fear 의심
        {"start": 40.241, "end": 41.161, "text": "suddenly explodes."},                # surprise 의심
        {"start": 63.826, "end": 64.848, "text": "I should leave you two alone."},    # sad/whisper 의심
    ]

def fixed_simpson_segments_set3() -> List[Dict[str, Any]]:
    """세그먼트 프리셋 3: 또 다른 5개 구간(표현/톤 다양화)."""
    return [
        {"start": 9.973,  "end": 11.494, "text": "I can read Marge like a book."},  # neutral/normal 의심
        {"start": 15.755, "end": 16.795, "text": "No, Mr. Simpson."},               # whisper/neutral 의심
        {"start": 34.299, "end": 37.18,  "text": "Well, if there's one consolation, it's that you'll feel no"}, # sad/whisper 의심
        {"start": 47.503, "end": 48.503, "text": "Because I'm not dying!"},         # angry 의심
        {"start": 68.472, "end": 70.215, "text": "So you're going to die."},        # shout/neutral 의심
    ]


def summarize_topk(dist: Dict[str, float], k: int = 3) -> str:
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:k]
    return " | ".join(f"{e}:{s:.3f}" for e, s in items)

def _generate_even_segments(audio: np.ndarray, sr: int, n: int = 15, seg_len: float = 1.6, phase: float = 0.0) -> List[Dict[str, Any]]:
    """오디오 전체 길이에 걸쳐 균일 간격으로 N개의 세그먼트를 생성.

    - 각 세그먼트 길이는 seg_len(초)
    - 시작점은 (i+1)/(n+1) 비율 지점에서 seg_len/2만큼 좌우로 배치
    - 텍스트는 빈 문자열로 둔다(표시는 시간 범위로 충분)
    """
    total_sec = len(audio) / sr if sr > 0 else 0.0
    if total_sec <= 0:
        return []
    segs: List[Dict[str, Any]] = []
    eff_len = min(seg_len, max(0.2, total_sec / max(1, n)))  # 과도한 길이 방지
    half = eff_len / 2.0
    # 균일 간격 간단 계산(step), phase(0~1)를 이용해 중심을 반칸 등으로 오프셋
    step = total_sec / float(n + 1)
    phi = max(0.0, min(1.0, phase))
    for i in range(n):
        center = step * (float(i + 1) + phi)
        start = max(0.0, center - half)
        end = min(total_sec, start + eff_len)
        # 다시 보정(끝이 초과된 경우 시작을 앞당김)
        start = max(0.0, min(start, end - 0.2))  # 최소 길이 0.2s 확보
        segs.append({"start": float(start), "end": float(end), "text": ""})
    return segs


def _run_asr_segments(audio: np.ndarray, device: str, whisper_model: str = "small", language: str | None = None, max_words: int = 10) -> Dict[str, Any]:
    """WhisperX로 전체 오디오를 전사하고 word-level 정렬과 subtitle 세그먼트를 함께 생성해 반환.

    반환 값:
      {
        "words": [ {start, end, text} ... ],
        "subtitles": [ {start, end, text, words:[...] } ... ]  # split_segment_by_max_words 로 분할
      }
    """
    # 1) WhisperX 음성 인식
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = whisperx.load_model(whisper_model, device, compute_type=compute_type, language=language)
    except Exception:
        # 실패 시 안전하게 float32로 재시도
        model = whisperx.load_model(whisper_model, device, compute_type="float32", language=language)

    result = model.transcribe(audio)

    # 2) 단어 정렬 모델 로드 및 정렬 수행(가능할 때)
    try:
        lang_code = language or result.get("language") or "en"
        align_model, metadata = whisperx.load_align_model(language_code=lang_code, device=device)
        aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)
        segments = aligned.get("segments", []) or []
    except Exception:
        # 정렬 실패 시 문장 세그먼트만 사용(후속 로직에서 최소한으로 잘라서 사용)
        segments = result.get("segments", []) or []

    # 3) 단어 리스트 추출(가능하면 단어 단위, 아니면 문장 단위로 대체)
    words: List[Dict[str, Any]] = []
    for seg in segments:
        # 단어 단위가 있을 때 우선 사용
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                text = (w.get("word") or w.get("text") or "").strip()
                if not text:
                    continue
                try:
                    ws = float(w.get("start")) if w.get("start") is not None else None
                    we = float(w.get("end")) if w.get("end") is not None else None
                except Exception:
                    ws, we = None, None
                if ws is None or we is None:
                    continue
                words.append({"start": ws, "end": we, "text": text})
        else:
            # 단어 정보가 없으면 문장 전체를 하나의 블록처럼 취급
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            try:
                ss = float(seg.get("start", 0.0))
                ee = float(seg.get("end", 0.0))
            except Exception:
                ss, ee = 0.0, 0.0
            words.append({"start": ss, "end": ee, "text": text})

    # 4) 실제 자막 세그먼트로 사용할 분할(메인 파이프라인과 동일 규칙)
    #    - 최대 단어 수 기준으로 분할
    #    - 너무 짧은 노이즈(0.2s 미만)는 제거 (main.py와 동일)
    try:
        sub_segments = split_segment_by_max_words(segments, max_words)
        sub_segments = [s for s in sub_segments if (float(s.get("end", 0)) - float(s.get("start", 0))) > 0.2]
        # 보조: text가 비어 있으면 words에서 생성
        for s in sub_segments:
            if not s.get("text"):
                wtxt = " ".join([w.get("word") or w.get("text") or "" for w in s.get("words", [])]).strip()
                s["text"] = wtxt
    except Exception:
        # 실패 시 문장 단위를 그대로 사용
        sub_segments = []

    return {"words": words, "subtitles": sub_segments}


def _fill_text_from_asr_words(windows: List[Dict[str, Any]], asr_words: List[Dict[str, Any]]) -> None:
    """윈도우 세그먼트에 ASR 결과를 매핑해 text를 채움(단어 단위 사용).

    - 기본: 윈도우 [start,end]와 겹치는 단어들만 이어붙임
    - 보정: 너무 짧으면 양 옆으로 작은 패딩(+/- 0.25s) 내 단어 추가
    - Fallback: 그래도 없으면 가장 가까운 단어 3개까지를 묶어 보여줌
    - 출력: 너무 길어지지 않게 최대 글자수 제한(예: 120자)
    """
    if not asr_words:
        return

    def preview_text(text: str, max_chars: int = 120) -> str:
        t = " ".join(text.split())
        return t if len(t) <= max_chars else (t[: max_chars - 1] + "…")

    # 단어 중심 시각 미리 계산
    centers = [0.5 * (w["start"] + w["end"]) for w in asr_words]

    for win in windows:
        ws, we = float(win["start"]), float(win["end"])
        pad = 0.25  # 초 단위 패딩(문맥 1~2 단어 정도)
        # 1) 겹치는 단어 수집
        selected = [w for w in asr_words if not (w["end"] <= ws or w["start"] >= we)]

        # 2) 너무 짧으면 패딩 범위 내 단어 추가
        if len(selected) < 2:
            sel_pad = [w for w in asr_words if not (w["end"] <= (ws - pad) or w["start"] >= (we + pad))]
            # 중복 제거(원순서 유지)
            seen = set()
            tmp = []
            for w in selected + sel_pad:
                tid = (w["start"], w["end"], w["text"])  # 간단한 키
                if tid in seen:
                    continue
                seen.add(tid)
                tmp.append(w)
            selected = tmp

        # 3) Fallback: 그래도 없으면 가장 가까운 단어 3개
        if not selected:
            c = 0.5 * (ws + we)
            idxs = np.argsort([abs(c - cc) for cc in centers])[:3]
            selected = [asr_words[int(i)] for i in idxs]

        # 시간순 정렬 후 이어붙임
        selected.sort(key=lambda x: x["start"])
        text = " ".join(w["text"] for w in selected).strip()
        win["text"] = preview_text(text)


def _fill_text_from_subtitles(windows: List[Dict[str, Any]], subtitle_segs: List[Dict[str, Any]]) -> None:
    """윈도우마다 '실제 자막 세그먼트' 한 개를 매핑해 전체 라인 텍스트를 채움.

    - 우선순위: 겹치는 세그먼트들 중 겹친 시간 길이가 가장 긴 한 개 선택
    - 없으면: 윈도우 중심과 가장 가까운 세그먼트 1개 선택
    - 텍스트: 세그먼트의 text가 비어 있으면 words로 생성
    """
    if not subtitle_segs:
        return

    # 미리 중심 계산
    centers = [0.5 * (s["start"] + s["end"]) for s in subtitle_segs]

    def seg_text(seg: Dict[str, Any]) -> str:
        t = (seg.get("text") or "").strip()
        if t:
            return t
        return " ".join([w.get("word") or w.get("text") or "" for w in seg.get("words", [])]).strip()

    for win in windows:
        ws, we = float(win["start"]), float(win["end"])
        # 1) 겹치는 세그먼트 후보와 overlap 길이 계산
        overlaps = []
        for seg in subtitle_segs:
            s, e = float(seg["start"]), float(seg["end"])
            ov = max(0.0, min(we, e) - max(ws, s))
            if ov > 0:
                overlaps.append((ov, seg))

        chosen = None
        if overlaps:
            overlaps.sort(key=lambda x: x[0], reverse=True)
            chosen = overlaps[0][1]
        else:
            # 2) 최근접 세그먼트
            c = 0.5 * (ws + we)
            idx = int(np.argmin([abs(c - cc) for cc in centers]))
            chosen = subtitle_segs[idx]

        win["text"] = seg_text(chosen)


def run_spotcheck(models: List[str], video_path: Path, device: str, batch_size: int, preset: str, auto: bool = False, num_segments: int = 15, seg_len: float = 1.6, asr: bool = False, whisper_model: str = "small", asr_lang: str | None = None, asr_text_mode: str = "subtitle", max_words: int = 10, phase: float = 0.0) -> None:
    # 오디오 로드 (16k)
    audio = whisperx.load_audio(str(video_path))
    if auto:
        segments = _generate_even_segments(audio, 16000, n=num_segments, seg_len=seg_len, phase=phase)
        if asr:
            asr_out = _run_asr_segments(audio, device, whisper_model=whisper_model, language=asr_lang, max_words=max_words)
            if asr_text_mode == "word":
                _fill_text_from_asr_words(segments, asr_out.get("words", []))
            else:
                _fill_text_from_subtitles(segments, asr_out.get("subtitles", []))
    else:
        if preset == "set2":
            segments = fixed_simpson_segments_set2()
        elif preset == "set3":
            segments = fixed_simpson_segments_set3()
        else:
            segments = fixed_simpson_segments_set1()

    print(f"\n대상 영상: {video_path.name}")
    print(f"비교 세그먼트({len(segments)}개):")
    for i, s in enumerate(segments, start=1):
        txt = (s.get("text") or "").strip()
        print(f"  {i}. [{s['start']:.3f} ~ {s['end']:.3f}] {txt}")

    for idx, model_name in enumerate(models, start=1):
        print("\n" + "=" * 100)
        print(f"[{idx}/{len(models)}] 모델: {model_name}")
        use_adapter = (model_name == "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
        if use_adapter:
            try:
                adapter = AudeeringDimAdapter(device=device)
                scores_list = adapter.predict_segments(segments, audio, 16000)
                # 어댑터는 7감정 분포를 반환하므로 EmotionResult 비슷한 출력 형식 만들기
                results = []
                for dist in scores_list:
                    # 최상위 선택 및 confidence
                    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
                    emotion, conf = items[0]
                    results.append(type("Tmp", (), {
                        "emotion": emotion,
                        "confidence": conf,
                        "audio_distribution": dist,
                    }))
            except Exception as e:
                print(f"  [어댑터 추론 실패] {e}")
                continue
        else:
            try:
                clf = EmotionClassifier(
                    device=device,
                    batch_size=batch_size,
                    audio_model_name=model_name,
                    enable_text=False,  # 텍스트 비활성화: 순수 오디오만
                )
            except Exception as e:
                print(f"  [로드 실패] {e}")
                continue

            try:
                results = clf.process_batch(segments, audio_data=audio, sr=16000)
            except Exception as e:
                print(f"  [추론 실패] {e}")
                continue

        # 출력: 세그먼트별 예측 요약 (오디오 분포 중심)
        for i, (seg, res) in enumerate(zip(segments, results), start=1):
            top_audio = summarize_topk(res.audio_distribution, k=3)
            # 최종 선택(emotion, confidence)은 내부 결합 로직 영향 가능 → 참고용 표기
            label_text = seg.get('text') or ''
            print(f"- S{i} [{seg['start']:.3f}-{seg['end']:.3f}] '{label_text}'")
            print(f"    · 오디오 Top3: {top_audio}")
            print(f"    · 최종 선택: {res.emotion} ({res.confidence:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="심슨 5개 세그먼트 스팟체크")
    ap.add_argument("--models", nargs="*", default=None, help="평가할 오디오 모델들")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--video", type=Path, default=ROOT / "assets" / "simpson.mp4")
    ap.add_argument("--preset", choices=["set1", "set2", "set3"], default="set1", help="세그먼트 프리셋 선택")
    ap.add_argument("--auto", action="store_true", help="프리셋 대신 자동 세그먼트 사용")
    ap.add_argument("--num-segments", type=int, default=15, help="--auto 사용 시 세그먼트 개수")
    ap.add_argument("--seg-len", type=float, default=1.6, help="--auto 사용 시 세그먼트 길이(초)")
    ap.add_argument("--asr", action="store_true", help="자동 세그먼트에 WhisperX 전사 텍스트를 채움")
    ap.add_argument("--whisper-model", default="small", help="WhisperX 모델 크기(ex. small, medium, large-v2)")
    ap.add_argument("--asr-lang", default=None, help="ASR 언어 코드(예: ko, en). 생략 시 자동")
    ap.add_argument("--asr-text-mode", choices=["subtitle", "word"], default="subtitle", help="세그먼트 텍스트 표시 방식: subtitle=실제 자막 세그먼트 1개, word=윈도우 내 단어 미리보기")
    ap.add_argument("--max-words", type=int, default=10, help="자막 분할 시 최대 단어 수(실제 파이프라인과 동일하게 유지 권장)")
    ap.add_argument("--phase", type=float, default=0.0, help="자동 세그먼트 균일 배치의 위상 오프셋(0~1). 0.5로 주면 이전과 겹침이 줄어듭니다.")
    args = ap.parse_args()

    dev = auto_device(args.device)
    models = args.models or load_candidates()

    if not args.video.exists():
        print(f"[에러] 영상 파일을 찾을 수 없습니다: {args.video}")
        sys.exit(2)

    print(f"장치: {dev}")
    print(f"모델 후보 수: {len(models)}")
    for m in models:
        print(f" - {m}")

    run_spotcheck(
        models,
        args.video,
        dev,
        args.batch_size,
        args.preset,
        auto=args.auto,
        num_segments=args.num_segments,
        seg_len=args.seg_len,
        asr=args.asr,
        whisper_model=args.whisper_model,
        asr_lang=args.asr_lang,
        asr_text_mode=args.asr_text_mode,
        max_words=args.max_words,
        phase=args.phase,
    )


if __name__ == "__main__":
    main()
