#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import librosa
import math
from collections import defaultdict
from config import config

class AudioAnalyzer:
    """오디오 분석 및 특성 추출 클래스"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        # 전역 통계/임계 저장 구조 초기화
        self.volume_stats = {
            'values': [],
            'mean': None,
            'std': None,
            'percentiles': {},
            'thresholds': {'soft': None, 'normal': None, 'loud': None}
        }
        self.pitch_stats = {
            'values': [],
            'p10': None,
            'p90': None
        }
        self.speech_rate_stats = {
            'values': [],
            'mean': None,
            'std': None,
            'thresholds': {'slow': None, 'normal': None, 'fast': None}
        }
        # 발화 속도별 자간 간격 (자막 스타일용)
        self.speech_rate_spacing = {'slow': 10, 'normal': 0, 'fast': -5}
        # 최소/이상적 세그먼트 길이
        self.min_duration = 0.2
        self.ideal_duration = 0.5
    def analyze_audio_features(self, segments, audio):
        """오디오 특성 분석 (최적화 버전)

        - 볼륨/피치/발화속도 전역 분포 분석
        - 세그먼트 단위 RMS + 히스테리시스
        - 세그먼트당 1회 pitch track 계산 후 단어 구간 추출
        - 단어 RMS 캐싱 / 중복 계산 제거
        """
        self.analyze_volume_distribution(audio)
        # 피치 분포는 아래 세그먼트 루프에서 한 번 계산한 YIN을 재활용하여 집계
        self.analyze_speech_rate_distribution(segments)

        # config 조회 캐시 (루프 내 반복 호출 최소화)
        fast_mode = bool(config.get('analysis', 'fast_mode', default=False))

        prev_volume_level = None
        hop_length_seg = 256
        # 세그먼트별 평균 피치 누적(중복 YIN 제거)
        _pitch_values = []
        for segment in segments:
            duration = segment['end'] - segment['start']
            if duration < self.min_duration:
                seg_audio = self._get_context_audio(audio, segment, pad_duration=0.1)
            else:
                start_idx = int(segment['start'] * self.sample_rate)
                end_idx = int(segment['end'] * self.sample_rate)
                seg_audio = audio[start_idx:end_idx]

            if len(seg_audio) == 0:
                segment['volume_level'] = 'normal'
                segment['volume_stats'] = {'mean': 0.0, 'levels': ['normal']}
                segment['pitch_stats'] = {'levels': ['normal']}
                segment['speech_rate_stats'] = {'levels': ['normal']}
                continue

            seg_rms = self.compute_rms(seg_audio)
            vol_level_raw = self._classify_volume([seg_rms])[0]
            if prev_volume_level and prev_volume_level != vol_level_raw:
                thr = self.volume_stats['thresholds']
                soft_thr = thr.get('soft') or 0
                loud_thr = thr.get('normal') or 0
                margin_soft = soft_thr * 0.1 if soft_thr else 0.01
                margin_loud = loud_thr * 0.1 if loud_thr else 0.03
                if vol_level_raw == 'soft' and soft_thr and (seg_rms > soft_thr - margin_soft):
                    vol_level_raw = prev_volume_level
                elif vol_level_raw == 'loud' and loud_thr and (seg_rms < loud_thr + margin_loud):
                    vol_level_raw = prev_volume_level
            segment['volume_level'] = vol_level_raw
            prev_volume_level = vol_level_raw

            words = segment.get('words', [])
            # 세그먼트 pitch track 1회 계산
            frame_f0 = None
            frame_times = None
            try:
                # 세그먼트당 1회만 YIN 수행 (단어/세그먼트 피치 모두 이 값으로 처리)
                frame_f0 = librosa.yin(seg_audio, fmin=50, fmax=600, hop_length=hop_length_seg)
                frame_times = (np.arange(len(frame_f0)) * hop_length_seg) / self.sample_rate
                # 세그먼트 평균 피치 집계 (유효 프레임만)
                valid_f0 = frame_f0[frame_f0 > 0]
                if valid_f0.size > 0:
                    _pitch_values.append(float(np.mean(valid_f0)))
            except Exception:
                frame_f0 = None
                frame_times = None

            word_rms_values = []
            for word in words:
                try:
                    if not isinstance(word, dict) or 'start' not in word or 'end' not in word:
                        continue
                    w_start_idx = int(word['start'] * self.sample_rate)
                    w_end_idx = int(word['end'] * self.sample_rate)
                    if w_end_idx <= w_start_idx or w_end_idx > len(audio):
                        continue
                    w_audio = audio[w_start_idx:w_end_idx]
                    if len(w_audio) == 0:
                        word['volume_level'] = 'normal'
                        word['pitch_level'] = 'normal'
                        word['speech_rate'] = 'normal'
                        continue
                    # RMS 캐시
                    rms = self.compute_rms(w_audio)
                    word['rms'] = rms
                    word['volume_level'] = self._classify_volume([rms])[0]
                    # Pitch from segment track
                    if frame_f0 is not None and frame_times is not None:
                        rel_start = word['start'] - segment['start']
                        rel_end = word['end'] - segment['start']
                        mask = (frame_times >= rel_start) & (frame_times <= rel_end)
                        sel = frame_f0[mask]
                        sel = sel[sel > 0]
                        if sel.size > 0:
                            avg_pitch = np.mean(sel)
                            word['pitch_level'] = self.assign_pitch_level(avg_pitch)
                        else:
                            word['pitch_level'] = 'normal'
                    else:
                        word['pitch_level'] = 'normal'
                    # 발화 속도 계산: fast_mode에서는 간이 계산으로 CPU 사용 절감
                    dur_w = word['end'] - word['start']
                    if dur_w > 0:
                        if fast_mode:
                            # 간이: 음절 복잡도 / 전체 단어 길이 시간
                            sr_val = self._estimate_syllable_complexity(word.get('word', '')) / max(dur_w, 1e-6)
                        else:
                            # 정밀: 실제 발성 시간 추정 기반
                            sr_val = self._calculate_phonetic_speech_rate(w_audio, word.get('word', ''), dur_w)
                        word['speech_rate'] = self.assign_speech_rate_level(float(sr_val))
                    else:
                        word['speech_rate'] = 'normal'
                    word_rms_values.append(rms)
                except Exception as e:
                    print(f"단어 처리 중 오류 발생: {str(e)}")
                    word['volume_level'] = 'normal'
                    word['pitch_level'] = 'normal'
                    word['speech_rate'] = 'normal'

            segment['volume_stats'] = {
                'mean': float(np.mean(word_rms_values)) if word_rms_values else seg_rms,
                'levels': [w.get('volume_level', 'normal') for w in words] if words else [vol_level_raw]
            }
            segment['pitch_stats'] = {
                'levels': [w.get('pitch_level', 'normal') for w in words] if words else ['normal']
            }
            segment['speech_rate_stats'] = {
                'levels': [w.get('speech_rate', 'normal') for w in words] if words else ['normal']
            }

        # 세그먼트 루프가 끝난 뒤 p10/p90 계산 (중복 YIN 제거)
        if _pitch_values:
            self.pitch_stats['values'] = _pitch_values
            self.pitch_stats['p10'] = float(np.percentile(_pitch_values, 10))
            self.pitch_stats['p90'] = float(np.percentile(_pitch_values, 90))

        return segments

    def compute_voice_spans(self, segments):
        """세그먼트 내부 하이브리드 음성 타입 스팬 산출

        하이브리드 규칙(사전 합의):
        - 기본은 세그먼트 레벨 voice_type(whisper/normal/shout)을 유지
        - 단어별 RMS(상대)와 전역 볼륨 레벨(soft/loud)을 이용해 강한 지역 패턴이
          일정 길이 이상 지속되면 그 구간만 부분 재태깅
        - 히스테리시스: std 기반 상대 임계로 토글 방지, 최소 단어수/지속시간 적용
        - 기본값 우선: 세그먼트가 whisper면 약한 shout 후보는 무시(반대도 동일)
        결과는 segment['voice_spans']에 기록: [{'label', 'start_word', 'end_word'}]
        """
        MIN_SPAN_WORDS = 2
        MIN_DUR = {"whisper": 0.50, "shout": 0.40}
        REL_STD = 0.75

        for seg in segments:
            words = seg.get('words') or []
            if not words:
                # 단어가 없으면 세그먼트 전역 라벨만 유지
                seg['voice_spans'] = [{'label': seg.get('voice_type', 'normal'), 'start_word': 0, 'end_word': -1}]
                continue

            # 단어 RMS 수집 (미존재 시 0)
            rms_vals = []
            for w in words:
                r = float(w.get('rms', 0.0))
                rms_vals.append(r)
            arr = np.asarray(rms_vals, dtype=np.float32)
            seg_mean = float(arr.mean()) if arr.size > 0 else 0.0
            seg_std = float(arr.std()) if arr.size > 0 else 0.0
            if seg_std < 1e-6:
                seg_std = 1.0  # 상대 임계가 무의미해지는 것을 방지

            # 단어별 후보 라벨 산출
            cand = [None] * len(words)
            for i, w in enumerate(words):
                r = float(w.get('rms', 0.0))
                vol_lvl = w.get('volume_level', 'normal')
                is_shout = (r >= seg_mean + REL_STD * seg_std) or (vol_lvl == 'loud')
                is_whisp = (r <= seg_mean - REL_STD * seg_std) or (vol_lvl == 'soft')
                if is_shout and not is_whisp:
                    cand[i] = 'shout'
                elif is_whisp and not is_shout:
                    cand[i] = 'whisper'
                else:
                    cand[i] = None

            # 연속 그룹화 및 최소 길이/지속시간 필터
            accepted_overrides = []  # (label, start_idx, end_idx)
            i = 0
            while i < len(words):
                if cand[i] is None:
                    i += 1
                    continue
                j = i
                lab = cand[i]
                while j + 1 < len(words) and cand[j + 1] == lab:
                    j += 1
                # i..j 그룹 지속시간 계산
                start_t = float(words[i].get('start', seg.get('start', 0.0)))
                end_t = float(words[j].get('end', seg.get('end', start_t)))
                dur = max(0.0, end_t - start_t)
                enough_words = (j - i + 1) >= MIN_SPAN_WORDS
                enough_dur = dur >= MIN_DUR.get(lab, 0.45)

                # 세그먼트 전역 라벨 대비 약한 반대 후보 억제
                seg_label = seg.get('voice_type', 'normal')
                if seg_label == 'whisper' and lab == 'shout':
                    # 더 엄격: 단어수가 3 이상 + 0.5s 이상
                    enough_words = (j - i + 1) >= max(3, MIN_SPAN_WORDS)
                    enough_dur = dur >= max(0.50, MIN_DUR['shout'])
                elif seg_label == 'shout' and lab == 'whisper':
                    enough_words = (j - i + 1) >= max(3, MIN_SPAN_WORDS)
                    enough_dur = dur >= max(0.55, MIN_DUR['whisper'])

                if enough_words and enough_dur:
                    accepted_overrides.append((lab, i, j))
                i = j + 1

            # 단어별 최종 라벨 적용(기본: 세그먼트 라벨)
            base = seg.get('voice_type', 'normal')
            final_labels = [base] * len(words)
            for lab, si, sj in accepted_overrides:
                for k in range(si, sj + 1):
                    final_labels[k] = lab

            # 동일 라벨 연속 구간을 voice_spans로 압축
            spans = []
            cur_lab = final_labels[0]
            span_start = 0
            for idx in range(1, len(words)):
                if final_labels[idx] != cur_lab:
                    spans.append({
                        'label': cur_lab,
                        'start_word': span_start,
                        'end_word': idx - 1
                    })
                    cur_lab = final_labels[idx]
                    span_start = idx
            # 꼬리 처리
            spans.append({'label': cur_lab, 'start_word': span_start, 'end_word': len(words) - 1})

            seg['voice_spans'] = spans

        return segments

    def classify_voice_types(self, segments, audio):
        """발화자 정규화 + 오디오 신호 기반 Whisper/Shout 분류(일반화 알고리즘)

        - 텍스트/특정 영상 특화 로직 없이, 오디오 특성만으로 판정
        - 화자별 정규화(RMS/crest/HF/LF/틸트/호기성)
        - 적응형 임계값(평균+표준편차)과 간단한 가드로 오탐 억제
        """
        MIN_DUR = 0.30
        FRAME_MS = 0.025  # 25ms 프레임
        HOP_MS = 0.010    # 10ms 홉
        # 위치 특화 휴리스틱 사용 여부(기본 False)
        use_peak_pos = bool(config.get('voice', 'use_peak_position_heuristics', default=False))

        def crest_factor(x: np.ndarray) -> float:
            if len(x) == 0:
                return 0.0
            rms_v = self.compute_rms(x)
            if rms_v == 0:
                return 0.0
            return float(np.max(np.abs(x)) / (rms_v + 1e-9))

        def hf_lf_ratio(x: np.ndarray) -> float:
            try:
                if len(x) > self.sample_rate:
                    mid = len(x) // 2
                    half = self.sample_rate // 2
                    x = x[mid-half:mid+half]
                window = np.hanning(len(x))
                spec = np.abs(np.fft.rfft(x * window))
                freqs = np.fft.rfftfreq(len(x), d=1.0/self.sample_rate)
                lf_energy = spec[freqs <= 1000].sum() + 1e-9
                hf_mask = (freqs >= 2000) & (freqs <= 5000)
                hf_energy = spec[hf_mask].sum() + 1e-9
                return float(hf_energy / lf_energy)
            except Exception:
                return 1.0

        def spectral_tilt_analysis(x: np.ndarray) -> float:
            try:
                if len(x) < 512:
                    return 0.0
                window = np.hanning(len(x))
                spec = np.abs(np.fft.rfft(x * window))
                freqs = np.fft.rfftfreq(len(x), d=1.0/self.sample_rate)
                hf_mask = (freqs >= 2000) & (freqs <= 4000)
                lf_mask = (freqs >= 300) & (freqs <= 1500)
                if not (hf_mask.any() and lf_mask.any()):
                    return 0.0
                hf_energy = spec[hf_mask].mean()
                lf_energy = spec[lf_mask].mean() + 1e-9
                spectral_tilt = hf_energy / lf_energy
                return float(np.clip((spectral_tilt - 0.3) / 0.7, 0.0, 1.0))
            except Exception:
                return 0.0

        def breathiness_detection(x: np.ndarray) -> float:
            try:
                if len(x) < 512:
                    return 0.0
                window = np.hanning(len(x))
                spec = np.abs(np.fft.rfft(x * window))
                freqs = np.fft.rfftfreq(len(x), d=1.0/self.sample_rate)
                noise_mask = (freqs >= 6000) & (freqs <= 8000)
                if not noise_mask.any():
                    return 0.0
                noise_energy = spec[noise_mask].mean()
                total_energy = spec.mean() + 1e-9
                noise_ratio = noise_energy / total_energy
                return float(np.clip((noise_ratio - 0.02) / 0.05, 0.0, 1.0))
            except Exception:
                return 0.0

        segment_features = []
        speaker_buckets = defaultdict(lambda: {
            'rms': [],
            'crest': [],
            'hf_lf': [],
            'breathiness': [],
            'tilt': []
        })

        for idx, seg in enumerate(segments):
            start_idx = int(seg['start'] * self.sample_rate)
            end_idx = int(seg['end'] * self.sample_rate)
            if end_idx <= start_idx or end_idx > len(audio):
                continue
            duration = seg['end'] - seg['start']
            seg_audio = audio[start_idx:end_idx]
            if seg_audio.size == 0:
                continue

            rms = self.compute_rms(seg_audio)
            c_factor = crest_factor(seg_audio)
            ratio_hf_lf = hf_lf_ratio(seg_audio)
            spectral_tilt = spectral_tilt_analysis(seg_audio)
            breathiness = breathiness_detection(seg_audio)
            speaker_id = seg.get('speaker', 'Unknown')

            segment_features.append({
                'index': idx,
                'speaker': speaker_id,
                'rms': rms,
                'crest': c_factor,
                'hf_lf': ratio_hf_lf,
                'tilt': spectral_tilt,
                'breathiness': breathiness,
                'duration': duration
            })

            speaker_buckets[speaker_id]['rms'].append(rms)
            speaker_buckets[speaker_id]['crest'].append(c_factor)
            speaker_buckets[speaker_id]['hf_lf'].append(ratio_hf_lf)
            speaker_buckets[speaker_id]['breathiness'].append(breathiness)
            speaker_buckets[speaker_id]['tilt'].append(spectral_tilt)

        if not segment_features:
            for seg in segments:
                seg['voice_type'] = 'normal'
                seg['voice_type_confidence'] = 0.0
            return segments

        def mean_std(values):
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == 0:
                return 0.0, 1.0
            mean = float(arr.mean())
            std = float(arr.std())
            if std < 1e-6:
                std = 1.0
            return mean, std

        speaker_stats = {}
        for speaker, feats in speaker_buckets.items():
            speaker_stats[speaker] = {
                'rms': mean_std(feats['rms']),
                'crest': mean_std(feats['crest']),
                'hf_lf': mean_std(feats['hf_lf']),
                'breathiness': mean_std(feats['breathiness']),
                'tilt': mean_std(feats['tilt'])
            }

        def normalize(value, mean, std):
            return (value - mean) / (std if std else 1.0)

        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))

        whisper_probs = []
        shout_probs = []

        # 프레임 기반 에너지/지속성 분석 헬퍼
        def _frame_rms(arr: np.ndarray) -> np.ndarray:
            """벡터화된 프레임 RMS 계산 (librosa.feature.rms 사용)."""
            frame_size = max(1, int(self.sample_rate * FRAME_MS))
            hop_size = max(1, int(self.sample_rate * HOP_MS))
            if arr.size == 0:
                return np.array([0.0], dtype=np.float32)
            # librosa.feature.rms는 2D 반환(shape: 1, n_frames)
            rms2d = librosa.feature.rms(
                y=arr.astype(np.float32),
                frame_length=frame_size,
                hop_length=hop_size,
                center=False
            )
            return rms2d.flatten().astype(np.float32)

        # 각 세그먼트별 정규화/확률 및 지속시간 특징 계산
        for feat in segment_features:
            stats = speaker_stats.get(feat['speaker'])
            if not stats:
                stats = {
                    'rms': (0.02, 0.01),
                    'crest': (5.5, 0.5),
                    'hf_lf': (1.0, 0.2),
                    'breathiness': (0.1, 0.05),
                    'tilt': (0.2, 0.1)
                }

            rms_norm = normalize(feat['rms'], *stats['rms'])
            crest_norm = normalize(feat['crest'], *stats['crest'])
            hf_lf_norm = normalize(feat['hf_lf'], *stats['hf_lf'])
            breath_norm = normalize(feat['breathiness'], *stats['breathiness'])
            tilt_norm = normalize(feat['tilt'], *stats['tilt'])

            duration = feat['duration']
            duration_bonus = np.clip(duration - 0.45, -0.4, 0.6)
            long_bonus = max(0.0, duration - 0.35)

            whisper_logit = (
                (-2.0 * rms_norm) +
                (1.4 * feat['breathiness']) +
                (0.9 * max(0.0, hf_lf_norm)) +
                (-0.8 * crest_norm) +
                (0.4 * breath_norm) +
                (0.3 * duration_bonus)
            ) - 1.0

            shout_logit = (
                (2.0 * rms_norm) +
                (1.2 * feat['tilt']) +
                (0.9 * max(0.0, crest_norm)) +
                (-0.7 * hf_lf_norm) +
                (0.3 * tilt_norm) +
                (0.4 * long_bonus)
            ) - 0.9

            whisper_prob = sigmoid(whisper_logit)
            shout_prob = sigmoid(shout_logit)

            feat['whisper_prob'] = whisper_prob
            feat['shout_prob'] = shout_prob

            # === AV(Valence/Arousal) 근사 계산 ===
            # - Arousal: 에너지/밝기/피크성 상관 (rms_norm, tilt, hf_lf_norm, crest_norm)
            # - Valence: 호기성(역상관), 거친 고주파/피크(역상관) 기반 보수적 근사
            # 범위를 좁히지 않도록 작은 가중으로 시그모이드 매핑
            arousal_logit = (
                1.6 * rms_norm +
                0.9 * feat['tilt'] +
                0.5 * max(0.0, hf_lf_norm) +
                0.4 * max(0.0, crest_norm)
            ) - 0.2 * max(0.0, breath_norm)
            arousal = float(np.clip(sigmoid(arousal_logit), 0.0, 1.0))

            valence_logit = (
                -1.3 * max(0.0, breath_norm) +
                -0.6 * max(0.0, hf_lf_norm) +
                -0.2 * max(0.0, crest_norm) +
                0.1
            )
            valence = float(np.clip(sigmoid(valence_logit), 0.0, 1.0))

            # === 지속성 기반 고/저 에너지 시간 추정 (shout/whisper 안정화) ===
            seg = segments[feat['index']]
            start_idx = int(seg['start'] * self.sample_rate)
            end_idx = int(seg['end'] * self.sample_rate)
            seg_audio = audio[start_idx:end_idx]
            frame_rms = _frame_rms(seg_audio)
            spk_mean, spk_std = speaker_stats[feat['speaker']]['rms']
            high_thr = max(spk_mean + 0.50 * spk_std, 0.028)
            low_thr = min(spk_mean - 0.50 * spk_std, 0.016)
            frame_time = max(HOP_MS, FRAME_MS)  # 대략적 프레임 시간 (초)
            high_frames = np.sum(frame_rms >= high_thr)
            low_frames = np.sum(frame_rms <= low_thr)
            high_dur = float(high_frames * frame_time)
            low_dur = float(low_frames * frame_time)
            total_dur = max(1e-6, feat['duration'])
            feat['high_energy_dur'] = high_dur
            feat['low_energy_dur'] = low_dur
            feat['high_energy_frac'] = float(np.clip(high_dur / total_dur, 0.0, 1.0))
            feat['low_energy_frac'] = float(np.clip(low_dur / total_dur, 0.0, 1.0))

            # 앞/뒤 구간의 고에너지 분포(초반 과도탐지 억제, 말미 강조 검증)
            n_frames = max(1, frame_rms.size)
            early_end = int(0.25 * n_frames)
            late_start = int(0.60 * n_frames)
            if early_end <= 0:
                early_end = min(1, n_frames)
            if late_start >= n_frames:
                late_start = max(0, n_frames - 1)
            early_high = int(np.sum(frame_rms[:early_end] >= high_thr)) if frame_rms.size > 0 else 0
            late_high = int(np.sum(frame_rms[late_start:] >= high_thr)) if frame_rms.size > 0 else 0
            feat['early_high_frac'] = float(np.clip((early_high * frame_time) / total_dur, 0.0, 1.0))
            feat['late_high_frac'] = float(np.clip((late_high * frame_time) / total_dur, 0.0, 1.0))

            # 단어 피크 정렬: 세그먼트 내 가장 큰 RMS 단어 위치(상대시간)와 RMS
            words = seg.get('words', []) or []
            peak_rms = -1.0
            peak_pos_rel = 0.0
            soft_word_frac = 0.0
            loud_word_frac = 0.0
            if words:
                wrms = []
                w_soft = 0
                w_loud = 0
                for w in words:
                    r = float(w.get('rms', 0.0)) if isinstance(w, dict) else 0.0
                    wrms.append(r)
                    # 화자 기준 상대적 soft/loud 추정 (분류 선행 여부와 무관하게 복원적 계산)
                    if r > 0.0:
                        if r <= (spk_mean - 0.50 * spk_std):
                            w_soft += 1
                        elif r >= (spk_mean + 0.50 * spk_std):
                            w_loud += 1
                if any(v > 0 for v in wrms):
                    j = int(np.argmax(wrms))
                    w = words[j]
                    w_mid = 0.5 * (float(w.get('start', seg.get('start', 0))) + float(w.get('end', seg.get('end', 0))))
                    peak_rms = float(wrms[j])
                    # 세그먼트 상대 위치 0~1
                    feat_start = float(seg.get('start', 0.0))
                    feat_end = float(seg.get('end', feat_start))
                    dur = max(1e-6, feat_end - feat_start)
                    peak_pos_rel = float(np.clip((w_mid - feat_start) / dur, 0.0, 1.0))
                soft_word_frac = float(w_soft / max(1, len(words)))
                loud_word_frac = float(w_loud / max(1, len(words)))
            feat['peak_word_rms'] = peak_rms
            feat['peak_pos_rel'] = peak_pos_rel
            feat['soft_word_frac'] = soft_word_frac
            feat['loud_word_frac'] = loud_word_frac

            whisper_probs.append(whisper_prob)
            shout_probs.append(shout_prob)

        whisper_probs_arr = np.array(whisper_probs)
        shout_probs_arr = np.array(shout_probs)

        base_whisper_thr = 0.60
        base_shout_thr = 0.62

        def adaptive_threshold(prob_arr, base):
            """보수적 상향만 허용하는 적응형 임계값.
            - 분포가 낮을수록 과도하게 올라가지 않도록 mean + 0.5*std만 반영
            - 상한을 조금 낮춰 0.88로 클립 (과도 보수화 방지)
            - 표본이 적으면 기본값 사용
            """
            if prob_arr.size < 4:
                return base
            mean = float(prob_arr.mean())
            std = float(prob_arr.std())
            candidate = mean + 0.5 * std
            return float(np.clip(max(base, candidate), base, 0.88))

        whisper_threshold = adaptive_threshold(whisper_probs_arr, base_whisper_thr)
        shout_threshold = adaptive_threshold(shout_probs_arr, base_shout_thr)

        # 1차 판정
        preliminary_types = {}
        for feat in segment_features:
            idx = feat['index']
            whisper_prob = feat['whisper_prob']
            shout_prob = feat['shout_prob']

            speaker_mean, speaker_std = speaker_stats[feat['speaker']]['rms']
            crest_mean, crest_std = speaker_stats[feat['speaker']]['crest']
            hf_mean, hf_std = speaker_stats[feat['speaker']]['hf_lf']
            rms_guard_whisper = (feat['rms'] <= speaker_mean - 0.45 * speaker_std) or (feat['rms'] <= 0.016)
            rms_guard_shout = (feat['rms'] >= speaker_mean + 0.45 * speaker_std) or (feat['rms'] >= 0.028)
            breath_guard = feat['breathiness'] >= 0.45
            tilt_guard = feat['tilt'] >= 0.5
            crest_guard = (feat['crest'] >= crest_mean + 0.35 * crest_std) or (feat['crest'] >= 6.2)
            hf_guard = feat['hf_lf'] <= (
                speaker_stats[feat['speaker']]['hf_lf'][0] - 0.35 * speaker_stats[feat['speaker']]['hf_lf'][1]
            )
            hf_high_guard = feat['hf_lf'] >= (hf_mean + 0.10 * hf_std)

            # 너무 짧은 구간은 normal 유지 (잡음 억제)
            seg = segments[idx]
            dur = seg.get('end', 0) - seg.get('start', 0)
            if dur < 0.30:
                voice_type = 'normal'
                confidence = max(0.0, max(whisper_prob, shout_prob) - 0.5)
            else:
                # 최종 판정: 오디오 신호만 사용 + 지속성 가드
                crest_upper = min(6.0, crest_mean + 0.15 * crest_std)
                sustained_low_ok = (feat.get('low_energy_dur', 0.0) >= 0.45) or (feat.get('low_energy_frac', 0.0) >= 0.60)
                sustained_high_ok = (feat.get('high_energy_dur', 0.0) >= 0.35) or (feat.get('high_energy_frac', 0.0) >= 0.40)
                # 짧은 감탄사(예: "No way!") 보정: 중간 길이(0.40~0.80s)에서 고에너지 비율이 0.30 이상이고
                # 스펙트럴 지표(tilt/crest)가 강하면 shout 허용
                if not sustained_high_ok and 0.40 <= dur <= 0.80:
                    if (feat.get('high_energy_frac', 0.0) >= 0.30) and (tilt_guard or crest_guard):
                        sustained_high_ok = True

                # Whisper 판정 강화: soft 단어 비율/저에너지 지속으로 일관성 확보
                soft_majority = feat.get('soft_word_frac', 0.0) >= 0.70
                whisper_like = (
                    whisper_prob >= (whisper_threshold + 0.01)
                    or (soft_majority and whisper_prob >= (whisper_threshold - 0.03))
                )
                if (
                    whisper_like and
                    (whisper_prob - shout_prob) >= 0.12 and
                    rms_guard_whisper and
                    (feat['breathiness'] >= 0.50 or soft_majority) and
                    feat['crest'] <= crest_upper and
                    dur >= 0.50 and
                    sustained_low_ok
                ):
                    voice_type = 'whisper'
                    confidence = whisper_prob
                else:
                    # Shout 판단: 길이에 따른 조건 가변화로 과도 억제 완화
                    spectral_combo = (1 if tilt_guard else 0) + (1 if crest_guard else 0) + (1 if hf_guard else 0)
                    sustained_ok = sustained_high_ok
                    if dur >= 1.0:
                        sustained_ok = (feat.get('high_energy_frac', 0.0) >= 0.50) or (feat.get('high_energy_dur', 0.0) >= 0.60)
                    else:
                        # 중/단길이 구간은 완화된 지속성 기준
                        sustained_ok = (feat.get('high_energy_frac', 0.0) >= 0.30) or (feat.get('high_energy_dur', 0.0) >= 0.30)

                    # 기본 규칙 (긴 구간: 스펙트럴 가드 2개, 짧은 구간: 1개로 허용)
                    min_spectral = 2 if dur >= 0.90 else 1
                    # 늦은 피크/말미 고에너지 가중(위치 휴리스틱 비활성 유지하되, 순수 에너지 기반)
                    late_emph_ok = feat.get('late_high_frac', 0.0) >= (0.12 if dur < 0.9 else 0.18)
                    base_shout_cond = (
                        shout_prob >= max(shout_threshold, 0.66) and
                        (shout_prob - whisper_prob) >= 0.10 and
                        rms_guard_shout and
                        spectral_combo >= min_spectral and
                        sustained_ok and
                        (late_emph_ok or use_peak_pos)
                    )

                    # 보조 규칙: 감탄사형(0.40~0.90s)에서 스펙트럴 강하고 에너지 비율 충분하면 약간 낮은 임계도 허용
                    exclaim_cond = (
                        0.40 <= dur <= 0.90 and
                        shout_prob >= (max(shout_threshold, 0.66) - 0.03) and
                        rms_guard_shout and
                        (tilt_guard or crest_guard) and
                        (feat.get('high_energy_frac', 0.0) >= 0.25) and
                        (feat.get('late_high_frac', 0.0) >= 0.12)
                    )

                    # 단어 피크 정렬 기반 보정(옵션): 기본 비활성화로 일반성 유지
                    peak_rel = feat.get('peak_pos_rel', 0.0)
                    end_peak_promote = False
                    early_peak_demote = False
                    if use_peak_pos:
                        # 끝부분(>=60%) 피크 + 고에너지면 승격 허용
                        end_peak_promote = (peak_rel >= 0.60) and (feat.get('high_energy_frac', 0.0) >= 0.30) and (tilt_guard or crest_guard)
                        # 초반(<=25%) 피크만 있고 고에너지 지속이 약하면 억제
                        early_peak_demote = (peak_rel <= 0.25) and (feat.get('high_energy_frac', 0.0) < 0.45) and (not sustained_ok)

                    # 초반 과도탐지 억제: 초반만 높고 말미 에너지가 빈약하면 억제
                    early_only = (
                        feat.get('early_high_frac', 0.0) >= 0.30 and
                        feat.get('late_high_frac', 0.0) < 0.08 and
                        not sustained_ok
                    )

                    if (base_shout_cond or exclaim_cond) and not early_only:
                        voice_type = 'shout'
                        confidence = shout_prob
                        if early_peak_demote:
                            voice_type = 'normal'
                            confidence = max(0.0, confidence - 0.1)
                    elif end_peak_promote and shout_prob >= (max(shout_threshold, 0.66) - 0.05) and rms_guard_shout:
                        voice_type = 'shout'
                        confidence = max(shout_prob, 0.66)
                    else:
                        voice_type = 'normal'
                        confidence = max(0.0, max(whisper_prob, shout_prob) - 0.5)

            preliminary_types[idx] = (voice_type, float(confidence))
            analysis = segments[idx].setdefault('voice_analysis', {})
            analysis.update({
                'rms': feat['rms'],
                'crest_factor': feat['crest'],
                'hf_lf_ratio': feat['hf_lf'],
                'spectral_tilt': feat['tilt'],
                'breathiness': feat['breathiness'],
                'whisper_prob': round(whisper_prob, 3),
                'shout_prob': round(shout_prob, 3),
                'speaker_rms_mean': speaker_stats[feat['speaker']]['rms'][0],
                'speaker_rms_std': speaker_stats[feat['speaker']]['rms'][1],
                'speaker_crest_mean': crest_mean,
                'speaker_crest_std': crest_std,
                'high_energy_frac': feat.get('high_energy_frac', 0.0),
                'low_energy_frac': feat.get('low_energy_frac', 0.0)
            })
            # AV 근사값을 세그먼트에 저장
            segments[idx]['av'] = {
                'arousal': float(np.clip(arousal, 0.0, 1.0)) if 'arousal' in locals() else None,
                'valence': float(np.clip(valence, 0.0, 1.0)) if 'valence' in locals() else None,
                'source': 'approx'
            }

        # 2차 스무딩: 동일 화자 인접 세그먼트 확산/억제
        # - 고립된 짧은 shout 억제: dur<0.45이고 양옆이 normal이며 high_energy_frac<0.3 → normal
        # - 인접 세그먼트 확산: shout인 세그먼트의 이웃(±0.25s)이 같은 화자이고 shout_prob 충분하며 high_energy_frac>=0.3 → shout로 승격
        index_to_feat = {f['index']: f for f in segment_features}
        for i, seg in enumerate(segments):
            if i not in preliminary_types:
                continue
            vtype, conf = preliminary_types[i]
            feat = index_to_feat.get(i)
            if not feat:
                continue
            dur = seg.get('end', 0) - seg.get('start', 0)
            spk = seg.get('speaker', 'Unknown')

            # 이웃 인덱스 후보
            prev_i = i - 1 if i - 1 >= 0 else None
            next_i = i + 1 if i + 1 < len(segments) else None
            neighbors = [j for j in [prev_i, next_i] if j is not None]

            # 시간 간격 체크 (0.25s 이내만 이웃으로 간주)
            valid_neighbors = []
            for j in neighbors:
                if segments[j].get('speaker', 'Unknown') != spk:
                    continue
                gap = 0.0
                if j == prev_i:
                    gap = max(0.0, seg.get('start', 0) - segments[j].get('end', 0))
                else:
                    gap = max(0.0, segments[j].get('start', 0) - seg.get('end', 0))
                if gap <= 0.25:
                    valid_neighbors.append(j)

            # 억제 규칙: 고립된 짧은 shout → normal
            if vtype == 'shout' and dur < 0.45:
                neighbor_types = [preliminary_types.get(j, ('normal', 0.0))[0] for j in valid_neighbors]
                high_frac = index_to_feat[i].get('high_energy_frac', 0.0)
                if all(nt != 'shout' for nt in neighbor_types) and high_frac < 0.30:
                    vtype = 'normal'
                    conf = max(0.0, conf - 0.1)

            # 추가 억제: 초반만 강한 에너지로 인한 오탐 방지
            if vtype == 'shout':
                efrac = index_to_feat[i].get('early_high_frac', 0.0)
                lfrac = index_to_feat[i].get('late_high_frac', 0.0)
                if dur >= 0.35 and efrac >= 0.35 and lfrac < 0.08:
                    # 이웃 중 하나가 shout이거나, 본인 late가 꽤 있으면 유지
                    neighbor_is_shout = any(preliminary_types.get(j, ('normal', 0.0))[0] == 'shout' for j in valid_neighbors)
                    if not neighbor_is_shout:
                        vtype = 'normal'
                        conf = max(0.0, conf - 0.1)

            # 확산 규칙: 이웃 승격
            if vtype == 'shout':
                for j in valid_neighbors:
                    ntype, nconf = preliminary_types.get(j, ('normal', 0.0))
                    nfeat = index_to_feat.get(j)
                    if not nfeat:
                        continue
                    if ntype != 'whisper' and nfeat.get('high_energy_frac', 0.0) >= 0.30:
                        # 확산 조건: shout_prob가 임계치에 근접
                        if (nfeat['shout_prob'] >= (max(shout_threshold, 0.68) - 0.05)) and (nfeat.get('late_high_frac', 0.0) >= 0.10):
                            preliminary_types[j] = ('shout', max(nconf, nfeat['shout_prob']))

            # 최종 기록
            preliminary_types[i] = (vtype, conf)

        # === 보수적 세그먼트 레벨 게이팅 + 카운트 보정 ===
        # 1) 초기 카운트 기록
        orig_whisper = sum(1 for v, _ in preliminary_types.values() if v == 'whisper')
        orig_shout = sum(1 for v, _ in preliminary_types.values() if v == 'shout')

        # 2) 게이팅 기준 (보수적)
        strict_margin = {'whisper': 0.18, 'shout': 0.16}
        # 에너지 지속 기준(기본값; 길이에 따라 상단에서 이미 조정됨)
        min_frac = {'whisper': 0.50, 'shout': 0.40}

        # 후보 저장
        demoted_whisper = []  # (idx, whisper_prob, margin)
        demoted_shout = []    # (idx, shout_prob, margin)

        final_types = dict(preliminary_types)
        for idx, (vtype, conf) in list(final_types.items()):
            feat = index_to_feat.get(idx, {})
            wprob = float(feat.get('whisper_prob', 0.0))
            sprob = float(feat.get('shout_prob', 0.0))
            if vtype == 'whisper':
                margin = wprob - sprob
                low_frac = float(feat.get('low_energy_frac', 0.0))
                if not (wprob >= whisper_threshold and margin >= strict_margin['whisper'] and low_frac >= min_frac['whisper']):
                    final_types[idx] = ('normal', max(0.0, conf - 0.1))
                    demoted_whisper.append((idx, wprob, margin))
            elif vtype == 'shout':
                margin = sprob - wprob
                high_frac = float(feat.get('high_energy_frac', 0.0))
                if not (sprob >= max(shout_threshold, 0.66) and margin >= strict_margin['shout'] and high_frac >= min_frac['shout']):
                    final_types[idx] = ('normal', max(0.0, conf - 0.1))
                    demoted_shout.append((idx, sprob, margin))

        # 3) 카운트 보정: 지나치게 줄어들면 상위 후보 일부 복구 (목표: 원래의 85%)
        def restore_top(demoted_list, target_count, cls):
            if target_count <= 0 or not demoted_list:
                return
            # 점수가 높은 순으로 복원 (확률 우선, 동률 시 margin)
            key_idx = 1 if cls == 'whisper' else 1
            demoted_sorted = sorted(demoted_list, key=lambda x: (x[1], x[2]), reverse=True)
            restored = 0
            for idx, prob, margin in demoted_sorted:
                if restored >= target_count:
                    break
                # 완화 기준으로 재허용
                final_types[idx] = (cls, prob)
                restored += 1

        new_whisper = sum(1 for v, _ in final_types.values() if v == 'whisper')
        new_shout = sum(1 for v, _ in final_types.values() if v == 'shout')
        target_whisper = max(0, int(orig_whisper * 0.85))
        target_shout = max(0, int(orig_shout * 0.85))
        if new_whisper < target_whisper:
            restore_top(demoted_whisper, target_whisper - new_whisper, 'whisper')
        if new_shout < target_shout:
            restore_top(demoted_shout, target_shout - new_shout, 'shout')

        # 결과 반영 (최종)
        for idx, (vtype, conf) in final_types.items():
            segments[idx]['voice_type'] = vtype
            segments[idx]['voice_type_confidence'] = round(float(conf), 3)

        total_segments = len(segments)
        whisper_count = sum(1 for seg in segments if seg.get('voice_type') == 'whisper')
        shout_count = sum(1 for seg in segments if seg.get('voice_type') == 'shout')
        normal_count = total_segments - whisper_count - shout_count

        whisper_pct = (whisper_count / total_segments * 100) if total_segments > 0 else 0
        shout_pct = (shout_count / total_segments * 100) if total_segments > 0 else 0
        normal_pct = (normal_count / total_segments * 100) if total_segments > 0 else 0

        # 비율 후처리 제거: 일반화 알고리즘 유지(영상 특화 억제 방지)

        try:
            print(f"🎯 Voice Type 분류 결과 (총 {total_segments}개 세그먼트):")
            print(f"   Whisper: {whisper_count}개 ({whisper_pct:.1f}%)")
            print(f"   Shout: {shout_count}개 ({shout_pct:.1f}%)")
            print(f"   Normal: {normal_count}개 ({normal_pct:.1f}%)")
            print(f"   동적 임계값: whisper_thr~{whisper_threshold:.2f}, shout_thr~{shout_threshold:.2f}")
        except Exception:
            # Windows 등 콘솔 인코딩 이슈 대비 (이모지/유니코드 없이 출력)
            print(f"Voice Type Result (total {total_segments} segments):")
            print(f"   Whisper: {whisper_count} ({whisper_pct:.1f}%)")
            print(f"   Shout: {shout_count} ({shout_pct:.1f}%)")
            print(f"   Normal: {normal_count} ({normal_pct:.1f}%)")
            print(f"   Thr: whisper~{whisper_threshold:.2f}, shout~{shout_threshold:.2f}")

        self.voice_type_stats = {
            'mean_rms': float(np.mean([f['rms'] for f in segment_features])) if segment_features else 0.0,
            'std_rms': float(np.std([f['rms'] for f in segment_features])) if segment_features else 0.0,
            'whisper_threshold': whisper_threshold,
            'shout_threshold': shout_threshold,
            'whisper_pct': whisper_pct,
            'shout_pct': shout_pct,
            'normal_pct': normal_pct
        }
        return segments

    def compute_rms(self, audio_segment):
        """오디오 세그먼트의 RMS(Root Mean Square) 볼륨 계산"""
        if isinstance(audio_segment, np.ndarray):
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
        return torch.sqrt(torch.mean(audio_segment**2)).item()

    

    def _estimate_syllables(self, text):
        """텍스트의 음절 수 추정 (한글 및 영어 지원)"""
        count = 0
        for ch in text:
            if '\uAC00' <= ch <= '\uD7A3':
                count += 1
            elif ch.lower() in 'aeiouy':
                count += 1
        return max(1, count)

    # 전역 통계는 analyze_volume_distribution를 사용

    def analyze_volume_distribution(self, audio):
        """전체 오디오의 볼륨 분포 분석 - 적응형 임계값 방식"""
        # 1초 단위로 청크 분할하여 RMS 볼륨 계산
        chunk_size = self.sample_rate
        volumes = []
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) > 0:
                rms = self.compute_rms(chunk)
                volumes.append(rms)

        if not volumes:
            return

        # 볼륨 분포 통계 계산
        volumes = np.array(volumes)
        self.volume_stats['values'] = volumes
        self.volume_stats['mean'] = np.mean(volumes)
        self.volume_stats['std'] = np.std(volumes)
        
        # 백분위수 기반 초기 임계값 (P30, P70의 실제 값을 구함)
        p30 = np.percentile(volumes, 30)
        p70 = np.percentile(volumes, 70)
        
        initial_thresholds = {
            'soft': p30,     # 하위 30%에 해당하는 볼륨 값
            'normal': p70,   # 상위 30%에 해당하는 볼륨 값
            'loud': float('inf')
        }
        
        # 적응형 임계값 조정 (각 레벨이 최소 10% 이상 되도록)
        # 이유: 데이터가 한쪽으로 치우쳐있거나 동일값이 많을 때 변별력 확보
        self._adjust_volume_thresholds(volumes, initial_thresholds)
        
        print("\n=== 볼륨 레벨 분포 (적응형 임계값) ===")
        for level in ['soft', 'normal', 'loud']:
            count = np.sum(self._classify_volume(volumes) == level)
            percentage = (count / len(volumes)) * 100
            threshold_val = self.volume_stats['thresholds'][level]
            threshold_str = f"{threshold_val:.3f}" if threshold_val != float('inf') else "∞"
            print(f"{level}: {percentage:.1f}% (임계값: {threshold_str})")
        print(f"P30값: {p30:.3f}, P70값: {p70:.3f} (백분위 기준점)")

    def _adjust_volume_thresholds(self, volumes, initial_thresholds):
        """적응형 임계값 조정 - 정확도 향상을 위한 핵심 로직"""
        def calculate_distribution(thresholds):
            levels = self._classify_volume(volumes, thresholds)
            unique, counts = np.unique(levels, return_counts=True)
            dist = dict(zip(unique, counts / len(volumes)))
            return {level: dist.get(level, 0.0) for level in ['soft', 'normal', 'loud']}

        # 초기 임계값으로 시작 (P30, P70 값들)
        current_thresholds = initial_thresholds.copy()
        distribution = calculate_distribution(current_thresholds)

        # 반복적 조정: 각 레벨이 최소 10% 이상 확보되도록
        # 목적: 극단 분포/동일값 집중 등에서도 변별력 유지
        MAX_ITERATIONS = 20
        for iteration in range(MAX_ITERATIONS):
            if all(v >= 0.1 for v in distribution.values()):
                break  # 모든 레벨이 10% 이상 확보됨

            # soft 레벨 조정
            if distribution['soft'] < 0.1:  # soft가 10% 미만
                current_thresholds['soft'] *= 1.1  # 임계값 상향 → 더 많은 구간이 soft
            elif distribution['soft'] > 0.4:  # soft가 40% 초과  
                current_thresholds['soft'] *= 0.9  # 임계값 하향 → soft 구간 축소

            # loud 레벨 조정 (normal 임계값 조정으로 제어)
            if distribution['loud'] < 0.1:  # loud가 10% 미만
                current_thresholds['normal'] *= 0.9  # normal 임계값 하향 → loud 구간 확대
            elif distribution['loud'] > 0.4:  # loud가 40% 초과
                current_thresholds['normal'] *= 1.1  # normal 임계값 상향 → loud 구간 축소

            distribution = calculate_distribution(current_thresholds)

        self.volume_stats['thresholds'] = current_thresholds

    def _classify_volume(self, volumes, thresholds=None):
        """볼륨값을 3단계로 분류"""
        if thresholds is None:
            thresholds = self.volume_stats['thresholds']
            
        if thresholds['soft'] is None:
            return np.full(len(volumes), 'normal')
            
        levels = np.full(len(volumes), 'normal', dtype='U10')
        levels[volumes < thresholds['soft']] = 'soft'
        levels[volumes >= thresholds['normal']] = 'loud'
        return levels

    

    def assign_pitch_level(self, pitch):
        """피치값에 따라 level 할당 (의미있는 극단값 보존)"""
        if not self.pitch_stats['p10']:
            return 'normal'
        
        # 1단계: 인간 청각 기준 절대값 체크 (극단값 보존)
        if pitch < 80:  # 매우 낮은 남성 음성
            return 'low'
        elif pitch > 400:  # 매우 높은 여성 음성 또는 감정적 발성
            return 'high'
        
        # 2단계: 상대적 백분위 분류 (일반적인 경우)
        if pitch <= self.pitch_stats['p10']:
            return 'low'
        elif pitch >= self.pitch_stats['p90']:
            return 'high'
        return 'normal'

    def analyze_speech_rate_distribution(self, segments):
        """전체 세그먼트의 발화 속도 분포 분석"""
        rates = []
        
        # 1. 모든 세그먼트의 발화 속도 수집
        for segment in segments:
            duration = segment['end'] - segment['start']
            word_count = len(segment.get('words', []))
            
            if duration > 0 and word_count > 0:
                # 초당 단어 수 계산
                rate = word_count / duration
                rates.append(rate)
                segment['_speech_rate'] = rate
        
        if rates:
            # 2. 통계값 계산
            self.speech_rate_stats['values'] = rates
            self.speech_rate_stats['mean'] = np.mean(rates)
            self.speech_rate_stats['std'] = np.std(rates)
            
            # 3. 25%, 75% 백분위수로 임계값 설정
            p25 = np.percentile(rates, 25)
            p75 = np.percentile(rates, 75)
            
            self.speech_rate_stats['thresholds'] = {
                'slow': p25,
                'normal': p75,
                'fast': float('inf')
            }

    def assign_speech_rate_level(self, rate):
        """발화 속도에 따른 level 할당"""
        if not self.speech_rate_stats['thresholds']['slow']:
            return 'normal'
        
        if rate <= self.speech_rate_stats['thresholds']['slow']:
            return 'slow'
        elif rate >= self.speech_rate_stats['thresholds']['normal']:
            return 'fast'
        return 'normal'

    def _get_context_audio(self, audio, segment, pad_duration=0.1):
        """짧은 세그먼트를 위한 컨텍스트 오디오 획득"""
        start = max(0, segment['start'] - pad_duration)
        end = min(len(audio) / self.sample_rate, segment['end'] + pad_duration)
        
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate)
        
        return audio[start_idx:end_idx]

    def _calculate_phonetic_speech_rate(self, audio, word_text, duration):
        """음성학적으로 개선된 발화속도 계산"""
        if duration <= 0:
            return 1.0
            
        # 1. 음성 에너지 기반 실제 발성 시간 계산
        actual_speech_time = self._estimate_actual_speech_time(audio, duration)
        
        # 2. 음절 복잡도 고려
        syllable_complexity = self._estimate_syllable_complexity(word_text)
        
        # 3. 조정된 발화속도 = (음절 복잡도) / (실제 발성 시간)
        if actual_speech_time > 0:
            adjusted_rate = syllable_complexity / actual_speech_time
        else:
            adjusted_rate = syllable_complexity / duration  # fallback
            
        return float(adjusted_rate)
    
    def _estimate_actual_speech_time(self, audio, total_duration):
        """무음 구간을 제외한 실제 발성 시간 추정"""
        if len(audio) == 0:
            return total_duration

        # 에너지 기반 음성 활동 감지 (벡터화)
        frame_size = int(self.sample_rate * 0.025)  # 25ms 프레임
        hop_size = int(self.sample_rate * 0.010)    # 10ms 홉

        # 프레임 RMS -> 에너지(=RMS^2)
        rms2d = librosa.feature.rms(
            y=audio.astype(np.float32),
            frame_length=frame_size,
            hop_length=hop_size,
            center=False
        )
        rms = rms2d.flatten()
        frame_energy = (rms ** 2).astype(np.float64)

        # 적응적 임계값 (전체 오디오 에너지의 5%)
        global_energy = float(np.mean(audio.astype(np.float32) ** 2))
        threshold = global_energy * 0.05

        if frame_energy.size == 0:
            return total_duration

        speech_frames = int(np.sum(frame_energy > threshold))
        total_frames = int(frame_energy.size)

        if total_frames > 0:
            speech_ratio = speech_frames / total_frames
            return total_duration * speech_ratio
        return total_duration
    
    def _estimate_syllable_complexity(self, text):
        """텍스트의 음절 복잡도 추정"""
        if not text:
            return 1.0
            
        # 기본 글자 수
        char_count = len(text.strip())
        
        # 언어별 가중치 적용
        complexity = 0
        for char in text:
            if char.isspace():
                continue
            elif 0x1100 <= ord(char) <= 0x11FF or 0x3130 <= ord(char) <= 0x318F or 0xAC00 <= ord(char) <= 0xD7AF:
                # 한글: 자음+모음 구조로 복잡
                complexity += 1.2
            elif char.isalpha():
                # 영어: 상대적으로 단순
                complexity += 1.0
            else:
                # 숫자, 기호
                complexity += 0.8
                
        return max(complexity, 1.0)