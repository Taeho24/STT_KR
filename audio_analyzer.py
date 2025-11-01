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
        self.volume_stats = {
            'values': [], 'mean': None, 'std': None, 'percentiles': {},
            'thresholds': {'soft': None, 'normal': None, 'loud': None}
        }
        self.pitch_stats = {'values': [], 'p10': None, 'p90': None}
        self.speech_rate_stats = {
            'values': [], 'mean': None, 'std': None,
            'thresholds': {'slow': None, 'normal': None, 'fast': None}
        }
        self.speech_rate_spacing = {'slow': 10, 'normal': 0, 'fast': -5}
        self.min_duration = 0.2
        self.ideal_duration = 0.5

    def compute_rms(self, audio_segment):
        if isinstance(audio_segment, np.ndarray):
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
        return torch.sqrt(torch.mean(audio_segment**2)).item()

    # volume
    def analyze_volume_distribution(self, audio):
        chunk_size = self.sample_rate
        volumes = []
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i+chunk_size]
            if len(chunk) > 0:
                volumes.append(self.compute_rms(chunk))
        if not volumes:
            return
        volumes = np.array(volumes)
        self.volume_stats['values'] = volumes
        self.volume_stats['mean'] = np.mean(volumes)
        self.volume_stats['std'] = np.std(volumes)
        p30 = np.percentile(volumes, 30)
        p70 = np.percentile(volumes, 70)
        initial_thresholds = {'soft': p30, 'normal': p70, 'loud': float('inf')}
        self._adjust_volume_thresholds(volumes, initial_thresholds)

    def _adjust_volume_thresholds(self, volumes, initial_thresholds):
        def calculate_distribution(thresholds):
            levels = self._classify_volume(volumes, thresholds)
            unique, counts = np.unique(levels, return_counts=True)
            dist = dict(zip(unique, counts / len(volumes)))
            return {level: dist.get(level, 0.0) for level in ['soft', 'normal', 'loud']}
        current_thresholds = initial_thresholds.copy()
        distribution = calculate_distribution(current_thresholds)
        MAX_ITERATIONS = 20
        for iteration in range(MAX_ITERATIONS):
            if all(v >= 0.1 for v in distribution.values()):
                break
            if distribution['soft'] < 0.1:
                current_thresholds['soft'] *= 1.1
            elif distribution['soft'] > 0.4:
                current_thresholds['soft'] *= 0.9
            if distribution['loud'] < 0.1:
                current_thresholds['normal'] *= 0.9
            elif distribution['loud'] > 0.4:
                current_thresholds['normal'] *= 1.1
            distribution = calculate_distribution(current_thresholds)
        self.volume_stats['thresholds'] = current_thresholds

    def _classify_volume(self, volumes, thresholds=None):
        if thresholds is None:
            thresholds = self.volume_stats['thresholds']
        if thresholds['soft'] is None:
            return np.full(len(volumes), 'normal')
        levels = np.full(len(volumes), 'normal', dtype='U10')
        levels[volumes < thresholds['soft']] = 'soft'
        levels[volumes >= thresholds['normal']] = 'loud'
        return levels

    # pitch
    def assign_pitch_level(self, pitch):
        if not self.pitch_stats['p10']:
            return 'normal'
        if pitch < 80:
            return 'low'
        elif pitch > 400:
            return 'high'
        if pitch <= self.pitch_stats['p10']:
            return 'low'
        elif pitch >= self.pitch_stats['p90']:
            return 'high'
        return 'normal'

    # speech rate
    def analyze_speech_rate_distribution(self, segments):
        rates = []
        for segment in segments:
            duration = segment['end'] - segment['start']
            word_count = len(segment.get('words', []))
            if duration > 0 and word_count > 0:
                rate = word_count / duration
                rates.append(rate)
                segment['_speech_rate'] = rate
        if rates:
            self.speech_rate_stats['values'] = rates
            self.speech_rate_stats['mean'] = np.mean(rates)
            self.speech_rate_stats['std'] = np.std(rates)
            p25 = np.percentile(rates, 25)
            p75 = np.percentile(rates, 75)
            self.speech_rate_stats['thresholds'] = {'slow': p25, 'normal': p75, 'fast': float('inf')}

    def assign_speech_rate_level(self, rate):
        if not self.speech_rate_stats['thresholds']['slow']:
            return 'normal'
        if rate <= self.speech_rate_stats['thresholds']['slow']:
            return 'slow'
        elif rate >= self.speech_rate_stats['thresholds']['normal']:
            return 'fast'
        return 'normal'

    # helpers
    def _get_context_audio(self, audio, segment, pad_duration=0.1):
        start = max(0, segment['start'] - pad_duration)
        end = min(len(audio) / self.sample_rate, segment['end'] + pad_duration)
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate)
        return audio[start_idx:end_idx]

    def _estimate_actual_speech_time(self, audio, total_duration):
        if len(audio) == 0:
            return total_duration
        frame_size = int(self.sample_rate * 0.025)
        hop_size = int(self.sample_rate * 0.010)
        rms2d = librosa.feature.rms(
            y=audio.astype(np.float32),
            frame_length=frame_size,
            hop_length=hop_size,
            center=False
        )
        rms = rms2d.flatten()
        frame_energy = (rms ** 2).astype(np.float64)
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
        if not text:
            return 1.0
        complexity = 0
        for char in text:
            if char.isspace():
                continue
            elif 0x1100 <= ord(char) <= 0x11FF or 0x3130 <= ord(char) <= 0x318F or 0xAC00 <= ord(char) <= 0xD7AF:
                complexity += 1.2
            elif char.isalpha():
                complexity += 1.0
            else:
                complexity += 0.8
        return max(complexity, 1.0)

    def _calculate_phonetic_speech_rate(self, audio, word_text, duration):
        if duration <= 0:
            return 1.0
        actual_speech_time = self._estimate_actual_speech_time(audio, duration)
        syllable_complexity = self._estimate_syllable_complexity(word_text)
        if actual_speech_time > 0:
            adjusted_rate = syllable_complexity / actual_speech_time
        else:
            adjusted_rate = syllable_complexity / duration
        return float(adjusted_rate)

    # classify_voice_types
    def classify_voice_types(self, segments, audio):
        MIN_DUR = 0.30
        FRAME_MS = 0.025
        HOP_MS = 0.010
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
        speaker_buckets = defaultdict(lambda: { 'rms': [], 'crest': [], 'hf_lf': [], 'breathiness': [], 'tilt': [] })
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
                'index': idx, 'speaker': speaker_id, 'rms': rms, 'crest': c_factor,
                'hf_lf': ratio_hf_lf, 'tilt': spectral_tilt, 'breathiness': breathiness, 'duration': duration
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
                'rms': mean_std(feats['rms']), 'crest': mean_std(feats['crest']),
                'hf_lf': mean_std(feats['hf_lf']), 'breathiness': mean_std(feats['breathiness']),
                'tilt': mean_std(feats['tilt'])
            }
        def normalize(value, mean, std):
            return (value - mean) / (std if std else 1.0)
        def sigmoid(x):
            return 1.0 / (1.0 + math.exp(-x))
        whisper_probs = []
        shout_probs = []
        def _frame_rms(arr: np.ndarray) -> np.ndarray:
            frame_size = max(1, int(self.sample_rate * 0.025))
            hop_size = max(1, int(self.sample_rate * 0.010))
            if arr.size == 0:
                return np.array([0.0], dtype=np.float32)
            rms2d = librosa.feature.rms(y=arr.astype(np.float32), frame_length=frame_size, hop_length=hop_size, center=False)
            return rms2d.flatten().astype(np.float32)
        for feat in segment_features:
            stats = speaker_stats.get(feat['speaker'])
            if not stats:
                stats = {'rms': (0.02, 0.01), 'crest': (5.5, 0.5), 'hf_lf': (1.0, 0.2), 'breathiness': (0.1, 0.05), 'tilt': (0.2, 0.1)}
            rms_norm = normalize(feat['rms'], *stats['rms'])
            crest_norm = normalize(feat['crest'], *stats['crest'])
            hf_lf_norm = normalize(feat['hf_lf'], *stats['hf_lf'])
            breath_norm = normalize(feat['breathiness'], *stats['breathiness'])
            tilt_norm = normalize(feat['tilt'], *stats['tilt'])
            duration = feat['duration']
            duration_bonus = np.clip(duration - 0.45, -0.4, 0.6)
            long_bonus = max(0.0, duration - 0.35)
            whisper_logit = ((-2.0 * rms_norm) + (1.4 * feat['breathiness']) + (0.9 * max(0.0, hf_lf_norm)) + (-0.8 * crest_norm) + (0.4 * breath_norm) + (0.3 * duration_bonus)) - 1.0
            shout_logit = ((2.0 * rms_norm) + (1.2 * feat['tilt']) + (0.9 * max(0.0, crest_norm)) + (-0.7 * hf_lf_norm) + (0.3 * tilt_norm) + (0.4 * long_bonus)) - 0.9
            whisper_prob = sigmoid(whisper_logit)
            shout_prob = sigmoid(shout_logit)
            feat['whisper_prob'] = whisper_prob
            feat['shout_prob'] = shout_prob
            arousal_logit = (1.6 * rms_norm + 0.9 * feat['tilt'] + 0.5 * max(0.0, hf_lf_norm) + 0.4 * max(0.0, crest_norm)) - 0.2 * max(0.0, breath_norm)
            arousal = float(np.clip(sigmoid(arousal_logit), 0.0, 1.0))
            valence_logit = (-1.3 * max(0.0, breath_norm) + -0.6 * max(0.0, hf_lf_norm) + -0.2 * max(0.0, crest_norm) + 0.1)
            valence = float(np.clip(sigmoid(valence_logit), 0.0, 1.0))
            seg = segments[feat['index']]
            start_idx = int(seg['start'] * self.sample_rate)
            end_idx = int(seg['end'] * self.sample_rate)
            seg_audio = audio[start_idx:end_idx]
            frame_rms = _frame_rms(seg_audio)
            spk_mean, spk_std = speaker_stats[feat['speaker']]['rms']
            high_thr = max(spk_mean + 0.50 * spk_std, 0.028)
            low_thr = min(spk_mean - 0.50 * spk_std, 0.016)
            frame_time = max(0.010, 0.025)
            high_frames = np.sum(frame_rms >= high_thr)
            low_frames = np.sum(frame_rms <= low_thr)
            high_dur = float(high_frames * frame_time)
            low_dur = float(low_frames * frame_time)
            total_dur = max(1e-6, feat['duration'])
            feat['high_energy_dur'] = high_dur
            feat['low_energy_dur'] = low_dur
            feat['high_energy_frac'] = float(np.clip(high_dur / total_dur, 0.0, 1.0))
            feat['low_energy_frac'] = float(np.clip(low_dur / total_dur, 0.0, 1.0))
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
            if prob_arr.size < 4:
                return base
            mean = float(prob_arr.mean())
            std = float(prob_arr.std())
            candidate = mean + 0.5 * std
            return float(np.clip(max(base, candidate), base, 0.88))
        whisper_threshold = adaptive_threshold(whisper_probs_arr, base_whisper_thr)
        shout_threshold = adaptive_threshold(shout_probs_arr, base_shout_thr)
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
            hf_guard = feat['hf_lf'] <= (speaker_stats[feat['speaker']]['hf_lf'][0] - 0.35 * speaker_stats[feat['speaker']]['hf_lf'][1])
            seg = segments[idx]
            dur = seg.get('end', 0) - seg.get('start', 0)
            if dur < 0.30:
                voice_type = 'normal'
                confidence = max(0.0, max(whisper_prob, shout_prob) - 0.5)
            else:
                crest_upper = min(6.0, crest_mean + 0.15 * crest_std)
                sustained_low_ok = (feat.get('low_energy_dur', 0.0) >= 0.45) or (feat.get('low_energy_frac', 0.0) >= 0.60)
                sustained_high_ok = (feat.get('high_energy_dur', 0.0) >= 0.35) or (feat.get('high_energy_frac', 0.0) >= 0.40)
                if not sustained_high_ok and 0.40 <= dur <= 0.80:
                    if (feat.get('high_energy_frac', 0.0) >= 0.30) and (tilt_guard or crest_guard):
                        sustained_high_ok = True
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
                    spectral_combo = (1 if tilt_guard else 0) + (1 if crest_guard else 0) + (1 if hf_guard else 0)
                    sustained_ok = sustained_high_ok
                    if dur >= 1.0:
                        sustained_ok = (feat.get('high_energy_frac', 0.0) >= 0.50) or (feat.get('high_energy_dur', 0.0) >= 0.60)
                    else:
                        sustained_ok = (feat.get('high_energy_frac', 0.0) >= 0.30) or (feat.get('high_energy_dur', 0.0) >= 0.30)
                    min_spectral = 2 if dur >= 0.90 else 1
                    late_emph_ok = feat.get('late_high_frac', 0.0) >= (0.12 if dur < 0.9 else 0.18)
                    base_shout_cond = (
                        shout_prob >= max(shout_threshold, 0.66) and
                        (shout_prob - whisper_prob) >= 0.10 and
                        rms_guard_shout and
                        spectral_combo >= min_spectral and
                        sustained_ok and
                        (late_emph_ok)
                    )
                    exclaim_cond = (
                        0.40 <= dur <= 0.90 and
                        shout_prob >= (max(shout_threshold, 0.66) - 0.03) and
                        rms_guard_shout and
                        (tilt_guard or crest_guard) and
                        (feat.get('high_energy_frac', 0.0) >= 0.25) and
                        (feat.get('late_high_frac', 0.0) >= 0.12)
                    )
                    early_only = (
                        feat.get('early_high_frac', 0.0) >= 0.30 and
                        feat.get('late_high_frac', 0.0) < 0.08 and
                        not sustained_ok
                    )
                    if (base_shout_cond or exclaim_cond) and not early_only:
                        voice_type = 'shout'
                        confidence = shout_prob
                    else:
                        voice_type = 'normal'
                        confidence = max(0.0, max(whisper_prob, shout_prob) - 0.5)
            preliminary_types[idx] = (voice_type, float(confidence))
            analysis = segments[idx].setdefault('voice_analysis', {})
            analysis.update({
                'rms': feat['rms'], 'crest_factor': feat['crest'], 'hf_lf_ratio': feat['hf_lf'],
                'spectral_tilt': feat['tilt'], 'breathiness': feat['breathiness'],
                'whisper_prob': round(whisper_prob, 3), 'shout_prob': round(shout_prob, 3)
            })
            segments[idx]['av'] = {'arousal': None, 'valence': None, 'source': 'approx'}
        for idx, (vtype, conf) in preliminary_types.items():
            segments[idx]['voice_type'] = vtype
            segments[idx]['voice_type_confidence'] = round(float(conf), 3)
        total_segments = len(segments)
        whisper_count = sum(1 for seg in segments if seg.get('voice_type') == 'whisper')
        shout_count = sum(1 for seg in segments if seg.get('voice_type') == 'shout')
        normal_count = total_segments - whisper_count - shout_count
        whisper_pct = (whisper_count / total_segments * 100) if total_segments > 0 else 0
        shout_pct = (shout_count / total_segments * 100) if total_segments > 0 else 0
        normal_pct = (normal_count / total_segments * 100) if total_segments > 0 else 0
        return segments
