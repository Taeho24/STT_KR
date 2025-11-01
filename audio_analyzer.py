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

    # --- volume ---
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

    # --- pitch ---
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

    # --- speech rate ---
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

    # --- helpers ---
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

    # --- analyze_audio_features ---
    def analyze_audio_features(self, segments, audio):
        self.analyze_volume_distribution(audio)
        self.analyze_speech_rate_distribution(segments)
        fast_mode = bool(config.get('analysis', 'fast_mode', default=False))
        prev_volume_level = None
        hop_length_seg = 256
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
            frame_f0 = None
            frame_times = None
            try:
                frame_f0 = librosa.yin(seg_audio, fmin=50, fmax=600, hop_length=hop_length_seg)
                frame_times = (np.arange(len(frame_f0)) * hop_length_seg) / self.sample_rate
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
                    rms = self.compute_rms(w_audio)
                    word['rms'] = rms
                    word['volume_level'] = self._classify_volume([rms])[0]
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
                    dur_w = word['end'] - word['start']
                    if dur_w > 0:
                        if fast_mode:
                            sr_val = self._estimate_syllable_complexity(word.get('word', '')) / max(dur_w, 1e-6)
                        else:
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
        if _pitch_values:
            self.pitch_stats['values'] = _pitch_values
            self.pitch_stats['p10'] = float(np.percentile(_pitch_values, 10))
            self.pitch_stats['p90'] = float(np.percentile(_pitch_values, 90))
        return segments
