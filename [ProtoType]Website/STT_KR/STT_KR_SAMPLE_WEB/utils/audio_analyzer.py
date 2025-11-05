#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np

class AudioAnalyzer:
    """오디오 분석 및 특성 추출 클래스"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    # 임시 함수
    def analyze_voice_type(self, segments, audio):
        def compute_rms(audio_segment):
            if isinstance(audio_segment, np.ndarray):
                audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
            return torch.sqrt(torch.mean(audio_segment ** 2)).item()

        for segment in segments:
            for word in segment.get("words", []):
                if "start" not in word or "end" not in word:
                    continue
                
                start_sample = int(word["start"] * self.sample_rate)
                end_sample = int(word["end"] * self.sample_rate)
                audio_segment = audio[start_sample:end_sample]

                if len(audio_segment) == 0:
                    word["voice_type"] = -1
                    continue

                rms = compute_rms(audio_segment)
                if isinstance(audio_segment, np.ndarray):
                    audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
                rms = torch.sqrt(torch.mean(audio_segment ** 2)).item()

                if rms < 0.02:
                    word["voice_type"] = "whisper"  # 속삭임
                elif rms > 0.07:
                    word["voice_type"] = "shout"  # 고함
                else:
                    word["voice_type"] = "normal"  # 일반
        return segments