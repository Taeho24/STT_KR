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

    def compute_rms(self, audio_segment):
        """오디오 세그먼트의 RMS(Root Mean Square) 볼륨 계산"""
        if isinstance(audio_segment, np.ndarray):
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
        return torch.sqrt(torch.mean(audio_segment**2)).item()
