#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import librosa
from torchaudio.functional import spectral_centroid
from .config import config

class AudioAnalyzer:
    """오디오 분석 및 특성 추출 클래스"""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        
        # config에서 임계값 설정 로드
        self.volume_thresholds = config.get('thresholds', 'volume')
        self.pitch_thresholds = config.get('thresholds', 'pitch')
        self.speech_rate_thresholds = config.get('thresholds', 'speech_rate')

        # 볼륨 분포 통계 저장용
        self.volume_stats = {
            'values': [],
            'mean': None,
            'std': None,
            'thresholds': {
                'soft': None,    # 하위 30% 기준
                'normal': None,  # 30%~70% 기준
                'loud': None     # 상위 30% 기준
            }
        }
        
        # 볼륨 레벨 정규화를 위한 기준값
        self.volume_percentiles = {
            'p10': None,  # 하위 10%
            'p90': None   # 상위 90%
        }
        
        # 음향 특성 임계값 (실험적으로 조정 가능)
        # 초기 임계값 설정 (분석 과정에서 동적으로 조정됨)
        self.volume_thresholds = {
            'whisper': 0.01,  # 속삭임 임계값 (RMS)
            'shout':   0.05   # 소리침 임계값 (RMS)
        }
        self.centroid_thresholds = {
            'low':  800,      # 낮은 피치 임계값 (Hz)
            'high': 2500      # 높은 피치 임계값 (Hz)
        }
        self.speech_rate_thresholds = {
            'slow': 2.5,      # 느린 발화 임계값 (음절/초)
            'fast': 5.0       # 빠른 발화 임계값 (음절/초)
        }

        # 전체 오디오 통계 (analyze_audio_statistics에서 설정됨)
        self.audio_stats = {
            'mean_volume': None,
            'std_volume': None,
            'volume_percentiles': None
        }

        # 화자별 통계 저장
        self.speaker_stats = {}
        
        # 3단계 분류를 위한 임계값
        self.volume_levels = {
            'soft': -0.8,    # -0.8σ 미만: 작은 소리
            'normal': 0.8,   # -0.8σ ~ 0.8σ: 보통
            'loud': 0.8      # 0.8σ 초과: 큰 소리
        }
        
        self.pitch_levels = {
            'low': -0.8,     # -0.8σ 미만: 낮은 피치
            'normal': 0.8,   # -0.8σ ~ 0.8σ: 보통
            'high': 0.8      # 0.8σ 초과: 높은 피치
        }

        # 볼륨 임계값 수정
        self.volume_thresholds = {
            'soft': 0.1,    # RMS < 0.1
            'normal': 0.3,  # 0.1 <= RMS < 0.3
            'loud': float('inf')  # RMS >= 0.3
        }

        # 피치 임계값 수정
        self.pitch_thresholds = {
            'low': 100,     # < 100 Hz
            'normal': 250,  # 100-250 Hz
            'high': float('inf')  # > 250 Hz
        }
        
        # 발화 속도 임계값 추가
        self.speech_rate_thresholds = {
            'slow': 2.5,    # < 2.5 음절/초
            'normal': 4.5,  # 2.5-4.5 음절/초
            'fast': float('inf')   # > 4.5 음절/초
        }

        # 볼륨 레벨 조정을 위한 속성 추가
        self.volume_stats = {
            'mean': None,
            'std': None,
            'percentiles': None,
            'thresholds': {
                'soft': None,    # 하위 30% 기준
                'normal': None,  # 30%~70% 기준
                'loud': None     # 상위 30% 기준
            }
        }

        # 피치 분포 저장용
        self.pitch_stats = {
            'values': [],
            'p10': None,
            'p90': None
        }

        # 발화 속도 통계 저장용
        self.speech_rate_stats = {
            'values': [],
            'mean': None,
            'std': None,
            'thresholds': {
                'slow': None,    # 하위 25% 기준
                'normal': None,  # 25%~75% 기준
                'fast': None     # 상위 25% 기준
            }
        }
        
        # 발화 속도별 자간 간격 수정 (-5 ~ 10 범위)
        self.speech_rate_spacing = {
            'slow': 10,     # 매우 느린 발화 → 자간 넓게
            'normal': 0,    # 보통 발화 → 기본 자간
            'fast': -5      # 매우 빠른 발화 → 자간 좁게
        }

        # 최소 지속 시간 설정 추가
        self.min_duration = 0.2  # 200ms
        self.ideal_duration = 0.5  # 500ms (음성 특성 분석에 이상적인 길이)

    def compute_rms(self, audio_segment):
        """오디오 세그먼트의 RMS(Root Mean Square) 볼륨 계산"""
        if isinstance(audio_segment, np.ndarray):
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
        return torch.sqrt(torch.mean(audio_segment**2)).item()

    def compute_centroid(self, audio_segment):
        """오디오 세그먼트의 스펙트럴 중심(spectral centroid) 계산"""
        if isinstance(audio_segment, np.ndarray):
            audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
        if audio_segment.dim() == 1:
            audio_segment = audio_segment.unsqueeze(0)  # (1, n)

        n_fft = 1024
        hop_length = 512
        win_length = n_fft
        pad = 0  # ▶ pad 인자 추가

        # 길이가 너무 짧으면 처리 불가
        if audio_segment.size(-1) < win_length:
            return 0.0

        centroids = spectral_centroid(
            audio_segment,
            sample_rate=self.sample_rate,
            pad=pad,
            window=torch.hann_window(win_length),
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length
        )
        return centroids.mean().item() if centroids.numel() > 0 else 0.0

    def compute_speech_rate(self, word_info):
        """단어의 발화 속도 계산 (초당 음절 수)"""
        word = word_info["word"].strip()
        duration = word_info["end"] - word_info["start"]
        syllable_count = self._estimate_syllables(word)
        if duration < 0.1 or syllable_count == 0:
            return 3.0  # 기본 발화 속도
        return syllable_count / duration

    def _estimate_syllables(self, text):
        """텍스트의 음절 수 추정 (한글 및 영어 지원)"""
        count = 0
        for ch in text:
            if '\uAC00' <= ch <= '\uD7A3':
                count += 1
            elif ch.lower() in 'aeiouy':
                count += 1
        return max(1, count)

    def analyze_audio_statistics(self, audio):
        """전체 오디오의 볼륨 통계 분석 및 임계값 동적 설정"""
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
        self.volume_stats['mean'] = np.mean(volumes)
        self.volume_stats['std'] = np.std(volumes)
        
        # 백분위수 계산 (0~100%)
        self.volume_stats['percentiles'] = {
            i: np.percentile(volumes, i) for i in range(0, 101, 5)
        }

        # 초기 임계값 설정 (30/70 백분위)
        initial_thresholds = {
            'soft': self.volume_stats['percentiles'][30],    # 하위 30%
            'normal': self.volume_stats['percentiles'][70],  # 상위 30%
            'loud': float('inf')
        }

        # 임계값 조정 (각 레벨이 최소 10% 이상 되도록)
        self._adjust_volume_thresholds(volumes, initial_thresholds)

    def analyze_volume_distribution(self, audio):
        """전체 오디오의 볼륨 분포 분석"""
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
        
        # 백분위수 계산
        p30 = np.percentile(volumes, 30)
        p70 = np.percentile(volumes, 70)
        
        # 초기 임계값 설정
        initial_thresholds = {
            'soft': p30,     # 하위 30%
            'normal': p70,   # 30%~70%
            'loud': float('inf')  # 상위 30%
        }
        
        # 임계값 조정 (각 레벨이 최소 10% 이상 되도록)
        self._adjust_volume_thresholds(volumes, initial_thresholds)
        
        print("\n=== 볼륨 레벨 분포 ===")
        for level in ['soft', 'normal', 'loud']:
            count = np.sum(self._classify_volume(volumes) == level)
            percentage = (count / len(volumes)) * 100
            print(f"{level}: {percentage:.1f}% (임계값: {self.volume_stats['thresholds'][level]:.3f})")

    def _adjust_volume_thresholds(self, volumes, initial_thresholds):
        """각 볼륨 레벨이 최소 10% 이상이 되도록 임계값 조정"""
        def calculate_distribution(thresholds):
            levels = self._classify_volume(volumes, thresholds)
            unique, counts = np.unique(levels, return_counts=True)
            dist = dict(zip(unique, counts / len(volumes)))
            return {level: dist.get(level, 0.0) for level in ['soft', 'normal', 'loud']}

        # 초기 임계값으로 시작
        current_thresholds = initial_thresholds.copy()
        distribution = calculate_distribution(current_thresholds)

        # 반복적으로 임계값 조정
        MAX_ITERATIONS = 20
        for _ in range(MAX_ITERATIONS):
            if all(v >= 0.1 for v in distribution.values()):
                break

            # soft 임계값 조정
            if distribution['soft'] < 0.1:
                current_thresholds['soft'] *= 1.1
            elif distribution['soft'] > 0.4:
                current_thresholds['soft'] *= 0.9

            # normal 임계값 조정
            if distribution['loud'] < 0.1:
                current_thresholds['normal'] *= 0.9
            elif distribution['loud'] > 0.4:
                current_thresholds['normal'] *= 1.1

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

    def compute_speaker_statistics(self, segments):
        """화자별 음성 특성 평균 계산"""
        speaker_data = {}
        
        # 화자별 데이터 수집
        for seg in segments:
            speaker = seg.get('speaker', 'UNKNOWN')
            if speaker not in speaker_data:
                speaker_data[speaker] = {
                    'volumes': [],
                    'pitches': [],
                    'speech_rates': []
                }
            
            speaker_data[speaker]['volumes'].append(seg.get('raw_volume', 0))
            speaker_data[speaker]['pitches'].append(seg.get('pitch', 0))
            for word in seg.get('words', []):
                if 'speech_rate' in word:
                    speaker_data[speaker]['speech_rates'].append(word['speech_rate'])
        
        # 화자별 평균/표준차 계산
        self.speaker_stats = {}
        for speaker, data in speaker_data.items():
            self.speaker_stats[speaker] = {
                'volume': {
                    'mean': np.mean(data['volumes']),
                    'std': np.std(data['volumes'])
                },
                'pitch': {
                    'mean': np.mean(data['pitches']),
                    'std': np.std(data['pitches'])
                },
                'speech_rate': {
                    'mean': np.mean(data['speech_rates']),
                    'std': np.std(data['speech_rates'])
                }
            }

    def classify_level(self, value, mean, std, thresholds):
        """값을 3단계로 분류"""
        if std == 0:
            return 'normal'
        z_score = (value - mean) / std
        if z_score < thresholds['soft'] if 'soft' in thresholds else thresholds['low']:
            return 'soft' if 'soft' in thresholds else 'low'
        elif z_score > thresholds['loud'] if 'loud' in thresholds else thresholds['high']:
            return 'loud' if 'loud' in thresholds else 'high'
        return 'normal'

    def analyze_pitch_distribution(self, segments, audio):
        """전체 세그먼트의 피치 분포를 분석하여 임계값 설정"""
        pitch_values = []
        
        # 1. 모든 세그먼트의 피치 수집
        for segment in segments:
            start_idx = int(segment['start'] * self.sample_rate)
            end_idx = int(segment['end'] * self.sample_rate)
            audio_slice = audio[start_idx:end_idx]
            
            if len(audio_slice) > 0:
                # librosa의 pitch 추정 사용
                frequencies = librosa.yin(audio_slice, fmin=50, fmax=600)
                valid_freqs = frequencies[frequencies > 0]
                if len(valid_freqs) > 0:
                    pitch = np.mean(valid_freqs)
                    pitch_values.append(pitch)
        
        if pitch_values:
            # 2. 10%와 90% 지점 계산
            self.pitch_stats['values'] = pitch_values
            self.pitch_stats['p10'] = np.percentile(pitch_values, 10)
            self.pitch_stats['p90'] = np.percentile(pitch_values, 90)

    def assign_pitch_level(self, pitch):
        """피치값에 따라 level 할당"""
        if not self.pitch_stats['p10']:
            return 'normal'
        
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

    def analyze_audio_features(self, segments, audio):
        """오디오 특성 분석 수정"""
        # 1. 전체 오디오 통계 분석
        self.analyze_volume_distribution(audio)
        self.analyze_pitch_distribution(segments, audio)
        self.analyze_speech_rate_distribution(segments)
        
        for segment in segments:
            duration = segment['end'] - segment['start']
            
            # 짧은 세그먼트 처리 개선
            if duration < self.min_duration:
                # 주변 컨텍스트 고려
                context_audio = self._get_context_audio(audio, segment, pad_duration=0.1)
                seg_audio = context_audio
            else:
                start_idx = int(segment['start'] * self.sample_rate)
                end_idx = int(segment['end'] * self.sample_rate)
                seg_audio = audio[start_idx:end_idx]

            if len(seg_audio) > 0:
                seg_volume = self.compute_rms(seg_audio)
                segment['volume_level'] = self._classify_volume([seg_volume])[0]
                
                # 단어별 특성 처리
                words = segment.get('words', [])
                for word in words:
                    try:
                        if not isinstance(word, dict) or 'start' not in word or 'end' not in word:
                            continue

                        # 단어의 시작/끝 시간을 샘플 인덱스로 변환
                        start_idx = int(word['start'] * self.sample_rate)
                        end_idx = int(word['end'] * self.sample_rate)
                        
                        # 단어 구간의 오디오 추출
                        word_audio = audio[start_idx:end_idx]
                        
                        if len(word_audio) > 0:
                            # 단어별 볼륨 레벨 분석
                            rms = self.compute_rms(word_audio)
                            word['volume_level'] = self._classify_volume([rms])[0]
                            
                            # 단어별 피치 레벨 분석
                            frequencies = librosa.yin(word_audio, fmin=50, fmax=600)
                            valid_freqs = frequencies[frequencies > 0]
                            if len(valid_freqs) > 0:
                                avg_pitch = np.mean(valid_freqs)
                                word['pitch_level'] = self.assign_pitch_level(avg_pitch)
                            else:
                                word['pitch_level'] = 'normal'
                        else:
                            word['volume_level'] = 'normal'
                            word['pitch_level'] = 'normal'
                        
                        # 발화 속도는 단어 단위로 계산
                        if word['end'] - word['start'] > 0:
                            speech_rate = len(word.get('word', '')) / (word['end'] - word['start'])
                            word['speech_rate'] = self.assign_speech_rate_level(speech_rate)
                        else:
                            word['speech_rate'] = 'normal'
                    except Exception as e:
                        print(f"단어 처리 중 오류 발생: {str(e)}")
                        word['volume_level'] = 'normal'
                        word['pitch_level'] = 'normal'
                        word['speech_rate'] = 'normal'

            # 세그먼트에 통계 정보 저장
            segment['volume_stats'] = {
                'mean': np.mean([self.compute_rms(audio[int(w['start']*self.sample_rate):int(w['end']*self.sample_rate)]) 
                               for w in words if isinstance(w, dict) and 'start' in w and 'end' in w]) if words else 0.0,
                'levels': [w.get('volume_level', 'normal') for w in words] if words else ['normal']
            }
            segment['pitch_stats'] = {
                'levels': [w.get('pitch_level', 'normal') for w in words] if words else ['normal']
            }
            segment['speech_rate_stats'] = {
                'levels': [w.get('speech_rate', 'normal') for w in words] if words else ['normal']
            }
        
        return segments

    def _get_context_audio(self, audio, segment, pad_duration=0.1):
        """짧은 세그먼트를 위한 컨텍스트 오디오 획득"""
        start = max(0, segment['start'] - pad_duration)
        end = min(len(audio) / self.sample_rate, segment['end'] + pad_duration)
        
        start_idx = int(start * self.sample_rate)
        end_idx = int(end * self.sample_rate)
        
        return audio[start_idx:end_idx]

    def analyze_volume(self, audio_segment):
        """볼륨 레벨 분석 (3단계)"""
        if self.volume_stats['thresholds']['soft'] is None:
            return 'normal'  # 통계가 없으면 기본값 반환
            
        rms = self.compute_rms(audio_segment)
        
        if rms < self.volume_stats['thresholds']['soft']:
            return 'soft'
        elif rms >= self.volume_stats['thresholds']['normal']:
            return 'loud'
        return 'normal'

    def analyze_pitch(self, audio_segment):
        """피치 레벨 분석 (3단계)"""
        if len(audio_segment) < 512:
            return 'normal'
            
        frequencies = librosa.yin(audio_segment, fmin=75, fmax=600)
        avg_pitch = np.mean(frequencies[frequencies > 0])
        
        if avg_pitch < 150:
            return 'low'
        elif avg_pitch > 250:
            return 'high'
        return 'normal'
    
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
                    word["type"] = -1
                    continue

                rms = compute_rms(audio_segment)
                if isinstance(audio_segment, np.ndarray):
                    audio_segment = torch.tensor(audio_segment, dtype=torch.float32)
                rms = torch.sqrt(torch.mean(audio_segment ** 2)).item()

                if rms < 0.02:
                    word["type"] = 0  # 속삭임
                elif rms > 0.07:
                    word["type"] = 2  # 고함
                else:
                    word["type"] = 1  # 일반
        return segments
