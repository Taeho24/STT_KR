import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)
import librosa
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import torch.nn.functional as F
from tqdm import tqdm

from .config import config
from .db_manager import DBManager

DEFAULT_AUDIO_MODEL = "xbgoose/hubert-large-speech-emotion-recognition-russian-dusha-finetuned"
DEFAULT_TEXT_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('emotion_classification.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    features: Dict[str, float]
    text_score: float
    audio_score: float
    text_distribution: Dict[str, float]
    audio_distribution: Dict[str, float]
    combined_distribution: Dict[str, float]

class EmotionClassifier:
    """감정 분류 모델과 앙상블 처리를 담당하는 클래스"""
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
        cache_dir: str = ".cache",
        audio_model_name: str | None = None,
        text_model_name: str | None = None,
        enable_text: bool | None = None, 
        task_id: str | None = None,
    ):
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_manager = DBManager(task_id=task_id)

        # 모델 이름 구성 (config 우선, 파라미터로 재정의 가능)
        configured_audio = config.get('models', 'audio', DEFAULT_AUDIO_MODEL)
        configured_text = config.get('models', 'text', DEFAULT_TEXT_MODEL)

        self.audio_model_name = audio_model_name or configured_audio or DEFAULT_AUDIO_MODEL
        resolved_text_name: Optional[str]
        if text_model_name is None:
            resolved_text_name = configured_text or DEFAULT_TEXT_MODEL
        else:
            resolved_text_name = text_model_name or None

        self.available_audio_models = config.get('models', 'audio_candidates', [])
        self.available_text_models = config.get('models', 'text_candidates', [])
        
        # === 최적화된 다언어 감정 분석 모델 ===
        # XLS-R 기반 - 한국어/영어 지원, 빠른 추론
        print(f"🔄 Loading audio emotion model: {self.audio_model_name}")
        
        try:
            # 1차 시도: XLS-R 기반 감정 인식 모델 (더 빠르고 효율적)
            # FP32 사용 - 정확도 우선 (FP16은 감정 분류에서 정밀도 손실 큼)
            self.audio_model = AutoModelForAudioClassification.from_pretrained(
                self.audio_model_name,
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float32  # 정확도 우선: FP32 고정
            ).to(device)
            
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.audio_model_name,
                cache_dir=str(self.cache_dir)
            )
            self.feature_sampling_rate = getattr(self.feature_extractor, "sampling_rate", 16000)
            print("✅ Audio emotion model loaded successfully (FP32 for accuracy)")
            
        except Exception as e:
            logging.error(f"Failed to load audio model: {str(e)}")
            raise

        # 모델별 보정(캘리브레이션) 제거: 단순화(현재 기본 xbgoose에는 불필요)
        self._calibration: Dict[str, float] = {}
        
        # 텍스트 모델은 보조 수단으로만 사용 (다언어 지원)
        self.weights = config.get('emotions', 'weights') or {}
        self.audio_weight = float(self.weights.get('audio', 1.0))
        self.text_weight = float(self.weights.get('text', 0.0))

        if enable_text is None:
            text_enabled = self.text_weight > 0 and resolved_text_name is not None
        else:
            text_enabled = enable_text and resolved_text_name is not None

        self.text_enabled = text_enabled
        if self.text_enabled:
            self.text_model_name = resolved_text_name
            print(f"🔄 Loading text emotion model: {self.text_model_name}")
            self.text_model = AutoModelForSequenceClassification.from_pretrained(
                self.text_model_name,  # 일단 유지 (텍스트는 보조)
                cache_dir=str(self.cache_dir),
                torch_dtype=torch.float32  # 정확도 우선: FP32 고정
            ).to(device)
            
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                self.text_model_name,
                cache_dir=str(self.cache_dir)
            )
        else:
            self.text_model_name = None
            self.text_model = None
            self.text_tokenizer = None
            self.text_weight = 0.0

        # config에서 모든 설정 로드
        self.emotion_mapping = config.get('emotions', 'mapping')
        self.emotion_weights = config.get('emotions', 'emotion_weights')
        self.excluded_emotions = config.get('emotions', 'exclude', default=[]) or []
        self.audio_temperature = config.get('emotions', 'audio_temperature', default=1.0) or 1.0
        self.ensemble_settings = config.get('emotions', 'ensemble', default={}) or {}
        self.emotion_colors = self.db_manager.load_config()['hex_colors']['emotion_colors']
        self.default_color = self.db_manager.load_config()['hex_colors']['default_color']

        self.audio_confidence_threshold = float(self.ensemble_settings.get('audio_confidence_threshold', 0.6))
        self.text_confidence_threshold = float(self.ensemble_settings.get('text_confidence_threshold', 0.55))
        self.dominance_margin = float(self.ensemble_settings.get('dominance_margin', 0.15))
        self.audio_confidence_boost = float(self.ensemble_settings.get('audio_confidence_boost', 1.3))
        self.audio_confidence_decay = float(self.ensemble_settings.get('audio_confidence_decay', 0.7))
        self.text_confidence_boost = float(self.ensemble_settings.get('text_confidence_boost', 1.15))
        self.text_confidence_decay = float(self.ensemble_settings.get('text_confidence_decay', 0.8))
        self.neutral_suppression = float(self.ensemble_settings.get('neutral_suppression', 0.7))
        self.neutral_floor = float(self.ensemble_settings.get('neutral_floor', 0.05))
        # 중립 선호 가드 및 시간 스무딩 설정
        self.neutral_guard = config.get('emotions', 'neutral_guard', default={}) or {}
        self.ng_enabled = bool(self.neutral_guard.get('enabled', True))
        self.ng_min_audio_conf = float(self.neutral_guard.get('min_audio_conf', 0.62))
        self.ng_min_audio_margin = float(self.neutral_guard.get('min_audio_margin', 0.18))
        self.ng_min_text_support = float(self.neutral_guard.get('min_text_support', 0.22))
        self.ng_damp_factor = float(self.neutral_guard.get('damp_factor', 0.35))
        self.ng_targets = set(self.neutral_guard.get('target_classes', ['happy', 'angry']))
        # AV 융합 설정(결핍 클래스 보강용)
        self.av_fusion = config.get('emotions', 'av_fusion', default={}) or {}
        self.av_enabled = bool(self.av_fusion.get('enabled', True))
        self.av_text_gain = float(self.av_fusion.get('text_gain', 0.4))
        self.av_audio_gain = float(self.av_fusion.get('audio_gain', 0.15))
        
        # Whisper 모델 레이블 매핑 (다양한 표현을 7개 감정으로 통합)
        self.whisper_emotion_mapping = {
            'happy': 'happy',
            'happiness': 'happy',
            'joy': 'happy',
            'excited': 'happy',
            'sad': 'sad',
            'sadness': 'sad',
            'angry': 'angry',
            'anger': 'angry',
            'fear': 'fear',
            'fearful': 'fear',
            'surprise': 'surprise',
            'surprised': 'surprise',
            'disgust': 'disgust',
            'disgusted': 'disgust',
            'neutral': 'neutral',
            'calm': 'neutral',
            'bored': 'neutral'
        }

        self._setup_memory_management()
        logging.info("Emotion classifier initialized successfully")

        # 텍스트/오디오 모델 레이블 출력
        if self.text_enabled and self.text_model is not None:
            print("Text model labels:", self.text_model.config.id2label)
        print("Audio model labels:", self.audio_model.config.id2label)

    def _apply_emotion_exclusions(self, scores: Dict[str, float]) -> Dict[str, float]:
        """제외 설정된 감정 레이블을 0으로 만들고 재정규화.

        - 모든 점수가 0이 되면 원본 유지(안정성), 또는 neutral=1.0로 대체할 수 있음.
        """
        if not self.excluded_emotions:
            return scores
        if not isinstance(scores, dict):
            return scores
        out = dict(scores)
        changed = False
        for ex in self.excluded_emotions:
            if ex in out:
                out[ex] = 0.0
                changed = True
        if not changed:
            return out
        s = sum(out.values())
        if s > 1e-12:
            return {k: (v / s) for k, v in out.items()}
        # 모든 점수가 0이면 neutral=1.0로 복구(보수적)
        out = {k: 0.0 for k in out.keys()}
        out['neutral'] = 1.0
        return out

    def _av_compatibility(self, segment: Dict[str, Any]) -> Dict[str, float]:
        """AV(Valence/Arousal) 근사값을 바탕으로 결핍 클래스(disgust/fear/surprise) 호환성을 계산.

        - arousal, valence ∈ [0,1] 가정(없으면 0.5로 대체)
        - 템플릿은 '넓은' 가우시안 형태로 유연성을 확보
          · surprise: arousal↑(μ=0.75, σ≈0.25), valence 중립(μ=0.5, σ≈0.35)
          · fear:    arousal↑(μ=0.75, σ≈0.25), valence 음(μ=0.25, σ≈0.25)
          · disgust: valence 음(μ=0.25, σ≈0.28), arousal 양봉형(저/고 모두 허용: μ≈0.3, 0.7, σ≈0.22)
        """
        try:
            av = segment.get('av') or segment.get('voice_analysis', {}).get('av')
            if not isinstance(av, dict):
                raise ValueError("No AV in segment")
            a = float(av.get('arousal', 0.5))
            v = float(av.get('valence', 0.5))
        except Exception:
            a, v = 0.5, 0.5

        def gauss(x: float, mu: float, sigma: float) -> float:
            sigma = max(1e-3, sigma)
            return float(np.exp(-((x - mu) / sigma) ** 2))

        # Surprise: 고각성 + 중립 밸런스
        s_ar = gauss(a, 0.75, 0.25)
        s_va = gauss(v, 0.50, 0.35)
        comp_surprise = 0.5 * (s_ar + s_va)

        # Fear: 고각성 + 부정 밸런스
        f_ar = gauss(a, 0.75, 0.25)
        f_va = gauss(v, 0.25, 0.25)
        comp_fear = 0.5 * (f_ar + f_va)

        # Disgust: 부정 밸런스 + (저/고) 양봉형 각성 허용
        d_va = gauss(v, 0.25, 0.28)
        d_ar_low = gauss(a, 0.30, 0.22)
        d_ar_high = gauss(a, 0.70, 0.22)
        d_ar = max(d_ar_low, d_ar_high)  # 둘 중 더 잘 맞는 쪽을 채택
        comp_disgust = 0.5 * (d_ar + d_va)

        # 안정성 차원에서 [0,1] 클램프
        def clamp01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        return {
            'surprise': clamp01(comp_surprise),
            'fear': clamp01(comp_fear),
            'disgust': clamp01(comp_disgust)
        }

    def _setup_memory_management(self):
        """메모리 관리 설정"""
        if torch.cuda.is_available():
            # GPU 메모리 캐시 정리 함수
            torch.cuda.empty_cache()
            # 그래디언트 계산 비활성화
            torch.set_grad_enabled(False)
        
        # 배치 처리를 위한 임계값 설정
        self.max_audio_length = 30  # 최대 30초
        self.max_text_length = 512  # BERT 모델 제한
        self.audio_sampling_rate = 16000

    def process_batch(self, segments: List[Dict[str, Any]], audio_data: np.ndarray, sr: int = 16000):
        """배치 단위 처리로 메모리 효율성 개선"""
        results = []
        
        # 진행률 표시와 함께 배치 처리
        for i in tqdm(range(0, len(segments), self.batch_size), desc="Processing segments"):
            batch = segments[i:i + self.batch_size]
            
            # 배치 데이터 준비
            batch_audio = []
            for seg in batch:
                try:
                    start_idx = int(seg['start'] * sr)
                    end_idx = int(seg['end'] * sr)
                    if start_idx < len(audio_data) and end_idx <= len(audio_data) and start_idx < end_idx:
                        audio_segment = audio_data[start_idx:end_idx]
                        batch_audio.append(audio_segment)
                    else:
                        # 인덱스 범위 초과 시 대체 데이터
                        batch_audio.append(np.zeros(1600, dtype=np.float32))
                except Exception as e:
                    logging.warning(f"Audio segment extraction error: {str(e)}")
                    batch_audio.append(np.zeros(1600, dtype=np.float32))
            batch_text = [seg.get('text', '') for seg in batch]

            # 배치 처리
            batch_results = self._process_segment_batch(batch, batch_audio, batch_text)
            results.extend(batch_results)

            # 메모리 관리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def _process_segment_batch(self, segments, batch_audio, batch_text):
        """배치 단위 세그먼트 처리"""
        results = []
        
        # 텍스트 감정 분석
        text_scores = self._analyze_text_batch(batch_text)
        
        # 오디오 감정 분석
        audio_scores = self._analyze_audio_batch(batch_audio)
        
        # 결과 결합
        for i, segment in enumerate(segments):
            text_score = text_scores[i]
            audio_score = audio_scores[i]
            
            # 최종 감정 결정
            final_emotion = self._combine_predictions(
                text_score,
                audio_score,
                segment
            )
            
            results.append(final_emotion)
            
            # 로그 기록
            self._log_segment_result(segment, final_emotion)
            
        return results

    def _analyze_text_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """텍스트 배치 감정 분석"""
        if not texts:
            return [{"neutral": 1.0}] * len(texts)

        if not self.text_enabled or self.text_model is None or self.text_tokenizer is None:
            return [{} for _ in texts]

        try:
            inputs = self.text_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                # 정확도 우선: FP32 추론 (FP16 autocast 제거)
                outputs = self.text_model(**inputs)
                scores = F.softmax(outputs.logits, dim=-1)

            results = []
            for score in scores:
                # 감정 표준화 추가
                emotion_scores = {}
                for i, emotion in enumerate(self.text_model.config.id2label.values()):
                    # 감정 레이블 소문자 변환 및 매핑
                    standardized_emotion = emotion.lower()
                    if standardized_emotion in ['sadness', 'sad']:
                        standardized_emotion = 'sad'
                    elif standardized_emotion in ['anger', 'angry']:
                        standardized_emotion = 'angry'
                    elif standardized_emotion in ['joy', 'happy', 'positive']:  # joy/positive를 happy로 매핑
                        standardized_emotion = 'happy'
                    elif standardized_emotion in ['other']:
                        standardized_emotion = 'neutral'
                    emotion_scores[standardized_emotion] = score[i].item()
                # 제외 레이블 적용 후 정규화
                cleaned = self._apply_emotion_exclusions(emotion_scores)
                results.append(cleaned)

            return results
        except Exception as e:
            logging.error(f"Text analysis error: {str(e)}")
            return [{"neutral": 1.0}] * len(texts)

    def _analyze_audio_batch(self, audio_segments: List[np.ndarray]) -> List[Dict[str, float]]:
        """오디오 배치 감정 분석"""
        if not audio_segments:
            return [{"neutral": 1.0}] * len(audio_segments)

        try:
            feature_sr = getattr(self, "feature_sampling_rate", 16000)
            base_sr = getattr(self, "audio_sampling_rate", feature_sr)
            max_length_samples = int(feature_sr * self.max_audio_length)
            min_length_samples = max(1, int(feature_sr * 0.1))
            model_type = getattr(self.audio_model.config, "model_type", "").lower()
            requires_fixed_length = model_type == "whisper"

            valid_segments = []
            for segment in audio_segments:
                if not isinstance(segment, np.ndarray) or segment.size == 0:
                    segment = np.zeros(min_length_samples, dtype=np.float32)
                else:
                    if segment.ndim > 1:
                        segment = np.squeeze(segment)
                    segment = np.asarray(segment, dtype=np.float32)

                if base_sr != feature_sr:
                    segment = librosa.resample(segment, orig_sr=base_sr, target_sr=feature_sr)

                if requires_fixed_length:
                    if len(segment) > max_length_samples:
                        segment = segment[:max_length_samples]
                    elif len(segment) < max_length_samples:
                        segment = np.pad(segment, (0, max_length_samples - len(segment)))
                else:
                    if len(segment) < min_length_samples:
                        segment = np.pad(segment, (0, min_length_samples - len(segment)))
                    if len(segment) > max_length_samples:
                        segment = segment[:max_length_samples]

                valid_segments.append(segment)

            if not valid_segments:
                return [{"neutral": 1.0}] * len(audio_segments)

            feature_kwargs = {
                "sampling_rate": feature_sr,
                "return_tensors": "pt",
            }
            if requires_fixed_length:
                feature_kwargs.update({
                    "padding": "max_length",
                    "max_length": max_length_samples,
                    "truncation": True,
                    "return_attention_mask": False,
                })
            else:
                feature_kwargs.update({
                    "padding": True,
                    "return_attention_mask": True,
                })

            features = self.feature_extractor(valid_segments, **feature_kwargs)

            if requires_fixed_length and isinstance(features, dict) and "attention_mask" in features:
                features.pop("attention_mask")

            features = features.to(self.device)

            with torch.no_grad():
                # 정확도 우선: FP32 추론 (FP16 autocast 제거)
                outputs = self.audio_model(**features)
                logits = outputs.logits
                temperature = max(float(self.audio_temperature), 1e-3)
                if temperature != 1.0:
                    logits = logits / temperature

                scores = F.softmax(logits, dim=-1)

                # 과도한 평준화를 방지하면서도 0 확률을 피하기 위해 아주 작은 하한만 적용
                scores = torch.clamp(scores, min=1e-6)
                scores = scores / scores.sum(dim=-1, keepdim=True)

            results = []
            emotion_aliases = {
                'happiness': 'happy',
                'joy': 'happy',
                'excited': 'happy',
                'positive': 'happy',
                'hap': 'happy',
                '기쁨': 'happy',
                '행복': 'happy',
                'anger': 'angry',
                'ang': 'angry',
                '분노': 'angry',
                '화남': 'angry',
                'sadness': 'sad',
                'sad': 'sad',
                '슬픔': 'sad',
                'fearful': 'fear',
                'fear': 'fear',
                '불안': 'fear',
                '공포': 'fear',
                'surprised': 'surprise',
                'surprise': 'surprise',
                '당황': 'surprise',
                '놀람': 'surprise',
                'disgusted': 'disgust',
                'disgust': 'disgust',
                '짜증': 'disgust',
                'neutral': 'neutral',
                'other': 'neutral',
                'neu': 'neutral',
                '중립': 'neutral',
                'calm': 'neutral',
                'bored': 'neutral'
            }

            for score in scores:
                # 원본 레이블을 7개 감정으로 매핑
                raw_emotion_scores = {
                    emotion: score[i].item()
                    for i, emotion in enumerate(self.audio_model.config.id2label.values())
                }
                
                # 표준 7개 감정으로 통합 (기존 매핑 사용)
                emotion_scores = {
                    'happy': 0.0, 'sad': 0.0, 'angry': 0.0, 'fear': 0.0,
                    'surprise': 0.0, 'disgust': 0.0, 'neutral': 0.0
                }
                
                for raw_emotion, raw_score in raw_emotion_scores.items():
                    normalized = raw_emotion.strip().lower()
                    mapped = emotion_aliases.get(raw_emotion, None)
                    if mapped is None:
                        mapped = emotion_aliases.get(normalized, None)
                    if mapped is None:
                        mapped = normalized

                    if mapped in emotion_scores:
                        emotion_scores[mapped] += raw_score
                    else:
                        emotion_scores['neutral'] += raw_score
                
                # === 모델별 후처리 보정 적용 (예: ehcalabres 중립 쏠림 완화) ===
                if self._calibration:
                    ns = float(self._calibration.get("neutral_scale", 1.0))
                    nns = float(self._calibration.get("non_neutral_scale", 1.0))
                    if ns != 1.0 or nns != 1.0:
                        # 중립은 ns 배, 비중립은 nns 배로 스케일 조정
                        for k in list(emotion_scores.keys()):
                            if k == 'neutral':
                                emotion_scores[k] *= ns
                            else:
                                emotion_scores[k] *= nns

                # 정규화
                total = sum(emotion_scores.values())
                if total > 0:
                    emotion_scores = {k: v/total for k, v in emotion_scores.items()}
                # 제외 레이블 적용 후 정규화
                cleaned = self._apply_emotion_exclusions(emotion_scores)
                results.append(cleaned)

            return results

        except Exception as e:
            logging.error(f"Audio analysis error: {str(e)}")
            return [{"neutral": 1.0}] * len(audio_segments)

    def _combine_predictions(
        self,
        text_scores: Dict[str, float],
        audio_scores: Dict[str, float],
        segment: Dict[str, Any]
    ) -> EmotionResult:
        """멀티모달 예측 결과 결합"""
        combined_scores = {}

        # 모달별 상위 감정 및 신뢰도 계산
        sorted_audio = sorted(audio_scores.items(), key=lambda x: x[1], reverse=True)
        sorted_text = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)

        audio_top = sorted_audio[0] if sorted_audio else ("neutral", 0.0)
        audio_second = sorted_audio[1] if len(sorted_audio) > 1 else ("neutral", 0.0)
        text_top = sorted_text[0] if sorted_text else ("neutral", 0.0)

        audio_margin = audio_top[1] - audio_second[1]
        high_audio = audio_top[1] >= self.audio_confidence_threshold and audio_margin >= self.dominance_margin
        high_text = text_top[1] >= self.text_confidence_threshold

        # 동적 가중치 조정
        adjusted_audio_weight = self.audio_weight
        adjusted_text_weight = self.text_weight

        if high_audio and audio_top[0] != "neutral":
            adjusted_audio_weight *= self.audio_confidence_boost
            adjusted_text_weight *= self.text_confidence_decay

        if (not high_audio or audio_top[0] == "neutral") and high_text and text_top[0] != "neutral":
            adjusted_text_weight *= self.text_confidence_boost
            adjusted_audio_weight *= self.audio_confidence_decay

        if audio_top[0] == text_top[0] and audio_top[0] != "neutral" and (high_audio or high_text):
            # 두 모달이 합치면 동일 비율로 소폭 강화
            adjusted_audio_weight *= 1.05
            adjusted_text_weight *= 1.05

        weight_sum = adjusted_audio_weight + adjusted_text_weight
        if weight_sum > 0:
            dynamic_audio_weight = adjusted_audio_weight / weight_sum
            dynamic_text_weight = adjusted_text_weight / weight_sum
        else:
            dynamic_audio_weight = self.audio_weight
            dynamic_text_weight = self.text_weight

        # 텍스트와 오디오 점수 결합
        for emotion in set(text_scores.keys()) | set(audio_scores.keys()):
            text_score = text_scores.get(emotion, 0.0)
            audio_score = audio_scores.get(emotion, 0.0)
            
            # 감정별 가중치 적용
            emotion_weight = self.emotion_weights.get(emotion, 1.0)
            
            combined_scores[emotion] = (
                (text_score * dynamic_text_weight +
                 audio_score * dynamic_audio_weight) * emotion_weight
            )

        # === AV 융합으로 결핍 클래스(disgust/fear/surprise) 보강 ===
        if self.av_enabled:
            av_comp = self._av_compatibility(segment)
            for k in ('disgust', 'fear', 'surprise'):
                c = float(av_comp.get(k, 0.5))
                # 텍스트 강한 주장에 호환성 보너스, 결핍 클래스의 의사-오디오 점수도 소폭 부여
                combined_scores[k] = combined_scores.get(k, 0.0) \
                    + self.av_text_gain * c * float(text_scores.get(k, 0.0)) \
                    + self.av_audio_gain * c

        # === Disgust 게이팅: 텍스트 신호가 충분하고 오디오가 불확실/중립일 때만 통과 ===
        if "disgust" in combined_scores:
            text_disgust = float(text_scores.get("disgust", 0.0)) if isinstance(text_scores, dict) else 0.0
            audio_neutral = float(audio_scores.get("neutral", 0.0)) if isinstance(audio_scores, dict) else 0.0
            # 오디오 불확실성: 1위-2위 격차가 작을 때
            audio_uncertain = (audio_margin < max(0.12, 0.5 * self.dominance_margin))
            # 키워드 부스팅(간단): 역겨/징그러/gross/disgust 등 포함 시 +0.05
            kw = ["역겨", "징그러", "혐오", "gross", "yuck", "ew", "disgust"]
            text_raw = (segment.get("text", "") or "").lower()
            if any(k in text_raw for k in kw):
                text_disgust += 0.05
            # 게이트 조건 (AV 호환성이 높을수록 게이트 완화)
            av_bonus = 0.0
            if self.av_enabled:
                av_bonus = 0.08 * float(self._av_compatibility(segment).get('disgust', 0.5))
            disgust_gate = ((text_disgust + av_bonus) >= 0.65) and (audio_neutral >= 0.60 or audio_uncertain)
            if not disgust_gate:
                # 통과 실패: disgust 억제 → 0으로 두고 재분배는 정규화로 처리
                combined_scores["disgust"] = 0.0

        # 중립 감정이 과도하게 지배하는 현상을 완화
        if 'neutral' in combined_scores:
            suppress_neutral = False
            if high_audio and audio_top[0] != 'neutral':
                suppress_neutral = True
            if high_text and text_top[0] != 'neutral' and audio_top[0] != 'neutral':
                suppress_neutral = True

            if suppress_neutral:
                combined_scores['neutral'] *= self.neutral_suppression

            total_before_floor = sum(combined_scores.values())
            if total_before_floor > 0:
                min_neutral = self.neutral_floor * total_before_floor
                if combined_scores['neutral'] < min_neutral:
                    combined_scores['neutral'] = min_neutral

        total_combined = sum(combined_scores.values())
        if total_combined > 0:
            normalized_combined = {k: v / total_combined for k, v in combined_scores.items()}
        else:
            normalized_combined = combined_scores

        # 최종 감정 선택
        # === Neutral guard: 불확실한 happy/angry는 중립으로 기울임 ===
        if self.ng_enabled:
            # 모달별 1,2위 재사용
            sorted_audio = sorted(audio_scores.items(), key=lambda x: x[1], reverse=True)
            audio_top = sorted_audio[0] if sorted_audio else ("neutral", 0.0)
            audio_second = sorted_audio[1] if len(sorted_audio) > 1 else ("neutral", 0.0)
            audio_margin = audio_top[1] - audio_second[1]
            sorted_text = sorted(text_scores.items(), key=lambda x: x[1], reverse=True) if isinstance(text_scores, dict) else []
            text_top = sorted_text[0] if sorted_text else ("neutral", 0.0)

            for cls in list(self.ng_targets):
                cur = normalized_combined.get(cls, 0.0)
                if cur <= 0:
                    continue
                text_support = float(text_scores.get(cls, 0.0)) if isinstance(text_scores, dict) else 0.0
                # 텍스트가 반대 정서(행복↔분노/슬픔/공포/혐오)로 강하게 주장하면 가중 감쇠
                neg_text = max(
                    float(text_scores.get('angry', 0.0)),
                    float(text_scores.get('sad', 0.0)),
                    float(text_scores.get('fear', 0.0)),
                    float(text_scores.get('disgust', 0.0))
                ) if isinstance(text_scores, dict) else 0.0
                conflict = (cls == 'happy' and neg_text >= 0.40) or (cls == 'angry' and float(text_scores.get('happy', 0.0)) >= 0.40)

                # 오디오/텍스트가 해당 cls를 강하게 주장하는지 여부
                audio_supports = (audio_top[0] == cls)
                text_supports = (text_top[0] == cls)
                audio_strong = audio_supports and (audio_top[1] >= self.ng_min_audio_conf) and (audio_margin >= self.ng_min_audio_margin)
                text_strong = text_supports and (text_top[1] >= max(0.35, self.text_confidence_threshold))
                bimodal_agree = audio_supports and text_supports

                # 가드 조건: 오디오가 강하지 않고 텍스트 지지가 약하거나, 정서 충돌이 크면 감쇠
                # 모달 동의(또는 한 모달 강확신) 시에는 감쇠하지 않음
                if bimodal_agree or audio_strong or text_strong:
                    continue
                if (text_support < self.ng_min_text_support) or conflict:
                    damp = self.ng_damp_factor
                    removed = cur * (1.0 - damp)
                    normalized_combined[cls] = cur * damp
                    # 제거분의 대부분을 neutral로 이동
                    normalized_combined['neutral'] = normalized_combined.get('neutral', 0.0) + removed

            # 재정규화
            sm = sum(max(0.0, v) for v in normalized_combined.values())
            if sm > 0:
                normalized_combined = {k: max(0.0, v) / sm for k, v in normalized_combined.items()}

        best_emotion = max(normalized_combined.items(), key=lambda x: x[1])
        
        # 상위 2개 감정 선택 (로그용)
        sorted_emotions = sorted(normalized_combined.items(), key=lambda x: x[1], reverse=True)[:2]

        result = EmotionResult(
            emotion=best_emotion[0],
            confidence=best_emotion[1],
            features=self._extract_audio_features(segment),
            text_score=text_scores.get(best_emotion[0], 0.0),
            audio_score=audio_scores.get(best_emotion[0], 0.0),
            text_distribution={k: float(v) for k, v in text_scores.items()},
            audio_distribution={k: float(v) for k, v in audio_scores.items()},
            combined_distribution={k: float(v) for k, v in normalized_combined.items()}
        )
        
        # 가독성 좋은 로그 출력
        self._log_segment_summary(segment, result, text_scores, audio_scores, combined_scores, sorted_emotions)
        
        return result

    def _extract_audio_features(self, segment: Dict[str, Any]) -> Dict[str, float]:
        """세그먼트별 오디오 특성 추출"""
        features = {}
        try:
            audio_segment = segment.get('audio', None)
            if audio_segment is not None:
                features['rms_energy'] = np.sqrt(np.mean(audio_segment**2))
                features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio_segment).mean()
                # 추가 특성은 필요에 따라 확장
        except Exception as e:
            logging.warning(f"Feature extraction warning: {str(e)}")
        return features

    def _log_segment_summary(self, segment: Dict[str, Any], result: EmotionResult, 
                                   text_scores: Dict[str, float], audio_scores: Dict[str, float],
                                   combined_scores: Dict[str, float], sorted_emotions: List):
        text = segment.get('text', '').strip()
        timestamp = segment.get('start', 0)
        
        if not text:  # 빈 텍스트 건너뛰기
            return
            
        # 각 분류별 독립적인 상위 2개 감정 계산
        # Text 기반 상위 2개
        sorted_text = sorted(text_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_text = ' | '.join([f"{emotion}: {score:.3f}" for emotion, score in sorted_text])
        
        # Audio 기반 상위 2개  
        sorted_audio = sorted(audio_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top2_audio = ' | '.join([f"{emotion}: {score:.3f}" for emotion, score in sorted_audio])
        
        # Combined 상위 2개 (기존 sorted_emotions 사용)
        top2_combined = ' | '.join([f"{emotion}: {score:.3f}" for emotion, score in sorted_emotions])
        
        # 3줄을 한 묶음으로 출력 (박스 형태)
        print(f"\n┌─ [{timestamp:.1f}s] {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"├─ Text:     {top2_text}")
        print(f"├─ Audio:    {top2_audio}")
        print(f"└─ Combined: {top2_combined}")
        
    def _log_segment_result(self, segment: Dict[str, Any], result: EmotionResult):
        """기존 JSON 로깅 (파일용)"""
        log_entry = {
            'timestamp': segment.get('start', 0),
            'text': segment.get('text', ''),
            'emotion': result.emotion,
            'confidence': result.confidence,
            'text_score': result.text_score,
            'audio_score': result.audio_score
        }
        logging.info(json.dumps(log_entry, ensure_ascii=False))

    def save_results(self, results: List[EmotionResult], filepath: str):
        """분석 결과 저장"""
        output = [{
            'emotion': r.emotion,
            'confidence': r.confidence,
            'features': r.features,
            'text_score': r.text_score,
            'audio_score': r.audio_score,
            'text_distribution': r.text_distribution,
            'audio_distribution': r.audio_distribution,
            'combined_distribution': r.combined_distribution
        } for r in results]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    def get_emotion_color(self, emotion: str) -> str:
        """감정별 색상 코드 반환"""
        try:
            config_data = self.db_manager.load_config()
            emotion_colors = config_data.get('hex_colors', {}).get('emotion_colors', {})
            default_color = config_data.get('hex_colors', {}).get('default_color', '#FFFFFF')

            resolved_color = emotion_colors.get(emotion, default_color)
            
            return resolved_color 
            
        except Exception as e:
            logging.error(f"Failed to resolve emotion color from DB: {str(e)}")
            # DB 접근 실패 시 기본값 반환
            return '#FFFFFF'

    def classify_emotions(self, segments, full_audio):
        """기존 코드와의 호환성을 위한 메서드"""
        try:
            # 진행 상황 초기 출력
            print("\n감정 분석 중...")
            print(f"감정 분류 진행: 0/{len(segments)}")

            # 배치 처리 수행
            results = self.process_batch(segments, full_audio)

            # 감정 분석 결과를 세그먼트에 반영
            for segment, result in zip(segments, results):
                segment['emotion'] = result.emotion
                segment['confidence'] = result.confidence
                segment['emotion_color'] = self.get_emotion_color(result.emotion)
                segment['features'] = result.features
                segment['text_score'] = result.text_score
                segment['audio_score'] = result.audio_score
                segment['text_scores'] = result.text_distribution
                segment['audio_scores'] = result.audio_distribution
                segment['combined_scores'] = result.combined_distribution

            # 시간 스무딩: 고립된 happy/angry 스파이크를 중립으로 완화
            ts_cfg = config.get('emotions', 'temporal_smoothing', default={}) or {}
            if bool(ts_cfg.get('enabled', True)) and len(segments) >= 3:
                window = int(ts_cfg.get('window', 3))
                min_conf = float(ts_cfg.get('min_conf', 0.70))
                targets = set(ts_cfg.get('target_classes', ['happy', 'angry']))
                half = max(1, window // 2)
                for i in range(len(segments)):
                    seg = segments[i]
                    emo = seg.get('emotion', 'neutral')
                    if emo not in targets or float(seg.get('confidence', 0.0)) >= min_conf:
                        continue
                    # 모달 강지지/동의가 있으면 스무딩 제외
                    ts = seg.get('text_scores') or {}
                    as_ = seg.get('audio_scores') or {}
                    if isinstance(ts, dict) and ts.get(emo, 0.0) >= 0.40:
                        continue
                    if isinstance(as_, dict) and as_.get(emo, 0.0) >= 0.60:
                        continue
                    # 양옆(같은 화자 우선) 확인
                    left = segments[i-1] if i-1 >= 0 else None
                    right = segments[i+1] if i+1 < len(segments) else None
                    neighbors = [s for s in [left, right] if s is not None]
                    if not neighbors:
                        continue
                    same_speaker_neighbors = [s for s in neighbors if s.get('speaker', 'Unknown') == seg.get('speaker', 'Unknown')]
                    nb = same_speaker_neighbors if same_speaker_neighbors else neighbors
                    # 양옆 모두 neutral이거나 상이한 감정이며 자신보다 신뢰도가 높으면 중립으로 완화
                    if all(n.get('emotion', 'neutral') == 'neutral' for n in nb) or all(float(n.get('confidence', 0.0)) >= float(seg.get('confidence', 0.0)) for n in nb):
                        seg['emotion'] = 'neutral'
                        seg['emotion_color'] = self.get_emotion_color('neutral')
                        seg['confidence'] = float(max(0.5, seg.get('confidence', 0.0)))
                        # 분포도도 중립에 더 싱크
                        cs = seg.get('combined_scores') or {}
                        neu_added = 0.15
                        cs['neutral'] = float(cs.get('neutral', 0.0) + neu_added)
                        ssum = sum(max(0.0, v) for v in cs.values())
                        if ssum > 0:
                            seg['combined_scores'] = {k: float(max(0.0, v) / ssum) for k, v in cs.items()}

            # 최종 진행 상황 출력
            print(f"감정 분류 진행: {len(segments)}/{len(segments)}")
            print("감정 분류 완료")

            return segments

        except Exception as e:
            logging.error(f"감정 분류 중 오류 발생: {str(e)}")
            return segments


    def classify_audio_only(self, segments: List[Dict[str, Any]], full_audio: np.ndarray) -> List[Dict[str, Any]]:
        """
        [NEW FUNCTION]
        오디오 모델만 사용하여 감정을 분류하고, 텍스트 모델 및 앙상블 로직을 완전히 무시합니다.
        (기존 SubtitleGenerator의 'en' 로직과 동일한 역할을 수행합니다.)
        """
        print("\n🎧 오디오 전용 감정 분석 시작...")

        try:
            emotion_results = self.process_batch(segments, full_audio)

            for i, segment in enumerate(segments):
                result = emotion_results[i]
            
                segment['emotion'] = result.emotion
                segment['confidence'] = result.confidence

            print(f"✅ 오디오 전용 감정 분석 완료. 총 {len(segments)}개 세그먼트 처리.")

            return segments

        except Exception as e:
            logging.error(f"Audio-Only Classification Error: {str(e)}")
            
            # 실패 시 원본 세그먼트 반환
            return segments