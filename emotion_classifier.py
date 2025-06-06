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
from typing import List, Dict, Any
from dataclasses import dataclass
import torch.nn.functional as F
from tqdm import tqdm
from config import config

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

class EmotionClassifier:
    """감정 분류 모델과 앙상블 처리를 담당하는 클래스"""
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 8,
        cache_dir: str = ".cache"
    ):
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # 텍스트 감정 분석 모델 (RoBERTa 기반)
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base"
        ).to(device)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "j-hartmann/emotion-english-distilroberta-base"
        )

        # 오디오 감정 분석 모델 (Wav2Vec2 기반) 
        self.audio_model = AutoModelForAudioClassification.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        ).to(device)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )

        # config에서 모든 설정 로드
        self.emotion_mapping = config.get('emotions', 'mapping')
        self.weights = config.get('emotions', 'weights')
        self.emotion_weights = config.get('emotions', 'emotion_weights')
        self.emotion_colors = config.get('colors', 'emotion_colors')
        self.default_color = config.get('colors', 'default_color')

        # 가중치 설정
        self.audio_weight = self.weights['audio']
        self.text_weight = self.weights['text']

        self._setup_memory_management()
        logging.info("Emotion classifier initialized successfully")

        # 텍스트/오디오 모델 레이블 출력
        print("Text model labels:", self.text_model.config.id2label)
        print("Audio model labels:", self.audio_model.config.id2label)

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

    def process_batch(self, segments: List[Dict[str, Any]], audio_data: np.ndarray, sr: int = 16000):
        """배치 단위 처리로 메모리 효율성 개선"""
        results = []
        
        # 진행률 표시와 함께 배치 처리
        for i in tqdm(range(0, len(segments), self.batch_size), desc="Processing segments"):
            batch = segments[i:i + self.batch_size]
            
            # 배치 데이터 준비
            batch_audio = [
                audio_data[int(seg['start'] * sr):int(seg['end'] * sr)]
                for seg in batch
            ]
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

        try:
            inputs = self.text_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
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
                    elif standardized_emotion in ['joy', 'happy']:  # joy를 happy로 매핑
                        standardized_emotion = 'happy'
                    emotion_scores[standardized_emotion] = score[i].item()
                results.append(emotion_scores)

            return results

        except Exception as e:
            logging.error(f"Text analysis error: {str(e)}")
            return [{"neutral": 1.0}] * len(texts)

    def _analyze_audio_batch(self, audio_segments: List[np.ndarray]) -> List[Dict[str, float]]:
        """오디오 배치 감정 분석"""
        if not audio_segments:
            return [{"neutral": 1.0}] * len(audio_segments)

        try:
            # 1. 오디오 특성 추출 및 분석
            features = self.feature_extractor(
                audio_segments,
                sampling_rate=16000,
                padding=True,
                truncation=True,  # 추가: 긴 오디오 잘라내기
                max_length=16000 * 10,  # 추가: 최대 10초로 제한
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.audio_model(**features)
                scores = F.softmax(outputs.logits, dim=-1)
                
                # 점수 정규화 추가
                min_score = 0.1  # 최소 점수를 0.1로 설정
                scores = torch.clamp(scores, min=min_score)
                scores = scores / scores.sum(dim=-1, keepdim=True)  # 재정규화

            results = []
            for score in scores:
                emotion_scores = {
                    emotion: score[i].item()
                    for i, emotion in enumerate(self.audio_model.config.id2label.values())
                }
                results.append(emotion_scores)

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

        # 텍스트와 오디오 점수 결합
        for emotion in set(text_scores.keys()) | set(audio_scores.keys()):
            text_score = text_scores.get(emotion, 0.0)
            audio_score = audio_scores.get(emotion, 0.1)  # 최소값 0.1 설정
            
            # 감정별 가중치 적용
            emotion_weight = self.emotion_weights.get(emotion, 1.0)
            
            combined_scores[emotion] = (
                (text_score * self.text_weight +
                 audio_score * self.audio_weight) * emotion_weight
            )

        # 최종 감정 선택
        best_emotion = max(combined_scores.items(), key=lambda x: x[1])

        # 디버그용 로깅 추가
        print(f"Debug - Text scores: {text_scores}")
        print(f"Debug - Audio scores: {audio_scores}")
        print(f"Debug - Combined scores: {combined_scores}")
        print(f"Debug - Best emotion: {best_emotion[0]} ({best_emotion[1]})")

        return EmotionResult(
            emotion=best_emotion[0],
            confidence=best_emotion[1],
            features=self._extract_audio_features(segment),
            text_score=text_scores.get(best_emotion[0], 0.0),
            audio_score=audio_scores.get(best_emotion[0], 0.1)  # 최소값 0.1 설정
        )

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

    def _log_segment_result(self, segment: Dict[str, Any], result: EmotionResult):
        """세그먼트 분석 결과 로깅"""
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
            'audio_score': r.audio_score
        } for r in results]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_emotion_color(emotion: str) -> str:
        """감정별 색상 코드 반환"""
        # 디버그용 로깅 추가
        emotion_colors = config.get('colors', 'emotion_colors')
        default_color = config.get('colors', 'default_color', '&HFFFFFF')
        resolved_color = emotion_colors.get(emotion, default_color)
        print(f"Debug - Emotion color resolution: {emotion} -> {resolved_color}")
        return resolved_color  # 기본값은 흰색

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

            # 최종 진행 상황 출력
            print(f"감정 분류 진행: {len(segments)}/{len(segments)}")
            print("감정 분류 완료")

            return segments

        except Exception as e:
            logging.error(f"감정 분류 중 오류 발생: {str(e)}")
            return segments
