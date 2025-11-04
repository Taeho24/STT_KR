# OpenSMILE 기반 감정 인식 통합 가이드

## 개요

OpenSMILE은 음성에서 수천 개의 특징을 추출하는 오픈소스 도구입니다.  
사전 학습된 감정 분류 모델과 결합하여 사용할 수 있습니다.

## 설치

```bash
pip install opensmile
pip install scikit-learn  # ML 모델용
```

## 사용 가능한 모델

### 1. ComParE Feature Set + SVM

```python
import opensmile
import numpy as np
import pickle

# OpenSMILE 초기화
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# 오디오 특징 추출
def extract_features(audio_path):
    features = smile.process_file(audio_path)
    return features.values[0]  # (6373,) 차원 벡터

# 사전 학습된 SVM 로드 (직접 학습 필요)
with open("compare_svm_model.pkl", "rb") as f:
    classifier = pickle.load(f)

# 예측
features = extract_features("audio.wav")
emotion = classifier.predict([features])[0]
probs = classifier.predict_proba([features])[0]
```

### 2. eGeMAPS Feature Set

더 가볍고 빠른 특징 세트:

```python
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
```

## 프로젝트 통합

`emotion_classifier.py`에 OpenSMILE 어댑터 추가:

```python
class OpenSMILEAdapter:
    def __init__(self, model_path, feature_set='ComParE_2016'):
        import opensmile
        import pickle
        
        self.smile = opensmile.Smile(
            feature_set=getattr(opensmile.FeatureSet, feature_set),
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        # 사전 학습된 분류기 로드
        with open(model_path, 'rb') as f:
            self.classifier = pickle.load(f)
        
        self.label_map = {
            0: 'neutral',
            1: 'happy',
            2: 'sad',
            3: 'angry',
            4: 'fear',
            5: 'surprise',
            6: 'disgust'
        }
    
    def predict(self, audio_segment, sr=16000):
        import tempfile
        import soundfile as sf
        
        # 임시 파일로 저장 (OpenSMILE은 파일 입력 필요)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_segment, sr)
            features = self.smile.process_file(tmp.name).values[0]
        
        # 예측
        probs = self.classifier.predict_proba([features])[0]
        
        # 감정 분포 반환
        emotions = {}
        for idx, prob in enumerate(probs):
            emotions[self.label_map[idx]] = float(prob)
        
        return emotions
```

## 사전 학습 모델 다운로드

공개된 ComParE 모델은 없으므로, 다음 중 하나를 선택:

### 옵션 1: 직접 학습
- RAVDESS, IEMOCAP 등 공개 데이터셋 사용
- OpenSMILE로 특징 추출 후 SVM/Random Forest 학습

### 옵션 2: 공개 모델 활용
- **auDeep** 프로젝트: https://github.com/auDeep/auDeep
  - OpenSMILE 기반 딥러닝 모델
  - 사전 학습 가중치 제공

### 옵션 3: EmotionRecognition-RAVDESS
- GitHub: https://github.com/marcogdepinto/emotion-recognition-using-voice
- ComParE + Random Forest 모델 포함
- 7개 감정 지원

## 성능 특징

**장점:**
- 전통적 ML 방식으로 가볍고 빠름
- 설명 가능한 특징 (음높이, 에너지, MFCC 등)
- CPU에서도 실시간 처리 가능

**단점:**
- 딥러닝 모델보다 정확도 낮음
- 특징 추출에 시간 소요
- 사전 학습 모델 구하기 어려움

## 실행 예시

```bash
# 1. auDeep 설치
pip install audeep

# 2. 평가 실행
python tools/model_evaluator.py \
    --video assets/simpson.mp4 \
    --labels labelled_simpson.jsonl \
    --disable-text \
    --audio-models opensmile/compare-svm \
    --model-type opensmile \
    --device cpu
```
