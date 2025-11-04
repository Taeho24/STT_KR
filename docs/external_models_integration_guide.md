# TensorFlow Hub 및 Kaggle 모델 통합 가이드

## 1. TensorFlow Hub - YAMNet (음성 분류)

YAMNet은 Google의 사전 학습 오디오 분류 모델입니다.  
직접적인 감정 분류는 아니지만, 전이 학습으로 활용 가능합니다.

### 설치
```bash
pip install tensorflow tensorflow-hub
```

### 사용 방법

```python
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

# YAMNet 로드
model = hub.load('https://tfhub.dev/google/yamnet/1')

def predict_emotion(audio_segment, sr=16000):
    # YAMNet은 16kHz 요구
    scores, embeddings, spectrogram = model(audio_segment)
    
    # embeddings를 감정 분류기에 입력 (전이 학습 필요)
    # 또는 AudioSet 클래스 중 감정 관련 항목 필터링
    
    emotion_classes = {
        'Speech': 'neutral',
        'Laughter': 'happy',
        'Crying, sobbing': 'sad',
        'Screaming': 'fear',
        'Shout': 'angry'
    }
    
    # AudioSet 레이블에서 감정 매핑
    class_names = model.class_names.numpy()
    top_indices = np.argsort(scores.numpy()[0])[-5:]
    
    emotions = {}
    for idx in top_indices:
        label = class_names[idx].decode('utf-8')
        if label in emotion_classes:
            emotions[emotion_classes[label]] = float(scores.numpy()[0][idx])
    
    return emotions
```

### 한계
- 직접 감정 분류 불가 (AudioSet은 일반 음향 분류)
- 전이 학습 필요

---

## 2. Kaggle 공개 모델

### a) RAVDESS 기반 CNN 모델

**출처**: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio

```python
import tensorflow as tf
import librosa
import numpy as np

# Kaggle에서 다운로드한 모델
model = tf.keras.models.load_model('ravdess_cnn_model.h5')

def extract_mfcc(audio, sr=16000, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

def predict_emotion(audio_segment, sr=16000):
    features = extract_mfcc(audio_segment, sr)
    features = features.reshape(1, -1)
    
    prediction = model.predict(features)
    
    emotions = {
        'neutral': float(prediction[0][0]),
        'happy': float(prediction[0][1]),
        'sad': float(prediction[0][2]),
        'angry': float(prediction[0][3]),
        'fear': float(prediction[0][4]),
        'disgust': float(prediction[0][5]),
        'surprise': float(prediction[0][6])
    }
    
    return emotions
```

### b) TESS + CREMA-D 앙상블

**출처**: https://www.kaggle.com/code/shivamburnwal/speech-emotion-recognition

```bash
# 모델 다운로드
kaggle kernels output shivamburnwal/speech-emotion-recognition -p models/
```

---

## 3. PyTorch Hub - Emotion2Vec (Alibaba)

Alibaba의 최신 음성 감정 표현 모델 (2024년 공개)

### 설치
```bash
pip install emotion2vec
```

### 사용 방법

```python
from emotion2vec import Emotion2Vec

# 모델 로드
model = Emotion2Vec.from_pretrained('emotion2vec_base_finetuned')

def predict_emotion(audio_path):
    # 감정 임베딩 추출
    embedding = model.extract_embedding(audio_path)
    
    # 감정 예측
    emotion_probs = model.predict(audio_path)
    
    return {
        'neutral': emotion_probs[0],
        'happy': emotion_probs[1],
        'sad': emotion_probs[2],
        'angry': emotion_probs[3],
        'fear': emotion_probs[4],
        'surprise': emotion_probs[5],
        'disgust': emotion_probs[6]
    }
```

**특징:**
- 2024년 최신 모델
- 중국어/영어 모두 지원
- SER 벤치마크 SOTA 성능

**설치 가능 여부**: 공개 예정 (2024 Q4)

---

## 4. Whisper Audio Understanding (OpenAI - 비공식)

Whisper 모델의 인코더를 감정 분류에 활용

```python
import whisper
import torch

# Whisper 로드
model = whisper.load_model("base")

def extract_whisper_features(audio):
    # Whisper 인코더로 특징 추출
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    
    with torch.no_grad():
        features = model.encoder(mel.unsqueeze(0))
    
    return features

# 감정 분류 헤드 학습 필요
emotion_head = torch.nn.Linear(512, 7)  # 7개 감정
```

---

## 프로젝트 통합 방법

`emotion_classifier.py`에 범용 어댑터 추가:

```python
class ExternalModelAdapter:
    def __init__(self, model_type, model_path, device="cuda"):
        self.model_type = model_type
        self.device = device
        
        if model_type == "tensorflow":
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
        elif model_type == "speechbrain":
            from speechbrain.inference.interfaces import foreign_class
            self.model = foreign_class(source=model_path)
        elif model_type == "opensmile":
            import pickle
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
    
    def predict(self, audio_segment):
        if self.model_type == "tensorflow":
            return self._predict_tensorflow(audio_segment)
        elif self.model_type == "speechbrain":
            return self._predict_speechbrain(audio_segment)
        elif self.model_type == "opensmile":
            return self._predict_opensmile(audio_segment)
```

## 다운로드 링크

1. **RAVDESS CNN**: https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio
2. **TESS Emotion**: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
3. **EmoDB (독일어)**: http://emodb.bilderbar.info/download/
4. **CREMA-D**: https://github.com/CheyneyComputerScience/CREMA-D

## 권장 사항

**즉시 사용 가능:**
- YAMNet (전이 학습 필요)
- Kaggle RAVDESS 모델 (다운로드 후 사용)

**고급 사용:**
- Emotion2Vec (최신, 공개 대기 중)
- Whisper 인코더 (직접 학습 필요)

**전통적 방법:**
- OpenSMILE + SVM/RF
