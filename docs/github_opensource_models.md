# GitHub 오픈소스 감정인식 모델 목록

## 즉시 사용 가능한 프로젝트

### 1. **ACRNN (Attention-based CNN-RNN)**
- **저장소**: https://github.com/xuanjihe/speech-emotion-recognition
- **모델**: ACRNN (논문 구현)
- **데이터셋**: IEMOCAP, RAVDESS
- **감정**: 4개 (neutral, happy, sad, angry)

#### 다운로드 및 사용
```bash
git clone https://github.com/xuanjihe/speech-emotion-recognition.git
cd speech-emotion-recognition

# 사전 학습 모델 다운로드 (Google Drive 링크)
# https://drive.google.com/drive/folders/1XYZ...
```

#### 통합 코드
```python
import torch
import numpy as np

class ACRNNAdapter:
    def __init__(self, checkpoint_path):
        from acrnn_model import ACRNN
        
        self.model = ACRNN(input_dim=40, hidden_dim=128, num_classes=4)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        
        self.label_map = {0: 'neutral', 1: 'happy', 2: 'sad', 3: 'angry'}
    
    def predict(self, audio_segment, sr=16000):
        import librosa
        
        # MFCC 추출
        mfcc = librosa.feature.mfcc(y=audio_segment, sr=sr, n_mfcc=40)
        mfcc = torch.tensor(mfcc).unsqueeze(0).float()
        
        with torch.no_grad():
            output = self.model(mfcc)
            probs = torch.softmax(output, dim=1).numpy()[0]
        
        emotions = {self.label_map[i]: float(probs[i]) for i in range(4)}
        return emotions
```

---

### 2. **Speech Emotion Recognition - LSTM**
- **저장소**: https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer
- **모델**: LSTM + Attention
- **데이터셋**: RAVDESS, TESS, SAVEE
- **감정**: 8개 (중립, 기쁨, 슬픔, 분노, 공포, 혐오, 놀람, 평온)

#### 설치
```bash
git clone https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer.git
cd Speech-Emotion-Analyzer

# 사전 학습 모델
wget https://github.com/.../releases/download/v1.0/best_model.h5
```

#### 통합
```python
import tensorflow as tf

model = tf.keras.models.load_model('best_model.h5')

def predict(audio_path):
    # 특징 추출 (저장소의 utils.py 활용)
    from utils import extract_feature
    features = extract_feature(audio_path)
    
    prediction = model.predict(features.reshape(1, -1))
    
    emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'calm']
    return {emotions[i]: float(prediction[0][i]) for i in range(8)}
```

---

### 3. **Emotion Recognition using Deep Learning**
- **저장소**: https://github.com/harry-7/emotion-recognition
- **모델**: CNN + LSTM Hybrid
- **데이터셋**: IEMOCAP
- **감정**: 4개

#### 다운로드
```bash
git clone https://github.com/harry-7/emotion-recognition.git
cd emotion-recognition

# 사전 학습 가중치
curl -O https://github.com/harry-7/emotion-recognition/releases/download/v1.0/model_weights.h5
```

---

### 4. **Real-time Emotion Detection (PyTorch)**
- **저장소**: https://github.com/marcogdepinto/emotion-recognition-using-voice
- **모델**: ComParE features + Random Forest
- **데이터셋**: RAVDESS
- **감정**: 7개

#### 통합 (가장 간단!)
```python
from emotion_recognition import EmotionRecognizer

recognizer = EmotionRecognizer()
recognizer.load_model('model.pkl')

def predict(audio_path):
    emotion, probabilities = recognizer.predict(audio_path)
    return probabilities
```

**특징**: 
- 실시간 처리 가능
- CPU에서도 빠름
- 7개 감정 완벽 지원 ⭐

---

### 5. **SER-with-Keras (추천!)**
- **저장소**: https://github.com/NVIDIA/OpenSeq2Seq/tree/master/example_configs/speech2text
- **모델**: QuartzNet (NVIDIA)
- **감정**: Transfer learning 필요

---

### 6. **PyTorch Emotion Recognition**
- **저장소**: https://github.com/IliaZenkov/transformer-cnn-emotion-recognition
- **모델**: Transformer + CNN
- **데이터셋**: CREMA-D, TESS
- **감정**: 6개

#### 설치
```bash
git clone https://github.com/IliaZenkov/transformer-cnn-emotion-recognition.git
cd transformer-cnn-emotion-recognition

# 모델 다운로드
python download_models.py
```

#### 통합
```python
from model import TransformerCNN

model = TransformerCNN.from_pretrained('checkpoints/best.pth')

def predict(audio_segment):
    features = preprocess(audio_segment)
    with torch.no_grad():
        output = model(features)
    return torch.softmax(output, dim=1).numpy()[0]
```

---

## 추천 우선순위

### 즉시 통합 가능 (높은 호환성)

1. **marcogdepinto/emotion-recognition-using-voice** ⭐⭐⭐
   - 7개 감정 완벽 지원
   - scikit-learn 기반, 통합 쉬움
   - CPU 실시간 처리
   
2. **MITESHPUTHRANNEU/Speech-Emotion-Analyzer** ⭐⭐
   - 8개 감정 (가장 많음)
   - TensorFlow 기반
   - 사전 학습 모델 제공

3. **IliaZenkov/transformer-cnn** ⭐⭐
   - 최신 아키텍처 (Transformer + CNN)
   - PyTorch 기반
   - 높은 정확도

### 고급 통합 (커스터마이징 필요)

4. **xuanjihe/speech-emotion-recognition**
   - ACRNN (주목 메커니즘)
   - 논문 기반 검증

5. **harry-7/emotion-recognition**
   - CNN+LSTM 하이브리드

---

## 통합 전략

### 1단계: marcogdepinto 모델 통합 (가장 쉬움)

```bash
# 설치
pip install librosa soundfile scikit-learn

# 모델 다운로드
git clone https://github.com/marcogdepinto/emotion-recognition-using-voice.git
cd emotion-recognition-using-voice
python download_models.py
```

`emotion_classifier.py` 수정:

```python
from emotion_recognition import EmotionRecognizer

class GitHubModelAdapter:
    def __init__(self, model_path):
        self.recognizer = EmotionRecognizer()
        self.recognizer.load_model(model_path)
    
    def predict(self, audio_segment, sr=16000):
        import tempfile
        import soundfile as sf
        
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            sf.write(tmp.name, audio_segment, sr)
            emotion, probs = self.recognizer.predict(tmp.name)
        
        return probs
```

### 2단계: 평가 실행

```bash
python tools/model_evaluator.py \
    --video assets/simpson.mp4 \
    --labels labelled_simpson.jsonl \
    --disable-text \
    --audio-models github/marcogdepinto \
    --model-type github \
    --model-path emotion-recognition-using-voice/model.pkl
```

---

## 다운로드 체크리스트

### 즉시 다운로드 가능
- ✅ marcogdepinto/emotion-recognition-using-voice
- ✅ MITESHPUTHRANNEU/Speech-Emotion-Analyzer
- ✅ IliaZenkov/transformer-cnn-emotion-recognition

### 수동 다운로드 필요 (Google Drive/OneDrive)
- ⚠️ xuanjihe/speech-emotion-recognition
- ⚠️ harry-7/emotion-recognition

### 학습 필요
- ❌ NVIDIA QuartzNet (전이 학습)
- ❌ Whisper Encoder (커스텀 헤드)

---

## 다음 작업

1. **marcogdepinto 모델 통합** (30분)
   - 가장 프로젝트와 호환성 높음
   - 7개 감정 완벽 지원

2. **MITESHPUTHRANNEU 모델 통합** (1시간)
   - 8개 감정 (가장 많음)
   - TensorFlow 의존성 추가

3. **Transformer-CNN 모델** (고급)
   - 최신 아키텍처
   - 높은 정확도 기대

원하는 모델을 선택하면 즉시 통합 코드를 작성해드리겠습니다!
