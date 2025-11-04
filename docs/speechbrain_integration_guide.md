# SpeechBrain 모델 통합 가이드

## 설치

```bash
pip install speechbrain
```

## 사용 방법

### 1. IEMOCAP 감정 인식 모델

```python
from speechbrain.inference.interfaces import foreign_class

# 모델 로드
classifier = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

# 추론
import torchaudio
signal, fs = torchaudio.load("audio.wav")
out_prob, score, index, text_lab = classifier.classify_batch(signal)
```

### 2. EmotionClassifier에 통합

`emotion_classifier.py`에 SpeechBrain 어댑터 추가:

```python
class SpeechBrainAdapter:
    def __init__(self, model_id, device="cuda"):
        from speechbrain.inference.interfaces import foreign_class
        
        self.classifier = foreign_class(
            source=model_id,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device}
        )
        
        # 감정 레이블 매핑
        self.label_map = {
            'neu': 'neutral',
            'hap': 'happy',
            'ang': 'angry',
            'sad': 'sad'
        }
    
    def predict(self, audio_segment):
        import torch
        
        # 오디오 텐서 준비
        audio_tensor = torch.tensor(audio_segment).unsqueeze(0)
        
        # 추론
        out_prob, score, index, text_lab = self.classifier.classify_batch(audio_tensor)
        
        # 확률 분포 추출
        probs = torch.softmax(out_prob[0], dim=0).cpu().numpy()
        
        # 레이블 매핑
        emotions = {}
        for i, label in enumerate(self.classifier.hparams.label_encoder.ind2lab):
            mapped = self.label_map.get(label, label)
            emotions[mapped] = float(probs[i])
        
        return emotions
```

### 3. 프로젝트에 통합

`emotion_classifier.py`의 `__init__` 수정:

```python
def __init__(self, ..., use_speechbrain=False):
    if use_speechbrain:
        self.audio_classifier = SpeechBrainAdapter(
            model_id="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            device=self.device
        )
    else:
        # 기존 HuggingFace 로딩
        ...
```

## 주의사항

1. **메모리**: SpeechBrain 모델은 별도 메모리 할당
2. **배치 처리**: SpeechBrain의 배치 인터페이스 활용
3. **레이블 불일치**: 모델별 감정 레이블이 다르므로 매핑 필수

## 평가 실행

```bash
python tools/model_evaluator.py \
    --video assets/simpson.mp4 \
    --labels labelled_simpson.jsonl \
    --disable-text \
    --audio-models speechbrain/emotion-recognition-wav2vec2-IEMOCAP \
    --use-speechbrain \
    --device auto
```

## 성능 예상

- **장점**: IEMOCAP 데이터셋으로 학습, 4개 감정 지원
- **단점**: 7개 감정 체계와 부분 호환, fear/surprise/disgust 누락
