# 이 프로젝트에 적합한 모델 선정 기준

## 프로젝트 요구사항 분석

### 핵심 기능
- **입력**: 영상 파일 (한국어/영어 음성)
- **처리**: WhisperX STT → 감정 분류 → 스타일링된 ASS 자막 생성
- **출력**: 감정별 색상/폰트 적용된 자막 파일 (.ass/.srt)

### 사용 시나리오
- 영화, 드라마, 유튜브 영상에 감정 표현 자막 추가
- 배치 처리 (실시간 아님)
- 긴 영상도 처리 가능해야 함 (메모리 효율성)

---

## ✅ 필수 조건 (Must-Have)

### 1. **7개 감정 클래스 지원**
프로젝트의 `config.py`에 정의된 감정:
```python
'neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust'
```

**제외 기준**:
- ❌ 4개 이하 감정 (angry, happy, sad, neutral만 지원)
- ❌ 차원 기반 출력 (arousal/valence/dominance)
- ❌ 바이너리 분류 (positive/negative)

**허용**:
- ✅ 7개 정확히 일치
- ✅ 8개 이상 (매핑 가능: calm → neutral)

---

### 2. **오디오 입력 지원**
**현재 파이프라인**:
```python
audio_segment = librosa.load(audio_path, sr=16000)
emotion = classifier.predict(audio_segment)
```

**제외 기준**:
- ❌ 텍스트 전용 모델 (DistilRoBERTa, BERT, GPT)
- ❌ 이미지 입력 (얼굴 표정 인식)
- ❌ 멀티모달 필수 (오디오+비디오 동시 필요)

**허용**:
- ✅ 순수 오디오 모델 (Wav2Vec2, HuBERT, WavLM)
- ✅ 특징 추출 + 분류기 (OpenSMILE + SVM)
- ✅ 멀티모달 선택적 (오디오만으로도 작동)

---

### 3. **다국어 지원 (한국어 + 영어)**
**사용자 요구사항**:
> "영어와 한국어 모두 입력 가능하게 할거고"

**우선순위**:
1. 🥇 **한국어 + 영어 동시 지원** (XLS-R 기반)
2. 🥈 **영어 전용 (한국어 전이 가능)** (대규모 데이터셋 학습)
3. 🥉 **한국어 전용** (jungjongho 모델)

**제외 기준**:
- ❌ 특정 언어 전용 (그리스어, 독일어만)
- ❌ 영어에서 한국어 전이 성능 0에 가까움

---

### 4. **HuggingFace Transformers 호환성 (우선)**
**현재 로딩 방식**:
```python
model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
```

**우선순위**:
- 🥇 **Transformers 라이브러리 직접 지원** (즉시 통합)
- 🥈 **Adapter 패턴으로 통합 가능** (SpeechBrain, TensorFlow)
- 🥉 **수동 통합 필요** (OpenSMILE, Kaggle 모델)

**제외 기준**:
- ❌ 폐쇄형 API (OpenAI, Google Cloud - 비용 발생)
- ❌ 온라인 전용 (오프라인 사용 불가)
- ❌ 상용 라이선스 (MIT/Apache 선호)

---

### 5. **배치 처리 효율성**
**현재 코드**:
```python
def classify_batch(self, segments: List[dict], batch_size: int = 4):
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        # 배치 처리
```

**요구사항**:
- ✅ 긴 영상 처리 (30분~2시간)
- ✅ 메모리 효율적 (8GB GPU에서 작동)
- ✅ 추론 속도 합리적 (실시간의 10배 이내)

**제외 기준**:
- ❌ 초대형 모델 (10GB+ 메모리 필요)
- ❌ 세그먼트당 5초 이상 처리 시간
- ❌ 배치 처리 미지원 (세그먼트마다 재로딩)

---

## 🎯 가산점 조건 (Nice-to-Have)

### 1. **높은 정확도** (Simpson 데이터셋 기준)
- 🥇 **Accuracy > 0.6** (현재 1위: superb 0.645)
- 🥈 **Macro F1 > 0.3** (감정 균형 고려)
- 🥉 **Neutral Rate < 50%** (중립 편향 방지)

### 2. **최신 아키텍처**
- Wav2Vec2 XLS-R (다국어)
- Whisper Encoder (음성 표현 강력)
- Transformer + CNN Hybrid

### 3. **사전 학습 가중치 제공**
- IEMOCAP (연기 감정)
- RAVDESS (감정 음성)
- MSP-Podcast (자연스러운 대화)

### 4. **활발한 커뮤니티**
- HuggingFace 다운로드 1000회 이상
- GitHub Stars 100개 이상
- 최근 1년 내 업데이트

---

## ❌ 제외 기준 (Exclusion Criteria)

### 자동 제외 대상

#### 1. **감정 클래스 불일치**
- `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
  - 이유: arousal/valence/dominance 출력 (차원 기반)
  - 영향: 7개 감정으로 매핑 불가능

#### 2. **언어 미지원**
- `m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition`
  - 이유: 그리스어 전용, 영어 전이 실패 (0.032 정확도)
  - 영향: 한국어/영어 처리 불가

#### 3. **형식 비호환 (수동 제외)**
- 구조적 비호환: `config.json` 없음
- 로딩 실패: Transformers 라이브러리 미지원

#### 4. **극단적 편향**
- `harshit345/xlsr-wav2vec-speech-emotion-recognition`
  - 이유: 100% sad 예측 (Simpson 데이터셋)
  - 영향: 실용성 없음

#### 5. **라이선스 문제**
- 상용 라이선스 (재배포 불가)
- 학술 전용 (상업적 사용 금지)

---

## 📊 현재 후보군 평가

### Tier S: 즉시 사용 가능 + 고성능
1. ✅ **superb/wav2vec2-large-superb-er**
   - 정확도: 0.645, F1: 0.211
   - 7개 감정 완벽 지원
   - 영어 기반 (한국어 전이 가능)

2. ✅ **marcogdepinto/emotion-recognition-using-voice** (GitHub)
   - 7개 감정
   - scikit-learn 기반 (CPU 효율적)
   - RAVDESS 학습

### Tier A: 통합 필요 + 잠재력 높음
3. 🔄 **speechbrain/emotion-recognition-wav2vec2-IEMOCAP**
   - Adapter 필요
   - IEMOCAP 사전 학습 (연기 감정)

4. 🔄 **IliaZenkov/transformer-cnn-emotion-recognition** (GitHub)
   - Transformer + CNN
   - 6개 감정 (매핑 필요)

### Tier B: 한국어 특화
5. ⚠️ **jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition**
   - 한국어 전용 (영어 0.065 정확도)
   - 한국어 데이터셋으로만 평가해야 함

### Tier C: 제외
- ❌ ehcalabres (100% neutral)
- ❌ audeering (차원 기반 출력)
- ❌ harshit345 (100% sad)
- ❌ m3hrdadfi (그리스어 전용)

---

## 🎯 최종 테스트 후보군 (10개)

### HuggingFace (2개)
1. superb/wav2vec2-large-superb-er ⭐ (현재 1위)
2. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (neutral 편향 수정 가능 시)

### GitHub 오픈소스 (3개)
3. marcogdepinto/emotion-recognition-using-voice
4. IliaZenkov/transformer-cnn-emotion-recognition
5. MITESHPUTHRANNEU/Speech-Emotion-Analyzer

### SpeechBrain (1개)
6. speechbrain/emotion-recognition-wav2vec2-IEMOCAP

### Kaggle/TensorFlow (2개)
7. Kaggle RAVDESS 사전 학습 모델
8. TensorFlow Hub YAMNet (전이 학습)

### 한국어 특화 (1개)
9. jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition

### OpenSMILE (1개)
10. OpenSMILE ComParE + SVM (학습 필요)

---

## 평가 전략

### 1단계: 즉시 평가 (HuggingFace + GitHub)
- 소요 시간: 2-3시간
- 모델 수: 5개
- 목표: 현재 1위 모델보다 우수한 후보 발견

### 2단계: 고급 통합 (SpeechBrain + Kaggle)
- 소요 시간: 3-4시간
- 모델 수: 3개
- 목표: 최신 아키텍처 성능 검증

### 3단계: 한국어 데이터셋 평가
- 소요 시간: 2-3시간
- 모델 수: jungjongho + Top 3 모델
- 목표: 한국어 성능 확인

---

## 요약

**적합한 모델 = 7개 감정 + 오디오 입력 + 다국어 지원 + Transformers 호환 + 배치 처리 효율**

**최우선 테스트 대상**:
1. marcogdepinto (GitHub, 즉시 사용)
2. speechbrain IEMOCAP (고성능 기대)
3. Kaggle RAVDESS (검증된 데이터셋)

다음 단계로 넘어가시겠습니까?
