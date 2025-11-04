# 모델 평가 전략 - 종합 가이드

## 📌 질문 요약

### 1. "이 프로젝트에 적합한 모델"의 정의는?
### 2. 수동 작업이 필요한 모델과 구체적인 단계는?
### 3. 라벨 없이 정확도를 평가하는 방법은?

---

## ✅ 1. 프로젝트에 적합한 모델의 정의

### 프로젝트 요구사항

**핵심 기능**:
- WhisperX STT → 감정 분류 → 감정별 스타일 적용된 ASS 자막 생성
- 영화/드라마/유튜브 영상 대상
- 배치 처리 (실시간 아님)

**입출력**:
```
입력: 영상 파일 (한국어/영어 음성)
출력: 7개 감정 (neutral, happy, sad, angry, fear, surprise, disgust)
      + 색상/폰트 스타일 적용된 .ass 자막
```

---

### ✅ 필수 조건 (Must-Have)

#### 1. **7개 감정 클래스 지원**
```python
# config.py에 정의된 감정
emotions = ['neutral', 'happy', 'sad', 'angry', 'fear', 'surprise', 'disgust']
```

**❌ 제외 대상**:
- 4개 이하 감정 모델
- 차원 기반 출력 (arousal/valence)
- 바이너리 분류 (positive/negative)

**✅ 허용**:
- 정확히 7개
- 8개 이상 (매핑 가능: calm → neutral)

---

#### 2. **오디오 입력 지원**
```python
audio_segment = librosa.load(audio_path, sr=16000)
emotion = classifier.predict(audio_segment)
```

**❌ 제외**:
- 텍스트 전용 (BERT, RoBERTa)
- 이미지 입력 (얼굴 표정)

**✅ 허용**:
- Wav2Vec2, HuBERT, WavLM
- 특징 추출 + ML (OpenSMILE + SVM)

---

#### 3. **다국어 지원 (한국어 + 영어)**

**우선순위**:
1. 🥇 한국어 + 영어 동시 (XLS-R 기반)
2. 🥈 영어 전용 (대규모 데이터)
3. 🥉 한국어 전용 (jungjongho)

**❌ 제외**:
- 단일 언어 전용 (그리스어, 독일어)

---

#### 4. **HuggingFace Transformers 호환 (우선)**
```python
model = AutoModelForAudioClassification.from_pretrained(model_name)
```

**우선순위**:
- 🥇 Transformers 직접 지원 (즉시 통합)
- 🥈 Adapter 패턴 (SpeechBrain, TensorFlow)
- 🥉 수동 통합 (OpenSMILE, Kaggle)

**❌ 제외**:
- 폐쇄형 API (OpenAI - 비용)
- 상용 라이선스

---

#### 5. **배치 처리 효율성**
```python
for i in range(0, len(segments), batch_size):
    batch = segments[i:i + batch_size]
    results = model.predict_batch(batch)
```

**요구사항**:
- 긴 영상 처리 (30분~2시간)
- 8GB GPU에서 작동
- 실시간의 10배 이내 속도

**❌ 제외**:
- 초대형 모델 (10GB+ 메모리)
- 세그먼트당 5초 이상 처리

---

### 🎯 가산점 (Nice-to-Have)

1. **높은 정확도** (Simpson 기준)
   - Accuracy > 0.6
   - Macro F1 > 0.3
   - Neutral Rate < 50%

2. **최신 아키텍처**
   - Wav2Vec2 XLS-R
   - Whisper Encoder
   - Transformer + CNN

3. **활발한 커뮤니티**
   - HuggingFace 다운로드 1000+
   - GitHub Stars 100+

---

### ❌ 자동 제외 기준

#### 실제 제외 사례

1. **audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim**
   - 이유: arousal/valence 출력 (차원 기반)
   - 결과: 7개 감정 매핑 불가

2. **m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition**
   - 이유: 그리스어 전용
   - 결과: 영어 정확도 0.032

3. **harshit345/xlsr-wav2vec-speech-emotion-recognition**
   - 이유: 100% sad 예측 (극단적 편향)
   - 결과: 실용성 없음

4. **형식 비호환**
   - `config.json` 없음
   - Transformers 로딩 실패

---

### 📊 최종 테스트 후보군 (10개)

#### Tier S: 즉시 사용 가능
1. ✅ **superb/wav2vec2-large-superb-er** (현재 1위, 0.645 정확도)
2. ✅ **marcogdepinto/emotion-recognition-using-voice** (GitHub, 7개 감정)

#### Tier A: 통합 필요
3. 🔄 **speechbrain/emotion-recognition-wav2vec2-IEMOCAP**
4. 🔄 **IliaZenkov/transformer-cnn-emotion-recognition**
5. 🔄 **MITESHPUTHRANNEU/Speech-Emotion-Analyzer**

#### Tier B: 고급
6. 🔴 **Kaggle RAVDESS** (계정 필요)
7. 🟣 **OpenSMILE + SVM** (학습 필요)

#### Tier C: 한국어 특화
8. ⚠️ **jungjongho/wav2vec2-xlsr-korean** (한국어 데이터 필요)

---

## 📥 2. 수동 작업이 필요한 모델

### 🟢 Tier 1: 자동 (작업 불필요)

#### HuggingFace 모델
- superb/wav2vec2-large-superb-er
- ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
- speechbrain/emotion-recognition-wav2vec2-IEMOCAP

**사용자 작업**: ❌ 없음 (자동 다운로드)

---

### 🟡 Tier 2: Git Clone (15분)

#### 1. marcogdepinto/emotion-recognition-using-voice

```powershell
# 1. 디렉토리 생성 및 클론
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption
mkdir external_models
cd external_models
git clone https://github.com/marcogdepinto/emotion-recognition-using-voice.git

# 2. 모델 파일 확인
cd emotion-recognition-using-voice
dir models\model.pkl

# 3. 없으면 다운로드
python download_models.py
```

**예상 시간**: 5분  
**완료 조건**: `models\model.pkl` 존재

---

#### 2. IliaZenkov/transformer-cnn-emotion-recognition

```powershell
# 1. 클론
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption\external_models
git clone https://github.com/IliaZenkov/transformer-cnn-emotion-recognition.git
cd transformer-cnn-emotion-recognition

# 2. 종속성 설치
..\..\venv\Scripts\pip.exe install -r requirements.txt

# 3. 사전 학습 모델 다운로드
..\..\venv\Scripts\python.exe download_models.py
```

**수동 다운로드 (스크립트 실패 시)**:
1. https://github.com/IliaZenkov/transformer-cnn-emotion-recognition/releases
2. `best_model.pth` 다운로드
3. `checkpoints/` 폴더에 저장

**예상 시간**: 10분 (다운로드 ~500MB)  
**완료 조건**: `checkpoints\best_model.pth` 존재

---

#### 3. MITESHPUTHRANNEU/Speech-Emotion-Analyzer

```powershell
# 1. 클론
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption\external_models
git clone https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer.git
cd Speech-Emotion-Analyzer

# 2. TensorFlow 설치
..\..\venv\Scripts\pip.exe install tensorflow

# 3. 모델 수동 다운로드
```

**수동 다운로드**:
1. https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/releases
2. `best_model.h5` 다운로드 (~100MB)
3. 프로젝트 루트에 저장

**예상 시간**: 10분  
**완료 조건**: `best_model.h5` 존재

---

### 🔴 Tier 3: Kaggle 계정 (30분)

#### Kaggle RAVDESS 모델

##### Step 1: 계정 생성
1. https://www.kaggle.com 접속
2. "Sign Up" → Google/이메일 가입
3. 이메일 인증

##### Step 2: API 토큰
1. 로그인 → 프로필 → "Settings"
2. "API" → "Create New Token"
3. `kaggle.json` 자동 다운로드

##### Step 3: 토큰 설정
```powershell
# 1. 디렉토리 생성
mkdir $env:USERPROFILE\.kaggle

# 2. 토큰 복사
copy "%USERPROFILE%\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\"

# 3. 확인
dir $env:USERPROFILE\.kaggle\kaggle.json
```

##### Step 4: 데이터셋 다운로드
```powershell
# 1. Kaggle CLI 설치
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption
.\venv\Scripts\pip.exe install kaggle

# 2. 다운로드
cd external_models
mkdir kaggle_models
cd kaggle_models
..\..\.venv\Scripts\kaggle.exe datasets download -d uwrfkaggle/ravdess-emotional-speech-audio

# 3. 압축 해제
tar -xf ravdess-emotional-speech-audio.zip
```

**예상 시간**: 20분 (다운로드 ~500MB)  
**완료 조건**: `kaggle.json` 설정 + RAVDESS 데이터 존재

---

### 🟣 Tier 4: 학습 필요 (1-2시간)

#### OpenSMILE + SVM

```powershell
# 1. 설치
.\venv\Scripts\pip.exe install opensmile scikit-learn

# 2. 데이터셋 준비 (위 RAVDESS 사용)

# 3. 특징 추출 및 학습 (스크립트 제공 예정)
```

⚠️ **주의**: 다른 모델 평가 후 필요 시 진행 권장

---

### 🟤 Tier 5: 한국어 데이터셋 (1-3일)

#### jungjongho 모델 평가용

##### Option 1: AI Hub
1. https://aihub.or.kr 가입
2. "한국어 멀티모달 감정 데이터셋" 신청
3. 승인 대기 (1-3일)
4. 다운로드 (~50GB)

##### Option 2: 직접 라벨링
```powershell
# 한국어 영상 준비
# labelled_korean.jsonl 생성 (Simpson 형식)
# 최소 30개 세그먼트
```

---

### 📋 체크리스트

#### 필수 작업 (5개 모델)
- [ ] **marcogdepinto** Git Clone (5분)
- [ ] **IliaZenkov** Git Clone + 모델 다운로드 (10분)
- [ ] **MITESHPUTHRANNEU** 수동 다운로드 (10분)

**총 소요 시간**: 25분

#### 선택 작업
- [ ] **Kaggle** 계정 + 토큰 (30분)
- [ ] **한국어 데이터셋** (1-3일)

---

## 📈 3. 라벨 없이 정확도 평가하는 방법

### 문제점
- 라벨링 시간 소요 (1시간+)
- 새 영상마다 수동 작업
- 주관적 판단

### 해결책: Unsupervised Evaluation

---

### ✅ 방법 1: Cross-Model Consistency (모델 간 일치도)

#### 원리
여러 모델이 **같은 세그먼트**에 일치하는 예측 → 높은 신뢰도

#### 사용법
```powershell
python tools/unsupervised_evaluator.py \
    --video assets/simpson.mp4 \
    --models \
        superb/wav2vec2-large-superb-er \
        ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition \
        speechbrain/emotion-recognition-wav2vec2-IEMOCAP
```

#### 평가 지표
- **평균 일치도**: > 0.6 (좋음)
- **고신뢰 비율**: > 50%
- **다양성 (Entropy)**: 1.5~2.0

---

### ✅ 방법 2: Confidence Distribution Analysis (신뢰도 분포)

#### 지표
```python
{
    'mean_confidence': 0.67,      # 평균 신뢰도
    'high_conf_ratio': 0.52,      # 고신뢰 비율 (>0.7)
    'neutral_ratio': 0.35,        # 중립 비율
    'diversity': 1.82             # 감정 다양성
}
```

#### 좋은 모델 기준
- 평균 신뢰도: 0.6~0.8
- 중립 비율: 20~40%
- 다양성: 1.5~2.0

---

### ✅ 방법 3: Entropy-Based Quality Score

#### 원리
예측 분포가 **적정 수준**의 확신도 유지

#### 품질 점수
```python
quality_score = (
    0.3 * consistency +           # 일치도
    0.3 * entropy_quality +       # 예측 품질
    0.2 * (1 - neutral_ratio) +   # 중립 패널티
    0.2 * mean_confidence         # 평균 신뢰도
)
```

| 점수 | 판단 |
|------|------|
| > 0.8 | 우수 |
| 0.6~0.8 | 양호 |
| 0.4~0.6 | 보통 |
| < 0.4 | 불량 |

---

### ✅ 방법 4: Perceptual Validation (지각적 검증)

#### 절차
1. 각 감정별 대표 샘플 선택 (10개)
2. HTML 리뷰 페이지 생성
3. 10개만 수동 확인 (10분)
4. 정확도 추정

#### 장점
- 빠름 (10분)
- 인간 직관 활용
- 극단적 오류 탐지

---

### 📊 통합 평가 프레임워크

#### 실제 사용 예시

```powershell
# 1. Unsupervised 평가 실행
python tools/unsupervised_evaluator.py \
    --video assets/simpson.mp4 \
    --models \
        superb/wav2vec2-large-superb-er \
        marcogdepinto/emotion-recognition \
        speechbrain/emotion-recognition-wav2vec2-IEMOCAP \
    --output-dir result

# 출력:
# 📊 UNSUPERVISED EVALUATION RESULTS
# 
# 🏆 Rank 1: superb/wav2vec2-large-superb-er
#    Overall Score: 0.687
#    ├─ Consistency: 0.723
#    ├─ Entropy Quality: 0.845
#    ├─ Mean Confidence: 0.671
#    └─ Neutral Ratio: 0.323
# 
# 🏆 Rank 2: speechbrain/emotion-recognition-wav2vec2-IEMOCAP
#    Overall Score: 0.642
#    ...
```

#### 생성 파일
- `result/unsupervised_eval_simpson.json` (상세 결과)
- `result/unsupervised_eval_simpson.png` (시각화)

---

### 검증 전략

#### 1단계: Simpson (라벨 있음)
```powershell
# Supervised 평가
python tools/model_evaluator.py \
    --video assets/simpson.mp4 \
    --labels labelled_simpson.jsonl \
    --disable-text

# 결과: superb = 0.645 정확도
```

#### 2단계: Simpson (라벨 숨김)
```powershell
# Unsupervised 평가
python tools/unsupervised_evaluator.py \
    --video assets/simpson.mp4 \
    --models superb/wav2vec2-large-superb-er ...

# 결과: superb = 0.687 품질 점수
```

#### 3단계: 상관관계 확인
```python
correlation = 0.85  # Supervised vs Unsupervised
# 0.85 이상이면 신뢰 가능
```

#### 4단계: 새 영상 적용
```powershell
# 라벨 없는 새 영상도 평가 가능
python tools/unsupervised_evaluator.py \
    --video assets/new_movie.mp4 \
    --models ...
```

---

### 비교표

| 방법 | 라벨 | 시간 | 신뢰도 | 용도 |
|------|------|------|--------|------|
| Cross-Model | ❌ | 자동 | ⭐⭐⭐⭐ | 기본 |
| Confidence | ❌ | 자동 | ⭐⭐⭐ | 품질 진단 |
| Entropy | ❌ | 자동 | ⭐⭐⭐⭐ | 정량 평가 |
| Perceptual | ❌ | 10분 | ⭐⭐⭐⭐⭐ | 최종 검증 |
| Supervised | ✅ | 1시간+ | ⭐⭐⭐⭐⭐ | 절대 기준 |

**권장 조합**:
1. Simpson으로 Supervised 1회
2. 모든 영상에 Unsupervised 적용
3. 10개 샘플로 Perceptual Validation

**라벨링 시간 90% 절감!**

---

## 🎯 실전 프로토콜

### Phase 1: 필수 모델 평가 (1시간)

```powershell
# 1. HuggingFace 모델 (자동)
python tools/model_evaluator.py \
    --video assets/simpson.mp4 \
    --labels labelled_simpson.jsonl \
    --disable-text \
    --audio-models \
        superb/wav2vec2-large-superb-er \
        ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
```

### Phase 2: GitHub 모델 (25분 설정 + 1시간 평가)

```powershell
# 1. 수동 작업 (25분)
# - marcogdepinto git clone
# - IliaZenkov git clone + 모델 다운로드
# - MITESHPUTHRANNEU 수동 다운로드

# 2. 평가 (자동 통합 후)
python tools/unsupervised_evaluator.py \
    --video assets/simpson.mp4 \
    --models \
        superb/wav2vec2-large-superb-er \
        marcogdepinto/emotion-recognition \
        IliaZenkov/transformer-cnn
```

### Phase 3: 최종 검증 (10분)

```powershell
# 10개 샘플 수동 확인
# HTML 리뷰 페이지 생성 (자동)
```

**총 소요 시간**: 2시간 35분

---

## 📚 생성된 문서

1. **docs/model_selection_criteria.md** - 모델 선정 기준 상세
2. **docs/manual_setup_guide.md** - 수동 작업 단계별 가이드
3. **docs/unsupervised_evaluation_methods.md** - 라벨 없는 평가 방법
4. **tools/unsupervised_evaluator.py** - 실행 가능한 평가 도구

---

## 🚀 다음 단계

원하는 작업을 선택하세요:

### Option 1: 즉시 평가 시작 (HuggingFace 모델만)
```powershell
python tools/unsupervised_evaluator.py \
    --video assets/simpson.mp4 \
    --models \
        superb/wav2vec2-large-superb-er \
        ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition \
        speechbrain/emotion-recognition-wav2vec2-IEMOCAP
```

### Option 2: GitHub 모델 설정 가이드
필수 작업 3개 (25분) 단계별 안내

### Option 3: 전체 10개 모델 평가 계획
Kaggle 포함 전체 파이프라인

어떤 옵션으로 진행할까요?
