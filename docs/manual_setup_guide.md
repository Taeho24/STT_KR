# 수동 작업이 필요한 모델 - 단계별 가이드

## 개요

테스트 후보 10개 모델 중 **사용자가 직접 작업해야 하는 모델**과 구체적인 단계를 설명합니다.

---

## 🟢 Tier 1: 자동 다운로드 가능 (사용자 작업 불필요)

### 1. superb/wav2vec2-large-superb-er
- ✅ **HuggingFace 자동 다운로드**
- ✅ 사용자 작업: **없음**

### 2. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
- ✅ **HuggingFace 자동 다운로드**
- ✅ 사용자 작업: **없음**

### 3. speechbrain/emotion-recognition-wav2vec2-IEMOCAP
- ✅ **HuggingFace 자동 다운로드**
- ✅ 사용자 작업: **없음** (Adapter 코드는 자동 생성)

---

## 🟡 Tier 2: Git Clone 필요 (간단한 작업)

### 4. marcogdepinto/emotion-recognition-using-voice

#### 필요 이유
GitHub 저장소에서 소스 코드와 모델을 함께 제공하므로 `git clone` 필요

#### 단계별 가이드

```powershell
# 1. 프로젝트 루트로 이동
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption

# 2. external_models 디렉토리 생성
mkdir external_models
cd external_models

# 3. 저장소 클론
git clone https://github.com/marcogdepinto/emotion-recognition-using-voice.git

# 4. 디렉토리 구조 확인
cd emotion-recognition-using-voice
dir
```

**예상 출력**:
```
emotion_recognition/
    __init__.py
    recognizer.py
models/
    model.pkl
requirements.txt
README.md
```

#### 모델 파일 확인

```powershell
# 5. 모델 파일이 있는지 확인
dir models\model.pkl
```

**만약 모델 파일이 없다면**:
```powershell
# 6. 사전 학습 모델 다운로드 (Python 스크립트 실행)
python download_models.py
```

#### 통합 확인

```powershell
# 7. 모델 로딩 테스트
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption
.\venv\Scripts\python.exe -c "from external_models.emotion-recognition-using-voice.emotion_recognition import EmotionRecognizer; print('✅ Import successful')"
```

✅ **완료 조건**: `✅ Import successful` 메시지 출력

---

### 5. IliaZenkov/transformer-cnn-emotion-recognition

#### 필요 이유
GitHub 전용, 사전 학습 가중치 다운로드 스크립트 실행 필요

#### 단계별 가이드

```powershell
# 1. external_models로 이동
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption\external_models

# 2. 저장소 클론
git clone https://github.com/IliaZenkov/transformer-cnn-emotion-recognition.git
cd transformer-cnn-emotion-recognition

# 3. 종속성 설치
..\..\venv\Scripts\pip.exe install -r requirements.txt

# 4. 사전 학습 모델 다운로드
..\..\venv\Scripts\python.exe download_models.py
```

**다운로드 시간**: 약 2-5분 (모델 크기: ~500MB)

#### 수동 다운로드 (스크립트 실패 시)

만약 `download_models.py` 실행이 실패하면:

1. 브라우저에서 열기: https://github.com/IliaZenkov/transformer-cnn-emotion-recognition/releases
2. `best_model.pth` 파일 다운로드
3. `checkpoints/` 폴더에 저장:
```powershell
mkdir checkpoints
# 다운로드 폴더에서 파일 복사
copy "%USERPROFILE%\Downloads\best_model.pth" checkpoints\
```

✅ **완료 조건**: `checkpoints\best_model.pth` 파일 존재

---

### 6. MITESHPUTHRANNEU/Speech-Emotion-Analyzer

#### 필요 이유
TensorFlow 모델 파일 수동 다운로드 필요 (GitHub Releases)

#### 단계별 가이드

```powershell
# 1. external_models로 이동
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption\external_models

# 2. 저장소 클론
git clone https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer.git
cd Speech-Emotion-Analyzer

# 3. TensorFlow 설치 (필요 시)
..\..\venv\Scripts\pip.exe install tensorflow
```

#### 모델 다운로드 (수동)

1. 브라우저에서 열기: https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/releases
2. `best_model.h5` 파일 다운로드 (크기: ~100MB)
3. 프로젝트 루트에 저장:
```powershell
copy "%USERPROFILE%\Downloads\best_model.h5" .
```

#### 대체 방법 (Google Drive)

README에 Google Drive 링크가 있을 수 있음:

1. Google Drive 링크 열기 (README 참조)
2. `best_model.h5` 다운로드
3. 위와 동일하게 저장

✅ **완료 조건**: `best_model.h5` 파일 존재 확인
```powershell
dir best_model.h5
```

---

## 🔴 Tier 3: 계정 생성 필요 (Kaggle)

### 7. Kaggle RAVDESS 사전 학습 모델

#### 필요 이유
Kaggle API 토큰 인증 필요

#### 단계별 가이드

##### Step 1: Kaggle 계정 생성 (없는 경우)

1. 브라우저에서 https://www.kaggle.com 접속
2. "Sign Up" 클릭
3. Google/Facebook 계정 또는 이메일로 가입
4. 이메일 인증 완료

##### Step 2: API 토큰 생성

1. Kaggle 로그인 후 우측 상단 프로필 아이콘 클릭
2. "Settings" 선택
3. "API" 섹션으로 스크롤
4. "Create New Token" 클릭
5. `kaggle.json` 파일 자동 다운로드

##### Step 3: API 토큰 설정

```powershell
# 1. Kaggle 설정 디렉토리 생성
mkdir $env:USERPROFILE\.kaggle

# 2. 다운로드된 kaggle.json 복사
copy "%USERPROFILE%\Downloads\kaggle.json" "$env:USERPROFILE\.kaggle\"

# 3. 권한 확인 (Windows에서는 자동)
dir $env:USERPROFILE\.kaggle\kaggle.json
```

##### Step 4: Kaggle CLI 설치

```powershell
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption
.\venv\Scripts\pip.exe install kaggle
```

##### Step 5: 데이터셋 다운로드

```powershell
# 1. external_models로 이동
cd external_models
mkdir kaggle_models
cd kaggle_models

# 2. RAVDESS 데이터셋 다운로드
..\..\.venv\Scripts\kaggle.exe datasets download -d uwrfkaggle/ravdess-emotional-speech-audio

# 3. 압축 해제
tar -xf ravdess-emotional-speech-audio.zip
```

**다운로드 크기**: ~500MB  
**예상 시간**: 3-10분 (인터넷 속도에 따라)

##### Step 6: 사전 학습 모델 검색

Kaggle에서 사전 학습 모델 검색:

```powershell
# 감정 인식 모델 검색
..\..\.venv\Scripts\kaggle.exe kernels list -s "emotion recognition model"
```

추천 모델:
- `marcogdepinto/speech-emotion-analyzer-model`
- `ejlok1/audio-emotion-recognition`

다운로드:
```powershell
..\..\.venv\Scripts\kaggle.exe kernels pull <username>/<kernel-name>
```

✅ **완료 조건**: 
- `kaggle.json` 파일이 `%USERPROFILE%\.kaggle\` 에 존재
- `ravdess-emotional-speech-audio` 폴더 존재

---

## 🟣 Tier 4: 수동 학습 필요 (OpenSMILE)

### 10. OpenSMILE ComParE + SVM

#### 필요 이유
사전 학습 모델이 공개되지 않음, 직접 학습 필요

#### 단계별 가이드 (고급)

##### Step 1: OpenSMILE 설치

```powershell
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption
.\venv\Scripts\pip.exe install opensmile
```

##### Step 2: 데이터셋 준비

RAVDESS 데이터셋 필요 (위 Kaggle에서 다운로드한 것 사용 가능)

##### Step 3: 특징 추출

```python
# extract_features.py 생성 필요
import opensmile

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# RAVDESS 데이터로 특징 추출 (자동 스크립트 제공 예정)
```

##### Step 4: SVM 학습

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# 특징 추출 결과 로드 → SVM 학습 (자동 스크립트 제공 예정)
```

⚠️ **주의**: 이 모델은 학습에 1-2시간 소요, 다른 모델 평가 후 고려 권장

---

## 🟤 Tier 5: 한국어 데이터셋 필요

### 9. jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition

#### 필요 이유
한국어 음성 데이터로 평가해야 정확한 성능 측정 가능

#### 단계별 가이드

##### Option 1: AI Hub 한국어 감정 음성 데이터셋

1. 브라우저에서 https://aihub.or.kr 접속
2. 회원가입 및 로그인
3. 검색: "감정 음성"
4. "한국어 멀티모달 감정 데이터셋" 다운로드 신청
5. 승인 대기 (1-3일)
6. 승인 후 다운로드

⚠️ **주의**: 다운로드 크기 ~50GB, 승인 대기 시간 있음

##### Option 2: 직접 라벨링

간단한 한국어 영상으로 직접 라벨링:

```powershell
# 한국어 영상 준비 (예: assets/korean_sample.mp4)
# labelled_korean.jsonl 생성 (Simpson과 동일한 형식)
```

**최소 요구사항**: 30개 이상 세그먼트

✅ **완료 조건**: `labelled_korean.jsonl` 파일 생성

---

## 📊 작업 우선순위 요약

### 즉시 가능 (5분 이내)
- ✅ HuggingFace 모델 (자동)
- ✅ Git clone 모델 (marcogdepinto, IliaZenkov)

### 30분 이내
- 🟡 MITESHPUTHRANNEU (수동 다운로드)
- 🟡 Kaggle 계정 생성 + 토큰 설정

### 1시간 이상
- 🔴 한국어 데이터셋 (AI Hub 신청)
- 🟣 OpenSMILE 학습

---

## 체크리스트

### 필수 작업 (최소 5개 모델 테스트)

- [ ] **marcogdepinto** Git Clone
```powershell
cd C:\Users\adap8\OneDrive\Desktop\STT_KR-liveCaption\external_models
git clone https://github.com/marcogdepinto/emotion-recognition-using-voice.git
```

- [ ] **IliaZenkov** Git Clone + 모델 다운로드
```powershell
git clone https://github.com/IliaZenkov/transformer-cnn-emotion-recognition.git
cd transformer-cnn-emotion-recognition
..\..\venv\Scripts\python.exe download_models.py
```

- [ ] **MITESHPUTHRANNEU** 수동 다운로드
1. https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer/releases 방문
2. `best_model.h5` 다운로드
3. 저장소에 복사

### 선택 작업 (추가 모델)

- [ ] **Kaggle** 계정 + API 토큰
1. https://www.kaggle.com 가입
2. Settings → API → Create Token
3. `kaggle.json` 을 `%USERPROFILE%\.kaggle\` 에 저장

- [ ] **한국어 데이터셋** (jungjongho 평가용)
1. AI Hub 가입
2. 한국어 감정 음성 데이터셋 신청

---

## 다음 단계

필수 작업을 완료하면 알려주세요. 그러면:

1. 각 모델의 Adapter 코드 자동 생성
2. `config.py`에 후보 목록 추가
3. 일괄 평가 스크립트 실행

**예상 소요 시간**: 
- 필수 작업: 15분
- 선택 작업: 30분
- 자동 통합: 10분
- **전체 평가: 2-3시간**

준비되면 시작하시겠습니까?
