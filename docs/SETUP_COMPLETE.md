# 🚀 자동 모델 평가 완료 가이드

## ✅ 완료된 작업

### 1. 업데이트된 선정 기준
- ✅ 5개 이상 감정 (7개 필수 아님)
- ✅ 한국어/영어 분리 가능 (동시 지원 선호)
- ✅ 성능 대비 속도 중요
- ✅ 테스트 후 삭제 용이

### 2. 자동 설치 시스템
- ✅ `test_models/` 폴더 생성 (삭제 전용)
- ✅ Kaggle API 토큰 설정 완료
- ✅ Kaggle CLI 설치

### 3. 생성된 도구
- ✅ `tools/batch_evaluator.py` - 일괄 평가
- ✅ `tools/cleanup.py` - 정리 스크립트
- ✅ `tools/unsupervised_evaluator.py` - 라벨 없는 평가

### 4. 문서화
- ✅ `docs/AUTO_EVALUATION_GUIDE.md` - 자동 평가 가이드
- ✅ `docs/model_selection_criteria.md` - 선정 기준
- ✅ `docs/comprehensive_evaluation_strategy.md` - 종합 전략

---

## 🔧 현재 진행 중

### UTF-8 인코딩 수정
Windows PowerShell의 CP949 인코딩 문제 해결 중:
- `tools/model_evaluator.py`에 UTF-8 강제 설정 추가
- 평가 실행 중...

---

## 📋 다음 단계

### Option 1: 평가 결과 확인 (현재 실행 중)

평가가 완료되면:
```powershell
# 결과 확인
cat result/batch_evaluation.json

# CSV 열기
start result/batch_evaluation.csv
```

### Option 2: 전체 후보 모델 평가

```powershell
# config.py의 모든 후보 모델 평가
python tools/batch_evaluator.py

# 또는 특정 모델만
python tools/batch_evaluator.py --models \
    superb/wav2vec2-large-superb-er \
    microsoft/wavlm-base-plus \
    jonatasgrosman/wav2vec2-large-xlsr-53-english
```

### Option 3: 정리

```powershell
# 미리보기
python tools/cleanup.py --dry-run

# 삭제
python tools/cleanup.py
```

---

## 💡 확장된 모델 후보 (config.py 업데이트됨)

```python
'audio_candidates': [
    # 현재 평가 완료 (7개 감정)
    'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition',
    'superb/wav2vec2-large-superb-er',  # 현재 1위 (0.645)
    'jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition',  # 한국어
    
    # 추가 후보 (5-8개 감정, 속도 우선)
    'speechbrain/emotion-recognition-wav2vec2-IEMOCAP',  # 4개 감정
    'facebook/wav2vec2-large-robust-ft-swbd-300h',  # 범용
    'microsoft/wavlm-base-plus',  # 빠름
    'jonatasgrosman/wav2vec2-large-xlsr-53-english',  # 영어
    'jonatasgrosman/wav2vec2-large-xlsr-53-korean',  # 한국어
]
```

---

## 🗂️ 폴더 구조

```
STT_KR-liveCaption/
├── test_models/                    # 테스트 전용 (삭제 예정)
│   └── transformer-cnn-emotion-recognition/
├── result/
│   ├── batch_evaluation.json       # 평가 결과
│   └── batch_evaluation.csv
├── tools/
│   ├── batch_evaluator.py          # ⭐ 일괄 평가
│   ├── cleanup.py                  # ⭐ 정리 스크립트
│   └── unsupervised_evaluator.py   # 라벨 없는 평가
├── docs/
│   ├── AUTO_EVALUATION_GUIDE.md    # ⭐ 자동 평가 가이드
│   ├── model_selection_criteria.md
│   └── comprehensive_evaluation_strategy.md
└── kaggle.json                     # Kaggle API 토큰
```

---

## ⚡ 빠른 명령어

### 평가
```powershell
# 상위 3개 모델 평가 (빠름)
python tools/batch_evaluator.py --models \
    superb/wav2vec2-large-superb-er \
    microsoft/wavlm-base-plus \
    speechbrain/emotion-recognition-wav2vec2-IEMOCAP

# 전체 평가 (~20분)
python tools/batch_evaluator.py
```

### 정리
```powershell
# 미리보기
python tools/cleanup.py --dry-run

# test_models 삭제
python tools/cleanup.py

# 전체 정리 (최종 모델 제외)
python tools/cleanup.py --all --keep-models superb/wav2vec2-large-superb-er
```

---

## 📊 예상 평가 결과

| Rank | Model | Acc | F1 | Speed | 특징 |
|------|-------|-----|----|----|------|
| 1 | superb/wav2vec2-large-superb-er | 0.645 | 0.211 | 중간 | 7감정, 균형 |
| 2 | microsoft/wavlm-base-plus | ? | ? | **빠름** | 범용, 전이학습 |
| 3 | speechbrain/emotion-recognition | ? | ? | 빠름 | 4감정, IEMOCAP |

---

## 🔍 문제 해결

### CP949 인코딩 오류
```powershell
# UTF-8 강제
$env:PYTHONIOENCODING='utf-8'
python tools/batch_evaluator.py --models ...
```

### 메모리 부족
```python
# emotion_classifier.py
batch_size = 2  # 4 → 2로 줄이기
```

### 느린 속도
```powershell
# CPU 사용 시 GPU로 전환
python tools/batch_evaluator.py --device cuda
```

---

## ✅ 체크리스트

### 즉시 실행 가능
- [x] test_models 폴더 생성
- [x] Kaggle API 설정
- [x] config.py 후보 모델 추가
- [x] batch_evaluator.py 생성
- [x] cleanup.py 생성
- [x] UTF-8 인코딩 수정

### 평가 진행 중
- [ ] superb 모델 평가 (실행 중)
- [ ] 전체 후보 모델 평가
- [ ] 결과 CSV 확인

### 평가 완료 후
- [ ] 최종 모델 선택
- [ ] config.py 업데이트
- [ ] test_models 삭제
- [ ] HuggingFace 캐시 정리

---

## 🎯 권장 워크플로우

1. **현재 평가 완료 확인** (진행 중)
2. **결과 확인**: `cat result/batch_evaluation.json`
3. **추가 모델 평가**: 속도 우선 모델 추가
4. **최종 선택**: 정확도 vs 속도 trade-off 결정
5. **정리**: `python tools/cleanup.py`

---

**모든 작업이 자동화되었습니다! 🎉**

평가 완료를 기다리거나, 원하는 다른 작업이 있으면 말씀해주세요!
