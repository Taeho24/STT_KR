# 자동 모델 평가 및 정리 가이드

## 🚀 빠른 시작

### 1. 일괄 평가 실행

```powershell
# 모든 후보 모델 평가 (config.py의 audio_candidates)
python tools/batch_evaluator.py

# 특정 모델만 평가
python tools/batch_evaluator.py --models superb/wav2vec2-large-superb-er ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition

# 다른 영상으로 평가
python tools/batch_evaluator.py --video assets/truman.mp4 --labels labelled_truman.jsonl
```

**출력**:
- 순위표 (정확도, F1, 중립 비율, 속도)
- `result/batch_evaluation.json` (상세 결과)
- `result/batch_evaluation.csv` (스프레드시트용)

---

### 2. 평가 후 정리

#### 미리보기 (삭제하지 않음)
```powershell
python tools/cleanup.py --dry-run
```

#### test_models 폴더만 삭제
```powershell
python tools/cleanup.py
```

#### HuggingFace 캐시까지 정리 (최종 모델 선택 후)
```powershell
# 최종 모델 제외하고 전부 삭제
python tools/cleanup.py --all --keep-models superb/wav2vec2-large-superb-er
```

---

## 📋 업데이트된 모델 선정 기준

### 필수 조건
1. ✅ **5개 이상 감정 클래스** (7개 아니어도 됨)
2. ✅ **오디오 입력 지원**
3. ✅ **성능 대비 속도 우수**
4. ⚠️ **한국어/영어 동시 지원 선호** (필수 아님, 분리 가능)

### 평가 지표
- **정확도** (Accuracy)
- **F1 Score** (균형 지표)
- **중립 비율** (낮을수록 좋음)
- **처리 속도** (초/영상)

### 순위 계산
```
Overall Score = 0.6 * Accuracy + 0.2 * F1 + 0.1 * (1 - Neutral_Rate) + 0.1 * (1 / Time)
```

---

## 🎯 현재 후보 모델 (config.py)

### Tier 1: 검증 완료
1. **superb/wav2vec2-large-superb-er** (현재 1위)
   - 정확도: 0.645
   - 7개 감정
   - 영어 기반

2. **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**
   - 정확도: 높음 (중립 편향)
   - 7개 감정
   - 다국어 지원

3. **jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition**
   - 한국어 특화
   - 7개 감정
   - 영어 성능 낮음 (0.065)

### Tier 2: 평가 대기
4. **speechbrain/emotion-recognition-wav2vec2-IEMOCAP**
   - 4개 감정 (IEMOCAP)
   - 고품질 사전 학습

5. **facebook/wav2vec2-large-robust-ft-swbd-300h**
   - 범용 모델 (전이 학습 가능)

6. **microsoft/wavlm-base-plus**
   - WavLM (빠른 속도)

7. **jonatasgrosman/wav2vec2-large-xlsr-53-english**
   - 영어 최적화

8. **jonatasgrosman/wav2vec2-large-xlsr-53-korean**
   - 한국어 최적화

---

## 🗂️ 폴더 구조

```
STT_KR-liveCaption/
├── test_models/          # 테스트용 모델 (삭제 예정)
│   ├── transformer-cnn-emotion-recognition/
│   └── ...
├── result/
│   ├── batch_evaluation.json
│   └── batch_evaluation.csv
├── tools/
│   ├── batch_evaluator.py    # 일괄 평가
│   └── cleanup.py             # 정리 스크립트
└── .cache/                    # HuggingFace 캐시
```

**정리 순서**:
1. 평가 완료
2. 최종 모델 선택
3. `test_models/` 삭제
4. HuggingFace 캐시 정리 (최종 모델 제외)

---

## 📊 평가 결과 예시

```
Rank  Model                                                      Acc     F1      Neutral   Time(s)
----  ----                                                       ---     --      -------   -------
1     wav2vec2-large-superb-er                                  0.645   0.211   0.323     45.2
2     wav2vec2-lg-xlsr-en-speech-emotion-recognition           0.581   0.179   0.484     38.7
3     emotion-recognition-wav2vec2-IEMOCAP                      0.523   0.312   0.210     29.3
```

**해석**:
- Rank 1: 정확도 최고 (0.645)
- Rank 3: F1 최고 (0.312), 중립 비율 최저 (0.210), 가장 빠름 (29.3s)

→ 정확도 우선이면 Rank 1, 속도 우선이면 Rank 3

---

## 🛠️ 고급 사용법

### 새 모델 추가

1. `config.py` 수정:
```python
'audio_candidates': [
    # 기존 모델...
    'new-org/new-emotion-model',  # 새 모델 추가
]
```

2. 평가 실행:
```powershell
python tools/batch_evaluator.py
```

### 특정 모델만 재평가
```powershell
python tools/batch_evaluator.py --models new-org/new-emotion-model
```

### 여러 영상으로 교차 검증
```powershell
# Simpson
python tools/batch_evaluator.py --video assets/simpson.mp4 --labels labelled_simpson.jsonl

# Truman
python tools/batch_evaluator.py --video assets/truman.mp4 --labels labelled_truman.jsonl

# 평균 성능 비교
```

---

## ⚠️ 주의사항

### 디스크 공간
- HuggingFace 캐시: ~5-10 GB (모델당 ~500MB-2GB)
- test_models: ~1-2 GB
- **권장**: 최소 15GB 여유 공간

### 평가 시간
- 모델당: 30초 - 2분 (Simpson 기준)
- 전체 8개 모델: ~15분

### GPU 메모리
- 필요: 4GB 이상 권장
- 부족 시: `--device cpu` (느려짐)

---

## 🎯 최종 워크플로우

### 1단계: 일괄 평가
```powershell
python tools/batch_evaluator.py
```

### 2단계: 결과 확인
```powershell
# CSV 열기
start result/batch_evaluation.csv

# 또는 JSON 확인
cat result/batch_evaluation.json
```

### 3단계: 최종 모델 선택
순위표에서 **정확도**와 **속도**를 고려해 선택

### 4단계: config 업데이트
```python
# config.py
'models': {
    'audio': 'superb/wav2vec2-large-superb-er',  # 최종 모델
    # ...
}
```

### 5단계: 정리
```powershell
# test_models 삭제
python tools/cleanup.py

# HuggingFace 캐시 정리 (최종 모델 제외)
python tools/cleanup.py --all --keep-models superb/wav2vec2-large-superb-er
```

---

## 💡 팁

### 빠른 프로토타이핑
```powershell
# 상위 3개만 평가
python tools/batch_evaluator.py --models \
    superb/wav2vec2-large-superb-er \
    microsoft/wavlm-base-plus \
    speechbrain/emotion-recognition-wav2vec2-IEMOCAP
```

### 한국어/영어 분리 전략
```powershell
# 영어 모델
python tools/batch_evaluator.py --models jonatasgrosman/wav2vec2-large-xlsr-53-english

# 한국어 모델
python tools/batch_evaluator.py --models jonatasgrosman/wav2vec2-large-xlsr-53-korean

# 프로젝트에서 언어 감지 후 다른 모델 사용
```

---

## 🔗 관련 문서

- `docs/model_selection_criteria.md` - 상세 선정 기준
- `docs/comprehensive_evaluation_strategy.md` - 평가 전략
- `docs/unsupervised_evaluation_methods.md` - 라벨 없는 평가

---

## ❓ FAQ

**Q: 평가가 너무 오래 걸려요**
A: `--models`로 일부만 선택하거나 `--device cpu` 제거

**Q: 메모리 부족 오류**
A: `batch_size` 줄이거나 작은 모델 선택

**Q: 한국어 성능이 낮아요**
A: `jungjongho` 모델 사용 또는 한국어 데이터로 파인튜닝

**Q: 정리 후 복구 가능한가요?**
A: HuggingFace 모델은 재다운로드 가능, test_models는 복구 불가

---

## 📞 지원

문제 발생 시:
1. `result/batch_evaluation.json` 확인
2. 터미널 출력 로그 확인
3. Issue 등록

Good luck! 🚀
