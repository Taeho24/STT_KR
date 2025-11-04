# 오디오 전용 감정분류 모델 평가 결과

날짜: 2025-10-20  
데이터셋: Simpson 영상 라벨셋 (31개 세그먼트)  
평가 방식: 텍스트 모델 비활성화(`--disable-text`), 순수 오디오 기반 정확도 측정

## 평가 요약

총 7개 모델 후보 중 3개가 유효한 결과를 도출했습니다.

### 1위: **superb/wav2vec2-large-superb-er** ⭐ 추천
- **정확도**: 0.645
- **Macro F1**: 0.211
- **Neutral 비율**: 0.677
- **장점**:
  - SUPERB 벤치마크 기반, 공식 검증된 모델
  - 다국어 XLS-R 백본으로 영어/한국어 모두 일정 수준 지원 가능
  - Neutral 편향이 있지만, 다른 감정도 일부 감지
  - 빠른 추론 속도
- **단점**:
  - 4개 감정만 출력(neutral, happy, angry, sad) - 7개 감정 체계와 부분 호환
  - Neutral 편향이 여전히 높음 (67.7%)
- **권장 사항**: 
  - 현재 프로젝트의 메인 오디오 모델로 사용
  - `emotion_weights` 튜닝으로 neutral 억제 강화 필요

### 2위: **ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition**
- **정확도**: 0.645 (1위와 동일)
- **Macro F1**: 0.112
- **Neutral 비율**: 1.000 ❌
- **장점**:
  - 8개 감정 출력으로 7개 감정 체계와 완벽 호환
  - XLSR-53 백본, 다국어 지원 가능성
- **단점**:
  - **모든 예측을 neutral로만 출력** - 실용성 없음
  - 과도한 neutral 편향
- **권장 사항**: 
  - 현재 상태로는 사용 불가
  - Temperature, emotion_weights 대폭 조정 필요

### 3위: **jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition** (한국어 전용)
- **정확도**: 0.065
- **Macro F1**: 0.018
- **Neutral 비율**: 0.000
- **장점**:
  - 한국어 데이터셋으로 학습된 유일한 모델
  - 5개 감정(기쁨, 당황, 분노, 불안, 슬픔) - 한국어 특화 레이블
- **단점**:
  - **영어 라벨셋에서 성능 극히 낮음** (정확도 6.5%)
  - 모든 예측이 슬픔으로 치우침
- **권장 사항**: 
  - 영어 컨텐츠에는 부적합
  - **한국어 전용 파이프라인**을 별도 구성할 경우에만 사용
  - 한국어 라벨셋으로 재평가 필요

---

## 평가 실패 모델

### harshit345/xlsr-wav2vec-speech-emotion-recognition
- **결과**: 정확도 0.065, 슬픔 편향
- **이유**: 7개 감정 지원하지만 Simpson 라벨셋과 미스매치

### m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition
- **결과**: 정확도 0.032
- **이유**: 그리스어 데이터셋 학습, 영어 전이 실패

### audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
- **결과**: 정확도 0.645 (모두 neutral 예측)
- **이유**: 차원 기반 출력(arousal, dominance, valence), 이산 감정 매핑 불가

### speechbrain/emotion-recognition-wav2vec2-IEMOCAP
- **결과**: 로드 실패
- **이유**: HuggingFace Transformers 비호환 형식 (SpeechBrain 전용)

---

## 최종 권장 사항

### 영어 컨텐츠용
**1순위**: `superb/wav2vec2-large-superb-er`
- 정확도와 F1 스코어 균형이 가장 좋음
- `config.py`에서 기본 오디오 모델로 설정 권장

**2순위**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` (조건부)
- 하이퍼파라미터 튜닝 후 재평가
- `audio_temperature`를 0.3 이하로 낮추고, `neutral_suppression`을 0.3 이하로 강화

### 한국어 컨텐츠용
- **현재로서는 적합한 모델 없음**
- `jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition`을 한국어 라벨셋으로 재평가 필요
- 또는 다국어 백본 모델(superb)로 한국어 컨텐츠 처리 후 성능 확인

### 멀티언어 지원 전략
1. **단일 모델 방식** (권장):
   - `superb/wav2vec2-large-superb-er` 사용
   - 영어/한국어 모두 처리 가능 (XLS-R 백본)
   - 성능은 영어에서 더 높지만 한국어도 일정 수준 지원

2. **언어별 모델 분기** (차선):
   - 영어: `superb/wav2vec2-large-superb-er`
   - 한국어: `jungjongho/...` (한국어 라벨셋 재평가 후)
   - 언어 감지 로직 필요 (Whisper STT의 언어 감지 활용 가능)

---

## 다음 실험 제안

1. **추가 데이터셋 평가**:
   - 다양한 감정 분포를 가진 영상으로 재평가
   - 한국어 컨텐츠 라벨링 및 평가

2. **하이퍼파라미터 최적화**:
   - `superb` 모델에 대해 `audio_temperature`, `emotion_weights` 그리드 서치
   - `tools/weight_tuner.py` 활용

3. **앙상블 전략**:
   - 텍스트 모델과 결합 시 성능 향상 여부 확인
   - 최적 가중치 탐색

4. **추가 모델 후보 탐색**:
   - `facebook/wav2vec2-large-robust-24-ft-emotion-recognition`
   - `facebook/mms-1b-all` (다국어 지원, 감정 분류 fine-tuning 필요)

---

## 평가 환경

- **Python**: 3.11
- **PyTorch**: GPU (CUDA)
- **Transformers**: Latest
- **평가 도구**: `tools/model_evaluator.py --disable-text`
- **데이터셋**: `labelled_simpson.jsonl` (31 segments)
- **측정 지표**: Accuracy, Macro F1, Neutral Rate, Cross Entropy
