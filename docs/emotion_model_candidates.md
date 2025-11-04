# 감정 분류 모델 후보 정리

이 문서는 STT_KR 파이프라인과 궁합이 좋은 감정 분류 모델 후보를 정리한 것입니다. 실제 성능은 데이터셋·화자·녹음 품질에 따라 달라지므로, `tools/model_evaluator.py`로 프로젝트에 맞는 지표를 직접 측정하세요.

## 오디오 기반 감정 분류 모델 후보

| 모델명 | 제공처 / 다운로드 | 라이선스* | 지원 언어 | 권장 사용 시나리오 | 파이프라인 통합 상태 |
| --- | --- | --- | --- | --- | --- |
| `superb/wav2vec2-large-superb-er` | Hugging Face (SUPERB 벤치마크 팀) | Apache-2.0 | EN | Wav2Vec2 Large, IEMOCAP 기반 4클래스 | `EmotionClassifier`에서 바로 사용 가능 |
| `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | Hugging Face (에릭 칼라브레스) | Apache-2.0 | EN | 현재 기본값, 중립 편향이 있어 재평가 필요 | 기본값 |
| `superb/hubert-large-superb-er` | Hugging Face (SUPERB) | Apache-2.0 | EN | HuBERT Large, 노이즈 견고성 높음 | `EmotionClassifier`에서 바로 사용 가능 |
| `jungjongho/wav2vec2-xlsr-korean-speech-emotion-recognition` | Hugging Face (정종호) | Apache-2.0 | KO | KES Dataset 기반, 한국어 감정 인식 | `EmotionClassifier`에서 바로 사용 가능 |
| `firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3` | Hugging Face (Firdhokk, 2024) | Apache-2.0 | EN 중심(다국어 데이터 일부) | Whisper Large V3 기반 7감정 분류 | `EmotionClassifier`에서 바로 사용 가능 (고 VRAM 권장) |
| `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | SpeechBrain Hub (HF mirror) | Apache-2.0 | EN | IEMOCAP 기반. `speechbrain` 인터페이스 필요 | **추가 래퍼 필요** (아래 참고) |
| `audEERING/DeepSpectrumLite (EffNet-B0 FER 체크포인트)` | GitHub Release / pip `deepspectrumlite` | Apache-2.0 | 다국어 | 모바일 기기 배포, 저자원 장치 | **추가 래퍼 필요** (스펙트로그램 기반)

\* 라이선스 표기는 2025-10-17 기준 공개 자료를 바탕으로 한 요약입니다. 실제 사용 전 각 저장소의 최신 라이선스를 확인하세요.

### 추가 연동 메모
- `speechbrain/*` 모델은 `speechbrain.pretrained.EncoderClassifier` API로 로드해야 하며, 현재 코드에는 별도 래퍼가 필요합니다.
- `DeepSpectrumLite`는 스펙트로그램을 CNN에 입력하는 구조이므로, WAV를 임시 파일로 변환한 뒤 CLI 혹은 Python API 호출이 필요합니다.
- `openSMILE` 기반 전통 특징 + SVM 모델(예: EmoBase 2010)도 무료로 사용 가능하지만 실행 바이너리 설치와 특징 추출 파이프라인이 필요해, 이번 표에서는 제외했습니다.

## 텍스트 기반 감정 분류 모델 후보

| 모델명 | 제공처 / 다운로드 | 라이선스 | 지원 언어 | 특징 | 활용 팁 |
| --- | --- | --- | --- | --- | --- |
| `j-hartmann/emotion-english-distilroberta-base` | Hugging Face | MIT | EN | 경량, 현재 기본값 | 1~2문장 길이의 자막에 적합 |
| `SamLowe/roberta-base-go_emotions` | Hugging Face | MIT | EN | Google GoEmotions 27클래스 기반 | 파이프라인에서 7개 감정으로 재매핑 필요 |

## 다음 단계
1. `config.py`의 `audio_candidates`, `text_candidates` 목록을 위 후보로 갱신하세요 (일부는 래퍼 개발 완료 후 추가).
2. `tools/model_evaluator.py`로 후보 조합을 평가하고 로그를 남기세요.
3. 필요한 경우 `speechbrain` 또는 `DeepSpectrumLite` 모델용 추론 래퍼를 구현해 자동 비교 범위를 확장하세요.
