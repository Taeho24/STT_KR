# STT_KR
준언어적 표현(감정·속삭임·외침 등)을 시각화하여 포함하는 자막 생성 프로젝트

## 윈도우 원클릭 웹 실행 (권장)

아무것도 몰라도 됩니다. 아래만 하세요.

- 파일 탐색기에서 `[ProtoType]Website\run.ps1` 더블 클릭
- 잠시 후 자동으로 브라우저가 열리고 `http://127.0.0.1:8001/` 접속됩니다.

무엇을 해주나요?

- 저장소 최상위 가상환경(`venv`)이 있으면 그대로 사용해 로컬 파이프라인과 완전 동일한 품질을 보장합니다.
- 없다면 웹 폴더에 `.venv`를 만들고 필요한 패키지를 설치합니다.
- 모델 캐시(HuggingFace/Torch)를 `%LOCALAPPDATA%\stt_kr_cache`로 설정해 디스크를 정리합니다.
- 마이그레이션 실행 후 장고 개발 서버를 새 창에서 시작하고, 준비되면 브라우저를 자동으로 엽니다.

빠른 문제 해결

- 브라우저가 바로 안 뜨면 2~5초 기다렸다가 새로고침(F5)하세요. 또는 주소창에 직접 `http://127.0.0.1:8001/` 입력.
- 8001 포트가 이미 사용 중이면 `run.ps1`에서 포트를 바꾸거나, 수동으로 `python manage.py runserver 127.0.0.1:8002`로 실행하세요.
- 화자 분리나 일부 모델에서 토큰이 필요하면 `private/hf_token.txt` 또는 환경변수 `HUGGINGFACE_TOKEN`를 사용합니다(둘 중 하나면 충분).

# 목차
- [설치](#설치)
- [활용](#활용)
- [Contact](#contact)

## 설치 방법

### 1. 요구 사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항)
- FFmpeg

### 2. 환경 설정

```powershell
# 1) 저장소 복제
git clone https://github.com/Taeho24/STT_KR.git
cd STT_KR

# 2) 가상환경 생성 및 활성화 (Windows PowerShell)
python -m venv venv
venv\Scripts\Activate

# 3) 의존성 설치 (단일 requirements)
pip install -r requirements.txt

# 4) Hugging Face 토큰 설정 (선택: 화자 분리 사용 시)
ni private\hf_token.txt -Value "your_token_here"
```

### 3. CUDA 설정 (권장사항)
```powershell
# CUDA (GPU) 지원 - 중요 안내
# 권장: CUDA Toolkit 12.8 (이 프로젝트의 네이티브 확장(예: ctranslate2 v4.x)이 cuBLAS v12을 필요로 합니다)
# PyTorch GPU 휠은 사용 환경(시스템에 설치된 CUDA 런타임)에 맞게 선택해야 합니다.
# PyTorch 공식 설치 페이지의 Local selector를 사용해 시스템과 CUDA 버전에 맞는 정확한 설치 명령을 복사해 실행하세요:
# https://pytorch.org/get-started/locally/
# 예시(실제 설치 명령은 위 페이지에서 생성한 명령을 사용하세요):
#   pip install torch torchvision torchaudio --index-url <PyTorch wheel index for your CUDA>

# 본 프로젝트는 CUDA Toolkit 12.8 설치를 권장합니다. 설치 후 아래 명령으로 cublas64_12.dll 존재 여부를 확인하세요.
```

### CUDA 설정 및 확인 (GPU 사용 시)

파이프라인을 GPU에서 실행하려면 아래 항목을 확인하세요:

- NVIDIA GPU 및 최신 드라이버 (PowerShell에서 `nvidia-smi`로 확인)
- CUDA Toolkit: 권장 버전 12.8
	- 이 프로젝트에서 사용하는 네이티브 확장(예: `ctranslate2` v4.x)은 cuBLAS v12을 필요로 하며, 시스템 런타임에 맞는 라이브러리(cublas64_12.dll)가 있어야 합니다.
- PyTorch GPU 휠: 시스템에 설치된 CUDA 런타임과 일치하는 휠을 사용해야 합니다.
	- PyTorch 공식 사이트의 로컬 셀렉터에서 운영체제·패키지·CUDA 버전에 맞는 설치 명령을 생성해 사용하세요: https://pytorch.org/get-started/locally/

간단한 확인 방법 (PowerShell):
```powershell
# CUDA 12용 cuBLAS 라이브러리 존재 확인
where.exe cublas64_12.dll

# PyTorch 및 CUDA 사용 가능 여부 확인
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"

# CTranslate2 버전 확인
python -c "import ctranslate2; print(ctranslate2.__version__)"
```

만약 시스템에 CUDA 12.8을 설치할 수 없는 경우, 다음 중 하나를 권장합니다:

- CPU 전용으로 실행 (예: `--device cpu`)
- `nvidia/cuda:12.8-runtime` 같은 CUDA 12.8 기반 Docker 이미지를 사용하여 GPU 실행 환경을 구성

```

## 사용 방법

### 코어 파이프라인 (자막 생성)
```powershell
# 비디오에서 STT + 정렬 + 감정/보이스 타입 분석 → SRT/ASS/JSONL 출력
python .\main.py .\assets\simpson.mp4
# 결과: .\result\simpson.srt, .\result\simpson.ass, .\result\simpson_segments_for_labeling.jsonl
```

# 활용

## 하이퍼파라미터 탐색 도구 (선택 사항)

텍스트/오디오 감정 결합 가중치를 소규모 라벨 데이터로 검증하려면 `tools/weight_tuner.py`를 사용하세요. JSON Lines 형식으로 `text_scores`, `audio_scores`, `label`을 저장한 뒤 아래와 같이 실행하면 정확도, 매크로 F1, 중립 비율 등을 비교할 수 있습니다.

```powershell
python tools/weight_tuner.py labelled_segments.jsonl --audio-weights 0.6 0.7 0.8 --emotion-grid neutral=0.8,0.9,1.0 sad=1.0,1.1,1.2
```

출력된 상위 조합을 `config.py` 혹은 `config.json`에 반영해 멀티모달 앙상블 성능을 조정할 수 있습니다.

라벨만 준비되어 있다면 `--auto` 옵션으로 기본 후보 범위를 자동 탐색할 수도 있습니다.

```powershell
python tools/weight_tuner.py labelled_segments.jsonl --auto
```

필요하다면 `--audio-weights`, `--emotion-grid` 옵션으로 탐색 범위를 좁히거나 넓힐 수 있습니다.

라벨 데이터를 빠르게 만들고 싶다면 `python main.py <video>` 실행 후 생성되는 `result/<video>_segments_for_labeling.jsonl` 파일을
`tools/interactive_labeler.py`에 입력하세요.

```powershell
python tools/interactive_labeler.py --segments result/<video>_segments_for_labeling.jsonl --output labelled_<video>.jsonl --resume
```

각 세그먼트의 문장과 모델 예측을 확인하며 정답 감정을 입력할 수 있고, 숫자(예: `1`=neutral, `2`=happy ...) 또는 영문 감정명을 그대로 입력하면 됩니다. 확신이 없으면 `s`를 눌러 건너뛸 수 있으며, 건너 뛴 세그먼트는 튜너가 자동으로 제외합니다. 결과는 곧바로 튜너 입력으로 사용할 수 있습니다.

## 모델 조합 평가 도구 (선택 사항)

다양한 오디오/텍스트 감정 모델 조합을 라벨 데이터로 비교하려면 `tools/model_evaluator.py`를 사용하세요. 비디오(또는 오디오) 파일과 JSON Lines 형식 라벨을 입력하면, 각 조합의 정확도·매크로 F1·중립 비율을 계산하고 상위 조합을 요약합니다.

```powershell
python tools/model_evaluator.py --video assets/<video>.mp4 --labels labelled_<video>.jsonl --audio-models ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition m3hrdadfi/wav2vec2-base-100k-emo --text-models j-hartmann/emotion-english-distilroberta-base --device auto
```

### 빠른 분석 도구 (선택 사항)
자막 JSONL에서 오디오 기반 상위 감정 점수 범위를 확인하려면:
```powershell
python tools/analyzers/audio_top_scores.py --segments result/<video>_segments_for_labeling.jsonl
```

- `--audio-models`와 `--text-models`를 생략하면 `config.py`에 정의된 기본 모델을 사용합니다.
- `--device auto`는 GPU가 가능하면 `cuda`, 그렇지 않으면 `cpu`를 선택합니다.
- 텍스트 모듈 없이 오디오 모델만 비교하고 싶다면 `--disable-text` 플래그를 추가하면 됩니다. 이 경우 텍스트 감정 모델을 로드하지 않고, 오디오 분류기만 사용해 정확도·F1을 측정합니다.
- 출력에는 혼동 행렬과 조합별 성능 지표가 포함되며, 상위 조합을 `config.py` 또는 실행 구성에 반영할 수 있습니다.
- `config.py`의 `audio_candidates`, `text_candidates` 목록을 업데이트하면, 모델을 인자로 넘기지 않은 경우 자동으로 모든 후보 조합을 평가합니다.

자세한 후보 목록과 특징은 `docs/emotion_model_candidates.md`에서 확인할 수 있습니다.

# Contact
| Maintainer | e-mail |
|---------|---------|
| marmot8080 | marmot8080@gmail.com |
| Taeho24 | teahotiger@gmail.com |
| adap8709 | adap8709@gmail.com |
|||
