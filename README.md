# STT_KR
준언어적 표현을 시각화하여 포함하는 자막 생성 프로젝트

## 목차
- [설치 방법](#설치-방법)
- [Contact](#contact)

## 설치 방법

### 1. 요구 사항

- Python 3.8 이상
- CUDA 지원 GPU (선택사항)
- FFmpeg

### 2. 환경 설정

#### 1) 저장소 복제
```bash
git clone https://github.com/Taeho24/STT_KR.git
cd STT_KR
```
#### 2) 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
#### 3) 의존성 설치
```bash
pip install -r requirements.txt
```
#### 4) Hugging Face 토큰 설정
```bash
echo "your_token_here" > private/hf_token.txt
```

### 3. CUDA 설정 (권장사항)

- CUDA (GPU) 지원 - 중요 안내
- 권장: CUDA Toolkit 12.8 (이 프로젝트의 네이티브 확장(예: ctranslate2 v4.x)이 cuBLAS v12을 필요로 합니다)
- PyTorch GPU 휠은 사용 환경(시스템에 설치된 CUDA 런타임)에 맞게 선택해야 합니다.
	- [PyTorch](https://pytorch.org/get-started/locally/) 공식 설치 페이지의 Local selector를 사용해 시스템과 CUDA 버전에 맞는 정확한 설치 명령을 복사해 실행하세요:
- https://pytorch.org/get-started/locally/
- 예시(실제 설치 명령은 위 페이지에서 생성한 명령을 사용하세요):
	- pip install torch torchvision torchaudio --index-url <PyTorch wheel index for your CUDA>

* 본 프로젝트는 CUDA Toolkit 12.8 설치를 권장합니다. 설치 후 아래 명령으로 cublas64_12.dll 존재 여부를 확인하세요.


#### CUDA 설정 및 확인 (GPU 사용 시)

파이프라인을 GPU에서 실행하려면 아래 항목을 확인하세요:

- NVIDIA GPU 및 최신 드라이버 (PowerShell에서 `nvidia-smi`로 확인)
- CUDA Toolkit: 권장 버전 12.8
	- 이 프로젝트에서 사용하는 네이티브 확장(예: `ctranslate2` v4.x)은 cuBLAS v12을 필요로 하며, 시스템 런타임에 맞는 라이브러리(cublas64_12.dll)가 있어야 합니다.
- PyTorch GPU 휠: 시스템에 설치된 CUDA 런타임과 일치하는 휠을 사용해야 합니다.
	- [PyTorch](https://pytorch.org/get-started/locally/) 공식 사이트의 로컬 셀렉터에서 운영체제·패키지·CUDA 버전에 맞는 설치 명령을 생성해 사용하세요: https://pytorch.org/get-started/locally/

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

---

### 기본 사용 방법
```bash
# manage.py 가 포함된 디텍토리로 이동
# cd [directory containing manage.py]
python manage.py runserver
```

# Contact
| Maintainer | e-mail |
|---------|---------|
| marmot8080 | marmot8080@gmail.com |
| Taeho24 | teahotiger@gmail.com |
| adap8709 | adap8709@gmail.com |
