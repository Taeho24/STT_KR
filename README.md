# STT_KR
준언어적 표현을 시각화하여 포함하는 자막 생성 프로젝트

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

```bash
# 1. 저장소 복제
git clone https://github.com/Taeho24/STT_KR.git
cd STT_KR

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 의존성 설치
pip install -r requirements.txt

# 4. Hugging Face 토큰 설정
echo "your_token_here" > private/hf_token.txt
```

### 3. CUDA 설정 (선택사항)
```bash
# CUDA 11.8 기준
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 사용 방법

### 기본 사용
```bash
python manage.py runserver
```

### 고급 옵션
```bash
python main.py "input_video.mp4" \
    --output_dir "result" \
    --batch_size 8 \
    --device "cuda" \
    --compute_type "float16" \
    --add_to_video
```

# 활용

# Contact
| Maintainer | e-mail |
|---------|---------|
| marmot8080 | marmot8080@gmail.com |
| Taeho24 | teahotiger@gmail.com |
| adap8709 | adap8709@gmail.com |
|||
