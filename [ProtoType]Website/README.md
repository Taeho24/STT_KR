# Prototype Website – One-click Run (Windows)

가장 쉬운 실행 방법:

1) 파일 탐색기에서 이 폴더의 `run.ps1`을 더블 클릭합니다.
2) 첫 실행 시 필요한 패키지를 설치하고, 마이그레이션을 적용합니다.
3) 장고 개발 서버가 `127.0.0.1:8001`에서 시작되며, 준비되면 브라우저가 자동으로 열립니다.

동작 원리(요약):

- 저장소 루트의 `venv`가 있으면 그대로 사용하여 로컬 파이프라인과 패키지 버전을 일치시킵니다.
- 루트 `venv`가 없으면 이 폴더에 `.venv`를 만들고 `requirements.txt`를 설치합니다.
- 모델 캐시를 `%LOCALAPPDATA%\stt_kr_cache`로 설정합니다.
- `manage.py migrate` 후 `manage.py runserver 127.0.0.1:8001`를 새 프로세스에서 실행합니다.
- 서버가 뜨면 기본 브라우저에서 자동으로 `http://127.0.0.1:8001/`을 엽니다.

문제 해결:

- 8001 포트가 사용 중이면 `run.ps1`의 포트를 변경하세요(예: 8002).
- 브라우저가 자동으로 안 열리면 수동으로 `http://127.0.0.1:8001/`을 여세요.
- 화자 분리 등에 필요한 토큰은 `../../private/hf_token.txt` 또는 환경변수 `HUGGINGFACE_TOKEN`로 설정합니다.

수동 실행(선택):

아래는 필요할 때만 사용하세요.

```powershell
python -m venv .venv
.venv\Scripts\Activate
pip install -r requirements.txt
cd STT_KR
python manage.py migrate
python manage.py runserver 127.0.0.1:8001
```