# utils (bundled core fallback)

이 폴더는 웹사이트가 레포 루트 코어 모듈(`audio_analyzer.py`, `emotion_classifier.py`, `subtitle_generator.py` 등)을 찾지 못할 때 사용되는 "폴백(fallback) 사본"을 담고 있습니다.

권장 사용 순서:
- 기본: 레포지토리 루트의 코어 모듈을 사용합니다. (웹 `settings.py`가 루트를 `sys.path`에 추가)
- 예외: 외부인이 웹 폴더만 다운로드한 경우, 여기의 사본을 자동으로 사용해 기능을 동작시킵니다.

임포트 동작:
- 우선 시도: `from audio_analyzer import AudioAnalyzer` (레포 루트)
- 실패 시: `from .audio_analyzer import AudioAnalyzer` (폴백: 이 폴더)
- `EmotionClassifier`도 동일한 방식으로 동작합니다.

협업자 안내:
- 변경의 기준(소스 오브 트루스)은 레포 루트의 코어 파일입니다. PR/리뷰는 루트 코어를 대상으로 해주세요.
- 이 폴더의 파일은 가능한 한 루트와 동기화 상태를 유지하지만, 독립 배포를 위해 존재합니다.
- 웹 실행 로그에 실제 사용된 모듈 경로가 출력됩니다.
  - 예: `[DEBUG] AudioAnalyzer module path: C:\...\STT_KR-liveCaption\audio_analyzer.py`
  - 폴백 사용 시 경고가 출력됩니다.

문서화/운영 팁:
- 최상 품질을 위해서는 저장소 전체(브랜치 ZIP)로 배포/다운로드를 권장합니다.
- 단독 웹 배포가 필요하면, 이 폴더의 사본이 자동으로 사용됩니다.
