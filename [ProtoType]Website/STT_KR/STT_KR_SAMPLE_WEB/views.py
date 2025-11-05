import os
import json
import uuid

from django.shortcuts import render
import traceback
from django.http import JsonResponse, HttpResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

# Celery is optional. Import only if available and enabled.
try:
    from celery.result import AsyncResult  # type: ignore
except Exception:  # ImportError or others
    AsyncResult = None  # type: ignore
from .utils.config import config
from .utils.subtitle_config import set_subtitle_settings
from .utils.subtitle_generator import SubtitleGenerator

@csrf_exempt
def generate_caption(request):
    if request.method != "POST" or not request.FILES.get("audio"):
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        audio_file = request.FILES["audio"]
        file_format = request.POST.get("file_format", "srt").strip().lower()

        # 파일명 충돌을 피하기 위해 고유한 파일명 사용
        id = uuid.uuid4()
        unique_filename = f"{id}_{audio_file.name}"
        audio_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'tmp', 'asset', unique_filename)

        # 업로드 대상 디렉토리 보장
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        # django 서버는 windows, celery는 리눅스 환경에서 실행했을 경우의 경로 교정 (django, celery, redis 모두 리눅스 환경에서의 실행을 권장)
        wsl_path = audio_path.replace('C:', '/mnt/c').replace('\\', '/')

        # 파일을 서버에 저장
        with open(audio_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # 고유명사 값 수집
        # 고유명사 값 수집(JSON 문자열 또는 리스트 허용)
        proper_nouns = request.POST.get("proper_nouns", [])
        if isinstance(proper_nouns, str) and proper_nouns:
            try:
                proper_nouns = json.loads(proper_nouns)
            except Exception:
                # 파싱 실패 시 안전하게 비움
                proper_nouns = []
        if not isinstance(proper_nouns, list):
            proper_nouns = []

        # 프론트엔드 main.js는 감정 색상 키를 영어로 전송합니다.
        # 사용자 입력 색상값(#RRGGBB)을 검증/정규화합니다. 유효하지 않으면 기본값 사용.
        def norm_hex_color(v, default):
            if not v:
                return default
            v = v.strip()
            if v.startswith('#') and (len(v) == 7 or len(v) == 4):
                return v
            # Allow hex without '#'
            if len(v) == 6:
                return f"#{v}"
            return default

        # 누락 시 안전한 기본값을 사용합니다.
        subtitle_settings = {
            "font": {
                "default_size": int(request.POST.get("default_font_size", "24")),
                "min_size": int(request.POST.get("min_font_size", "20")),
                "max_size": int(request.POST.get("max_font_size", "28"))
            },
            "hex_colors": {
                "emotion_colors": {
                    "neutral": norm_hex_color(request.POST.get("neutral", "#FFFFFF"), "#FFFFFF"),
                    "happy": norm_hex_color(request.POST.get("happy", "#00FF00"), "#00FF00"),
                    "sad": norm_hex_color(request.POST.get("sad", "#0000FF"), "#0000FF"),
                    "angry": norm_hex_color(request.POST.get("angry", "#FF0000"), "#FF0000"),
                    "fear": norm_hex_color(request.POST.get("fear", "#800080"), "#800080"),
                    "surprise": norm_hex_color(request.POST.get("surprise", "#00FFFF"), "#00FFFF"),
                    "disgust": norm_hex_color(request.POST.get("disgust", "#008080"), "#008080"),
                },
                # default_color는 보통 중립과 동일하게 설정
                "default_color": norm_hex_color(request.POST.get("neutral", "#FFFFFF"), "#FFFFFF"),
                "highlight_color": norm_hex_color(request.POST.get("highlight_color", "#FFFF00"), "#FFFF00")
            }
        }

        # 성능/기능 토글 수집(기본: 안전·보수적으로 ON)
        enable_diarization = request.POST.get("enable_diarization", "1") in ("1", "true", "True")
        enable_ser = request.POST.get("enable_ser", "1") in ("1", "true", "True")
        enable_temporal_smoothing = request.POST.get("enable_temporal_smoothing", "1") in ("1", "true", "True")
        fast_mode = request.POST.get("fast_mode", "0") in ("1", "true", "True")

        # 비동기/동기 전환: settings.USE_CELERY 플래그로 제어 (기본 False - 개발 편의)
        if getattr(settings, 'USE_CELERY', False):
            # 비동기 방식: 필요할 때만 import하여 무거운 의존성 로딩을 지연
            from .tasks import process_and_generate_srt_task
            task = process_and_generate_srt_task.delay(audio_path=wsl_path, subtitle_settings=subtitle_settings, proper_nouns=proper_nouns)
            return JsonResponse({"task_id": task.id}, status=202)
        else:
            # 동기 처리 (개발/테스트 용)
            generator = SubtitleGenerator(
                audio_path=audio_path,
                enable_diarization=enable_diarization,
                enable_ser=enable_ser,
                enable_temporal_smoothing=enable_temporal_smoothing,
                fast_mode=fast_mode
            )
            generator.process_video()
            # generator 내부 id(파일 베이스명) 기준으로 설정 파일을 저장해야
            # 이후 SRT 생성 시 일관되게 로딩됩니다.
            set_subtitle_settings(generator.id, subtitle_settings)
            if proper_nouns:
                generator.modify_proper_nouns(proper_nouns)
            if file_format == 'ass':
                ass_text = generator.generate_ass_subtitle()
                resp = HttpResponse(ass_text, content_type="text/plain; charset=utf-8")
                # 프론트엔드에서 생성 결과 요약을 조회할 수 있도록 ID/형식 헤더 제공
                resp["X-Subtitle-ID"] = generator.id
                resp["X-Format"] = "ass"
                return resp
            else:
                srt_text = generator.generate_srt_subtitle()
                resp = HttpResponse(srt_text, content_type="text/plain; charset=utf-8")
                resp["X-Subtitle-ID"] = generator.id
                resp["X-Format"] = "srt"
                return resp
    except Exception as e:
        print("자막 생성 중 오류:", e)
        traceback.print_exc()
        return HttpResponseServerError("자막 생성 실패")

@csrf_exempt
def get_caption_status(request, task_id):
    # If Celery is not installed/used, return 404 for status endpoint
    if not getattr(settings, 'USE_CELERY', False) or AsyncResult is None:
        return JsonResponse({"error": "Celery not enabled"}, status=404)

    result = AsyncResult(task_id)  # type: ignore

    if result.ready(): # 작업이 완료되었는지 확인
        if result.successful(): # 작업이 성공했는지 확인
            srt_text = result.get() # 결과를 가져옴
            return HttpResponse(srt_text, content_type="text/plain; charset=utf-8")
        else: # 작업 실패
            return HttpResponseServerError("자막 생성 실패: 태스크 오류")
    else: # 작업이 아직 진행 중
        return JsonResponse({"status": "processing"}, status=200)

def index(request):
    return render(request, 'index.html')

def _load_segments_json(base_id: str):
    try:
        out_dir = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'tmp', 'result')
        json_path = os.path.join(out_dir, f"{base_id}_segments.json")
        if not os.path.exists(json_path):
            return None
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None

@csrf_exempt
def get_caption_summary(request):
    """세그먼트 JSON을 기반으로 간단한 분석 요약을 반환합니다.
    Query: ?id=<generator.id>
    Returns: {
      id, total_segments, emotions: {label: {count, pct}}, voice_types: {whisper/normal/shout}
    }
    """
    base_id = request.GET.get('id')
    if not base_id:
        return JsonResponse({"error": "missing id"}, status=400)
    segments = _load_segments_json(base_id)
    if segments is None:
        return JsonResponse({"error": "not found"}, status=404)

    total = len(segments) if isinstance(segments, list) else 0
    emotions = {}
    vtypes = {"whisper": 0, "normal": 0, "shout": 0}
    for s in segments or []:
        em = s.get('emotion', 'unknown')
        emotions[em] = emotions.get(em, 0) + 1
        vt = s.get('voice_type', 'normal')
        if vt not in vtypes:
            vt = 'normal'
        vtypes[vt] += 1

    # 비율 계산
    def pct(n):
        return round((n / total * 100.0), 1) if total > 0 else 0.0

    emotions_pct = {
        k: {"count": v, "pct": pct(v)} for k, v in sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    }
    voice_types_pct = {
        k: {"count": v, "pct": pct(v)} for k, v in vtypes.items()
    }

    return JsonResponse({
        "id": base_id,
        "total_segments": total,
        "emotions": emotions_pct,
        "voice_types": voice_types_pct,
    }, status=200)