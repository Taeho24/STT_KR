import os
import uuid

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from celery.result import AsyncResult

from .tasks import process_and_generate_srt_task
from .utils.config import config
from .utils.subtitle_config import set_subtitle_settings
from .utils.subtitle_generator import SubtitleGenerator

@csrf_exempt
def generate_caption(request):
    if request.method != "POST" or not request.FILES.get("audio"):
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        audio_file = request.FILES["audio"]

        # 파일명 충돌을 피하기 위해 고유한 파일명 사용
        id = uuid.uuid4()
        unique_filename = f"{id}_{audio_file.name}"
        audio_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'tmp', 'asset', unique_filename)

        # django 서버는 windows, celery는 리눅스 환경에서 실행했을 경우의 경로 교정 (django, celery, redis 모두 리눅스 환경에서의 실행을 권장)
        wsl_path = audio_path.replace('C:', '/mnt/c').replace('\\', '/')

        # 파일을 서버에 저장
        with open(audio_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # 고유명사 값 수집
        proper_nouns = request.POST.get("proper_nouns", [])

        # 프론트엔드 main.js는 감정 색상 키를 영어로 전송합니다.
        # 누락 시 안전한 기본값을 사용합니다.
        subtitle_settings = {
            "font": {
                "default_size": int(request.POST.get("default_font_size", "24")),
                "min_size": int(request.POST.get("min_font_size", "20")),
                "max_size": int(request.POST.get("max_font_size", "28"))
            },
            "hex_colors": {
                "emotion_colors": {
                    "neutral": request.POST.get("neutral", "#FFFFFF"),
                    "happy": request.POST.get("happy", "#00FF00"),
                    "sad": request.POST.get("sad", "#0000FF"),
                    "angry": request.POST.get("angry", "#FF0000"),
                    "fear": request.POST.get("fear", "#800080"),
                    "surprise": request.POST.get("surprise", "#00FFFF"),
                    "disgust": request.POST.get("disgust", "#008080"),
                },
                # default_color는 보통 중립과 동일하게 설정
                "default_color": request.POST.get("neutral", "#FFFFFF"),
                "highlight_color": request.POST.get("highlight_color", "#FFFF00")
            }
        }

        # 비동기/동기 전환: settings.USE_CELERY 플래그로 제어 (기본 False - 개발 편의)
        if getattr(settings, 'USE_CELERY', False):
            # 비동기 방식
            task = process_and_generate_srt_task.delay(audio_path=wsl_path, subtitle_settings=subtitle_settings, proper_nouns=proper_nouns)
            return JsonResponse({"task_id": task.id}, status=202)
        else:
            # 동기 처리 (개발/테스트 용)
            generator = SubtitleGenerator(audio_path=audio_path)
            generator.process_video()
            set_subtitle_settings(id, subtitle_settings)
            if proper_nouns:
                generator.modify_proper_nouns(proper_nouns)
            srt_text = generator.generate_srt_subtitle()
            return HttpResponse(srt_text, content_type="text/plain; charset=utf-8")
    except Exception as e:
        print("자막 생성 중 오류:", e)
        return HttpResponseServerError("자막 생성 실패")

@csrf_exempt
def get_caption_status(request, task_id):
    result = AsyncResult(task_id)

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