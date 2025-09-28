import os
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .utils.subtitle_generator import SubtitleGenerator
from .tasks import process_and_generate_srt_task
from .utils.config import config
from celery.result import AsyncResult
import uuid

@csrf_exempt
def generate_caption(request):
    if request.method != "POST" or not request.FILES.get("audio"):
        return JsonResponse({"error": "Invalid request"}, status=400)

    try:
        audio_file = request.FILES["audio"]
        
        # 파일명 충돌을 피하기 위해 고유한 파일명 사용
        unique_filename = f"{uuid.uuid4()}_{audio_file.name}"
        audio_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'assets', unique_filename)

        # django 서버는 windows, celery는 리눅스 환경에서 실행했을 경우의 경로 교정 (django, celery, redis 모두 리눅스 환경에서의 실행을 권장)
        wsl_path = audio_path.replace('C:', '/mnt/c').replace('\\', '/')

        # 파일을 서버에 저장
        with open(audio_path, 'wb+') as destination:
            for chunk in audio_file.chunks():
                destination.write(chunk)

        # 스타일 관련 값 수집
        default_font_size = int(request.POST.get("default_font_size", "12"))
        min_font_size = int(request.POST.get("min_font_size", "10"))
        max_font_size = int(request.POST.get("max_font_size", "14"))
        hex_highlight_color = request.POST.get("highlight_color", "#FFFF00")
        ass_highlight_color = config.hex_to_ass(hex_highlight_color)

        hex_emotion_colors = {
            "neutral": request.POST.get("중립", "#FFFFFF"),
            "happy": request.POST.get("행복", "#00FF00"),
            "sad": request.POST.get("슬픔", "#0000FF"),
            "angry": request.POST.get("분노", "#FF0000"),
            "fear": request.POST.get("공포", "#800080"),
            "surprise": request.POST.get("놀람", "#00FFFF"),
            "disgust": request.POST.get("혐오", "#008080"),
        }

        ass_emotion_colors = {
            "neutral": config.hex_to_ass(hex_emotion_colors["neutral"]),
            "happy": config.hex_to_ass(hex_emotion_colors["happy"]),
            "sad": config.hex_to_ass(hex_emotion_colors["sad"]),
            "angry": config.hex_to_ass(hex_emotion_colors["angry"]),
            "fear": config.hex_to_ass(hex_emotion_colors["fear"]),
            "surprise": config.hex_to_ass(hex_emotion_colors["surprise"]),
            "disgust": config.hex_to_ass(hex_emotion_colors["disgust"]),
        }

        config.set(default_font_size, 'font', 'default_size')
        config.set(min_font_size, 'font', 'min_size')
        config.set(max_font_size, 'font', 'max_size')
        config.set(hex_highlight_color, 'hex_colors', 'highlight_color')
        config.set(hex_emotion_colors, 'hex_colors', 'emotion_colors')
        config.set(ass_highlight_color, 'ass_colors', 'highlight_color')
        config.set(ass_emotion_colors, 'ass_colors', 'emotion_colors')

        #==========================================================
        # 비동기 방식
        #==========================================================
        # 오디오 처리
        task = process_and_generate_srt_task.delay(wsl_path)

        # 클라이언트에게 즉시 202 응답과 작업 ID를 전달
        return JsonResponse({"task_id": task.id}, status=202)
        #==========================================================

        #==========================================================
        # 비동기 서버 설정이 되지 않았을 경우 테스트용 기존 방식
        #==========================================================
        # generator = SubtitleGenerator(audio_path=audio_path)
        # generator.process_video()
        # srt_text = generator.generate_srt_subtitle()

        # return HttpResponse(srt_text, content_type="text/plain")
        #==========================================================
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