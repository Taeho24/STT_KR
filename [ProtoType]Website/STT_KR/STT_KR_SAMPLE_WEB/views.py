import os
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .utils import video_to_srt


@csrf_exempt
def generate_caption(request):
    try:
        if request.method == "POST" and request.FILES.get("audio"):
            audio_file = request.FILES["audio"]
            audio_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'assets', 'extracted.wav')

            with open(audio_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            # 오디오 처리
            video_to_srt.process_video(video_path=audio_path)

            # SRT 읽기
            with open(video_to_srt.output_path, 'r', encoding='utf-8') as f:
                srt_text = f.read()

            return HttpResponse(srt_text, content_type="text/plain")
        return HttpResponse("Invalid request", status=400)
    except Exception as e:
        print("자막 생성 중 오류:", e)
        return HttpResponseServerError("자막 생성 실패")

def index(request):
    return render(request, 'index.html')