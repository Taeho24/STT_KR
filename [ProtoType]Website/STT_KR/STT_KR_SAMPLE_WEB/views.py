import os
from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseServerError
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .utils.subtitle_generator import SubtitleGenerator
from .utils.config import config


@csrf_exempt
def generate_caption(request):
    try:
        if request.method == "POST" and request.FILES.get("audio"):
            audio_file = request.FILES["audio"]
            audio_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'assets', 'extracted.wav')

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

            # 오디오 처리
            generator = SubtitleGenerator(audio_path=audio_path)
            srt_text = generator.generate_srt_subtitle()

            return HttpResponse(srt_text, content_type="text/plain")
        return HttpResponse("Invalid request", status=400)
    except Exception as e:
        print("자막 생성 중 오류:", e)
        return HttpResponseServerError("자막 생성 실패")

def index(request):
    return render(request, 'index.html')