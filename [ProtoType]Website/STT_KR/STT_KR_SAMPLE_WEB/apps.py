import sys
import os
import torch

from django.apps import AppConfig
from .utils.model_cache import ModelCache
from .utils.utils import read_auth_token

class SttKrSampleWebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'STT_KR_SAMPLE_WEB'

    def ready(self):
        # 서버 시작 시에만 모델을 로드
        # 'manage.py makemigrations' 같은 다른 명령어 실행 시에는 로드되지 않도록 설정
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv or 'uwsgi' in sys.argv:
            from django.conf import settings
            hf_token_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'hf_token.txt')
            gemini_api_key_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'gemini_api_key.txt')
            auth_token = read_auth_token(hf_token_path)
            gemini_api_key = read_gemini_api_key()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "float32"

            ModelCache.load_models(device, compute_type, auth_token, gemini_api_key)