from django.apps import AppConfig

class SttKrSampleWebConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'STT_KR_SAMPLE_WEB'

    #===================================================================================================================================
    # 비동기 서버 설정이 되지 않았을 경우 테스트용 기존 방식
    #===================================================================================================================================
    # import sys
    # import os
    # import torch
    # 
    # from django.conf import settings
    # from .utils.model_cache import ModelCache
    # from .utils.utils import read_auth_token, read_gemini_api_key
    # 
    # def ready(self):
    #     # 'runserver', 'gunicorn', 'uwsgi' 명령어일 때만 모델 로드
    #     if 'runserver' in sys.argv or 'gunicorn' in sys.argv or 'uwsgi' in sys.argv:
    #         # os.environ.get('RUN_MAIN')을 사용하여 실제 서버 프로세스에서만 모델을 로드하도록 처리
    #         if os.environ.get('RUN_MAIN') != 'true':
    #             return
    #         
    #         hf_token_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'hf_token.txt')
    #         gemini_api_key_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'gemini_api_key.txt')
    #         auth_token = read_auth_token(hf_token_path)
    #         gemini_api_key = read_gemini_api_key(gemini_api_key_path)
    # 
    #         device = "cuda" if torch.cuda.is_available() else "cpu"
    #         compute_type = "float16" if device == "cuda" else "float32"
    # 
    #         ModelCache.load_models(device, compute_type, auth_token, gemini_api_key)
    #===================================================================================================================================