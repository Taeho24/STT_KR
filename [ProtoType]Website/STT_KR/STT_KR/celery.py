import os
import torch

from celery import Celery
from celery.signals import worker_process_init

from django.conf import settings
from STT_KR_SAMPLE_WEB.utils.model_cache import ModelCache
from STT_KR_SAMPLE_WEB.utils.utils import read_auth_token, read_gemini_api_key

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'STT_KR.settings')

app = Celery('STT_KR_SAMPLE_WEB')

app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()

@worker_process_init.connect
def load_models_on_worker_process_init(sender, **kwargs):
    """Celery 워커가 준비될 때 모델을 미리 로드합니다."""
    print("Celery 워커 시작. 모델을 미리 로드합니다...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"

        hf_token_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'hf_token.txt')
        gemini_api_key_path = os.path.join(settings.BASE_DIR, 'STT_KR_SAMPLE_WEB', 'static', 'private', 'gemini_api_key.txt')

        auth_token = read_auth_token(hf_token_path)
        gemini_api_key = read_gemini_api_key(gemini_api_key_path)

        ModelCache.load_models(device, compute_type, auth_token, gemini_api_key)
    except Exception as e:
        print(f"ERROR: Celery 워커에서 모델 로드 실패 - {e}")

@app.task(bind=True)
def debug_task(self):
    print("Request: {0!r}".format(self.request))