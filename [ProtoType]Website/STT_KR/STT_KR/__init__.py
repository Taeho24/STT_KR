try:
	from django.conf import settings
	if getattr(settings, 'USE_CELERY', False):
		from .celery import app as celery_app
		__all__ = ['celery_app']
	else:
		# Celery 미사용 시, 더미 심볼만 노출하여 Django 시작 시 무거운 의존성 로딩을 피함
		celery_app = None
		__all__ = ['celery_app']
except Exception:
	# settings 초기화 전 import될 수도 있으므로 안전 가드
	celery_app = None
	__all__ = ['celery_app']