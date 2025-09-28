from django.urls import path
from STT_KR_SAMPLE_WEB import views

urlpatterns = [
    path('',views.index, name='index'),
    path('generate-caption/', views.generate_caption, name='generate_caption'),
    path('status/<str:task_id>/', views.get_caption_status, name='get_caption_status'),
]
