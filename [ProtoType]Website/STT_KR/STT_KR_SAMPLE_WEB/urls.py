from django.urls import path
from STT_KR_SAMPLE_WEB import views

urlpatterns = [
    path('',views.index, name='index'),
    path('generate-caption/', views.generate_caption, name='generate_caption'),
    path('task-list/<int:user_id>/', views.get_task_list, name='get_task_list'),
    path('task-detail/<str:task_id>/', views.task_detail, name='task_detail'),
    path('status/<str:task_id>/', views.get_task_status, name='get_task_status'),
    path('delete-task/<str:task_id>/', views.delete_task, name='delete_task'),
]
