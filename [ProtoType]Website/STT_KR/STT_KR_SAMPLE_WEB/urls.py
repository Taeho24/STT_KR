from django.urls import path
from STT_KR_SAMPLE_WEB import views

urlpatterns = {
    path('',views.index, name='index'),
}
