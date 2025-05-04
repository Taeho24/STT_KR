"""
URL configuration for STT_KR project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from STT_KR_SAMPLE_WEB import views
# Use static() to add url mapping to serve static files during development (only)
from django.conf.urls.static import static
from django.conf import settings
#Add URL maps to redirect the vase URL to our application
from django.views.generic import RedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('STT/', include('STT_KR_SAMPLE_WEB.urls')),
    path('', RedirectView.as_view(url='/STT/', permanent=True)),
]

urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)