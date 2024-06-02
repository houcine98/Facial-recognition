from django.contrib import admin
from django.urls import path 
from . import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('/session',views.session,name="session"),
    path('patient_details',views.patient_details,name="patient_details"),
    path('',views.record,name="record"),
]
if settings.DEBUG:
    urlpatterns +=static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

