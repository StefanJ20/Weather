from django.urls import path #type: ignore
from . import views  # type: ignore

urlpatterns = [
    path('', views.index, name='index'),
    path("api/highest-full/", views.api_highest_full, name="api_highest_full"),
    path("api/weather-ai-impression/", views.weather_ai_impression, name="weather_ai_impression"),
]