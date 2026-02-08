from django.urls import path
from . import views  # type: ignore

urlpatterns = [
    path('', views.index, name='index'),
    path("api/highest-full/", views.api_highest_full, name="api_highest_full"),
]