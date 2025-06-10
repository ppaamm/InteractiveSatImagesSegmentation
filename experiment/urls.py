from django.urls import path
from . import views

app_name = 'experiment'

urlpatterns = [
    path('', views.index, name='index'),
    path('select/', views.image_selection, name='image_selection'),
]