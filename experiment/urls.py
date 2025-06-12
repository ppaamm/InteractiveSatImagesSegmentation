from django.urls import path
from . import views

app_name = 'experiment'

urlpatterns = [
    path('select/', views.image_selection, name='select'),
    path('', views.index, name='index'),
    path('load_segmentation/', views.load_segmentation, name='load_segmentation'),
]