from django.urls import path
from . import views

app_name = 'instructions'

urlpatterns = [
    #path("", views.index, name="index"),
    path("step/<int:step>/", views.instruction_step, name='instruction_step')
]