from django.urls import path
from .views import IrisClassificationView

urlpatterns = [
    path('classification', IrisClassificationView.as_view(), name='iris_classification'),
]