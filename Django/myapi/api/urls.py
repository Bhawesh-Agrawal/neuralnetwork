from django.urls import path
from .views import linear_regression

urlpatterns = [
    path('linear/', linear_regression, name = "linear_regression"),
]

