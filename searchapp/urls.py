from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.IndexView.as_view(), name='index_url'),
    path('search/', views.SearchView.as_view(), name='search_url'),
]
