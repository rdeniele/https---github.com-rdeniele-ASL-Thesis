from django.urls import path
from . import views

app_name = 'myapp'

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),    #new
    path('detection/video_feed/', views.video_feed, name='video_feed'),
    path('detection/cleanup/', views.cleanup_view, name='cleanup'),
    path('detection/get_prediction/', views.get_prediction, name='get_prediction'),
]
