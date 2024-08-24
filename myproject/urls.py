from django.contrib import admin
from django.urls import path, include
from myapp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # Home page view
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),  # Ensure this is using the correct view
    path('logout/', views.user_logout, name='logout'),
]
