from django.contrib import admin
from django.urls import path, include
from django.contrib.auth import views as auth_views
from myapp import views as myapp_views

urlpatterns = [
    path('admin/', admin.site.urls),  # Admin site
    path('register/', myapp_views.register, name='register'),  # Register view
    path('login/', auth_views.LoginView.as_view(template_name='login.html'), name='login'),  # Login view
    path('logout/', myapp_views.user_logout, name='logout'),
    path('', include('myapp.urls')),  # Include myapp's URLs
]