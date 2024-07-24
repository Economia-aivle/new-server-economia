from django.contrib import admin
from django.urls import path, include
from django.shortcuts import render
from django.urls import include
from django.conf import settings
from . import views
from django.conf.urls.static import static
from django.contrib.auth.views import LogoutView
from rest_framework_simplejwt.views import (TokenObtainPairView, TokenRefreshView)

def index(request):
    return render(request, 'index.html')

def chapter(request):
    return render(request, 'chapter.html')

def home(request):
    return render(request, 'home.html')

def mypage(request):
    return render(request, 'mypage.html')

def onboarding(request):
    return render(request, 'onboarding.html')

def update_info(request):
    return render(request, 'update_info.html')

def admin_dashboard(request):
    return render(request,'admin_dashboard.html')
def admin_login(request):
    return render(request,'admin_login.html')

def ask(request):
    return render(request,'ask.html')
def service_agree(request):
    return render(request,'terms/service_agree.html')
def account_term(request):
    return render(request,'terms/account_term.html')
def community_term(request):
    return render(request,'terms/community_term.html')
def service_term(request):
    return render(request,'terms/service_term.html')
def teen_term(request):
    return render(request,'terms/teen_term.html')



urlpatterns = [
    path("", views.onboarding, name='onboarding'),  # Combined and named "home"
    path('users/', include('users.urls')),
    path('educations/', include('educations.urls')),
    path('scenarios/', include('scenarios.urls')),
    path("chapter", views.chapter),
    path("home/<int:subject_id>", views.home, name='home'),
    path("mypage", views.mypage, name='mypage'),
    path("onboarding/re", views.LoginView.as_view(), name='login'),
    path("onboarding/logout", views.LogoutView.as_view(), name='logout'),
    path("update_info", views.update_info),
    path('admin/', admin.site.urls),
    path('community_term/', community_term, name='community_term'),
    path('account_term/', account_term, name='account_term'),
    path('service_term/', service_term, name='service_term'),
    path('teen_term/', teen_term, name='teen_term'),
    path('service_agree/', service_agree, name='service_agree'),
    path('ask', ask, name='ask'),
    path('register', views.register, name='register'),
    
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
