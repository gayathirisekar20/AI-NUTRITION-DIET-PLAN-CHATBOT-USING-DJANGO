"""project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import url
from django.urls import path
from app.views import insert,addcommant,input,bmi,bmi_cal,food,img,chat,diet,ProfileDetailView,user_logout, register,user_login,login

urlpatterns = [
    # url(r'^register/$', register, name='register'),
    # url(r'^logout/$',user_logout,name='logout'),
    url(r'^profile/(?P<pk>\d+)/$', ProfileDetailView.as_view(), name='profile'),
    path('login/', login, name='login'),
    path('user_login/', user_login, name='login'),
    path("",register, name='register'),
    # path("profile/",ProfileDetailView.as_view(), name='register'),
    path("logout/",user_logout, name='logout'),
    path('bmi/',bmi),
    path('addcommant',addcommant),
    path('chat/insert',insert),
    path('admin/', admin.site.urls),
    path('input',input),
    path('bmi_cal',bmi_cal),
    path('food/',food),
    path('food/img',img),
    path('chat/',chat),
    path('diet/',diet),
]
