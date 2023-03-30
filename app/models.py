from unicodedata import name
from django.db import models

# Create your models here.
class store(models.Model):

    user=models.TextField(max_length=20)
    response=models.TextField(max_length=20)


from django.contrib.auth.models import User
from django.conf.urls import url
from django.shortcuts import resolve_url


class UserProfileInfo(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    profile_pic = models.ImageField(upload_to='profile_pics', default="1.jpg")

    def __str__(self):
        return self.user.username