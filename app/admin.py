from django.contrib import admin

# Register your models here.

from app.models import UserProfileInfo

class UserProfile(admin.ModelAdmin):
    list_display = ('user', 'profile_pic')
admin.site.register(UserProfileInfo,UserProfile)
# Register your models here.
