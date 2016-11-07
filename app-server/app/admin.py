from django.contrib import admin
from app.models import AlertappLogin
# Register your models here.
class AlertappLoginAdmin(admin.ModelAdmin):
    list_display = ('id','user_id','first_login','last_login','time_length','is_paid')

admin.site.register(AlertappLogin,AlertappLoginAdmin)