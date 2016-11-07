from django.conf.urls import patterns, include, url

from django.contrib import admin
from app.views import *
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'alert_server.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^login/',testLogin),
    url(r'^stations/',get_locat),
    url(r'^feedback/',feedbackbody),   
    url(r'^alert/',get_alert_information),
    url(r'^device/',get_device_infomation),
    url(r'^chart/',get_for_chart),
    url(r'^alert_search/',alert_search),
    url(r'^position/',get_position),
    
)
