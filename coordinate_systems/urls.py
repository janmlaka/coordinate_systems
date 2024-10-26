from django.urls import path
from . import views

urlpatterns = [
    path('D48/', views.D48_GK, name="D48_GK"),
    path('D96/', views.D96_TM, name="D96_TM"),
    path('UTM/', views.UTM, name="UTM"),
    path("", views.starting_page, name="starting-page"),
    path('Transformations/', views.upload_files, name='upload_files'),
    path('success/', views.success_view, name='success'),
    path('Transformations/success/', views.success_view, name='coordinates'),
    path('Transformations/success/download', views.download_text_file, name='download_text'),
    path('Transformations/success/2Dplot', views.differences_2d, name='plot_data'),
]
