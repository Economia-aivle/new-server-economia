# blog/urls.py
from django.urls import path
from django.contrib import admin
from . import views

app_name = 'educations'
urlpatterns = [
    path('level_choice/<int:characters>/<int:subjects_id>/<str:chapter>/', views.level_choice, name='level_choice'),
    path('chapter_summary', views.chapter_summary, name='chapter_summary'),
    path('blank/<int:characters>/<int:subjects_id>/<int:chapter>/<int:num>', views.blank, name='blank'),
    path('multiple/<int:characters>/<int:subjects_id>/<int:chapter>/<int:num>', views.multiple, name='multiple'),
    path('tf_quiz/', views.tf_quiz_view, name='tf_quiz'),
    path('tf_quiz/<int:question_id>/', views.tf_quiz_view, name='tf_quiz_detail'),
    path('tf_quiz_page/<int:subjects_id>/<int:chapter>', views.tf_quiz_page, name='tf_quiz_page'),
    path('choose_tf_chapter/', views.choose_tf_chapter_view, name='choose_tf_chapter'),
    path('previous_quiz_answer', views.previous_quiz_answer, name='previous_quiz_answer'),
    path('previous_quiz/<int:characters>', views.previous_quiz, name='previous_quiz'),
    path('study/<int:subjects_id>/', views.study, name='study'),
    path('wrong_explanation', views.wrong_explanation, name='wrong_explanation'),
    path('study_video/', views.study_view, name='study_video'),
    path('summary_anime', views.summary_anime, name='summary_anime'),
    path('blankdatas/<int:characters>', views.getBlankDatas, name="BlankDatas"),
    path('multipledatas/<int:characters>', views.getMultipleDatas, name="MultipleDatas"),
    path('tfdatas/<int:characters>', views.getTfDatas, name="TfDatas"),
    path('getSubjectDatas/<str:subjects>/', views.getSubjectDatas, name='get_subject_datas'),
    path('chapter/<str:subjects>/', views.chapter, name='chapter'),
    path('getStageDatas/<int:characters>/', views.getStageDatas, name='StageDatas'),
    path('update_stage/', views.update_stage, name='update_stage'),
    path('success', views.success, name='success'),
]
