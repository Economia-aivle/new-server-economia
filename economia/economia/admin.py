from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Player, NoticeBoard, Qna
from django import forms

class PlayerAdmin(UserAdmin):
    model = Player
    list_display = ('player_id', 'player_name', 'nickname', 'email', 'is_staff', 'is_active', 'is_superuser')
    search_fields = ('player_id', 'player_name', 'nickname', 'email')
    ordering = ('player_id',)

    # 필터와 관련된 부분 수정
    list_filter = ('is_staff', 'is_superuser', 'is_active')  # 'groups' 제거
    filter_horizontal = ()  # 'groups', 'user_permissions' 제거

    fieldsets = (
        (None, {'fields': ('player_id', 'password')}),
        ('Personal info', {'fields': ('player_name', 'nickname', 'email', 'school')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser')}),
        ('Important dates', {'fields': ('last_login',)}),
    )
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('player_id', 'email', 'password1', 'password2', 'is_active', 'is_staff', 'is_superuser'),
        }),
    )

@admin.register(NoticeBoard)
class NoticeBoardAdmin(admin.ModelAdmin):
    list_display = ('title', 'texts', 'write_time', 'admin')
    search_fields = ('title', 'texts', 'admin__player_id')  # 제목, 내용, 관리자의 player_id로 검색
    list_filter = ('write_time', 'admin')  # 작성 시간, 관리자 필터링
    date_hierarchy = 'write_time'  # 날짜 계층을 추가하여 날짜 기반 필터링을 제공
    
class QnaForm(forms.ModelForm):
    class Meta:
        model = Qna
        fields = '__all__'

@admin.register(Qna)
class QnaAdmin(admin.ModelAdmin):
    form = QnaForm
    list_display = ('title', 'question_text', 'admin_answer', 'player', 'time')
    search_fields = ('title', 'question_text', 'admin_answer', 'player__player_id')
    list_filter = ('time', 'player')
    date_hierarchy = 'time'

admin.site.register(Player, PlayerAdmin)
