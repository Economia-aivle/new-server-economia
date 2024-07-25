from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Player, NoticeBoard, Qna, Subjects
from django import forms


import numpy as np
import openai
import faiss
import pickle
import random
import re

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


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

admin.site.register(Player, PlayerAdmin)

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


embedding = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 임베딩 모델 가져오기
faiss_vectorstore = FAISS.load_local('./faiss_jiwoo', embedding, allow_dangerous_deserialization=True)
faiss_index = faiss.read_index('./faiss_index.bin')

with open('./documents.pkl', 'rb') as file:
    documents = pickle.load(file)

def make_chapter(cate):
    num_samples = 5  # 추출할 샘플 수
    n_total_vectors = faiss_index.ntotal  # 인덱스에 저장된 벡터 수
    
    cate_indices = []
    for doc_id, doc in faiss_vectorstore.docstore.items():
        if cate in doc.metadata.get('Cate', ''):
            cate_indices.append(doc_id)

    random_indices = np.random.choice(cate_indices, num_samples, replace=False)  # 중복 없이 무작위 인덱스 선택

    summary_texts = []
    for idx in random_indices:
        content_parts = documents[idx].page_content.split('\n', 1)
        if len(content_parts) == 2:
            doc_title, doc_description = content_parts
        else:
            doc_title = "No title available"
            doc_description = content_parts[0]

        doc_category = documents[idx].metadata['Cate']
        summary_texts.append(f"{doc_category}: {doc_title} - {doc_description}")


    summary_text = "\n".join(summary_texts)

    chapter = ''
    for i in range(8):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 경제 데이터를 기반으로 청소년 대상 교육 1개 주제만을 생성하도록 훈련된 AI입니다. 1) 주제는 **으로시작해서**으로 끝나야합니다. 2) 중학생을 대상으로 주제를 사용해야 합니다."},
                {"role": "user", "content": f"다음 요약을 기반으로 주제 생성:\n{summary_text}"}
            ],
            max_tokens=300
        )

        # 생성된 주제 출력
        topic = response.choices[0].message.content
        pattern = r'\*\*(.+?)\*\*'
        matches = re.findall(pattern, topic)
        
        if matches:
            ai_chapter = matches[0]
        else:
            ai_chapter = "No valid topic found"
        
        if i == 0: chapter += ai_chapter
        else: chapter = chapter + ', ' + ai_chapter
    
    return chapter
        
    

class SubjectsForm(forms.ModelForm):
    class Meta:
        model = Subjects
        fields = ['subjects']

    def save(self, commit=True):
        instance = super().save(commit=False)
        instance.chapters = make_chapter(instance.subjects)
        if commit:
            instance.save()
        return instance

@admin.register(Subjects)
class SubjectsAdmin(admin.ModelAdmin):
    list_display = ('id', 'subjects', 'chapters')
    search_fields = ('subjects', 'chapters')
    
    # 폼을 커스터마이징하여 chapters 필드를 자동으로 생성하도록 함
    def get_form(self, request, obj=None, **kwargs):
        kwargs['form'] = SubjectsForm
        return super().get_form(request, obj, **kwargs)
