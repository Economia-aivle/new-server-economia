from django.shortcuts import render, redirect
from rest_framework.response import Response
from rest_framework.decorators import api_view
import requests
from economia.models import *
from .serializers import *
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
import random
import jwt
import json
from django.contrib.sessions.models import Session

import numpy as np
import openai
import faiss
import pickle
import re

from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from transformers import TextStreamer, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

# #챗봇
# # 모델과 토크나이져 로드
# model_name = "harangstar/Llama3"
# model = AutoModelForCausalLM.from_pretrained(model_name).to("cpu")
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Response:
# {}"""

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from google.colab import drive 
# drive.mount('/content/drive')
 
# model_name = "harangstar/Llama3"  # 모델 이름을 확인하세요.
 
# # 모델 로드
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# save_path = '/content/drive/MyDrive/jiu/'
 
# # 모델과 토크나이저 저장
# model.save_pretrained(save_path)
# tokenizer.save_pretrained(save_path)

# alpaca_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

# ### Instruction:
# {}

# ### Response:
# {}"""


embedding = SentenceTransformer('jhgan/ko-sroberta-multitask')  # 임베딩 모델 가져오기
faiss_vectorstore = FAISS.load_local('./faiss_jiwoo', embedding, allow_dangerous_deserialization=True)
faiss_index = faiss.read_index('./faiss_index.bin')

with open('./documents.pkl', 'rb') as file:
    documents = pickle.load(file)

class Retriever:
    @staticmethod
    def retrieve(cate, chap):

        cate_indices = []
        for doc_id, doc in faiss_vectorstore.docstore.items():
            if cate in doc.metadata.get('Cate', ''):
                cate_indices.append(doc_id)

        print(f"{cate} 카테고리에 속하는 문서 수: {len(cate_indices)}")

        # '금융' 카테고리에 속하는 문서들의 임베딩만 추출
        filtered_embeddings = np.array([faiss_vectorstore.index.reconstruct(doc_id) for doc_id in cate_indices])

        # 쿼리 임베딩 생성
        query_embedding = embedding.encode([chap]).reshape(1, -1)

        # '금융' 카테고리에 속하는 문서들 중에서 검색 수행
        D, I = faiss_vectorstore.index.search(query_embedding, len(filtered_embeddings))

        # 거리 D를 유사도로 변환
        similarities = 1 / (1 + D[0])

        # 유사도 기반으로 '금융' 카테고리 문서 필터링 및 결과 저장
        results = []
        for i in range(len(I[0])):
            if I[0][i] < len(cate_indices):
                doc_id = cate_indices[I[0][i]]  # 필터링된 인덱스에서 원래 문서 인덱스 가져오기
                doc = faiss_vectorstore.docstore[doc_id]
                similarity = similarities[i]

                results.append({
                    'score': similarity,
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

        print(f"최종 결과 문서 수: {len(results)}")

        random_results = random.sample(results, min(5, len(results)))

        results_dict = {'documents': []}
        # 랜덤으로 선택된 결과 출력
        for result in random_results:
            results_dict['documents'].append(result['content'])

        return results_dict

class selc_question_form(BaseModel):
    question: list = Field(description="문제")
    exam: list = Field(description="보기")
    ans: list = Field(description="답")
    expl : list = Field(description="설명")
    
class question_form(BaseModel):
    question: list = Field(description="문제")
    ans: list = Field(description="답")
    expl : list = Field(description="설명")

query_set = query_set = ['''Documents마다 OX 문제와 답, 문제를 푸는 설명 한 개씩 총 8문제를 만들어줘.
             문제는 question에, 답은 ans에, 설명은 expl에 넣어줘.
             문제는 번호없이 만들어줘.''', 
             '''Documents마다 보기가 4개인 문제와 답, 문제를 푸는 설명 한 개씩 총 8문제를 한국어로 만들어줘. 
             보기는 숫자를 포함해서 표시해줘. 답은 int형으로 숫자만 알려줘. 중복 답은 없도록 해줘.
             문제는 question에, 보기는 exam에, 답은 ans에, 설명은 expl에 넣어줘.
             문제는 번호없이 만들어줘.''',
             '''Documents마다 빈칸 맞추는 문제와 답, 문제를 푸는 설명 한 개씩 총 8문제를 만들어줘. 
             답은 영어가 없는 명사로 만들어줘. 중복 답은 없도록 해줘.
             문제는 question에, 답은 ans에, 설명은 expl에 넣어줘.
             문제는 번호없이 만들어줘.''',
             ]


form_set = [question_form, selc_question_form, question_form]



@api_view(['GET'])
def getBlankDatas(request, characters):
    datas = Blank.objects.filter(characters=characters)
    serializer = BlankSerializer(datas, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def getMultipleDatas(request, characters):
    datas = Multiple.objects.filter(characters=characters)
    serializer = MultipleSerializer(datas, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def getTfDatas(request, characters):
    datas = Tf.objects.filter(characters=characters)
    serializer = TfSerializer(datas, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def getSubjectDatas(request, subjects):
    datas = Subjects.objects.filter(subjects=subjects)
    serializer = SubjectSerializer(datas, many=True)
    return Response(serializer.data)

@api_view(['GET'])
def getStageDatas(request, characters):
    datas = Stage.objects.filter(characters=characters)
    serializer = StageSerializer(datas, many=True)
    return Response(serializer.data)

def previous_quiz(request, characters):
    
    characters = get_player(request, 'characters')
    
    blank_response = requests.get(f'http://127.0.0.1:8000/educations/blankdatas/{characters}')
    blank_data = blank_response.json()

    # 두 번째 API 요청
    multiple_response = requests.get(f'http://127.0.0.1:8000/educations/multipledatas/{characters}')
    multiple_data = multiple_response.json()
    
    tf_response = requests.get(f'http://127.0.0.1:8000/educations/tfdatas/{characters}')
    tf_data = tf_response.json()
    
    for item in blank_data:
        subject_id = item.get('subjects')
        if subject_id:
            subject_instance = Subjects.objects.filter(id=subject_id).first()
            if subject_instance:
                item['subjects'] = subject_instance.subjects
                
    for item in tf_data:
        subject_id = item.get('subjects')
        if subject_id:
            subject_instance = Subjects.objects.filter(id=subject_id).first()
            if subject_instance:
                item['subjects'] = subject_instance.subjects
                
    for item in multiple_data:
        subject_id = item.get('subjects')
        if subject_id:
            subject_instance = Subjects.objects.filter(id=subject_id).first()
            if subject_instance:
                item['subjects'] = subject_instance.subjects
    
    subject_list = Subjects.objects.values_list('subjects', flat=True)

    
    # 템플릿에 데이터를 전달
    context = {
        'blank': blank_data,
        'multiple': multiple_data,
        'tf' : tf_data,
        'subject_list': subject_list,
    }
    return render(request, 'previous_quiz.html', context)

def success(request):
    return render(request, 'success.html')


def previous_quiz_answer(request, characters):
    response = requests.get(f'http://127.0.0.1:8000/educations/multipledatas/{characters}')
    data = response.json()
    return render(request, 'previous_quiz_answer.html', {'multiple': data})

# Create your views here.
def level_choice(request, characters, subjects, chapter):
    chapter = int(chapter)
    response = requests.get(f'http://127.0.0.1:8000/educations/getSubjectDatas/{subjects}/')
    data = response.json()
    sub = Subjects.objects.get(subjects=subjects)
    subjects_id = sub.id
    # characters 정보 가져오기
    characters = get_player(request, 'characters')
    
    # subjects_id와 일치하는 chapter 찾기
    chapter_item = next((item for item in data if item['id'] == subjects_id), None)
    
    if chapter_item:
        chapter_content = chapter_item['chapters']
        chapter_list = chapter_content.split(', ')
        # chapter 값은 1-based index이므로 -1 해줌
        part = chapter_list[chapter - 1] if chapter - 1 < len(chapter_list) else None
    else:
        part = None
    try:
        stage_data = Stage.objects.get(characters_id=characters, subjects_id=subjects_id, chapter=chapter)
        chapter_sub = stage_data.chapter_sub
    except Stage.DoesNotExist:
        # Stage가 없을 경우 새로운 Stage 객체를 생성합니다.
        stage_data = Stage.objects.create(characters_id=characters, chapter=chapter, chapter_sub=1, subjects_id=subjects_id)
        chapter_sub = stage_data.chapter_sub
    
    context = {
        'characters': characters,
        'subjects_id': subjects_id,
        'chapter': chapter,
        'chapter_sub': chapter_sub,
        'part': part,
    }
    
    return render(request, 'level_choice.html', context)

def make_qa(cate, chap, type_n):
# 출력 파서 정의
    output_parser = JsonOutputParser(pydantic_object=form_set[type_n])    # 지정된 pydantic 모델에 맞게 데이터를 구조화하여 제공
    format_instructions = output_parser.get_format_instructions()
    query = query_set[type_n]

    template = '''
        아래의 자료만을 사용하여 질문에 답하세요: 
        {context}
        답변은 해당 형식에 맞게 모아서 만들어주세요:
        {form}

        질문: 중학교 수준의 문제를 만들어줘. {question}
        '''


    model = ChatOpenAI(model="gpt-4o", temperature=0.5)

    # query = "수출입은행의 역할과 중요성 교육"
    results = Retriever.retrieve(cate, chap)
    prompt = PromptTemplate.from_template(template)

    chain = prompt | model | output_parser
    res = chain.invoke({'context': results['documents'], 'form':format_instructions, 'question': query})
    return res


@csrf_exempt
def tf_quiz_view(request):
    if request.method == 'GET':
        used_question_ids = request.GET.getlist('used_question_ids[]')
        chapter = request.GET.get('chapter')
        subjects_id = request.GET.get('subjects_id')  # 수정된 부분
        # characters = request.GET.get('characters')
        
        if not chapter or not subjects_id:
            return JsonResponse({"error": "Chapter and subjects are required."}, status=status.HTTP_400_BAD_REQUEST)

        questions = Tf.objects.filter(chapter=chapter, subjects_id=subjects_id).exclude(id__in=used_question_ids)
        if not questions:
            return JsonResponse({"error": "No questions available."}, status=status.HTTP_404_NOT_FOUND)
        # question = random.choice(questions)
        num = request.session.get('num', 0) 
        question = questions.order_by('-id')[num]
        num = request.session.get('num', 0) + 1
        request.session['num'] = num
        print(num)
        context = {
            "question_id": question.id,
            "question_text": question.question_text,
            "correct_answer": question.correct_answer,
            "explanation": question.explanation,
        }
        return JsonResponse(context, status=status.HTTP_200_OK)

    elif request.method == 'POST':
        question_id = request.POST.get('question_id')
        submitted_answer = request.POST.get('submitted_answer')

        if not question_id:
            return JsonResponse({"error": "Question ID is required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            question = Tf.objects.get(id=question_id)
        except Tf.DoesNotExist:
            return JsonResponse({"error": "Question not found."}, status=status.HTTP_404_NOT_FOUND)

        correct_answer = question.correct_answer
        is_correct = submitted_answer.upper() == correct_answer.upper()
        
        response_data = {
            "question_id": question.id,
            "submitted_answer": submitted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question.explanation if not is_correct else None
        }
        
        return JsonResponse(response_data, status=status.HTTP_200_OK)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
def update_stage(request):
    if request.method == 'POST':
        characters = request.POST.get('characters')
        subjects_id = request.POST.get('subjects_id')
        chapter = request.POST.get('chapter')

        if not characters or not subjects_id or not chapter:
            return JsonResponse({"error": "Characters, subjects, and chapter are required."}, status=400)

        try:
            stage = Stage.objects.get(characters_id=characters, subjects_id=subjects_id, chapter=chapter)
            if stage.chapter_sub == 1:
                stage.chapter_sub = 2
                stage.save()
                return JsonResponse({'status': 'success', 'message': 'Stage updated successfully!'})
        except Stage.DoesNotExist:
            return JsonResponse({"error": "Stage not found."}, status=404)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=400)

def tf_quiz_page(request, subjects_id, chapter):
    characters = get_player(request, 'characters')
    character = Characters.objects.get(id=characters)
    character_img = character.kind_url
    sounds = ['sounds/back_sound1.mp3', 'sounds/back_sound2.mp3', 'sounds/back_sound3.mp3']
    random_sound = random.choice(sounds)
    print(character_img)
    
    if 'char1' in character_img:
        character_img_f = character_img.replace('char1', 'char1 f')
    elif 'char2' in character_img:
        character_img_f = character_img.replace('char2', 'char2 f')
    elif 'char3' in character_img:
        character_img_f = character_img.replace('char3', 'char3 f')
    elif 'char4' in character_img:
        character_img_f = character_img.replace('char4', 'char4 f')
    elif 'char5' in character_img:
        character_img_f = character_img.replace('char5', 'char5 f')
    else:
        character_img_f = character_img  # 위 조건에 해당하지 않는 경우 원본 URL 유지
    
    sub = Subjects.objects.get(id=subjects_id)
    subjects = sub.subjects
    chapters_list = sub.chapters.split(', ')
    chapter_name = chapters_list[int(chapter) - 1] 
    result = make_qa(subjects, chapter_name, 0)
    m_question = result['question']
    m_ans = result['ans']
    m_expl = result['expl']
    print(m_question)
    print(m_ans)
    
    problem_count = Tf.objects.filter(characters_id=characters).count()
    if problem_count > 50:
        old_problem = Tf.objects.filter(characters_id=characters).order_by('id')[:8]
        for obj in old_problem:
            obj.delete()
        print('오래된 문제 삭제')
        
    
    for i in range(8):
        Tf.objects.create(characters_id=characters, question_text=m_question[i], correct_answer=m_ans[i],
                          subjects_id=subjects_id, chapter=chapter, explanation=m_expl[i])
    context = {
            'characters': characters,
              'subjects_id': subjects_id,
              'chapter': chapter,
              'character_img':character_img,
              'random_sound': random_sound,
              'character_img_f': character_img_f,
              'subjects':subjects,
              'chapter_name':chapter_name,
    } 
    return render(request, 'tfquiz.html', context)

def choose_tf_chapter_view(request):
    if request.method == 'POST':
        chapter = request.POST.get('chapter')
        if not chapter:
            return JsonResponse({"error": "Chapter number is required."}, status=status.HTTP_400_BAD_REQUEST)

        questions = Tf.objects.filter(chapter=chapter)[:8]  # 챕터별 문제 8개 가져오기

        if not questions:
            return JsonResponse({"error": "No questions found for the selected chapter."},
                                status=status.HTTP_404_NOT_FOUND)

        question_data = [{
            "id": question.id,
            "question_text": question.question_text,
            "correct_answer": question.correct_answer,
            "explanation": question.explanation
        } for question in questions]

        return JsonResponse({"questions": question_data}, status=status.HTTP_200_OK)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=status.HTTP_400_BAD_REQUEST)
    



def multiple(request, characters, subjects_id, chapter, num):
    # characters, subject, chapter에 해당하는 데이터를 필터링합니다.
    characters = get_player(request, 'characters')
    player_id = get_player(request, 'player')
    character = Characters.objects.get(id=characters)
    character_img = character.kind_url
    print(character_img)
    sub = Subjects.objects.get(id=subjects_id)
    subjects = sub.subjects
    
    if 'char1' in character_img:
        character_img_f = character_img.replace('char1', 'char1 f')
    elif 'char2' in character_img:
        character_img_f = character_img.replace('char2', 'char2 f')
    elif 'char3' in character_img:
        character_img_f = character_img.replace('char3', 'char3 f')
    elif 'char4' in character_img:
        character_img_f = character_img.replace('char4', 'char4 f')
    elif 'char5' in character_img:
        character_img_f = character_img.replace('char5', 'char5 f')
    else:
        character_img_f = character_img  # 위 조건에 해당하지 않는 경우 원본 URL 유지
    
    random_sound = 'sounds/back_sound3.mp3'
    multiple_response = requests.get(f'http://127.0.0.1:8000/educations/multipledatas/{characters}')
    multiple_data = multiple_response.json()
    problem_count = Multiple.objects.filter(characters_id=characters).count()
    if problem_count > 50:
        old_problem = Multiple.objects.filter(characters_id=characters).order_by('id')[:8]
        for obj in old_problem:
            obj.delete()
        print('오래된 문제 삭제')

    if num == 9:
        return redirect('educations:level_choice', characters=characters, subjects_id=subjects_id, chapter=chapter)
    # # POST 요청 처리
    if request.method == 'POST':     
        multiple_response = requests.get(f'http://127.0.0.1:8000/educations/multipledatas/{characters}')
        multiple_data = multiple_response.json()
        multiple_list = [item for item in multiple_data if item['characters'] == characters and item['subjects'] == subjects_id and item['chapter'] == chapter]
        print(multiple_list)
        user_answer = request.POST.get('answer')
        correct_answer = multiple_list[-num]['correct_answer']
        
        if user_answer == correct_answer:
            # 정답인 경우
            correct_count = request.session.get('correct_count', 0) + 1
            request.session['correct_count'] = correct_count
            print(correct_count)
            if correct_count == 5:
                # 모든 문제를 맞춘 경우 Stage 모델의 chapter_sub를 3으로 업데이트
                try:
                    stage_data = Stage.objects.get(characters_id=characters, subjects_id=subjects_id, chapter=chapter)
                    character_score = SubjectsScore.objects.get(characters_id=characters, subjects_id=subjects_id)
                    character_exp = Characters.objects.get(id = characters)
                    stage_data.chapter_sub = 3
                    character_score.score +=10
                    character_exp.exp +=10
                    character_score.save()
                    stage_data.save()
                    character_exp.save()
                except Stage.DoesNotExist:
                    pass
                
                # 세션 초기화
                request.session['correct_count'] = 0

                # JSON 응답 전송
                return JsonResponse({'status': 'complete', 'message': '모든 문제를 맞췄습니다!'})
            else:
                # 아직 모든 문제를 맞추지 않은 경우
                return JsonResponse({'status': 'correct', 'message': '정답입니다.'})
        elif user_answer != correct_answer:
            wrong_count = request.session.get('wrong_count', 0) + 1
            request.session['wrong_count'] = wrong_count
            # 오답인 경우
            return JsonResponse({'status': 'wrong', 'message': '오답입니다.'})
    else:
        if num == 1:
            sub = Subjects.objects.get(id=subjects_id)
            subjects = sub.subjects
            print(subjects)
            chapters_list = sub.chapters.split(', ')
            chapter_name = chapters_list[int(chapter) - 1] 
            result = make_qa(subjects, chapter_name, 1)
            m_question = result['question']
            m_exam = result['exam']
            m_ans = result['ans']
            m_expl = result['expl']
            print(m_question)
            print(m_exam)
            print(m_ans)
            
            for i in range(8):
                a = m_exam[i][0]
                b = m_exam[i][1]
                c = m_exam[i][2]
                d = m_exam[i][3]
                Multiple.objects.create(characters_id=characters, question_text=m_question[i],
                                option_a = a, option_b = b, option_c = c, option_d = d,
                                correct_answer=int(m_ans[i]), subjects_id=subjects_id, chapter=chapter, explanation=m_expl[i])

    correct_count = request.session.get('correct_count', 0)
    wrong_count = request.session.get('wrong_count', 0)
    hp_percentage = max(0, 100 - (correct_count * 20))  # 체력 퍼센트 계산
    
    multiple_response = requests.get(f'http://127.0.0.1:8000/educations/multipledatas/{characters}')
    multiple_data = multiple_response.json()
    
    multiple_list = [item for item in multiple_data if item['characters'] == characters and item['subjects'] == subjects_id and item['chapter'] == chapter]
    question = multiple_list[-num]
    
    
    context ={'question': question,
              'num': num,
              'characters': characters,
              'subjects_id': subjects_id,
              'chapter': chapter,
              'correct_count': correct_count,
              'hp_percentage': hp_percentage,
              'wrong_count' : wrong_count,
              'character_img' : character_img,
              'random_sound' :random_sound,
              'character_img_f':character_img_f,
              'subjects':subjects,
              }
    
    
    # GET 요청 처리
    return render(request, 'multiple.html', context )



def blank(request, characters, subjects_id, chapter, num):
    characters = get_player(request, 'characters')
    character = Characters.objects.get(id=characters)
    character_img = character.kind_url
    print(character_img)
    sub = Subjects.objects.get(id=subjects_id)
    subjects = sub.subjects
    
    if 'char1' in character_img:
        character_img_f = character_img.replace('char1', 'char1 f')
    elif 'char2' in character_img:
        character_img_f = character_img.replace('char2', 'char2 f')
    elif 'char3' in character_img:
        character_img_f = character_img.replace('char3', 'char3 f')
    elif 'char4' in character_img:
        character_img_f = character_img.replace('char4', 'char4 f')
    elif 'char5' in character_img:
        character_img_f = character_img.replace('char5', 'char5 f')
    else:
        character_img_f = character_img  # 위 조건에 해당하지 않는 경우 원본 URL 유지
    
    
    random_sound = 'sounds/back_sound2.mp3'
    problem_count = Multiple.objects.filter(characters_id=characters).count()
    if problem_count > 50:
        old_problem = Blank.objects.filter(characters_id=characters).order_by('id')[:8]
        for obj in old_problem:
            obj.delete()
        print('오래된 문제 삭제')
   
    if num == 9:
        return redirect('educations:level_choice', characters=characters, subjects_id=subjects_id, chapter=chapter)
    
    if request.method == 'POST':
        blank_response = requests.get(f'http://127.0.0.1:8000/educations/blankdatas/{characters}')
        blank_data = blank_response.json()
        blank_list = [item for item in blank_data if item['characters'] == characters and item['subjects'] == subjects_id and item['chapter'] == chapter]
        
        user_answer = request.POST.get('answer')
        correct_answer = blank_list[-num]['correct_answer']
        
        if user_answer == correct_answer:
            # 정답인 경우
            blank_correct_count = request.session.get('blank_correct_count', 0) + 1
            request.session['blank_correct_count'] = blank_correct_count
            request.session.modified = True  # 세션 데이터가 변경되었음을 명시
            print(f"Correct count updated: {blank_correct_count}")
            if blank_correct_count == 5:
                # 모든 문제를 맞춘 경우 Stage 모델의 chapter_sub를 3으로 업데이트
                try:
                    stage_data = Stage.objects.get(characters_id=characters, subjects_id=subjects_id, chapter=chapter)
                    character_exp = Characters.objects.get(id=characters)
                    character_score = SubjectsScore.objects.get(characters_id=characters, subjects_id=subjects_id)
                    character_exp.exp +=10
                    character_score.score +=10
                    stage_data.chapter_sub = 3
                    character_exp.save()
                    character_score.save()
                    stage_data.save()
                except Stage.DoesNotExist:
                    pass
                
                # 세션 초기화
                request.session['blank_correct_count'] = 0
                request.session.modified = True  # 세션 데이터가 변경되었음을 명시

                # JSON 응답 전송
                return JsonResponse({'status': 'complete', 'message': '모든 문제를 맞췄습니다!'})
            else:
                # 아직 모든 문제를 맞추지 않은 경우
                return JsonResponse({'status': 'correct', 'message': '정답입니다.'})
        elif user_answer != correct_answer:
            blank_wrong_count = request.session.get('blank_wrong_count', 0) + 1
            request.session['blank_wrong_count'] = blank_wrong_count
            request.session.modified = True  # 세션 데이터가 변경되었음을 명시
            print(f"Wrong count updated: {blank_wrong_count}")
            # 오답인 경우
            return JsonResponse({'status': 'wrong', 'message': '오답입니다.'})
    
    else:
        if num == 1:
            sub = Subjects.objects.get(id=subjects_id)
            subjects = sub.subjects
            chapters_list = sub.chapters.split(', ')
            chapter_name = chapters_list[int(chapter) - 1] 
            result = result = make_qa(subjects, chapter_name, 2)
            m_question = result['question']
            m_ans = result['ans']
            m_expl = result['expl']
            print(m_question)
            print(m_ans)
        
            for i in range(8):
                Blank.objects.create(characters_id=characters, question_text=m_question[i], correct_answer=m_ans[i],
                                subjects_id=subjects_id, chapter=chapter, explanation=m_expl[i])
    blank_response = requests.get(f'http://127.0.0.1:8000/educations/blankdatas/{characters}')
    blank_data = blank_response.json()

    blank_list = [item for item in blank_data if item['characters'] == characters and item['subjects'] == subjects_id and item['chapter'] == chapter]
    question = blank_list[-num]

    blank_correct_count = request.session.get('blank_correct_count', 0)
    blank_wrong_count = request.session.get('blank_wrong_count', 0)
    hp_percentage = max(0, 100 - (blank_correct_count * 20))  # 체력 퍼센트 계산
    context = {
        'character_img_f':character_img_f,
        'question': question,
        'num': num,
        'characters': characters,
        'subjects_id': subjects_id,
        'chapter': chapter,
        'correct_count': blank_correct_count,
        'hp_percentage': hp_percentage,
        'wrong_count': blank_wrong_count,
        'character_img': character_img,
        'random_sound': random_sound,
        'subjects':subjects,
    }

    return render(request, 'blank.html', context)

    
def study(request, subjects_id):
    subject = Subjects.objects.get(id=subjects_id)
    subject_name = subject.subjects
    context = {
        'subject_name': subject_name,
    }
    return render(request,'study.html', context)

def summary_anime(request):
    return render(request,'summary_anime.html')

def wrong_explanation(request):
    return render(request,'wrong_explanation.html')

def chapter_summary(request):
    return render(request,'chapter_summary.html')

def chatbot(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            message = data.get('message', '')
            inputs = tokenizer(
                [
                    alpaca_prompt.format(
                        message,  # 지시사항
                        "",  # 출력 - 생성을 위해 이 부분을 비워둡니다!
                    )
                ],
                return_tensors="pt",
            ).to("cuda")

            text_streamer = TextStreamer(tokenizer)
            output = model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=50,  # 생성할 토큰 수
            )
            
            return JsonResponse({'response': output})
        except json.JSONDecodeError:
            return JsonResponse({'error': '잘못된 JSON 형식입니다.'}, status=400)
    return JsonResponse({'error': '잘못된 요청 방법입니다.'}, status=405)

def chapter(request, subjects):
    response = requests.get(f'http://127.0.0.1:8000/educations/getSubjectDatas/{subjects}/')
    data = response.json()
    characters = get_player(request, 'characters')
    for chapter in data:
        chapter_content = chapter['chapters']
        chapter['chapters_list'] = chapter_content.split(', ')
    context = {
        'chapter': data,
        'characters':characters,
        
    }
    print(data)
    return render(request,'chapter.html', context)

@csrf_exempt
def study_view(request):
    if request.method == 'GET':
        print("Rendering study.html")
        return render(request, 'study.html')
    print("Invalid request method")
    return JsonResponse({"error": "Invalid request method"}, status=400)


def summary_anime(request):
    return render(request, 'summary_anime.html')

def get_player(request, id):
    access_token = request.COOKIES.get('access_token')
    refresh_token = request.COOKIES.get('refresh_token')

    # 디버깅 로그 추가
    print("Access Token:", access_token)
    print("Refresh Token:", refresh_token)
    
    if not access_token:
        return JsonResponse({"error": "토큰이 없습니다."}, status=400)
    
    decoded = jwt.decode(access_token, 'economia', algorithms=['HS256'])
    decoded['access_token'] = access_token
    player = Player.objects.get(player_id=decoded['player_id'])
    player_id = player.id
    character = get_object_or_404(Characters, player_id=player_id)
    characters_id = character.id
    print(characters_id)
    print(player_id)
    if id == 'player':
        return player_id
    elif id == 'characters':
        return characters_id
    else:
        return player_id
    