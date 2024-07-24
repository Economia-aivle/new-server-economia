# myapp/middleware.py
from django.shortcuts import redirect

class ResetCorrectCountMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if not request.path.startswith('/educations/multiple/'):
            request.session['correct_count'] = 0

        return response

class ResetWrongCountMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if not request.path.startswith('/educations/multiple/'):
            request.session['wrong_count'] = 0

        return response
    
class ResetBlankWrongCountMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if not request.path.startswith('/educations/blank/'):
            request.session['blank_wrong_count'] = 0

        return response

class ResetBlankCorrectCountMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if not request.path.startswith('/educations/blank/'):
            request.session['blank_correct_count'] = 0

        return response
    
class ResetNumMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)

        if not request.path.startswith('/educations/tf_quiz/'):
            request.session['num'] = 0

        return response