# backends.py
from django.contrib.auth.backends import BaseBackend
from django.contrib.auth import get_user_model
from economia.models import Player

Player = get_user_model()

class CustomBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            user = Player.objects.get(player_id=username)
        except Player.DoesNotExist:
            return None
        print("pwd : ", user.password)
        if password == user.password:
            return user
        return None

    def get_user(self, user_id):
        try:
            return Player.objects.get(pk=user_id)
        except Player.DoesNotExist:
            return None