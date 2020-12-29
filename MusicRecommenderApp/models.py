from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class UsersTracks (models.Model):
    track_id = models.CharField(max_length=60)
    name= models.CharField(max_length=60)
    album = models.CharField(max_length=200)
    artist = models.CharField(max_length=200)
    artist_id = models.CharField(max_length=60)
    release_date = models.CharField(max_length=200)
    song_length = models.DecimalField(decimal_places=10,max_digits=50)
    popularity=models.DecimalField(decimal_places=10,max_digits=50)
    acousticness=models.DecimalField(decimal_places=10,max_digits=50)
    danceability = models.DecimalField(decimal_places=10,max_digits=50)
    energy = models.DecimalField(decimal_places=10,max_digits=50)
    instrumentalness = models.DecimalField(decimal_places=10,max_digits=50)
    key = models.DecimalField(decimal_places=10,max_digits=50)
    liveness  =models.DecimalField(decimal_places=10,max_digits=50)
    loudness  =models.DecimalField(decimal_places=10,max_digits=50)
    speechiness = models.DecimalField(decimal_places=10,max_digits=50)
    tempo = models.DecimalField(decimal_places=10,max_digits=50)
    time_signature = models.DecimalField(decimal_places=10,max_digits=50)
    valence = models.DecimalField(decimal_places=10,max_digits=50)
    mode =models.DecimalField(decimal_places=10,max_digits=50)
    #user_id = models.ForeignKey(User,on_delete=models.CASCADE)
    user_id = models.TextField(max_length=60)
    

class test(models.Model):
    name = models.CharField(max_length=100)