from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse
from django.views import View
import requests
from django import forms 
from django.http import HttpResponseRedirect
from django.contrib.auth import logout,login
from social_django.utils import load_strategy, load_backend
import time
from django.contrib.auth.models import User
from MusicRecommenderApp.models import UsersTracks,test
import pickle
import os
from django.conf import settings
import glob
from django.db.models import Q
from functools import wraps
import time
from MusicRecommenderApp.tasks import *
# get user info
'''
class Token :
    def __init__(self,client_id='8bb89c78e01147559a8e3abdcdf84f4e',client_secret='fdae390db3e14974bfd77b31b55d67c7',redirect_uri='http://127.0.0.1:8000/'):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
    
    def get_token(self,user_id):
        scope = 'user-library-read user-top-read playlist-modify-public playlist-read-private user-follow-read user-read-recently-played'
        try:
            token = util.prompt_for_user_token(user_id, scope,client_id=self.client_id, client_secret=self.client_secret, redirect_uri=self.redirect_uri)  
            sp=spotipy.Spotify(auth= token)
        except:
            print('Token is not accesible for ' + user_id)
        return(sp)
'''
########################################################## Page Views ###################################################################

class homepageView(View):

    def get(self,request,*args,**kwargs):
        
        return render(request,'MusicRecommenderApp/index.html')


class profileView(View):
    def get(self,request):
    
        if request.user.is_authenticated:

            # get the currently logged in user
            user_id = request.user
            user = User.objects.get(username=user_id)
            social = user.social_auth.get(provider='spotify')

            # refresh access token if it expires
            if (social.extra_data['auth_time'] + 3600 - 10) <= int(time.time()):
                strategy = load_strategy()
                social.refresh_token(strategy)
            
            # get the access token and set user's spotify object
            token=social.extra_data['access_token']
            user_sp=spotipy.Spotify(auth= token)

            #df_playlist = self.get_userplaylists()
            # get the current user's saved and top tracks and add them to database
            cf = CollaborativeFiltering(user_id,user_sp)
            #saved_tracks = cf.get_savedTracks()
            #top_tracks = cf.get_topTracks()
            #recent_tracks = cf.get_recentPlays()
            #entries = [] 

            '''
            for e in saved_tracks.T.to_dict().values():
                entries.append(UsersTracks(**e))  
            for f in top_tracks.T.to_dict().values():
                entries.append(UsersTracks(**f))     
     
            UsersTracks.objects.bulk_create(entries)
           
            
            # add only the saved and top tracks to database because thats what we are sure that the user likes
            for i in saved_tracks.itertuples():
                if not UsersTracks.objects.filter(user_id=user_id, track_id=i.track_id).exists():
                    UsersTracks.objects.create(track_id=i.track_id,name=i.name,album=i.album,artist=i.artist,artist_id=i.artist_id,
                                               release_date=i.release_date,song_length=i.song_length,popularity=i.popularity,
                                                acousticness=i.acousticness,danceability=i.danceability,energy=i.energy,
                                                instrumentalness=i.instrumentalness,key=i.key,liveness=i.liveness,loudness=i.loudness,
                                                speechiness=i.speechiness,tempo=i.tempo,time_signature=i.time_signature,valence=i.valence,
                                                mode=i.mode,user_id=i.user_id)
            
            for j in top_tracks.itertuples():
                if not UsersTracks.objects.filter(user_id=user_id, track_id=j.track_id).exists():
                    UsersTracks.objects.create(track_id=j.track_id,name=j.name,album=j.album,artist=j.artist,artist_id=j.artist_id,
                                               release_date=j.release_date,song_length=j.song_length,popularity=j.popularity,
                                                acousticness=j.acousticness,danceability=j.danceability,energy=j.energy,
                                                instrumentalness=j.instrumentalness,key=j.key,liveness=j.liveness,loudness=j.loudness,
                                                speechiness=j.speechiness,tempo=j.tempo,time_signature=j.time_signature,valence=j.valence,
                                                mode=j.mode,user_id=j.user_id)
           '''
            #UsersTracks.objects.all().delete()
     
            #als=ALS(user_id)
            #recent_recommendations= als.compute_als(recent_tracks)
            #saved_recommendations = als.compute_als(saved_tracks)
            #top_recommendations = als.compute_als(top_tracks)
            
            #test1.name='meee'
            #test1.save()  
            
            choices = cf.get_playlistnames()
              
         
            context = {
                'user':user,
                'choices' :choices,
                }         
            return render(request,'MusicRecommenderApp/profile.html', context)
        else:
            return render(request,'MusicRecommenderApp/profile.html', {'error_message':"Log in to an Spotify account to access this page."})

    def post(self,request):

        user_id = request.user
        user = User.objects.get(username=user_id)
        social = user.social_auth.get(provider='spotify')

            # refresh access token if it expires
        if (social.extra_data['auth_time'] + 3600 - 10) <= int(time.time()):
            strategy = load_strategy()
            social.refresh_token(strategy)
            
            # get the access token and set user's spotify object
        token=social.extra_data['access_token']
        user_sp=spotipy.Spotify(auth= token)
    
        cf = CollaborativeFiltering(user_id,user_sp)
        choices = cf.get_playlistnames()
        als=ALS(user_id)

        option = request.POST.get('selected_category','')
        searched = True
        if option =='Saved Tracks':
            playlist_name = option
            saved_tracks = cf.get_savedTracks()
            recommendations= als.compute_als(saved_tracks)
             # add only the saved and top tracks to database because thats what we are sure that the user likes
            for i in saved_tracks.itertuples():
                if not UsersTracks.objects.filter(user_id=user_id, track_id=i.track_id).exists():
                    UsersTracks.objects.create(track_id=i.track_id,name=i.name,album=i.album,artist=i.artist,artist_id=i.artist_id,
                                               release_date=i.release_date,song_length=i.song_length,popularity=i.popularity,
                                                acousticness=i.acousticness,danceability=i.danceability,energy=i.energy,
                                                instrumentalness=i.instrumentalness,key=i.key,liveness=i.liveness,loudness=i.loudness,
                                                speechiness=i.speechiness,tempo=i.tempo,time_signature=i.time_signature,valence=i.valence,
                                                mode=i.mode,user_id=i.user_id)
        elif option =='Recent Plays':
            playlist_name = option
            recent_tracks = cf.get_recentPlays()
            recommendations= als.compute_als(recent_tracks)

        elif option =='Top Tracks':
            playlist_name = option
            top_tracks = cf.get_topTracks()
            recommendations= als.compute_als(top_tracks)
            for j in top_tracks.itertuples():
                if not UsersTracks.objects.filter(user_id=user_id, track_id=j.track_id).exists():
                    UsersTracks.objects.create(track_id=j.track_id,name=j.name,album=j.album,artist=j.artist,artist_id=j.artist_id,
                                               release_date=j.release_date,song_length=j.song_length,popularity=j.popularity,
                                                acousticness=j.acousticness,danceability=j.danceability,energy=j.energy,
                                                instrumentalness=j.instrumentalness,key=j.key,liveness=j.liveness,loudness=j.loudness,
                                                speechiness=j.speechiness,tempo=j.tempo,time_signature=j.time_signature,valence=j.valence,
                                                mode=j.mode,user_id=j.user_id)

        elif ((option !='Top Tracks') and (option !='Saved Tracks') and (option !='Recent Plays')):
            playlist_name = sp.playlist(option)['name']
            playlist_tracks = cf.get_playlistTracks(option)
            recommendations = als.compute_als(playlist_tracks)
            
        context = {
                'user':user,
                'choices' :choices,
                'recommendations':recommendations,
                'playlist_name':playlist_name,
                'option' : option,
                }         

        return render(request,'MusicRecommenderApp/profile.html',context)

class aboutView(View):
    def get(self,request):

        return render(request,'MusicRecommenderApp/about.html')

             
class contactView(View):
    def get(self,request):

        return render(request,'MusicRecommenderApp/contact.html')

class feedbackView(View):
    def get(self,request):

        return render(request,'MusicRecommenderApp/feedback.html')

class trackPageView(View):
    search_type = 'track'
    searched=False

    def get(self,request,*args,**kwargs):

        return render(request,'MusicRecommenderApp/track.html')

    def post(self,request,*args,**kwargs):

        searched=True
        track_name=request.POST.get('bar_track')
        result = sp.search(q=track_name, type=self.search_type,limit=50)
        results = result['tracks']['items']
            
        context = {"results" : results,"searched":searched}
        
        return render(request,'MusicRecommenderApp/track.html',context) 
 
class artistPageView(View):

    search_type = 'artist'
    searched=False

    def get(self,request,*args,**kwargs):

        return render(request,'MusicRecommenderApp/artist.html')

    def post(self,request,*args,**kwargs):
        searched=True
        artist_name = request.POST.get('bar_artist')
        result = sp.search(q=artist_name, type=self.search_type)
        results = result['artists']['items']
            
        context = {"results" : results,"searched":searched}
        
        return render(request,'MusicRecommenderApp/artist.html',context) 
 
class genrePageView(View):

    search_type = 'genre'
    searched=False

    def get(self,request,*args,**kwargs):

        genres = ['pop','hiphop','r&b','edm','jazz & chill','country']

        return render(request,'MusicRecommenderApp/genre.html',{"genres":genres})

class recommendationsTrackView(View):

    def post(self,request,song_id):

        song_id = request.POST.get('song_id')
        song_details = sp.track(song_id)
        
        content_based = ContentBasedRecommender_track(song_id)
        recommendations = content_based.get_top()
        request.session['recommendations_track'] = recommendations
        context={
            'song_details' :song_details, # to get the pic of the song
            'song_id' :song_id,
            'recommendations' :recommendations,
        }

        return render(request,'MusicRecommenderApp/recommendationsTrack.html',context)

class recommendationsArtistView(View):

    def post(self,request,artist_id):

        artist_id = request.POST.get('artist_id')
        artist=[]
        artist.append(artist_id)
        contentbased_a = ContentBasedRecommender_artist(artist_id)
        recommendations = contentbased_a.get_top()
        request.session['recommendations_track'] = recommendations
        artist_details = sp.artist(artist_id)
        artist_albums = sp.artist_albums(artist_id)
        artist_toptracks = sp.artist_top_tracks(artist_id)
        similar_artists =sp.artist_related_artists(artist_id)
        context={
            'artist_id' :artist_id,
            'recommendations' :recommendations,
            'artist_details' :artist_details,
            'artist_albums':  artist_albums,
            'artist_toptracks' : artist_toptracks,
            'similar_artists' : similar_artists,
        }
        return render(request,'MusicRecommenderApp/recommendationsArtist.html',context)

class recommendationsGenreView(View):
    
    def get(self,request):

        # ATTN = CHG THIS AND THW NAME OF GENRE IN DATASET
        # get the selected genre 
        genre_id = request.GET['genre_id']
        genres=[]
        genres.append(genre_id)

        # a random jazz song, DELETE THIS
        song=[]
        song_id = '21HsI1RpRHEjbUjmzHtIyj'
        song.append(song_id)
        
        # open the pickle files with min and max featurs of each genre
        with open(os.path.join(settings.BASE_DIR, 'datasets/min_features.pkl'), 'rb') as a:
            min_features = pickle.load(a)
        with open(os.path.join(settings.BASE_DIR, 'datasets/max_features.pkl'), 'rb') as b:
            max_features = pickle.load(b)

        genre_min = pd.DataFrame(min_features)
        genre_min =genre_min[genre_min.index == genre_id]
        genre_max = pd.DataFrame(max_features)
        genre_max =genre_max[genre_max.index == genre_id]

        # get the recommended songs based on the genre's features
        recommendations = sp.recommendations(seed_tracks=song,limit=100,
                                     min_acousticness=genre_min.iloc[0][2], max_acousticness =genre_max.iloc[0][2],
                                     min_danceability=genre_min.iloc[0][3], max_danceability =genre_max.iloc[0][3],
                                     min_energy=genre_min.iloc[0][4], max_energy =genre_max.iloc[0][4],
                                     min_instrumentalness=genre_min.iloc[0][5], max_instrumentalness =genre_max.iloc[0][5],
                                     min_liveness=genre_min.iloc[0][7], max_liveness=genre_max.iloc[0][7],
                                     min_loudness=genre_min.iloc[0][8], max_loudness=genre_max.iloc[0][8],
                                     min_speechiness=genre_min.iloc[0][9], max_speechiness=genre_max.iloc[0][9],
                                     min_tempo=genre_min.iloc[0][10], max_tempo=genre_max.iloc[0][10],
                                     min_valence=genre_min.iloc[0][12], max_valence=genre_max.iloc[0][12])['tracks']

        request.session['recommendations_track'] = recommendations

        context={
            'genre_id' :genre_id,
            'recommendations' :recommendations,
        }
        return render(request,'MusicRecommenderApp/recommendationsGenre.html',context)

class createPlaylistView(View):

    def post(self,request,*args,**kwargs):

          if request.user.is_authenticated:

            # get the currently logged in user
            user_id = request.user
            user = User.objects.get(username=user_id)
            social = user.social_auth.get(provider='spotify')

            # refresh access token if it expires
            #if (social.extra_data['auth_time'] + 3600 - 10) <= int(time.time()):
                #strategy = load_strategy()
                #social.refresh_token(strategy)
            
            # get the access token and set user's spotify object
            token=social.extra_data['access_token']
            user_sp=spotipy.Spotify(auth= token)

            playlist_name=request.POST.get('playlist_name')
            new_playlist = user_sp.user_playlist_create(user=user_id,name=playlist_name)
            new_playlist_id = new_playlist['id']
            playlist_songs=[]
            recommendations_track = request.session['recommendations_track']
            for i in range(len(recommendations_track)):

                playlist_songs.append(recommendations_track[i]['id'])

            user_sp.user_playlist_add_tracks(user=user_id,playlist_id=new_playlist_id,tracks=playlist_songs)
            context={
                    'new_playlist' : new_playlist,
                    }
            return render(request,'MusicRecommenderApp/playlistCreated.html',context)

