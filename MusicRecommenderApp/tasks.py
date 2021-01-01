import spotipy 
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from decimal import Decimal
import operator
from statistics import mean 
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import implicit
import random
from MusicRecommenderApp.models import UsersTracks


def ccm_token():
    client_id='8bb89c78e01147559a8e3abdcdf84f4e'
    client_secret='fdae390db3e14974bfd77b31b55d67c7'
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id,client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return(sp)

sp=ccm_token()
SPOTIFY_CLIENT_ID = '8bb89c78e01147559a8e3abdcdf84f4e'
SPOTIFY_CLIENT_SECRET = 'fdae390db3e14974bfd77b31b55d67c7'
SPOTIFY_REDIRECT_URI = 'http://127.0.0.1:8000/'
SCOPE = 'user-library-read user-top-read playlist-modify-public playlist-read-private user-follow-read user-read-recently-played'

def get_token(user_id):
    token = util.prompt_for_user_token(user_id, SCOPE,client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET, redirect_uri=SPOTIFY_REDIRECT_URI)  
    sp=spotipy.Spotify(auth= token,requests_timeout=None)
    return sp

def getTrackFeatures(id):
    meta = sp.track(id)
    features = sp.audio_features(id)

  # meta
    track_id = meta['id']
    name = meta['name']
    album = meta['album']['name']
    artist = meta['album']['artists'][0]['name']
    artist_id =  meta['album']['artists'][0]['id']
    release_date = meta['album']['release_date']
    song_length = meta['duration_ms']
    popularity = meta['popularity']

  # features
    acousticness = features[0]['acousticness']
    danceability = features[0]['danceability']
    energy = features[0]['energy']
    instrumentalness = features[0]['instrumentalness']
    key=features[0]['key']
    liveness = features[0]['liveness']
    loudness = features[0]['loudness']
    speechiness = features[0]['speechiness']
    tempo = features[0]['tempo']
    time_signature = features[0]['time_signature']
    valence = features[0]['valence']
    mode=features[0]['mode']

    
    track = [track_id,name, album, artist,artist_id,release_date, song_length, popularity, acousticness, danceability,
             energy, instrumentalness, key, liveness, loudness, speechiness, tempo, time_signature,valence, mode]

    return track

def getArtistFeatures(id):
    meta = sp.artist(id)
    albums = sp.artist_albums(id)
    top_tracks = sp.artist_top_tracks(id)
    related_artist = sp.artist_related_artists(id)

  # meta
    artist_id = meta['id']
    artist_name = meta['name']
    genres = meta['genres']
    popularity = meta['popularity']

    artist = [artist_id,artist_name, genres, popularity, albums,top_tracks,related_artist]
    return artist

class ContentBasedRecommender_track():

    def __init__(self, song_id):
        self.song_id = song_id

    def getSpRec(self):

        song_features = getTrackFeatures(self.song_id)
        a=[]
        # use seed artist,can't get recommendations with seed_tracks cuz the features are too specific
        a.append(song_features[4]) 

        recommendations = sp.recommendations(seed_artists=a,limit=100,acousticness=song_features[7],danceability=song_features[8],
                                        energy=song_features[9],instrumentalness=song_features[10],key=song_features[11],
                                        liveness=song_features[12],loudness=song_features[13],speechiness=song_features[14],
                                        tempo=song_features[15],time_signature=song_features[16],mode=song_features[17])


        return recommendations

    # create the df for the recommendations features
    def create_dffeatures(self):
        
        min_max_scaler = preprocessing.MinMaxScaler()
        recommendations = self.getSpRec()

        song_features = getTrackFeatures(self.song_id)
        # add the track and artist names we got from spotify into a list
        tracks_features = []

        for i in range(len(recommendations['tracks'])):
            track_features = getTrackFeatures(recommendations['tracks'][i]['id'])
            tracks_features.append(track_features)
        
        # the dataframe of the songs we got from spotify 
        df_features = pd.DataFrame(tracks_features, columns = ['track_id','name', 'album', 'artist','artist_id', 'release_date', 
                                                                'song_length','popularity', 'acousticness', 'danceability', 'energy', 
                                                                'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 
                                                                'tempo', 'time_signature','valence','mode'])
        df_features.loc[0] = song_features # the first track in the df is the selected song
        df_features['index']= df_features.index

        features = df_features.drop(columns=['index','track_id','name','artist','album','release_date','artist_id'])
        features = min_max_scaler.fit_transform(features)
        features = pd.DataFrame(features)
        return df_features, features

    def nth_root(self,value, n_root):
    
        root_value = 1/float(n_root)
        return  (Decimal(value) ** Decimal(root_value))
    
    def minkowski_distance(self,x,y,p_value):
    
        return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)
    
    def get_top(self):
        
        list_minkowski ={}
        top_id = []
        df_features,features = self.create_dffeatures()

        for i,j in features.iterrows():
            minkowski = self.minkowski_distance(features.iloc[0], features.iloc[i],2)
            list_minkowski[i] = minkowski
            
        sorted_minkowski = sorted(list_minkowski.items(), key=operator.itemgetter(1))[2:]
        
        for i,j in enumerate(sorted_minkowski):
            if i == 15 : break
            top_id.append(df_features.iloc[j[0]]['track_id'])

        top_rec = sp.tracks(top_id)['tracks']

        return top_rec  

class ContentBasedRecommender_artist():

    def __init__(self, artist_id):
        self.artist_id = artist_id
        
    def getSpRec(self):
        
        artist_top = []
        artist_toptracks = sp.artist_top_tracks(self.artist_id)['tracks']
        song_features = []

        for i,j in enumerate(artist_toptracks):
            toptrack_id = artist_toptracks[i]['id']
            song_features.append(getTrackFeatures(toptrack_id))

        [track_id,name, album, artist,artist_id,release_date, song_length, popularity, acousticness, danceability,
        energy, instrumentalness, key, liveness, loudness, speechiness, tempo, time_signature,valence, mode] = [[row[i] for row in song_features] for i in range(len(song_features[0]))]
        
        a=[]
        a.append(artist_id[0]) 
        
        '''
        recommendations = sp.recommendations(seed_artists=a,limit=100,acousticness=mean(acousticness),
                                             danceability=mean(danceability),energy=mean(energy),
                                             instrumentalness=mean(instrumentalness),key=mean(key),
                                             liveness=mean(liveness),loudness=mean(loudness),speechiness=mean(speechiness),
                                             tempo=mean(tempo),time_signature=mean(time_signature),mode=mean(mode),
                                             valence=mean(valence))
        '''
        recommendations = sp.recommendations(seed_artists=a,limit=100)
        return song_features,recommendations
     
    def create_dffeatures(self):
        
            min_max_scaler = preprocessing.MinMaxScaler()
            song_features,recommendations = self.getSpRec()

            tracks_features = []

            for i in range(len(recommendations['tracks'])):
                track_features = getTrackFeatures(recommendations['tracks'][i]['id'])
                tracks_features.append(track_features)

            # the dataframe of the songs we got from spotify 
            df_features = pd.DataFrame(tracks_features, columns = ['track_id','name', 'album', 'artist','artist_id', 'release_date', 
                                                                    'song_length','popularity', 'acousticness', 'danceability', 'energy', 
                                                                    'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 
                                                                    'tempo', 'time_signature','valence','mode'])
            df_top = pd.DataFrame(song_features)
            df_features['index']= df_features.index

            features = df_features.drop(columns=['index','track_id','name','artist','album','release_date','artist_id'])
            features = pd.DataFrame(min_max_scaler.fit_transform(features))
            top_features = df_top.drop(columns=[0,1,2,3,4,5])
            top_features = pd.DataFrame(min_max_scaler.fit_transform(top_features))
            return (df_top,df_features,features,top_features)
    
    def get_top(self):
        
        df_top,df_features,features,top_features = self.create_dffeatures()
        top_features.append(features)

        # convert features to dense matrix
        matrix= csr_matrix(top_features.values)
        matrix = matrix.todense()

        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'auto') # use cosine and brute to get the nearest neighbour
        model_knn.fit(matrix)
        query_index = [i for i in range(len(df_top))]
        list_rec = []
        for i in query_index:

            distances, indices = model_knn.kneighbors(features.iloc[i,:].values.reshape(1, -1), n_neighbors = 10)
            for i in range(0, len(distances.flatten())):
                rec = df_features.iloc[indices[0][i]]['track_id']
                if rec not in list_rec:
                    list_rec.append(rec)

            top_rec = sp.tracks(list_rec)['tracks']
        
        return top_rec

class CollaborativeFiltering():

    def __init__(self,user_id,token):
        self.user_id = user_id
        self.token =token
    
    def get_playlistnames(self):
        sp = self.token
        choices = []
        total = sp.current_user_playlists()['total']
        offset_index = 0
        while offset_index < total:
            playlist = sp.current_user_playlists(limit=50, offset=offset_index)
            for i, item in enumerate(playlist['items']):
                choices.append(item)
            offset_index = offset_index +50

        return choices

    def get_playlistTracks(self,playlist_id):
        sp=self.token
        tracks = []
        playlist_items = sp.playlist_tracks(playlist_id)['items']
        random.shuffle(playlist_items)
        playlist_tracks = playlist_items[:50] # get 50 random tracks from the playlist
        for i in playlist_tracks:
            if i['is_local'] == False and i['track']['type']=='track':
                track = getTrackFeatures(i['track']['id'])
                tracks.append(track)
        df_playlist = pd.DataFrame(tracks, columns = ['track_id','name', 'album', 'artist','artist_id', 'release_date', 
                                                           'song_length','popularity', 'acousticness', 'danceability', 'energy', 
                                                           'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 
                                                           'tempo', 'time_signature','valence','mode'])
        if df_playlist.empty == True:
            return df_playlist
        else:
            df_playlist = df_playlist.drop_duplicates(subset=['track_id'])
            df_playlist['user_id'] = self.user_id
            df_playlist.drop_duplicates(inplace=True)
            return df_playlist

    def get_topTracks(self):

        sp = self.token
        # get user's top tracks based on short, medium and long term. 
        top_track_id=[]
        song_count_s = sp.current_user_top_tracks(time_range = 'short_term')['total']
        song_count_m = sp.current_user_top_tracks(time_range = 'medium_term')['total']
        song_count_l = sp.current_user_top_tracks(time_range = 'long_term')['total']
        offset_index = 0
        
        while offset_index < song_count_s:

            short_tracks = sp.current_user_top_tracks(time_range = 'short_term',offset=offset_index)
            for i, item in enumerate(short_tracks['items']):
                    top_track_id.append (item['id'])
            offset_index = offset_index + 20
        
        offset_index = 0

        while offset_index < song_count_m:
            medium_tracks = sp.current_user_top_tracks(time_range = 'medium_term',offset=offset_index)
            for i, item in enumerate(medium_tracks['items']):
                    top_track_id.append (item['id'])
            offset_index = offset_index + 20
        
        offset_index = 0

        while offset_index < song_count_l:
            long_tracks = sp.current_user_top_tracks(time_range = 'long_term',offset=offset_index)
            for i, item in enumerate(long_tracks['items']):
                    top_track_id.append (item['id'])
            offset_index = offset_index + 20
        
            # loop over track ids 
        tracks = []
        for i in range(len(top_track_id)):
            track = getTrackFeatures(top_track_id[i])
            tracks.append(track)

            # create dataset
        df_topTracks = pd.DataFrame(tracks, columns = ['track_id','name', 'album', 'artist','artist_id', 'release_date', 
                                                           'song_length','popularity', 'acousticness', 'danceability', 'energy', 
                                                           'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 
                                                           'tempo', 'time_signature','valence','mode'])
        if df_topTracks.empty == True:
            return df_topTracks
        else:
            df_topTracks = df_topTracks.drop_duplicates(subset=['track_id'])
            df_topTracks['user_id'] = self.user_id
            df_topTracks.drop_duplicates(inplace=True)
            return df_topTracks

    def get_recentPlays(self) :
        sp = self.token

        recent_plays_id = []

        recent_plays = sp.current_user_recently_played(limit=50)
        for i, item in enumerate(recent_plays['items']):
            recent_plays_id.append (item['track']['id'])

        # loop over track ids 
        recent_tracks = []
        for i in range(len(recent_plays_id)):
            #time.sleep(.5)
            recent_track = getTrackFeatures(recent_plays_id[i])
            recent_tracks.append(recent_track)

        # create dataset
        df_recentPlays = pd.DataFrame(recent_tracks, columns = ['track_id','name', 'album', 'artist','artist_id', 'release_date', 
                                                            'song_length','popularity', 'acousticness', 'danceability', 'energy', 
                                                            'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 
                                                            'tempo', 'time_signature','valence','mode'])

        if df_recentPlays.empty == True:
            return df_recentPlays
        else:
            df_recentPlays = df_recentPlays.drop_duplicates(subset=['track_id'])
            df_recentPlays['user_id'] = self.user_id
            df_recentPlays.drop_duplicates(inplace=True)
            return df_recentPlays
           
    def get_savedTracks(self):
        sp = self.token
        saved_tracks_id = []
        song_count = sp.current_user_saved_tracks()['total']
        offset_index = 0
        while offset_index < song_count:
            saved_tracks = sp.current_user_saved_tracks(offset=offset_index)
            for i, item in enumerate(saved_tracks['items']):
                    saved_tracks_id.append (item['track']['id'])
            offset_index = offset_index + 20

        # loop over track ids 
        tracks = []
        for i in range(len(saved_tracks_id)):
            #time.sleep(1.5)
            track = getTrackFeatures(saved_tracks_id[i])
            tracks.append(track)

        # create dataset
        df_savedTracks = pd.DataFrame(tracks, columns = ['track_id','name', 'album', 'artist','artist_id', 'release_date', 
                                                            'song_length','popularity', 'acousticness', 'danceability', 'energy', 
                                                            'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 
                                                            'tempo', 'time_signature','valence','mode'])
     

       
        if df_savedTracks.empty == True:
            return df_savedTracks
        else:
            df_savedTracks = df_savedTracks.drop_duplicates(subset=['track_id'])
            df_savedTracks['user_id'] = self.user_id
            df_savedTracks.drop_duplicates(inplace=True)
            return df_savedTracks
    
class ALS():
    
    def __init__(self,user_id):
        self.user_id = user_id

    def compute_als(self,user_dataset):
        user_id = str(self.user_id)
        #df_others = pd.DataFrame(list(UsersTracks.objects.filter(~Q(user_id=user_id)).values()))
        all_dataset = pd.DataFrame(list(UsersTracks.objects.all().values()))
        
        df_als = all_dataset.copy()
        df_als['listened'] = 1
        # convert user_id and track_id to numeric categories
        df_als['userid'] = df_als['user_id'].astype('category').cat.codes
        df_als['itemid'] = df_als['track_id'].astype('category').cat.codes

        # create user-item and item-user matrix
        user_items = csr_matrix((df_als['listened'].astype(int),(df_als['userid'], df_als['itemid'])))
        item_users = csr_matrix((df_als['listened'].astype(int),(df_als['itemid'], df_als['userid'])))

        #Building the als model
        als = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=20)
        alpha_val = 40
        data_conf = (item_users * alpha_val).astype('double')
        als.fit(data_conf)
        # get the top 15 recommended item based on ALS
        # get the current user's category code
        df_user = df_als[df_als['user_id']==user_id]
        
        user_num = df_user.userid.values[0]
        
        #Get top 15 recommendations
        recommended = als.recommend(user_num, user_items,filter_already_liked_items=True,N=user_items.shape[1])[:15]
        list_als = []
        for i,j in enumerate(recommended):
            a = df_als.loc[df_als['itemid']==j[0]].track_id
            list_als.append(a.values[0])

        # get the top 2 users similar to current user
        similar_users = als.similar_users(user_num,3)
        similar_userlist = []

        for a in similar_users:
            user_cat  =a[0]
            username = df_als[df_als['userid']==user_cat]
            username1 = username.user_id.values[0]
            similar_userlist.append(username1)
        
        # create another df storing only the songs of similar users

        temp = all_dataset[all_dataset['user_id']==similar_userlist[1]]
        temp2 =  all_dataset[all_dataset['user_id']==similar_userlist[2]]
        #temp = pd.DataFrame(list(UsersTracks.objects.filter(user_id=similar_userlist[1])))
        #temp2 =  pd.DataFrame(list(UsersTracks.objects.filter(user_id=similar_userlist[2])))
        similaruser_dataset = pd.concat([temp,temp2],axis=0,ignore_index=True)
        
        # normalize the features
        min_max_scaler = preprocessing.MinMaxScaler()

        user_features = user_dataset.drop(columns=['track_id','name','artist','artist_id','album','release_date','user_id'])
        user_features = pd.DataFrame(min_max_scaler.fit_transform(user_features))

        others_features = similaruser_dataset.drop(columns=['id','track_id','name','artist','artist_id','album','release_date','user_id'])
        others_features = pd.DataFrame(min_max_scaler.fit_transform(others_features))

        # get the most similar songs 

        cosine_similarities_count = cosine_similarity(user_features,others_features)

        top={}
        cosine_score = {}
        similar_list = []
        for i,j in user_features.iterrows():
            similar_indices = cosine_similarities_count[i].argsort()[:-cosine_similarities_count.shape[1]:-1]
            top[i]=similar_indices
            cosine_score[i] = cosine_similarities_count[i]

        for i in top:
            most_similar = top[i][0]
            similar_list.append(similaruser_dataset.iloc[most_similar]['track_id'])
       
        als_rec = list_als + similar_list
        als_rec_filt = []
        for ids in als_rec:
            if ids not in als_rec_filt:
                als_rec_filt.append(ids)
        
        if len(als_rec_filt)>50:
            
            top_rec = []
            for i in als_rec_filt:
                top_rec.append(sp.track(i))
            
            #top_rec = sp.tracks(als_rec_filt[:50])['tracks']
        else:
            top_rec = sp.tracks(als_rec_filt)['tracks']
        
        return top_rec