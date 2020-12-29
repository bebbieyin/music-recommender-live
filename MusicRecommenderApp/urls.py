
from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from . import views

urlpatterns = [
    path('',views.homepageView.as_view(),name='home-page'),
    path('track',views.trackPageView.as_view(),name='track'),
    path('artist',views.artistPageView.as_view(),name='artist'),
    path('genre',views.genrePageView.as_view(),name='genre'),
    path('recommendations-track/<song_id>',views.recommendationsTrackView.as_view(),name='recommendations-track'),
    path('recommendations-artist/<artist_id>',views.recommendationsArtistView.as_view(),name='recommendations-artist'),
    path('recommendations-genre',views.recommendationsGenreView.as_view(),name='recommendations-genre'),
    path('playlist-created',views.createPlaylistView.as_view(),name='playlist-created'),
    path('profile',views.profileView.as_view(),name='profile'),
    path('about',views.aboutView.as_view(),name='about'),
    path('contact',views.contactView.as_view(),name='contact'),
    path('feedback',views.feedbackView.as_view(),name='feedback'),

    path('', include('social_django.urls')),
]
