import spotipy
import pandas as pd
import time
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='df7dade4bdb441ff828575ef7c958089', client_secret='7cd1ea040ce5466f969e2ee3cc15e2af'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='f4b2f1282b6741dabf3a80a192e93129', client_secret='3058861b6b074d8597342c78f45af998'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='6354015b866d4c78a6e470d60ce24103', client_secret='65498c0c4674420cbf2e0f866c08dba7'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='efc7df0a61284315affdc4d2082f2119', client_secret='d99def372b13439ba8eba4b769980159'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='d130601ff30641d5af8832b4221014e2', client_secret='aaa2484ad3834adab16322244cdaf37a'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='949f078fb1a24aa492f288cdd32a2919', client_secret='ae585fc53353491c901e75ae349861aa'))

df_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv')
df_Top_100_Billboard = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Top_100_Billboard_with_Spotify_Features.csv')
df_The_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_The_Hot_100_with_Spotify_Features.csv')
df_Top_40 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Top_40_with_Spotify_Features.csv')
df_Weekly_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Weekly_Hot_100_with_Spotify_Features.csv')
dfs = [df_Hot_100, df_Top_100_Billboard, df_The_Hot_100, df_Top_40, df_Weekly_Hot_100]

track_name = []
track_primary_artist = []
release_year = []

for i in range(0, len(df_Hot_100), 50):
    current_partition = df_Hot_100.iloc[i:i+50]
    current_tracks = sp.tracks(tracks=list(current_partition['Spotify ID']))

    for j in range(len(current_partition)):
        track_name.append(current_tracks['tracks'][j]['name'])
        track_primary_artist.append(current_tracks['tracks'][j]['artists'][0]['name'])
        release_year.append(int(current_tracks['tracks'][j]['album']['release_date'].split('-')[0]))

    time.sleep(0.5)

for df in dfs:
    df['Spotify Song Title'] = track_name
    df['Spotify Primary Artist'] = track_primary_artist
    df['Release Year'] = release_year

df_Hot_100.to_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv', index=False)
df_The_Hot_100.to_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_The_Hot_100_with_Spotify_Features.csv', index=False)
df_Top_40.to_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Top_40_with_Spotify_Features.csv', index=False)
df_Top_100_Billboard.to_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Top_100_Billboard_with_Spotify_Features.csv', index=False)
df_Weekly_Hot_100.to_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Weekly_Hot_100_with_Spotify_Features.csv', index=False)
