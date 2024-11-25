import spotipy
import pandas as pd
import time
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='b34802dab95a426fa84e61b33622a7dc', client_secret='5c009c91c7434070b2ba505e4ee860e2'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='f4b2f1282b6741dabf3a80a192e93129', client_secret='3058861b6b074d8597342c78f45af998'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='6354015b866d4c78a6e470d60ce24103', client_secret='65498c0c4674420cbf2e0f866c08dba7'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='efc7df0a61284315affdc4d2082f2119', client_secret='d99def372b13439ba8eba4b769980159'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='d130601ff30641d5af8832b4221014e2', client_secret='aaa2484ad3834adab16322244cdaf37a'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='949f078fb1a24aa492f288cdd32a2919', client_secret='ae585fc53353491c901e75ae349861aa'))

df_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Hot_100.csv')
df_The_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_The_Hot_100.csv')
df_Top_40 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Top_40.csv')
df_Top_100_Billboard = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Top_100_Billboard.csv')
df_Weekly_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Weekly_Hot_100.csv')
dfs = [df_Hot_100, df_The_Hot_100, df_Top_40, df_Top_100_Billboard, df_Weekly_Hot_100]

track_ids = []
track_durations = []
track_danceability = []
track_energy = []
track_key = []
track_loudness = []
track_mode = []
track_speechiness = []
track_acousticness = []
track_instrumentalness = []
track_liveness = []
track_valence = []
track_tempo = []

no_track_found = 0

for i, r in df_Hot_100.iterrows():
    print(f"Index: {i}")

    search_response = sp.search(q='artist:' + r['Artist'] + ' ' + 'track:' + r['Song Title'], limit=1)

    if len(search_response['tracks']['items']) == 0:
        track_id = ''
        track_duration = 0
        no_track_found += 1
    else:
        track_id = search_response['tracks']['items'][0]['id']
        track_duration = search_response['tracks']['items'][0]['duration_ms']

    track_ids.append(track_id)
    track_durations.append(track_duration)

    time.sleep(0.5)

print(f"Number of tracks not found in Spotify: {no_track_found}")

for df in dfs:
    df['Spotify ID'] = track_ids
    df['Spotify Duration'] = track_durations

    # track_features = sp.audio_features([track_id])

    # track_danceability.append(track_features[0]['danceability'])
    # track_energy.append(track_features[0]['energy'])
    # track_key.append(track_features[0]['key'])
    # track_loudness.append(track_features[0]['loudness'])
    # track_mode.append(track_features[0]['mode'])
    # track_speechiness.append(track_features[0]['speechiness'])
    # track_acousticness.append(track_features[0]['acousticness'])
    # track_instrumentalness.append(track_features[0]['instrumentalness'])
    # track_liveness.append(track_features[0]['liveness'])
    # track_valence.append(track_features[0]['valence'])
    # track_tempo.append(track_features[0]['tempo'])

# for df in dfs:
#     df['Spotify ID'] = track_ids
#     df['Spotify Duration'] = track_durations
#     df['Spotify Danceability'] = track_danceability
#     df['Spotify Energy'] = track_energy
#     df['Spotify Key'] = track_key
#     df['Spotify Loudness'] = track_loudness
#     df['Spotify Mode'] = track_mode
#     df['Spotify Speechiness'] = track_speechiness
#     df['Spotify Acousticness'] = track_acousticness
#     df['Spotify Instrumentalness'] = track_instrumentalness
#     df['Spotify Liveness'] = track_liveness
#     df['Spotify Valence'] = track_valence
#     df['Spotify Tempo'] = track_tempo

df_Hot_100.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv')
df_The_Hot_100.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_The_Hot_100_with_Spotify_Features.csv')
df_Top_40.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Top_40_with_Spotify_Features.csv')
df_Top_100_Billboard.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Top_100_Billboard_with_Spotify_Features.csv')
df_Weekly_Hot_100.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Weekly_Hot_100_with_Spotify_Features.csv')
