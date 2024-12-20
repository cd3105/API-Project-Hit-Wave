import spotipy
import pandas as pd
import time
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='CLIENT_ID', client_secret='CLIENT_SECRET')) # Add Client ID and Client Secret

# Load in Data

df_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Hot_100.csv')
df_Top_100_Billboard = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Top_100_Billboard.csv')
df_The_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_The_Hot_100.csv')
df_Top_40 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Top_40.csv')
df_Weekly_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Weekly_Hot_100.csv')
dfs = [df_Hot_100, df_Top_100_Billboard, df_The_Hot_100, df_Top_40, df_Weekly_Hot_100]

# Extract Audio Features

track_ids_file = open(r"Necessary Files for Spotify Extraction\track_ids.txt", "r")
track_durations_file = open(r"Necessary Files for Spotify Extraction\track_durations.txt", "r")
track_ids = track_ids_file.read().split('\n')[:-1]
track_durations = [int(td) for td in track_durations_file.read().split('\n')[:-1]]

for df in dfs:
    df['Spotify ID'] = track_ids
    df['Spotify Duration'] = track_durations

df_Hot_100 = df_Hot_100[df_Hot_100['Spotify ID'] != '-'].reset_index(drop=True)
df_Top_100_Billboard = df_Top_100_Billboard[df_Top_100_Billboard['Spotify ID'] != '-'].reset_index(drop=True)
df_The_Hot_100 = df_The_Hot_100[df_The_Hot_100['Spotify ID'] != '-'].reset_index(drop=True)
df_Top_40 = df_Top_40[df_Top_40['Spotify ID'] != '-'].reset_index(drop=True)
df_Weekly_Hot_100 = df_Weekly_Hot_100[df_Weekly_Hot_100['Spotify ID'] != '-'].reset_index(drop=True)

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

for i in range(0, len(df_Hot_100), 100):
    current_partition = df_Hot_100.iloc[i:i+100]
    current_audio_features = sp.audio_features(tracks=list(current_partition['Spotify ID']))

    for j in range(len(current_partition)):
        try:
            track_danceability.append(current_audio_features[j]['danceability'])
            track_energy.append(current_audio_features[j]['energy'])
            track_key.append(current_audio_features[j]['key'])
            track_loudness.append(current_audio_features[j]['loudness'])
            track_mode.append(current_audio_features[j]['mode'])
            track_speechiness.append(current_audio_features[j]['speechiness'])
            track_acousticness.append(current_audio_features[j]['acousticness'])
            track_instrumentalness.append(current_audio_features[j]['instrumentalness'])
            track_liveness.append(current_audio_features[j]['liveness'])
            track_valence.append(current_audio_features[j]['valence'])
            track_tempo.append(current_audio_features[j]['tempo'])
        except TypeError:
            track_danceability.append(-100)
            track_energy.append(-100)
            track_key.append(-100)
            track_loudness.append(-100)
            track_mode.append(-100)
            track_speechiness.append(-100)
            track_acousticness.append(-100)
            track_instrumentalness.append(-100)
            track_liveness.append(-100)
            track_valence.append(-100)
            track_tempo.append(-100)

    time.sleep(0.5)

dfs = [df_Hot_100, df_Top_100_Billboard, df_The_Hot_100, df_Top_40, df_Weekly_Hot_100]

for df in dfs:
    df['Spotify Danceability'] = track_danceability
    df['Spotify Energy'] = track_energy
    df['Spotify Key'] = track_key
    df['Spotify Loudness'] = track_loudness
    df['Spotify Mode'] = track_mode
    df['Spotify Speechiness'] = track_speechiness
    df['Spotify Acousticness'] = track_acousticness
    df['Spotify Instrumentalness'] = track_instrumentalness
    df['Spotify Liveness'] = track_liveness
    df['Spotify Valence'] = track_valence
    df['Spotify Tempo'] = track_tempo

df_Hot_100 = df_Hot_100[df_Hot_100['Spotify Danceability'] != -100].reset_index(drop=True)
df_Top_100_Billboard = df_Top_100_Billboard[df_Top_100_Billboard['Spotify Danceability'] != -100].reset_index(drop=True)
df_The_Hot_100 = df_The_Hot_100[df_The_Hot_100['Spotify Danceability'] != -100].reset_index(drop=True)
df_Top_40 = df_Top_40[df_Top_40['Spotify Danceability'] != -100].reset_index(drop=True)
df_Weekly_Hot_100 = df_Weekly_Hot_100[df_Weekly_Hot_100['Spotify Danceability'] != -100].reset_index(drop=True)

df_Hot_100.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv', index=False)
df_The_Hot_100.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_The_Hot_100_with_Spotify_Features.csv', index=False)
df_Top_40.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Top_40_with_Spotify_Features.csv', index=False)
df_Top_100_Billboard.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Top_100_Billboard_with_Spotify_Features.csv', index=False)
df_Weekly_Hot_100.to_csv('Preprocessed and Labeled Datasets with Features\Labeled_Songs_per_Weekly_Hot_100_with_Spotify_Features.csv', index=False)
