import spotipy
import pandas as pd
import time
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='CLIENT_ID', client_secret='CLIENT_SECRET')) # Add Client ID and Client Secret

# Load in Data

df_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv')
df_Top_100_Billboard = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Top_100_Billboard_with_Spotify_Features.csv')
df_The_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_The_Hot_100_with_Spotify_Features.csv')
df_Top_40 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Top_40_with_Spotify_Features.csv')
df_Weekly_Hot_100 = pd.read_csv('Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Weekly_Hot_100_with_Spotify_Features.csv')
dfs = [df_Hot_100, df_Top_100_Billboard, df_The_Hot_100, df_Top_40, df_Weekly_Hot_100]

# Extract Spotify Info

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
