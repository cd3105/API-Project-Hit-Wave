import spotipy
import pandas as pd
import time
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='CLIENT_ID', client_secret='CLIENT_SECRET')) # Add Client ID and Client Secret

# Load in Data

df_Top_100_Billboard = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Top_100_Billboard.csv')

# Extract Spotify IDs

starting_track_index_file = open("Necessary Files for Spotify Extraction\starting_track_index.txt", "r")
starting_track_index = int(starting_track_index_file.read())

for i in range(starting_track_index, len(df_Top_100_Billboard)):
    search_response = sp.search(q='artist:' + df_Top_100_Billboard.iloc[i]['Artist'] + ' ' + 'track:' + df_Top_100_Billboard.iloc[i]['Song Title'], limit=1)

    if len(search_response['tracks']['items']) == 0:
        track_id = '-'
        track_duration = 0
    else:
        track_id = search_response['tracks']['items'][0]['id']
        track_duration = search_response['tracks']['items'][0]['duration_ms']

    with open(r"Necessary Files for Spotify Extraction\track_ids.txt", "a") as track_id_file:
        track_id_file.write(track_id + '\n')

    with open(r"Necessary Files for Spotify Extraction\track_durations.txt", "a") as track_duration_file:
        track_duration_file.write(str(track_duration) + '\n')

    with open(r"Necessary Files for Spotify Extraction\starting_track_index.txt", "w") as starting_track_index_file:
        starting_track_index_file.write(str(i+1))

    time.sleep(0.5)
