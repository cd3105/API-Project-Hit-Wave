import spotipy
import pandas as pd
import time
from spotipy.oauth2 import SpotifyClientCredentials

#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='b34802dab95a426fa84e61b33622a7dc', client_secret='5c009c91c7434070b2ba505e4ee860e2'))
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='f4b2f1282b6741dabf3a80a192e93129', client_secret='3058861b6b074d8597342c78f45af998'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='6354015b866d4c78a6e470d60ce24103', client_secret='65498c0c4674420cbf2e0f866c08dba7'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='efc7df0a61284315affdc4d2082f2119', client_secret='d99def372b13439ba8eba4b769980159'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='d130601ff30641d5af8832b4221014e2', client_secret='aaa2484ad3834adab16322244cdaf37a'))
#sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id='949f078fb1a24aa492f288cdd32a2919', client_secret='ae585fc53353491c901e75ae349861aa'))

df_Top_100_Billboard = pd.read_csv('Preprocessed and Labeled Datasets\Labeled_Songs_per_Top_100_Billboard.csv')

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
