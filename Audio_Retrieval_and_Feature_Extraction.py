import os
import pandas as pd
from pytube import Search
from pydub import AudioSegment
from yt_dlp import YoutubeDL
from Feature_Extraction import Features

def find_song_url(query):
    yt_search = Search(query)
    yt_search_results = yt_search.results
    selected_yt_search_result = yt_search_results[0]

    return selected_yt_search_result.watch_url, selected_yt_search_result.title

def extract_audio_of_song(youtube_url, title, path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': path + title + '.%(ext)s',
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info)

    base, _ = os.path.splitext(filename)
    mp3_file = base + '.mp3'
    wav_file = base + '.wav'

    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format='wav')

    os.remove(mp3_file)

    return wav_file

def extract_audios(s_idx, df, feat_df):
    for i in range(s_idx, len(df)):
        artist = df.iloc[i]['Artist']
        song_title = df.iloc[i]['Song Title']

        found_url, title_video = find_song_url(artist + " - " + song_title)
        audio_file = extract_audio_of_song(youtube_url=found_url, 
                                           title=song_title.replace('/', ' ') + ' by ' + artist.replace('/', ' '), 
                                           path='./Retrieved_Audio/')
        feat_df.loc[i] = {"Video Title of Audio": title_video} | Features(audio_file).get_all()
        feat_df.to_csv("Audio Features Datasets\Audio_Features_Dataset.csv", index=False)

        os.remove(audio_file)

    return feat_df

songs_df = pd.read_csv("Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv")
features_df = pd.read_csv("Audio Features Datasets\Audio_Features_Dataset.csv")
start_idx = len(features_df)

features_df = extract_audios(start_idx, songs_df[:2], features_df)
print(features_df)
print(songs_df[:2].join(features_df))
