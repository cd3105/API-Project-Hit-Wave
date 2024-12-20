import os
import pandas as pd
from pytube import Search
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError
from Feature_Extraction import Features
from mutagen.mp3 import MP3


def find_youtube_videos_of_song(query): # Function for querying Youtube and retrieving the results of the query
    yt_search = Search(query)

    return yt_search.results


def extract_audio_of_song(youtube_url, title, path): # Function to extract audio from YouTube
    ydl_opts_1 = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': path + title + '.%(ext)s',
    }

    ydl_opts_2 = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': path + title + '.%(ext)s',
        'cookiefile': r'"C:\Users\Gebruiker\Downloads\www.youtube.com_cookies.txt"', # Add Path to Your Cookie.txt file containing your YouTube cookies
    }

    try:
        with YoutubeDL(ydl_opts_1) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
    except DownloadError: # Try with Cookies
        try:
            with YoutubeDL(ydl_opts_2) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                filename = ydl.prepare_filename(info)
        except DownloadError:
            return "ERROR", "ERROR"

    base, _ = os.path.splitext(filename)
    mp3_file = base + '.mp3'
    wav_file = base + '.wav'

    return mp3_file, wav_file


def extract_audios(s_idx, df, feat_df): # Extract audio from all songs and extract corresponding audio features
    for i in range(s_idx, len(df)):
        artist = df.iloc[i]['Artist']
        song_title = df.iloc[i]['Song Title']

        if artist == "Halloween" and song_title == "HALLOWEEN WIND": # Handles specific query as it only found extremely long videos that did not match the Spotify track
            found_videos = find_youtube_videos_of_song("Howling Wind 2 - Scary Halloween Sound Effects - Halloween Sound Effects")
        elif artist == "Death" and song_title == "SUICIDE MACHINE": # Handles specific query as it let to no search results due to Suicide Prevention blocking it
            found_videos = find_youtube_videos_of_song("Death - SUICIDE MACHINE Lyrics")
        elif artist == "Snoop Dogg" and song_title == "DOGGYLAND": # Handles specific query as it results in a live stream
            found_videos = find_youtube_videos_of_song("Snoop Doggy Dogg - Doggyland King of Rap")
        elif artist == "Bing Crosby" and song_title == "CHRISTMAS IS": # Handles specific query as it results in a live stream
            found_videos = find_youtube_videos_of_song("bing crosby christmas is a comin")
        elif artist == "Nirvana" and song_title == "I HATE MYSELF AND WANT TO DIE": # Handles specific query as it let to no search results due to Suicide Prevention blocking it
            found_videos = find_youtube_videos_of_song("Nirvana - I HATE MYSELF AND WANT TO DIE Lyrics")
        else:
            found_videos = find_youtube_videos_of_song(artist + " - " + song_title)

        for video in found_videos:
            found_url = video.watch_url
            title_video = video.title

            audio_file_mp3, audio_file_wav = extract_audio_of_song(youtube_url=found_url,
                                                                   title=song_title.replace('/', ' ') + ' by ' + artist.replace('/', ' '), 
                                                                   path='./Retrieved_Audio/')

            try:
                if audio_file_mp3 != "ERROR":         
                    if MP3(audio_file_mp3).info.length < 1200:
                        audio = AudioSegment.from_mp3(audio_file_mp3)
                        audio.export(audio_file_wav, format='wav')
                        os.remove(audio_file_mp3)

                        break
                
                    os.remove(audio_file_mp3)
            except CouldntDecodeError:
                os.remove(audio_file_mp3)
                continue

        feat_df.loc[i] = {"Video Title of Audio": title_video} | Features(audio_file_wav).get_all()
        feat_df.to_csv("Audio Features Datasets\Audio_Features_Dataset.csv", index=False)

        os.remove(audio_file_wav)

    return


if not os.path.exists("Audio Features Datasets\Audio_Features_Dataset.csv"): # Define audio features dataset
    pd.DataFrame(columns=['Video Title of Audio', 
                          'Duration', 
                          'Tempo', 

                          'MEAN Spectrogram over all Bins', 
                          'MEDIAN Spectrogram over all Bins', 
                          'STD Spectrogram over all Bins', 
                          'MIN Spectrogram over all Bins', 
                          'MAX Spectrogram over all Bins', 

                          'MEAN Mel Spectrogram over all Mels', 
                          'MEDIAN Mel Spectrogram over all Mels', 
                          'STD Mel Spectrogram over all Mels', 
                          'MIN Mel Spectrogram over all Mels', 
                          'MAX Mel Spectrogram over all Mels', 
                          
                          'MEAN Chroma 1', 
                          'MEDIAN Chroma 1', 
                          'STD Chroma 1', 
                          'MIN Chroma 1', 
                          'MAX Chroma 1', 
                          
                          'MEAN Chroma 2', 
                          'MEDIAN Chroma 2', 
                          'STD Chroma 2', 
                          'MIN Chroma 2', 
                          'MAX Chroma 2', 

                          'MEAN Chroma 3', 
                          'MEDIAN Chroma 3', 
                          'STD Chroma 3', 
                          'MIN Chroma 3', 
                          'MAX Chroma 3', 
                          
                          'MEAN Chroma 4', 
                          'MEDIAN Chroma 4', 
                          'STD Chroma 4', 
                          'MIN Chroma 4', 
                          'MAX Chroma 4', 
                          
                          'MEAN Chroma 5', 
                          'MEDIAN Chroma 5', 
                          'STD Chroma 5', 
                          'MIN Chroma 5', 
                          'MAX Chroma 5', 
                          
                          'MEAN Chroma 6', 
                          'MEDIAN Chroma 6', 
                          'STD Chroma 6', 
                          'MIN Chroma 6', 
                          'MAX Chroma 6', 
                          
                          'MEAN Chroma 7', 
                          'MEDIAN Chroma 7', 
                          'STD Chroma 7', 
                          'MIN Chroma 7', 
                          'MAX Chroma 7', 
                          
                          'MEAN Chroma 8', 
                          'MEDIAN Chroma 8', 
                          'STD Chroma 8', 
                          'MIN Chroma 8', 
                          'MAX Chroma 8', 
                          
                          'MEAN Chroma 9', 
                          'MEDIAN Chroma 9', 
                          'STD Chroma 9', 
                          'MIN Chroma 9', 
                          'MAX Chroma 9', 
                          
                          'MEAN Chroma 10', 
                          'MEDIAN Chroma 10', 
                          'STD Chroma 10', 
                          'MIN Chroma 10', 
                          'MAX Chroma 10', 
                          
                          'MEAN Chroma 11', 
                          'MEDIAN Chroma 11', 
                          'STD Chroma 11', 
                          'MIN Chroma 11', 
                          'MAX Chroma 11', 
                          
                          'MEAN Chroma 12', 
                          'MEDIAN Chroma 12', 
                          'STD Chroma 12', 
                          'MIN Chroma 12', 
                          'MAX Chroma 12', 
                          
                          'MEAN MFCC 1 (Timbre)', 
                          'MEDIAN MFCC 1 (Timbre)', 
                          'STD MFCC 1 (Timbre)', 
                          'MIN MFCC 1 (Timbre)', 
                          'MAX MFCC 1 (Timbre)', 
                          
                          'MEAN MFCC 2 (Timbre)', 
                          'MEDIAN MFCC 2 (Timbre)', 
                          'STD MFCC 2 (Timbre)', 
                          'MIN MFCC 2 (Timbre)', 
                          'MAX MFCC 2 (Timbre)', 
                          
                          'MEAN MFCC 3 (Timbre)', 
                          'MEDIAN MFCC 3 (Timbre)', 
                          'STD MFCC 3 (Timbre)', 
                          'MIN MFCC 3 (Timbre)', 
                          'MAX MFCC 3 (Timbre)', 
                          
                          'MEAN MFCC 4 (Timbre)', 
                          'MEDIAN MFCC 4 (Timbre)', 
                          'STD MFCC 4 (Timbre)', 
                          'MIN MFCC 4 (Timbre)', 
                          'MAX MFCC 4 (Timbre)', 
                          
                          'MEAN MFCC 5 (Timbre)', 
                          'MEDIAN MFCC 5 (Timbre)', 
                          'STD MFCC 5 (Timbre)', 
                          'MIN MFCC 5 (Timbre)', 
                          'MAX MFCC 5 (Timbre)', 
                          
                          'MEAN MFCC 6 (Timbre)', 
                          'MEDIAN MFCC 6 (Timbre)', 
                          'STD MFCC 6 (Timbre)', 
                          'MIN MFCC 6 (Timbre)', 
                          'MAX MFCC 6 (Timbre)', 
                          
                          'MEAN MFCC 7 (Timbre)', 
                          'MEDIAN MFCC 7 (Timbre)', 
                          'STD MFCC 7 (Timbre)', 
                          'MIN MFCC 7 (Timbre)', 
                          'MAX MFCC 7 (Timbre)', 
                          
                          'MEAN MFCC 8 (Timbre)', 
                          'MEDIAN MFCC 8 (Timbre)', 
                          'STD MFCC 8 (Timbre)', 
                          'MIN MFCC 8 (Timbre)', 
                          'MAX MFCC 8 (Timbre)', 
                          
                          'MEAN MFCC 9 (Timbre)', 
                          'MEDIAN MFCC 9 (Timbre)', 
                          'STD MFCC 9 (Timbre)', 
                          'MIN MFCC 9 (Timbre)', 
                          'MAX MFCC 9 (Timbre)', 

                          'MEAN MFCC 10 (Timbre)', 
                          'MEDIAN MFCC 10 (Timbre)', 
                          'STD MFCC 10 (Timbre)', 
                          'MIN MFCC 10 (Timbre)', 
                          'MAX MFCC 10 (Timbre)', 
                          
                          'MEAN MFCC 11 (Timbre)', 
                          'MEDIAN MFCC 11 (Timbre)', 
                          'STD MFCC 11 (Timbre)', 
                          'MIN MFCC 11 (Timbre)', 
                          'MAX MFCC 11 (Timbre)', 
                          
                          'MEAN MFCC 12 (Timbre)', 
                          'MEDIAN MFCC 12 (Timbre)', 
                          'STD MFCC 12 (Timbre)', 
                          'MIN MFCC 12 (Timbre)', 
                          'MAX MFCC 12 (Timbre)', 
                          
                          'MEAN MFCC 13 (Timbre)', 
                          'MEDIAN MFCC 13 (Timbre)', 
                          'STD MFCC 13 (Timbre)', 
                          'MIN MFCC 13 (Timbre)', 
                          'MAX MFCC 13 (Timbre)', 
                          
                          'MEAN Tonnetz Fifth X (Instrumentation)', 
                          'MEDIAN Tonnetz Fifth X (Instrumentation)', 
                          'STD Tonnetz Fifth X (Instrumentation)', 
                          'MIN Tonnetz Fifth X (Instrumentation)', 
                          'MAX Tonnetz Fifth X (Instrumentation)', 
                          
                          'MEAN Tonnetz Fifth Y (Instrumentation)', 
                          'MEDIAN Tonnetz Fifth Y (Instrumentation)', 
                          'STD Tonnetz Fifth Y (Instrumentation)', 
                          'MIN Tonnetz Fifth Y (Instrumentation)', 
                          'MAX Tonnetz Fifth Y (Instrumentation)', 
                          
                          'MEAN Tonnetz Minor X (Instrumentation)', 
                          'MEDIAN Tonnetz Minor X (Instrumentation)', 
                          'STD Tonnetz Minor X (Instrumentation)', 
                          'MIN Tonnetz Minor X (Instrumentation)', 
                          'MAX Tonnetz Minor X (Instrumentation)', 
                          
                          'MEAN Tonnetz Minor Y (Instrumentation)', 
                          'MEDIAN Tonnetz Minor Y (Instrumentation)', 
                          'STD Tonnetz Minor Y (Instrumentation)', 
                          'MIN Tonnetz Minor Y (Instrumentation)', 
                          'MAX Tonnetz Minor Y (Instrumentation)', 
                          
                          'MEAN Tonnetz Major X (Instrumentation)', 
                          'MEDIAN Tonnetz Major X (Instrumentation)', 
                          'STD Tonnetz Major X (Instrumentation)', 
                          'MIN Tonnetz Major X (Instrumentation)', 
                          'MAX Tonnetz Major X (Instrumentation)', 
                          
                          'MEAN Tonnetz Major Y (Instrumentation)', 
                          'MEDIAN Tonnetz Major Y (Instrumentation)', 
                          'STD Tonnetz Major Y (Instrumentation)', 
                          'MIN Tonnetz Major Y (Instrumentation)', 
                          'MAX Tonnetz Major Y (Instrumentation)', 
                          
                          'MEAN RMS', 
                          'MEDIAN RMS', 
                          'STD RMS', 
                          'MIN RMS', 
                          'MAX RMS', 
                          
                          'MEAN Spectral Centroid', 
                          'MEDIAN Spectral Centroid', 
                          'STD Spectral Centroid', 
                          'MIN Spectral Centroid', 
                          'MAX Spectral Centroid', 
                          
                          'MEAN Spectral Bandwidth', 
                          'MEDIAN Spectral Bandwidth', 
                          'STD Spectral Bandwidth', 
                          'MIN Spectral Bandwidth', 
                          'MAX Spectral Bandwidth', 
                          
                          'MEAN Zero Crossing Rate', 
                          'MEDIAN Zero Crossing Rate', 
                          'STD Zero Crossing Rate', 
                          'MIN Zero Crossing Rate', 
                          'MAX Zero Crossing Rate']).to_csv("Audio Features Datasets\Audio_Features_Dataset.csv", 
                                                            index=False)
    
songs_df = pd.read_csv("Reordered Preprocessed and Labeled Datasets with Spotify Features\Reordered_Labeled_Songs_per_Hot_100_with_Spotify_Features.csv")
features_df = pd.read_csv("Audio Features Datasets\Audio_Features_Dataset.csv")
start_idx = len(features_df)

extract_audios(start_idx, songs_df, features_df)
