import os
import pandas as pd
from pytube import Search
from pydub import AudioSegment
from yt_dlp import YoutubeDL

def find_song_url(query):
    yt_search = Search(query)
    yt_search_results = yt_search.results
    selected_yt_search_result = yt_search_results[0]

    print(f"\nQuery: {query}, URL: {selected_yt_search_result.watch_url}, Title {selected_yt_search_result.title}\n")

    return selected_yt_search_result.watch_url

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

    # The file is already in MP3 format, so we just need to convert it to WAV
    base, _ = os.path.splitext(filename)
    mp3_file = base + '.mp3'
    wav_file = base + '.wav'

    # Convert to WAV
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format='wav')

    # Optional: Remove the MP3 file
    # os.remove(mp3_file)

    return 

# TO DO: Add Duration of Video Check (if too long, select next result) + Add Delete Wav File

def extract_audios(df):
    for artist, song_title in zip(df['Artist'], df['Song Title']):
        print(f"Current Artist: {artist}, Current Song: {song_title}")

        found_url = find_song_url(artist + " - " + song_title)
        print(f"Query: {artist + ' - ' + song_title}, Found URL: {found_url}")

        extract_audio_of_song(youtube_url=found_url, 
                              title=song_title.replace('/', ' ') + ' by ' + artist.replace('/', ' '), 
                              path='./Retrieved_Audio/')

songs_df = pd.read_csv("Preprocessed and Labeled Datasets\Labeled_Songs_per_Top_200_Billboard.csv")
extract_audios(songs_df)
