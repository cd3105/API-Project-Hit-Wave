import os
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

def extract_audios(titles=["Gangnam Style", "Smells Like Teen Spirit", "All Summer Long", "London Calling"]):
    for current_song_title in titles:
        found_url = find_song_url(current_song_title)
        extract_audio_of_song(youtube_url=found_url, 
                              title=current_song_title, 
                              path='./Retrieved_Audio/')
