import librosa
import numpy as np
import pandas as pd

class Features:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.y, self.sr = librosa.load(audio_file)

    def compute_statistics(self, feature_matrix):
        return [np.mean(feature_matrix), np.std(feature_matrix), np.min(feature_matrix), np.max(feature_matrix)]

    def spectrogram(self):
        D = librosa.amplitude_to_db(librosa.stft(self.y), ref=np.max)
        return self.compute_statistics(D)

    def mel_spectrogram(self):
        mel = librosa.feature.melspectrogram(y=self.y, sr=self.sr, n_mels=128, fmax=8000)
        mel_db = librosa.amplitude_to_db(mel, ref=np.max)
        return self.compute_statistics(mel_db)

    def chroma(self):
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        return self.compute_statistics(chroma)

    def tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.y, sr=self.sr)
        return tempo  # Return BPM

    def timbre(self):
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)
        return self.compute_statistics(mfcc)

    def instrumentation(self):
        tonnetz = librosa.feature.tonnetz(y=self.y, sr=self.sr)
        return self.compute_statistics(tonnetz)

    def get_all(self):

        features = []
        # Total of 21 Features
        features.extend(self.spectrogram())      
        features.extend(self.mel_spectrogram())  
        features.extend(self.chroma())           
        features.extend(self.tempo())           
        features.extend(self.timbre())           
        features.extend(self.instrumentation())  

        return features
    
