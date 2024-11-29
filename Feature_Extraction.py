import librosa
import numpy as np
import pandas as pd

class Features:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.y, self.sr = librosa.load(audio_file)

    def compute_statistics(self, feature_matrix):
        return [np.mean(feature_matrix), np.std(feature_matrix), np.min(feature_matrix), np.max(feature_matrix)]

    def duration(self):
        duration = librosa.get_duration(y=self.y, 
                                        sr=self.sr)
        return duration

    def spectrogram(self):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), 
                                    ref=np.max)
        return D #self.compute_statistics(D)

    def mel_spectrogram(self):
        mel = librosa.feature.melspectrogram(y=self.y, 
                                             sr=self.sr, 
                                             n_mels=128, 
                                             fmax=8000)
        mel_db = librosa.amplitude_to_db(mel, 
                                         ref=np.max)
        return mel_db #self.compute_statistics(mel_db)

    def chroma(self):
        chroma = librosa.feature.chroma_stft(y=self.y, 
                                             sr=self.sr)
        return chroma #self.compute_statistics(chroma)

    def tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.y, 
                                           sr=self.sr)
        return tempo  # Return BPM

    def mfcc(self):
        mfcc = librosa.feature.mfcc(y=self.y, 
                                    sr=self.sr, 
                                    n_mfcc=13)
        return mfcc #self.compute_statistics(mfcc)

    def tonnetz(self):
        tonnetz = librosa.feature.tonnetz(y=self.y, 
                                          sr=self.sr)
        return tonnetz #self.compute_statistics(tonnetz)

    def rms(self):
        rms = librosa.feature.rms(y=self.y)
        return rms

    def spectral_centroid(self):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y,
                                                              sr=self.sr)
        return spectral_centroid

    def spectral_bandwidth(self):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, 
                                                                sr=self.sr)
        return spectral_bandwidth

    def zero_crossing_rate(self):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=self.y)
        return zero_crossing_rate

    def get_all(self):

        features = {
            "Duration": self.duration(),
            "Spectrogram": self.spectrogram(),
            "Spectrogram": self.spectrogram(),
            "Mel Spectrogram": self.mel_spectrogram(),
            "Chroma": self.chroma(),
            "Tempo": self.tempo(),
            "MFCC (Timbre)": self.mfcc(),
            "Tonnetz (Instrumentation)": self.tonnetz(),
            "RMS": self.rms(),
            "Spectral Centroid": self.spectral_centroid(),
            "Spectral Bandwidth": self.spectral_bandwidth(),
            "Zero Crossing Rate": self.zero_crossing_rate()
            }      

        return features
    
