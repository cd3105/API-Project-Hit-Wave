import librosa
import numpy as np
import pandas as pd

class Features:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.y, self.sr = librosa.load(audio_file)

    def compute_statistics(self, feature_matrix):
        return (np.mean(feature_matrix), 
                np.median(feature_matrix), 
                np.std(feature_matrix), 
                np.min(feature_matrix), 
                np.max(feature_matrix))

    def compute_statistics_1(self, feature_matrix):
        return (np.mean(feature_matrix, axis=1), 
                np.median(feature_matrix, axis=1), 
                np.std(feature_matrix, axis=1), 
                np.min(feature_matrix, axis=1), 
                np.max(feature_matrix, axis=1))

    def duration(self):
        duration = librosa.get_duration(y=self.y, 
                                        sr=self.sr)
        return duration

    def tempo(self):
        tempo, _ = librosa.beat.beat_track(y=self.y, 
                                           sr=self.sr)
        return tempo  

    def spectrogram(self):
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.y)), 
                                    ref=np.max)
        return self.compute_statistics(D)

    def mel_spectrogram(self):
        mel = librosa.feature.melspectrogram(y=self.y, 
                                             sr=self.sr, 
                                             n_mels=128, 
                                             fmax=8000)
        mel_db = librosa.amplitude_to_db(mel, 
                                         ref=np.max)
        return self.compute_statistics(mel_db)

    def chroma(self):
        chroma = librosa.feature.chroma_stft(y=self.y, 
                                             sr=self.sr)
        return self.compute_statistics_1(chroma)

    def mfcc(self):
        mfcc = librosa.feature.mfcc(y=self.y, 
                                    sr=self.sr, 
                                    n_mfcc=13)
        return self.compute_statistics_1(mfcc)

    def tonnetz(self):
        tonnetz = librosa.feature.tonnetz(y=self.y, 
                                          sr=self.sr)
        return self.compute_statistics_1(tonnetz)

    def rms(self):
        rms = librosa.feature.rms(y=self.y)
        return self.compute_statistics(rms)

    def spectral_centroid(self):
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y,
                                                              sr=self.sr)
        return self.compute_statistics(spectral_centroid)

    def spectral_bandwidth(self):
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, 
                                                                sr=self.sr)
        return self.compute_statistics(spectral_bandwidth)

    def zero_crossing_rate(self):
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=self.y)
        return self.compute_statistics(zero_crossing_rate)

    def get_all(self):
        spectrogram_features = self.spectrogram()
        mel_spectrogram_features = self.mel_spectrogram()
        chroma_features = self.chroma()
        mffc_features = self.mfcc()
        tonnetz_features = self.tonnetz()
        rms_features = self.rms()
        spectral_centroid_features = self.spectral_centroid()
        spectral_bandwidth_features = self.spectral_bandwidth()
        zero_crossing_rate_features = self.zero_crossing_rate()

        features = {
            "Duration": self.duration(),
            "Tempo": self.tempo(),

            "MEAN Spectrogram over all Bins": spectrogram_features[0],
            "MEDIAN Spectrogram over all Bins": spectrogram_features[1],
            "STD Spectrogram over all Bins": spectrogram_features[2],
            "MIN Spectrogram over all Bins": spectrogram_features[3],
            "MAX Spectrogram over all Bins": spectrogram_features[4],

            "MEAN Mel Spectrogram over all Mels": mel_spectrogram_features[0],
            "MEDIAN Mel Spectrogram over all Mels": mel_spectrogram_features[1],
            "STD Mel Spectrogram over all Mels": mel_spectrogram_features[2],
            "MIN Mel Spectrogram over all Mels": mel_spectrogram_features[3],
            "MAX Mel Spectrogram over all Mels": mel_spectrogram_features[4],

            "MEAN Chroma 1": chroma_features[0][0],
            "MEDIAN Chroma 1": chroma_features[1][0],
            "STD Chroma 1": chroma_features[2][0],
            "MIN Chroma 1": chroma_features[3][0],
            "MAX Chroma 1": chroma_features[4][0],

            "MEAN Chroma 2": chroma_features[0][1],
            "MEDIAN Chroma 2": chroma_features[1][1],
            "STD Chroma 2": chroma_features[2][1],
            "MIN Chroma 2": chroma_features[3][1],
            "MAX Chroma 2": chroma_features[4][1],

            "MEAN Chroma 3": chroma_features[0][2],
            "MEDIAN Chroma 3": chroma_features[1][2],
            "STD Chroma 3": chroma_features[2][2],
            "MIN Chroma 3": chroma_features[3][2],
            "MAX Chroma 3": chroma_features[4][2],

            "MEAN Chroma 4": chroma_features[0][3],
            "MEDIAN Chroma 4": chroma_features[1][3],
            "STD Chroma 4": chroma_features[2][3],
            "MIN Chroma 4": chroma_features[3][3],
            "MAX Chroma 4": chroma_features[4][3],

            "MEAN Chroma 5": chroma_features[0][4],
            "MEDIAN Chroma 5": chroma_features[1][4],
            "STD Chroma 5": chroma_features[2][4],
            "MIN Chroma 5": chroma_features[3][4],
            "MAX Chroma 5": chroma_features[4][4],

            "MEAN Chroma 6": chroma_features[0][5],
            "MEDIAN Chroma 6": chroma_features[1][5],
            "STD Chroma 6": chroma_features[2][5],
            "MIN Chroma 6": chroma_features[3][5],
            "MAX Chroma 6": chroma_features[4][5],

            "MEAN Chroma 7": chroma_features[0][6],
            "MEDIAN Chroma 7": chroma_features[1][6],
            "STD Chroma 7": chroma_features[2][6],
            "MIN Chroma 7": chroma_features[3][6],
            "MAX Chroma 7": chroma_features[4][6],

            "MEAN Chroma 8": chroma_features[0][7],
            "MEDIAN Chroma 8": chroma_features[1][7],
            "STD Chroma 8": chroma_features[2][7],
            "MIN Chroma 8": chroma_features[3][7],
            "MAX Chroma 8": chroma_features[4][7],

            "MEAN Chroma 9": chroma_features[0][8],
            "MEDIAN Chroma 9": chroma_features[1][8],
            "STD Chroma 9": chroma_features[2][8],
            "MIN Chroma 9": chroma_features[3][8],
            "MAX Chroma 9": chroma_features[4][8],

            "MEAN Chroma 10": chroma_features[0][9],
            "MEDIAN Chroma 10": chroma_features[1][9],
            "STD Chroma 10": chroma_features[2][9],
            "MIN Chroma 10": chroma_features[3][9],
            "MAX Chroma 10": chroma_features[4][9],

            "MEAN Chroma 11": chroma_features[0][10],
            "MEDIAN Chroma 11": chroma_features[1][10],
            "STD Chroma 11": chroma_features[2][10],
            "MIN Chroma 11": chroma_features[3][10],
            "MAX Chroma 11": chroma_features[4][10],

            "MEAN Chroma 12": chroma_features[0][11],
            "MEDIAN Chroma 12": chroma_features[1][11],
            "STD Chroma 12": chroma_features[2][11],
            "MIN Chroma 12": chroma_features[3][11],
            "MAX Chroma 12": chroma_features[4][11],

            "MEAN MFCC 1 (Timbre)": mffc_features[0][0],
            "MEDIAN MFCC 1 (Timbre)": mffc_features[1][0],
            "STD MFCC 1 (Timbre)": mffc_features[2][0],
            "MIN MFCC 1 (Timbre)": mffc_features[3][0],
            "MAX MFCC 1 (Timbre)": mffc_features[4][0],

            "MEAN MFCC 2 (Timbre)": mffc_features[0][1],
            "MEDIAN MFCC 2 (Timbre)": mffc_features[1][1],
            "STD MFCC 2 (Timbre)": mffc_features[2][1],
            "MIN MFCC 2 (Timbre)": mffc_features[3][1],
            "MAX MFCC 2 (Timbre)": mffc_features[4][1],

            "MEAN MFCC 3 (Timbre)": mffc_features[0][2],
            "MEDIAN MFCC 3 (Timbre)": mffc_features[1][2],
            "STD MFCC 3 (Timbre)": mffc_features[2][2],
            "MIN MFCC 3 (Timbre)": mffc_features[3][2],
            "MAX MFCC 3 (Timbre)": mffc_features[4][2],

            "MEAN MFCC 4 (Timbre)": mffc_features[0][3],
            "MEDIAN MFCC 4 (Timbre)": mffc_features[1][3],
            "STD MFCC 4 (Timbre)": mffc_features[2][3],
            "MIN MFCC 4 (Timbre)": mffc_features[3][3],
            "MAX MFCC 4 (Timbre)": mffc_features[4][3],

            "MEAN MFCC 5 (Timbre)": mffc_features[0][4],
            "MEDIAN MFCC 5 (Timbre)": mffc_features[1][4],
            "STD MFCC 5 (Timbre)": mffc_features[2][4],
            "MIN MFCC 5 (Timbre)": mffc_features[3][4],
            "MAX MFCC 5 (Timbre)": mffc_features[4][4],

            "MEAN MFCC 6 (Timbre)": mffc_features[0][5],
            "MEDIAN MFCC 6 (Timbre)": mffc_features[1][5],
            "STD MFCC 6 (Timbre)": mffc_features[2][5],
            "MIN MFCC 6 (Timbre)": mffc_features[3][5],
            "MAX MFCC 6 (Timbre)": mffc_features[4][5],

            "MEAN MFCC 7 (Timbre)": mffc_features[0][6],
            "MEDIAN MFCC 7 (Timbre)": mffc_features[1][6],
            "STD MFCC 7 (Timbre)": mffc_features[2][6],
            "MIN MFCC 7 (Timbre)": mffc_features[3][6],
            "MAX MFCC 7 (Timbre)": mffc_features[4][6],

            "MEAN MFCC 8 (Timbre)": mffc_features[0][7],
            "MEDIAN MFCC 8 (Timbre)": mffc_features[1][7],
            "STD MFCC 8 (Timbre)": mffc_features[2][7],
            "MIN MFCC 8 (Timbre)": mffc_features[3][7],
            "MAX MFCC 8 (Timbre)": mffc_features[4][7],

            "MEAN MFCC 9 (Timbre)": mffc_features[0][8],
            "MEDIAN MFCC 9 (Timbre)": mffc_features[1][8],
            "STD MFCC 9 (Timbre)": mffc_features[2][8],
            "MIN MFCC 9 (Timbre)": mffc_features[3][8],
            "MAX MFCC 9 (Timbre)": mffc_features[4][8],

            "MEAN MFCC 10 (Timbre)": mffc_features[0][9],
            "MEDIAN MFCC 10 (Timbre)": mffc_features[1][9],
            "STD MFCC 10 (Timbre)": mffc_features[2][9],
            "MIN MFCC 10 (Timbre)": mffc_features[3][9],
            "MAX MFCC 10 (Timbre)": mffc_features[4][9],

            "MEAN MFCC 11 (Timbre)": mffc_features[0][10],
            "MEDIAN MFCC 11 (Timbre)": mffc_features[1][10],
            "STD MFCC 11 (Timbre)": mffc_features[2][10],
            "MIN MFCC 11 (Timbre)": mffc_features[3][10],
            "MAX MFCC 11 (Timbre)": mffc_features[4][10],

            "MEAN MFCC 12 (Timbre)": mffc_features[0][11],
            "MEDIAN MFCC 12 (Timbre)": mffc_features[1][11],
            "STD MFCC 12 (Timbre)": mffc_features[2][11],
            "MIN MFCC 12 (Timbre)": mffc_features[3][11],
            "MAX MFCC 12 (Timbre)": mffc_features[4][11],

            "MEAN MFCC 13 (Timbre)": mffc_features[0][12],
            "MEDIAN MFCC 13 (Timbre)": mffc_features[1][12],
            "STD MFCC 13 (Timbre)": mffc_features[2][12],
            "MIN MFCC 13 (Timbre)": mffc_features[3][12],
            "MAX MFCC 13 (Timbre)": mffc_features[4][12],

            "MEAN Tonnetz Fifth X (Instrumentation)": tonnetz_features[0][0],
            "MEDIAN Tonnetz Fifth X (Instrumentation)": tonnetz_features[1][0],
            "STD Tonnetz Fifth X (Instrumentation)": tonnetz_features[2][0],
            "MIN Tonnetz Fifth X (Instrumentation)": tonnetz_features[3][0],
            "MAX Tonnetz Fifth X (Instrumentation)": tonnetz_features[4][0],

            "MEAN Tonnetz Fifth Y (Instrumentation)": tonnetz_features[0][1],
            "MEDIAN Tonnetz Fifth Y (Instrumentation)": tonnetz_features[1][1],
            "STD Tonnetz Fifth Y (Instrumentation)": tonnetz_features[2][1],
            "MIN Tonnetz Fifth Y (Instrumentation)": tonnetz_features[3][1],
            "MAX Tonnetz Fifth Y (Instrumentation)": tonnetz_features[4][1],

            "MEAN Tonnetz Minor X (Instrumentation)": tonnetz_features[0][2],
            "MEDIAN Tonnetz Minor X (Instrumentation)": tonnetz_features[1][2],
            "STD Tonnetz Minor X (Instrumentation)": tonnetz_features[2][2],
            "MIN Tonnetz Minor X (Instrumentation)": tonnetz_features[3][2],
            "MAX Tonnetz Minor X (Instrumentation)": tonnetz_features[4][2],

            "MEAN Tonnetz Minor Y (Instrumentation)": tonnetz_features[0][3],
            "MEDIAN Tonnetz Minor Y (Instrumentation)": tonnetz_features[1][3],
            "STD Tonnetz Minor Y (Instrumentation)": tonnetz_features[2][3],
            "MIN Tonnetz Minor Y (Instrumentation)": tonnetz_features[3][3],
            "MAX Tonnetz Minor Y (Instrumentation)": tonnetz_features[4][3],

            "MEAN Tonnetz Major X (Instrumentation)": tonnetz_features[0][4],
            "MEDIAN Tonnetz Major X (Instrumentation)": tonnetz_features[1][4],
            "STD Tonnetz Major X (Instrumentation)": tonnetz_features[2][4],
            "MIN Tonnetz Major X (Instrumentation)": tonnetz_features[3][4],
            "MAX Tonnetz Major X (Instrumentation)": tonnetz_features[4][4],

            "MEAN Tonnetz Major Y (Instrumentation)": tonnetz_features[0][5],
            "MEDIAN Tonnetz Major Y (Instrumentation)": tonnetz_features[1][5],
            "STD Tonnetz Major Y (Instrumentation)": tonnetz_features[2][5],
            "MIN Tonnetz Major Y (Instrumentation)": tonnetz_features[3][5],
            "MAX Tonnetz Major Y (Instrumentation)": tonnetz_features[4][5],

            "MEAN RMS": rms_features[0],
            "MEDIAN RMS": rms_features[1],
            "STD RMS": rms_features[2],
            "MIN RMS": rms_features[3],
            "MAX RMS": rms_features[4],

            "MEAN Spectral Centroid": spectral_centroid_features[0],
            "MEDIAN Spectral Centroid": spectral_centroid_features[1],
            "STD Spectral Centroid": spectral_centroid_features[2],
            "MIN Spectral Centroid": spectral_centroid_features[3],
            "MAX Spectral Centroid": spectral_centroid_features[4],

            "MEAN Spectral Bandwidth": spectral_bandwidth_features[0],
            "MEDIAN Spectral Bandwidth": spectral_bandwidth_features[1],
            "STD Spectral Bandwidth": spectral_bandwidth_features[2],
            "MIN Spectral Bandwidth": spectral_bandwidth_features[3],
            "MAX Spectral Bandwidth": spectral_bandwidth_features[4],

            "MEAN Zero Crossing Rate": zero_crossing_rate_features[0],
            "MEDIAN Zero Crossing Rate": zero_crossing_rate_features[1],
            "STD Zero Crossing Rate": zero_crossing_rate_features[2],
            "MIN Zero Crossing Rate": zero_crossing_rate_features[3],
            "MAX Zero Crossing Rate": zero_crossing_rate_features[4],
            }      

        return features  
