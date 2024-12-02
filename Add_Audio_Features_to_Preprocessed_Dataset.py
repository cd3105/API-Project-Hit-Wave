import pandas as pd

songs_df = pd.read_csv("Reordered Preprocessed and Labeled Datasets with Spotify Features\Reordered_Labeled_Songs_per_Hot_100_with_Spotify_Features.csv")
features_df = pd.read_csv("Audio Features Datasets\Audio_Features_Dataset.csv")

if len(features_df) != 0:
    if len(features_df) < len(songs_df):
        songs_df_with_audio_features = songs_df[:len(features_df)].join(features_df)
    else:
        songs_df_with_audio_features = songs_df.join(features_df)
    
    songs_df_with_audio_features.to_csv(f"Final Datasets without Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_{len(songs_df_with_audio_features)}.csv", 
                                        index=False)
    
    filtered_songs_df_with_audio_features = songs_df_with_audio_features.sort_values(by="Hit", ascending=False).reset_index(drop=True)
    filtered_songs_df_with_audio_features = filtered_songs_df_with_audio_features.drop_duplicates(subset=["Spotify ID"]).reset_index(drop=True)
    filtered_songs_df_with_audio_features = filtered_songs_df_with_audio_features.drop_duplicates(subset=["Video Title of Audio"]).reset_index(drop=True)

    filtered_songs_df_with_audio_features.to_csv(f"Final Datasets after Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_{len(filtered_songs_df_with_audio_features)}.csv", 
                                                 index=False)
