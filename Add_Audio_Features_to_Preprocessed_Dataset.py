import pandas as pd
import unicodedata
import re

def preprocess_text(text):
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

songs_df = pd.read_csv("Reordered Preprocessed and Labeled Datasets with Spotify Features\Reordered_Labeled_Songs_per_Hot_100_with_Spotify_Features.csv")
features_df = pd.read_csv("Audio Features Datasets\Audio_Features_Dataset.csv")

if len(features_df) != 0:
    if len(features_df) < len(songs_df):
        songs_df_with_audio_features = songs_df[:len(features_df)].join(features_df)
    else:
        songs_df_with_audio_features = songs_df.join(features_df)
    
    songs_df_with_audio_features.to_csv(f"Final Datasets without Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_{len(songs_df_with_audio_features)}.csv", 
                                        index=False)
    
    print(f"- Number of Hits before Filtering: {len(songs_df_with_audio_features[songs_df_with_audio_features['Hit'] == 1])}\n- Number of No Hits before Filtering: {len(songs_df_with_audio_features[songs_df_with_audio_features['Hit'] == 0])}\n- Total Length Dataset before Filtering: {len(songs_df_with_audio_features)}\n")

    filtered_songs_df_with_audio_features = songs_df_with_audio_features.sort_values(by="Hit", ascending=False).reset_index(drop=True)
    filtered_songs_df_with_audio_features = filtered_songs_df_with_audio_features.drop_duplicates(subset=["Spotify ID"]).reset_index(drop=True)
    filtered_songs_df_with_audio_features = filtered_songs_df_with_audio_features.drop_duplicates(subset=["Video Title of Audio"]).reset_index(drop=True)
    filtered_songs_df_with_audio_features = filtered_songs_df_with_audio_features[filtered_songs_df_with_audio_features['Release Year'] >= 1958].reset_index(drop=True)
    filtered_songs_df_with_audio_features = filtered_songs_df_with_audio_features.sample(frac=1, ignore_index=True)

    filtered_songs_df_with_audio_features.to_csv(f"Final Datasets after Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_{len(filtered_songs_df_with_audio_features)}.csv", 
                                                 index=False)

    print(f"- Number of Hits before Filtering: {len(filtered_songs_df_with_audio_features[filtered_songs_df_with_audio_features['Hit'] == 1])}\n- Number of No Hits before Filtering: {len(filtered_songs_df_with_audio_features[filtered_songs_df_with_audio_features['Hit'] == 0])}\n- Total Length Dataset after Filtering: {len(filtered_songs_df_with_audio_features)}\n")

    indexes_to_keep = []

    for i, r in filtered_songs_df_with_audio_features.iterrows():
        if (preprocess_text(r['Song Title']) in preprocess_text(r['Spotify Song Title'])) or (preprocess_text(r['Spotify Song Title']) in preprocess_text(r['Song Title'])):
            if (preprocess_text(r['Spotify Primary Artist']) in preprocess_text(r['Artist'])) or (preprocess_text(r['Artist']) in preprocess_text(r['Spotify Primary Artist'])):
                if (preprocess_text(r['Song Title']) in preprocess_text(r['Video Title of Audio'])) or (preprocess_text(r['Video Title of Audio']) in preprocess_text(r['Song Title'])) or (preprocess_text(r['Artist']) in preprocess_text(r['Video Title of Audio'])) or (preprocess_text(r['Spotify Primary Artist']) in preprocess_text(r['Video Title of Audio'])):
                    indexes_to_keep.append(i)
    
    further_filtered_songs_df_with_audio_features = filtered_songs_df_with_audio_features.loc[indexes_to_keep].reset_index(drop=True)

    print(f"- Number of Hits after Further Filtering: {len(further_filtered_songs_df_with_audio_features[further_filtered_songs_df_with_audio_features['Hit'] == 1])}\n- Number of No Hits after Further Filtering: {len(further_filtered_songs_df_with_audio_features[further_filtered_songs_df_with_audio_features['Hit'] == 0])}\n- Total Length Dataset after Further Filtering: {len(further_filtered_songs_df_with_audio_features)}")

    further_filtered_songs_df_with_audio_features.to_csv(f"Final Datasets after Further Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_{len(further_filtered_songs_df_with_audio_features)}.csv", 
                                                         index=False)
