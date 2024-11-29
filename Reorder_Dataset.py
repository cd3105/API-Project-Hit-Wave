import pandas as pd

# df_hot_100 = pd.read_csv("Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv")

# print(df_hot_100.sample(frac=1, ignore_index=True).reset_index(drop=True))

df = pd.read_csv("Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_The_Hot_100_with_Spotify_Features.csv", index_col=0)

print(len(df[df['Hit'] == 0]))
print(len(df[df['Hit'] == 1]))

df.to_csv("Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_The_Hot_100_with_Spotify_Features.csv", index=False)


