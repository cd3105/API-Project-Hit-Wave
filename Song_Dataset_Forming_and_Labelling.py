import pandas as pd
import numpy as np

top_40_df = pd.read_csv("Datasets/Top_40_Dataset.csv", sep="\t", encoding='latin-1')
top_100_billboard_df = pd.read_csv("Datasets/Top_100_Billboard.csv")
million_spotify_df = pd.read_csv("Datasets/Million_Spotify_Dataset.csv")
hot_100_df = pd.read_csv("Datasets/Hot_100_Dataset.csv")
the_hot_100_df = pd.read_csv("Datasets/The_Hot_100_Dataset.csv")
weekly_hot_100_df = pd.read_csv("Datasets/Weekly_Hot_100_Dataset.csv", index_col="index")

top_40_df["Artiest"] = top_40_df["Artiest"].apply(lambda x: str(x).split('/')[0])
top_40_df["Titel"] = top_40_df["Titel"].apply(lambda x: str(x).split('/')[0])
top_100_billboard_df["Year"] = top_100_billboard_df["week_id"].apply(lambda x: np.int64(x.split("/")[-1]))
hot_100_df["Year"] = hot_100_df["Date"].apply(lambda x: np.int64(x.split("-")[0]))
the_hot_100_df["Year"] = the_hot_100_df["date"].apply(lambda x: np.int64(x.split("-")[0]))

unique_top_40_df = top_40_df.sort_values(by="Jaar").drop_duplicates(subset=["Artiest", "Titel"]).reset_index(drop=True)
unique_top_100_billboard_df = top_100_billboard_df.sort_values(by="Year").drop_duplicates(subset=["performer", "song"]).reset_index(drop=True)
unique_million_spotify_df = million_spotify_df.drop_duplicates(subset=["artist", "song"]).reset_index(drop=True)
unique_hot_100_df = hot_100_df.sort_values(by="Year").drop_duplicates(subset=["Artist", "Song"]).reset_index(drop=True)
unique_the_hot_100_df = the_hot_100_df.sort_values(by="Year").drop_duplicates(subset=["artist", "song"]).reset_index(drop=True)
unique_weekly_hot_100_df = weekly_hot_100_df.drop_duplicates(subset=["Performer", "Song"]).reset_index(drop=True)

new_top_40_df = pd.DataFrame({"Artist": unique_top_40_df["Artiest"], "Song Title": unique_top_40_df["Titel"]})
new_top_100_billboard_df = pd.DataFrame({"Artist": unique_top_100_billboard_df["performer"], "Song Title": unique_top_100_billboard_df["song"].apply(lambda x: x.upper())})
new_million_spotify_df = pd.DataFrame({"Artist": unique_million_spotify_df["artist"], "Song Title": unique_million_spotify_df["song"].apply(lambda x: x.upper())})
new_hot_100_df = pd.DataFrame({"Artist": unique_hot_100_df["Artist"], "Song Title": unique_hot_100_df["Song"].apply(lambda x: x.upper())})
new_the_hot_100_df = pd.DataFrame({"Artist": unique_the_hot_100_df["artist"], "Song Title": unique_the_hot_100_df["song"].apply(lambda x: x.upper())})
new_weekly_hot_100_df = pd.DataFrame({"Artist": unique_weekly_hot_100_df["Performer"], "Song Title": unique_weekly_hot_100_df["Song"].apply(lambda x: x.upper())})

all_unique_songs = pd.concat([new_top_40_df, new_top_100_billboard_df, new_hot_100_df, new_the_hot_100_df, new_weekly_hot_100_df, new_million_spotify_df]).sort_values(by="Artist").drop_duplicates(subset=["Song Title"]).reset_index(drop=True)

all_unique_songs_top_40 = all_unique_songs.copy()
all_unique_songs_top_40["Hit"] = all_unique_songs_top_40["Song Title"].apply(lambda x: 1 if x in list(new_top_40_df['Song Title']) else 0)
all_unique_songs_top_40.to_csv('Preprocessed and Labeled Datasets/Labeled_Songs_per_Top_40.csv', index=False)

all_unique_songs_top_100_billboard = all_unique_songs.copy()
all_unique_songs_top_100_billboard["Hit"] = all_unique_songs_top_100_billboard["Song Title"].apply(lambda x: 1 if x in list(new_top_100_billboard_df['Song Title']) else 0)
all_unique_songs_top_100_billboard.to_csv('Preprocessed and Labeled Datasets/Labeled_Songs_per_Top_100_Billboard.csv', index=False)

all_unique_songs_hot_100 = all_unique_songs.copy()
all_unique_songs_hot_100["Hit"] = all_unique_songs_hot_100["Song Title"].apply(lambda x: 1 if x in list(new_hot_100_df['Song Title']) else 0)
all_unique_songs_hot_100.to_csv('Preprocessed and Labeled Datasets/Labeled_Songs_per_Hot_100.csv', index=False)

all_unique_songs_the_hot_100 = all_unique_songs.copy()
all_unique_songs_the_hot_100["Hit"] = all_unique_songs_the_hot_100["Song Title"].apply(lambda x: 1 if x in list(new_the_hot_100_df['Song Title']) else 0)
all_unique_songs_the_hot_100.to_csv('Preprocessed and Labeled Datasets/Labeled_Songs_per_The_Hot_100.csv', index=False)

all_unique_songs_weekly_hot_100 = all_unique_songs.copy()
all_unique_songs_weekly_hot_100["Hit"] = all_unique_songs_weekly_hot_100["Song Title"].apply(lambda x: 1 if x in list(new_weekly_hot_100_df['Song Title']) else 0)
all_unique_songs_weekly_hot_100.to_csv('Preprocessed and Labeled Datasets/Labeled_Songs_per_Weekly_Hot_100.csv', index=False)

print(f'Songs in Dataset: {len(all_unique_songs)}')
print(f'Number of Songs in Top 40: {len(new_top_40_df)}')
print(f'Number of Hit Songs in Dataset based on Top 40: {sum(all_unique_songs_top_40["Hit"])}')

print(f'Number of Songs in Top 100 Billboard: {len(new_top_100_billboard_df)}')
print(f'Number of Hit Songs in Dataset based on Top 100 Billboard: {sum(all_unique_songs_top_100_billboard["Hit"])}')

print(f'Number of Songs in Hot 100: {len(new_hot_100_df)}')
print(f'Number of Hit Songs in Dataset based on Hot 100: {sum(all_unique_songs_hot_100["Hit"])}')

print(f'Number of Songs in The Hot 100: {len(new_the_hot_100_df)}')
print(f'Number of Hit Songs in Dataset based on The Hot 100: {sum(all_unique_songs_the_hot_100["Hit"])}')

print(f'Number of Songs in Weekly Hot 100: {len(new_weekly_hot_100_df)}')
print(f'Number of Hit Songs in Dataset based on Weekly Hot 100: {sum(all_unique_songs_weekly_hot_100["Hit"])}')
