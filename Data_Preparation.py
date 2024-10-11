import pandas as pd
import numpy as np

top_40_df = pd.read_csv("Datasets/Top_40_Dataset.csv", sep="\t", encoding='latin-1')
top_100_billboard_df = pd.read_csv("Datasets/Top_100_Billboard.csv")
top_200_billboard_df = pd.read_csv("Datasets/Top_200_Billboard.csv")
million_spotify_df = pd.read_csv("Datasets/Million_Spotify_Dataset.csv")
hot_100_df = pd.read_csv("Datasets/Hot_100_Dataset.csv")
the_hot_100_df = pd.read_csv("Datasets/The_Hot_100_Dataset.csv")
weekly_hot_100_df = pd.read_csv("Datasets/Weekly_Hot_100_Dataset.csv", index_col="index")

top_100_billboard_df["Year"] = top_100_billboard_df["week_id"].apply(lambda x: np.int64(x.split("/")[-1]))
top_200_billboard_df["Year"] = top_200_billboard_df["Date"].apply(lambda x: np.int64(x.split("-")[0]))
hot_100_df["Year"] = hot_100_df["Date"].apply(lambda x: np.int64(x.split("-")[0]))
the_hot_100_df["Year"] = the_hot_100_df["date"].apply(lambda x: np.int64(x.split("-")[0]))

unique_top_40_df = top_40_df.sort_values(by="Jaar").drop_duplicates(subset=["Artiest", "Titel"]).reset_index(drop=True)
unique_top_100_billboard_df = top_100_billboard_df.sort_values(by="Year").drop_duplicates(subset=["performer", "song"]).reset_index(drop=True)
unique_top_200_billboard_df = top_200_billboard_df.sort_values(by="Year").drop_duplicates(subset=["Artist", "Song"]).reset_index(drop=True)
unique_million_spotify_df = million_spotify_df.drop_duplicates(subset=["artist", "song"]).reset_index(drop=True)
unique_hot_100_df = hot_100_df.sort_values(by="Year").drop_duplicates(subset=["Artist", "Song"]).reset_index(drop=True)
unique_the_hot_100_df = the_hot_100_df.sort_values(by="Year").drop_duplicates(subset=["artist", "song"]).reset_index(drop=True)
unique_weekly_hot_100_df = weekly_hot_100_df.drop_duplicates(subset=["Performer", "Song"]).reset_index(drop=True)

new_top_40_df = pd.DataFrame({"Year": unique_top_40_df["Jaar"], "Artist": unique_top_40_df["Artiest"], "Song Title": unique_top_40_df["Titel"]})
new_top_100_billboard_df = pd.DataFrame({"Year": unique_top_100_billboard_df["Year"], "Artist": unique_top_100_billboard_df["performer"], "Song Title": unique_top_100_billboard_df["song"].apply(lambda x: x.upper())})
new_top_200_billboard_df = pd.DataFrame({"Year": unique_top_200_billboard_df["Year"], "Artist": unique_top_200_billboard_df["Artist"], "Song Title": unique_top_200_billboard_df["Song"].apply(lambda x: x.upper())})
new_million_spotify_df = pd.DataFrame({"Artist": unique_million_spotify_df["artist"], "Song Title": unique_million_spotify_df["song"].apply(lambda x: x.upper())})
new_hot_100_df = pd.DataFrame({"Year": unique_hot_100_df["Year"], "Artist": unique_hot_100_df["Artist"], "Song Title": unique_hot_100_df["Song"].apply(lambda x: x.upper())})
new_the_hot_100_df = pd.DataFrame({"Year": unique_the_hot_100_df["Year"], "Artist": unique_the_hot_100_df["artist"], "Song Title": unique_the_hot_100_df["song"].apply(lambda x: x.upper())})
new_weekly_hot_100_df = pd.DataFrame({"Artist": unique_weekly_hot_100_df["Performer"], "Song Title": unique_weekly_hot_100_df["Song"].apply(lambda x: x.upper())})

print(new_top_40_df.merge(new_top_100_billboard_df, on=["Artist", "Song Title"]))
