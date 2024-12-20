import pandas as pd

# Load in Data

df = pd.read_csv("Preprocessed and Labeled Datasets with Spotify Features\Labeled_Songs_per_Hot_100_with_Spotify_Features.csv")
df_hits = df[df['Hit'] == 1]
df_no_hits = df[df['Hit'] == 0]

print(f"DF with only hits (Length: {len(df_hits)}):\n{df_hits}\n")
print(f"DF with no hits (Length: {len(df_no_hits)}):\n{df_no_hits}\n")

df_hits_randomized_order = df_hits.sample(frac=1, ignore_index=True)
df_no_hits_randomized_order = df_no_hits.sample(frac=1, ignore_index=True)

print(f"Randomized DF with only hits (Length: {len(df_hits_randomized_order)}):\n{df_hits_randomized_order}\n")
print(f"Randomized DF with no hits (Length: {len(df_no_hits_randomized_order)}):\n{df_no_hits_randomized_order}\n")

df_w_new_order = pd.DataFrame(columns=df.columns)

# Reorder Dataset

for i in range(min(len(df_no_hits_randomized_order), len(df_hits_randomized_order))):
    df_w_new_order.loc[i*2] = df_hits_randomized_order.iloc[i]
    df_w_new_order.loc[i*2+1] = df_no_hits_randomized_order.iloc[i]

print(f"Newly ordered DF (Length: {len(df_w_new_order)}):\n{df_w_new_order}\n")

if len(df_no_hits) > len(df_hits):
    df_w_new_order = pd.concat([df_w_new_order, df_no_hits_randomized_order[len(df_hits):]], ignore_index=True)

if len(df_no_hits) < len(df_hits):
    df_w_new_order = pd.concat([df_w_new_order, df_hits_randomized_order[len(df_no_hits):]], ignore_index=True)

print(f"Newly ordered DF (Length: {len(df_w_new_order)}):\n{df_w_new_order}\n")
print(f"DF without duplicates: {df_w_new_order.drop_duplicates().reset_index(drop=True)}")

df_w_new_order.to_csv("Reordered Preprocessed and Labeled Datasets with Spotify Features\Reordered_Labeled_Songs_per_Hot_100_with_Spotify_Features.csv", index=False)
