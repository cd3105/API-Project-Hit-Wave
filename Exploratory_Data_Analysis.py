import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Final Datasets after Further Filtering\Reordered_Preprocessed_Labeled_Songs_per_Hot_100_with_Spotify_Features_and_Audio_Features_52896.csv")

pd.DataFrame({'Split': ['1960s', 
                        '1970s', 
                        '1980s', 
                        '1990s', 
                        '2000s', 
                        '2010s', 
                        '2020s', 
                        'Full'],
              'Hits': [len(df[(df['Release Year'] >= 1960) & (df['Release Year'] < 1970) & (df['Hit'] == 1)]),
                       len(df[(df['Release Year'] >= 1970) & (df['Release Year'] < 1980) & (df['Hit'] == 1)]),
                       len(df[(df['Release Year'] >= 1980) & (df['Release Year'] < 1990) & (df['Hit'] == 1)]),
                       len(df[(df['Release Year'] >= 1990) & (df['Release Year'] < 2000) & (df['Hit'] == 1)]),
                       len(df[(df['Release Year'] >= 2000) & (df['Release Year'] < 2010) & (df['Hit'] == 1)]),
                       len(df[(df['Release Year'] >= 2010) & (df['Release Year'] < 2020) & (df['Hit'] == 1)]),
                       len(df[(df['Release Year'] >= 2020) & (df['Hit'] == 1)]),
                       len(df[df['Hit'] == 1])],
              'No Hits': [len(df[(df['Release Year'] >= 1960) & (df['Release Year'] < 1970) & (df['Hit'] == 0)]),
                          len(df[(df['Release Year'] >= 1970) & (df['Release Year'] < 1980) & (df['Hit'] == 0)]),
                          len(df[(df['Release Year'] >= 1980) & (df['Release Year'] < 1990) & (df['Hit'] == 0)]),
                          len(df[(df['Release Year'] >= 1990) & (df['Release Year'] < 2000) & (df['Hit'] == 0)]),
                          len(df[(df['Release Year'] >= 2000) & (df['Release Year'] < 2010) & (df['Hit'] == 0)]),
                          len(df[(df['Release Year'] >= 2010) & (df['Release Year'] < 2020) & (df['Hit'] == 0)]),
                          len(df[(df['Release Year'] >= 2020) & (df['Hit'] == 0)]),
                          len(df[df['Hit'] == 0])]}).to_csv("EDA Datasets/Hits_n_No_Hits_per_Split.csv")

sns.countplot(x='Hit', data=df, palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in Complete Dataset')
plt.savefig("Bar Plots/Bar_Plot_Complete_Dataset.png")
plt.show()

sns.countplot(x='Hit', data=df[(df['Release Year'] >= 1960) & (df['Release Year'] < 1970)], palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in 1960s Partition of Dataset')
plt.savefig("Bar Plots/Bar_Plot_1960s.png")
plt.show()

sns.countplot(x='Hit', data=df[(df['Release Year'] >= 1970) & (df['Release Year'] < 1980)], palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in 1970s Partition of Dataset')
plt.savefig("Bar Plots/Bar_Plot_1970s.png")
plt.show()

sns.countplot(x='Hit', data=df[(df['Release Year'] >= 1980) & (df['Release Year'] < 1990)], palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in 1980s Partition of Dataset')
plt.savefig("Bar Plots/Bar_Plot_1980s.png")
plt.show()

sns.countplot(x='Hit', data=df[(df['Release Year'] >= 1990) & (df['Release Year'] < 2000)], palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in 1990s Partition of Dataset')
plt.savefig("Bar Plots/Bar_Plot_1990s.png")
plt.show()

sns.countplot(x='Hit', data=df[(df['Release Year'] >= 2000) & (df['Release Year'] < 2010)], palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in 2000s Partition of Dataset')
plt.savefig("Bar Plots/Bar_Plot_2000s.png")
plt.show()

sns.countplot(x='Hit', data=df[(df['Release Year'] >= 2010) & (df['Release Year'] < 2020)], palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in 2010s Partition of Dataset')
plt.savefig("Bar Plots/Bar_Plot_2010s.png")
plt.show()

sns.countplot(x='Hit', data=df[df['Release Year'] >= 2020], palette='Blues')
plt.xticks([0, 1], ['No Hit', 'Hit'])
plt.xlabel('Hit / No Hit')
plt.ylabel('Amount of Songs')
plt.title('Overview of Hits and Non-Hits in 2020s Partition of Dataset')
plt.savefig("Bar Plots/Bar_Plot_2020s.png")
plt.show()

