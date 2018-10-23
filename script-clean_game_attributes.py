import pandas as pd

# Load csv we had saved
ga = pd.read_csv('bgg game attributes.csv', header=1)
# Drop the first column (just indices)
ga.drop(ga.columns[0], inplace=True, axis=1)

# Find the rows that have 0 categories or 0 mechanics
to_remove = ga[(ga['total categories'] == 0) | (ga['total mechanics'] == 0)]

# Remove the rows from the ga df
ga.drop(to_remove.index, inplace=True)

manual = ga[(ga['total mechanics'].apply(lambda x: int(x)) > 6) |
            (ga['total categories'].apply(lambda x: int(x)) > 6)]

manual.to_csv('bgg_games_for_manual.csv')
ga.to_csv('bgg_games_curated.csv')
