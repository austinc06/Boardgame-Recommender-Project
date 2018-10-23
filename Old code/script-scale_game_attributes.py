import pandas as pd
#Import sklearn standardscaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Load csv of games after manually adjusting mechanisms and categories as
#  mentioned in "Cleaning game_attributes dataset"
ga = pd.read_csv('bgg game attributes edited.csv', header = 1)
ga.drop(ga.columns[0], inplace=True, axis=1)

#Set up PCA-dataset; remove text
ga_to_pca = ga.drop(['Game name','Game rank','Game ID','total categories',
                     'total mechanics'], axis=1)
