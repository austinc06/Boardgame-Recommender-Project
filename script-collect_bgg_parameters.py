# Load in functions and libraries
%run functions-bgg_get.py
#General libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#For webscraping
from bs4 import BeautifulSoup
import requests
from time import time, sleep
#Regular expression
import re

# Get Mechanisms ---------------------------------------------------------------
mech_list = get_bgg_mechanics()
print("---")
print("%s mechanics total" %len(mech_list))

# Get Categories ---------------------------------------------------------------
cat_list = get_bgg_categories()
print("---")
print("%s categories total" %len(cat_list))

# Get gamelist; separate script made to do so ----------------------------------
%run script-pull_gamelist.py

# Get game attributes for every game pulled ------------------------------------
# Get the games list we'll use
games_list = pd.read_csv('bgg_id_output.csv')
# Remove all NaN rows
games_list.dropna(axis=0,how='any',inplace=True)
games_list.reset_index(drop=True,inplace=True)
# There are repeat titles in the list. Remove them.
rep_games_idx = games_list[games_list['Game'].duplicated()].index.tolist() #Returns the indices of all repeat titles. This list does NOT include the first appearance
games_list.drop(games_list.index[rep_games_idx],inplace=True)
games_list.reset_index(drop=True,inplace=True)
# Convert GameID to int
games_list['GameID'] = games_list['GameID'].apply(lambda x: int(x))
# Convert BGG Rank to int
games_list['BGG Rank'] = games_list['BGG Rank'].apply(lambda x: int(x))

# Get category and mechanisms list
cat_list = pd.read_csv('BGG categories.csv',sep='\t')
cat_list = cat_list['Categories'].tolist()
mech_list = pd.read_csv('BGG mechanics.csv',sep='\t')
mech_list = mech_list['Mechanics'].tolist()

if len(cat_list) == 0:
    cat_list = pd.read_csv('BGG categories.csv',sep='\t')
    cat_list = cat_list['Categories'].tolist()

if len(mech_list) == 0:
    mech_list = pd.read_csv('BGG mechanics.csv',sep='\t')
    mech_list = mech_list['Mechanics'].tolist()

# Establish column headings
columns = ['Game name', 'Game rank', 'Game ID', '#players', 'playtime',
'weight'] + cat_list + mech_list + ['total categories', 'total mechanics']

# Create dataframe filled with 0's
game_attributes = pd.DataFrame(0,
                               index=np.arange(games_list.shape[0]),
                               columns=columns)

# Iterate through games list and fill in game attributes df
for i in range(0, len(games_list)):
    name = games_list.loc[i,"Game"]
    rank = games_list.loc[i,"BGG Rank"]
    ID = games_list.loc[i,"GameID"]

    current_game = get_game_attributes(name, ID, rank, cat_list, mech_list)

    game_attributes.iloc[i,:] = current_game.iloc[0,:]

    #Set a time limit between each loop to reduce bgg load
    sleep(2)

# Save the game attribute dataframe
#  Took hours to create, so save as csv so we can load it quickly
game_attributes.to_csv('bgg game attributes.csv',sep='\t')
