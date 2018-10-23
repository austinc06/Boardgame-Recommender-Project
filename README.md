<<<<<<< HEAD
Boardgame-Recommender\
Author: Austin Chou\
Update: 2018-10-21\

# Table of Contents
1) [Purpose](#purpose)\
2) [Topic of Interest](#topic-of-interest)\
3) [Project Outline](#project-outline)\
4) [Code Documentation](#code-documentation)
5) [Additional Notes](#additional-notes)

# Purpose
The purpose of the project is to practice programming and data manipulation in
python and learn some modeling techniques.

# Topic of Interest
The focus of the project is developing a boardgame recommendation system. The website
"http://boardgamegeek.com" has over 10000 ranked boardgames, making the choices for
new hobbyists quite astronomical. The primary problem is to identify patterns to
help recommend boardgames to people who are looking to grow their boardgame collection.

# Project Outline
1) Pull Total Game List (off bgg ranked list; 2017)
2) Measure Game Similarity
3) Get a collection of user collections
4) Determine "similarity" measure for user collections
5) Implement a CBOW-like neural network as a recommendation system for boardgames based off existing collections

# Code Documentation
## Notebooks
1) Boardgame Collection Similarity Analysis
The primary notebook containing the data collection, curation, and initial analysis
of the problem. Specific sections of the notebook call section-specific scripts.  Functions are also called from the functions script where they are defined. The code is meant to be run sequentially from top to bottom.\

2) Boardgame Recommender neural network
This is a test notebook to set up the collected datasets as a Training Matrix and
try implementing a neural network approach. Still needs to be cleaned and organized.\

## Scripts
### functions-bgg_get.py
Contains the following functions:\
get_bgg_mechanics: Scrape the mechanisms used by boardgamegeek and produce a csv containing the list.\
  -Input: None\
  -Output: List of Mechanisms (written csv)\

get_bgg_categories: Scrape the categories used by boardgamegeek and produce a csv containing the list.\
  -Input: None\
  -Output: List of Categories (written csv)\

get_game_attributes: Given a game name, id, and rank from bgg, create and return a dataframe storing the game's attributes: Name, Rank, ID, #Players, Playtime, Weight, Categories, Mechanisms, and the total number of categories and mechanisms.\
  -Input: Game name (str), Game id (str or int), Game rank (str or int), List of categories (list of strings), List of mechanisms (list of strings)\
  -Output: A dataframe containing a single row\

scale_dataframe: Given a bgg attributes dataframe, remove the label columns (name, rank, id) and scale the non-binary columns (#players, playtime, weight)\
  -Input: Game attributes dataframe (df)\
  -Output: Scaled game attributes dataframe (df)\

game_tsne: Take a scaled game attribute dataframe and run the tsne function to generate tsne coordinates. Print the plot and return a dataframe of coordinates for each game.\
  -Input: Scaled game attribute dataframe (df), Perplexity value for tsne (int), Steps value for tsne (int)\
  -Output: TSNE coordinate dataframe (df)\

RatingThreshold: Filter a collection based on a rating threshold (user or bgg score) of 7.\
  -Input: Dataframe of games representing a collection (df)\
  -Output: Curated dataframe of games (df)\

collection_distance: Calculate the average distance (dissimilarity) score of the user's collection given a set of tsne coordinates for each game.\
  -Input: List of games (list), Dataframe of dimensions for each game (df)\
  -Output: Mean distance of games in the list (dbl)\

collection_euc_distance: Calculate the average distance score of the user's collection given a set of coordinates using euclidean distance.\
  -Input: List of games (list), Dataframe of dimensions for each game (df)\
  -Output: Mean distance of games in the list (dbl)\

calc_distance: Take a set of target coordinates and a matrix containing all sets of coordinates. Calculate the distance the target from all other points in space.\
  -Input: Target coordinate (dataframe row), Array of all other coordinates (df)\
  -Output: Array of calculated distances (df)\

find_closest_game: Take the target game name, the dataframe of coordinates, and a number of games to return. Game coordinates can be euclidean or tsne or any other set of coordinates. First column must be game name.\
  -Input: Game name (str), Dataframe of coordinates (df), Number of games to return (int)\
  -Output: Dataframe with CLOSEST boardgame names and calculated distances\

find_furthest_game: Take the target game name, the dataframe of coordinates, adn a number of games to return. Game coordinates can be euclidean or tsne or any other set of coordinates. First column must be game name.\
  -Input: Game name (str), Dataframe of coordinates (df), Number of games to return (int)\
  -Output: Dataframe with FURTHEST boardgame names and calculated distances\

### script-collect_bgg_parameters
Collect parameters from bgg including Mechanisms, Categories, and the full game list. Additionally, create a dataframe containing the game attributes for each game in the game list and create relevant csvs.\

### script-clean_game_attributes
Data cleaning code to remove games that have no categories or no mechanisms listed on bgg. Also flag entries that need manual adjustment of attributes; the collection function only grabs the first six mechanisms and categories of each game.\

### script-pull_gamelist
Script to scrape the gamelist from bgg.\

### BGG_csvsettings
Settings passed to the webscraper to generate the appropriate csvs.\

# Additional Notes
1) Vectorized bgg data.csv is not provided. The csv contains every user collection split into training vectors for each game in the collection (includes the input (all other games in the collection) and output (the game itself)). File is too large to upload.
=======
# Boardgame-Recommender-Project
>>>>>>> 405629261ac0befebe31d69e75a637bd1b2edb9c
