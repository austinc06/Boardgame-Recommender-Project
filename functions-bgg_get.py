# Get BGG Mechanics ------------------------------------------------------------
def get_bgg_mechanics(write = False):
    """get_bgg_mechanics pulls the mechanics categories from boardgamegeek and
    outputs a csv.
    Input: None
    Output: List of Mechanics (csv written)
    """
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd

    # Get category website in xml format
    url = "https://boardgamegeek.com/browse/boardgamemechanic"

    r = requests.get(url)
    page = r.text

    soup = BeautifulSoup(page, "lxml")

    #Categories are stored in a table, so get all the tables
    tables = soup.find_all(lambda tag: tag.name=='table')

    #The sixth table contains the game categories
    tables = tables[5]

    #Get all the category tags
    mech = tables.find_all(lambda tag: tag.name=='a')

    #Store categories into a list
    mech_list = []

    for i in mech:
        mech_list.append(i.contents[0])
        print(i.contents[0])

    #Create pd dataframe for export
    mech_list = pd.DataFrame(mech_list, columns=['Mechanics'])

    #Save category list to csv
    if write:
        mech_list.to_csv('BGG mechanics.csv',sep='\t')
    return mech_list

# Get BGG Categories -----------------------------------------------------------
def get_bgg_categories(write = False):
    """get_bgg_categories pulls the mechanics categories from boardgamegeek and
    outputs a csv.
    Input: None
    Output: List of Categories (csv written)
    """
    # Use beautiful soup
    #Import necessary libraries
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd

    # Get category website in xml format
    url = "https://boardgamegeek.com/browse/boardgamecategory"

    r = requests.get(url)
    page = r.text

    soup = BeautifulSoup(page, "lxml")

    #Categories are stored in a table, so get all the tables
    tables = soup.find_all(lambda tag: tag.name=='table')
    #The sixth table contains the game categories
    tables = tables[5]

    #Get all the category tags
    cat = tables.find_all(lambda tag: tag.name=='a')

    #Store categories into a list
    cat_list = []

    for i in cat:
        cat_list.append(i.contents[0])
        print(i.contents[0])

    #Create pd dataframe for export
    cat_list = pd.DataFrame(cat_list, columns=['Categories'])

    #Save category list to csv
    if write:
        cat_list.to_csv('BGG categories.csv',sep='\t')
    return cat_list

# Get Mechanics from bgg -------------------------------------------------------
# def GetMechanics(gameid):
#     """GetMechanics takes the gameid of the game on boardgamegeek and outputs a
#     dictionary with the Title, Year, and (boardgame)Mechanics associated with the
#     game"""
#
#     import requests
#     import re
#
#     # Use Requests package to pull from BGG API
#     game = requests.get("https://boardgamegeek.com/xmlapi2/thing?id="+str(gameid))
#     #For reference, print output code;; Should write error if value != 200
#     print("API request status: " + str(game.status_code))
#
#     # Decodes byte object to string for interpretation
#     game_info = game.content.decode("utf-8")
#
#     # Get game name
#     #  Find the index of the primary Title
#     Title_idx = game_info.find('value=',game_info.find('name type="primary"'))+len('value="')
#     # Get the Title string
#     title = game_info[Title_idx:game_info.find('"',Title_idx)]
#     # Print title
#     print(title)
#
#     # Get year published
#     Year_idx = game_info.find('yearpublished value=') + len('yearpublished value="')
#     year = game_info[Year_idx:game_info.find('"',Year_idx)]
#
#     # Get game mechanics
#     game_mechs = list()
#
#     # Finds all mentions of boardgamemechanic
#     Mechanic_mentions = [m.start() for m in re.finditer('boardgamemechanic',game_info)]
#
#     for x in Mechanic_mentions:
#         # Find index of the word "value=" and adds the length of value
#         #  That should be the starting index of the Mechanic term
#         Game_mech_idx = game_info.find('value=',x)+len('value="')
#         # Find the index of the closing quotation " for the Mechanic term
#         last_letter = game_info.find('"',Game_mech_idx)
#         # Output the term
#         game_mechs.append(game_info[Game_mech_idx:last_letter])
#
#     return {'Title':title,'Year':year,'Mechanics':game_mechs}

# Get Game Attributes ----------------------------------------------------------
def get_game_attributes(game_name,
                        game_id,
                        game_rank,
                        cat_list = [],
                        mech_list = []):
    '''Given a game's name, id, and rank from bgg, create and return a DataFrame
    storing the game attributes: Name, Rank, ID, #Players, Playtime, Weight,
    Categories, Mechanisms, and the total number of categories and mechanisms

    Input: Game name (string),
            Game id (string or numeric),
            Game rank (string or numeric),
            List of categories (list of strings),
            List of mechanisms (list of strings)
    Output: Dataframe row
    '''
    #General libraries
    import pandas as pd
    import numpy as np
    #For webscraping
    from bs4 import BeautifulSoup
    import requests
    #Regular expression
    import re
    from time import time, sleep

    if len(cat_list) == 0:
        cat_list = pd.read_csv('BGG categories.csv',sep='\t')
        cat_list = cat_list['Categories'].tolist()

    if len(mech_list) == 0:
        mech_list = pd.read_csv('BGG mechanics.csv',sep='\t')
        mech_list = mech_list['Mechanics'].tolist()

    #Establish column headings
    columns = ['Game name', 'Game rank', 'Game ID', '#players', 'playtime',
    'weight'] + cat_list + mech_list + ['total categories', 'total mechanics']

    #Create dataframe filled with 0's
    game_attributes = pd.DataFrame(0, index = [1], columns=columns)

    game_id = game_id
    #print(game_id)
    game_rank = game_rank
    #print(game_rank)
    game_name = game_name
    #print(game_name)

    #Use just the GameID to get the true credits url
    url = "https://boardgamegeek.com/boardgame/"+str(game_id)
    r = requests.get(url)
    page = r.text

    soup = BeautifulSoup(page, "lxml")

    #Update url; e.g.
    url = soup.find('head').find('link')['href']
    url = url+'/credits'
    r = requests.get(url)
    page = r.text

    soup = BeautifulSoup(page, "lxml")

    #Get the game script that contains all the relevant info
    script = soup.find_all(lambda tag: tag.name=='script')

    script_num = 0
    script_get = 0

    #Check each script on the page; find the script that contains 'maxplayers' which will include all other relevant info
    while (script_num < len(script)) and script_get == 0:
        if 'maxplayers' in script[script_num].text:
            script_get = 1
        else:
            script_num += 1

    if script_num < len(script):
        current_game = script[script_num]
        game = current_game.text


        #----#PLAYERS----#
        player_pos = [m.start() for m in re.finditer('maxplayer',game)]
        maxplayer = game[player_pos[2]:game.find(',',player_pos[2])]
        maxplayer = maxplayer[13:-1]
    #    print(maxplayer)

        #----MAX PLAYTIME----#
        time_pos = [m.start() for m in re.finditer('maxplaytime',game)]
        maxtime = game[time_pos[2]:game.find(',',time_pos[2])]
        maxtime = maxtime[14:-1]
    #    print(maxtime)

        #----WEIGHT----#
        weight_pos = [m.start() for m in re.finditer('averageweight',game)]
        weight = game[weight_pos[0]:game.find(',',weight_pos[0])]
        weight = weight[15:]
    #    print(weight)


        #----CATEGORIES----#
        #Get the category text
        boardgamecategory_pos = [m.start() for m in re.finditer('boardgamecategory',game)]

        #We can see that the fourth pos value is the script segment that has all the actual categories
        #Get the string that contains all the categories
        categories = game[boardgamecategory_pos[4]:game.find(']',boardgamecategory_pos[4])]

        #Split by '{' character
        categories = categories.split('{')

        Categories = [] #Empty list to store the categories

        #First value is just the heading. Start with second
        for k in range(1,len(categories)):
            #Can see the first pair is "name":"category". So we pull category out using.
            #First split again by ',' then by '"', then grab the appropriate string
            #Because of how we converted the html to a string, there are forward-slash breaks: e.g. "Action \\/ Movement Programming"
            #Use regular expression (re) to remove them: sub out all '\\' values ('\\\\' to '')
            cat = re.sub('\\\\','',categories[k].split(',')[0].split('"')[3])
            Categories.append(cat)
    #    print(Categories)

        #Get total categories for given boardgame
        #Get the string that contains the number
        total_cat = game[boardgamecategory_pos[-1]:game.find(',',boardgamecategory_pos[-1])]
        #Get the number and convert to int
        total_cat = int(total_cat.split(':')[1])

        #----MECHANICS----#
        #Get the mechanics text
        mechanic_pos = [m.start() for m in re.finditer('boardgamemechanic',game)]

        #Same as Categories; fourth pos is the script segment we want
        mechanic = game[mechanic_pos[4]:game.find(']',mechanic_pos[4])]

        #Split by '{' character
        mechanic = mechanic.split('{')

        Mechanics = [] #Empty list to store the mechanics

        #First value is just the heading. Start with the second
        for k in range(1,len(mechanic)):
            #Because of how we converted the html to a string, there are forward-slash breaks: e.g. "Action \\/ Movement Programming"
            #Use regular expression (re) to remove them
            #1) From mechanic list, take the current string
            #2) Split string by ','; the mechanics name is in the first [0] item
            #3) Split by '"'; the actual mechanics name is in the fourth [4] item
            #4) sub out all '\\' values ('\\\\' to '')
            mech = re.sub('\\\\','',mechanic[k].split(',')[0].split('"')[3])
            Mechanics.append(mech)
    #    print(Mechanics)

        #Get total mechanics for given boardgame
        #Get the string that contains the number
        total_mech = game[mechanic_pos[-1]:game.find(',',mechanic_pos[-1])]
        #Get the number and convert to int
        total_mech = int(total_mech.split(':')[1])


        #----Add data to game_attributes df----#
        game_attributes.loc[1, 'Game name'] = game_name
        game_attributes.loc[1, 'Game rank'] = game_rank
        game_attributes.loc[1, 'Game ID'] = game_id
        game_attributes.loc[1, '#players'] = maxplayer
        game_attributes.loc[1, 'playtime'] = maxtime
        game_attributes.loc[1, 'weight'] = weight

        for cat in Categories:
            game_attributes.loc[1, cat] = 1

        for mech in Mechanics:
            game_attributes.loc[1, mech] = 1

        game_attributes.loc[1, 'total categories'] = total_cat
        game_attributes.loc[1, 'total mechanics'] = total_mech

    return game_attributes

# Scale non-binary columns of game attributes dataframe ------------------------
def scale_dataframe(ga):
    '''Take bgg game attributes dataframe, remove the label columns (name, rank,
    id) and scale the non-binary columns (#players, playtime, weight)

    Input: game attributes DataFrame
    Output: scaled game attributes Dataframe
    '''
    import pandas as pd
    #Import sklearn standardscaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    #Set up PCA-dataset; remove text
    ga_to_pca = ga.drop(['Game name','Game rank','Game ID','total categories',
                         'total mechanics'], axis=1)

    #Fit and Transform via standard scaler
    scaler.fit(ga_to_pca[['#players','playtime','weight']])
    scaled_ga = scaler.transform(ga_to_pca[['#players','playtime','weight']])

    #Concatenate the normalized columns to the binary columns
    scaled_ga = pd.concat([pd.DataFrame(scaled_ga,
                                        columns=['#players','playtime','weight']
                                       ),
                           ga_to_pca.drop(['#players','playtime','weight'],
                                          axis=1)
                          ],
                          axis=1)

    return scaled_ga

# Run TSNE, plot, and output TSNE coordinates ----------------------------------
def game_tsne(original_ga, scaled_ga, perp=100, steps=2000, printout=False):
    '''Take a scaled game attribute dataframe and run the tsne function to find
    the mapped coordinates. Print a plot and return the coordinates.

    Input: Scaled parameter dataframe, perplexity and steps parameters for tsne
    Output: TSNE coordinate dataframe
    '''
    import pandas as pd
    from sklearn.manifold import TSNE # Import t-sne

    tsne = TSNE(n_components=2, verbose=1, perplexity=perp, n_iter=steps)
    tsne_results = tsne.fit_transform(scaled_ga)

    df = pd.concat([original_ga['Game name'],
                    pd.DataFrame(tsne_results, columns=['Ax1','Ax2'])],
                    axis=1)

    # See the t-sne plot and save it
    out = str(perp) + ' Perplexity, ' + str(steps) + ' Steps'
    if printout:
        ts= df.plot(x='Ax1',y='Ax2',kind='scatter',title=out)

    out = 'game_tsne '+out+'.png'
    #ts.get_figure().savefig(out)

    out = 'bgg tsne coordinates ' + str(perp) + 'Perplexity ' + str(steps) + 'Steps.csv'
    #df.to_csv(out, encoding='utf-8')

    return df

# Threshold a collection based on game ratings ---------------------------------
def RatingThreshold(collection, printout=False):
    '''Filter a collection based on a rating threshold (user or bgg) of 7

    Input: Dataframe of games representing a collection
    Output: Curated dataframe of games
    '''
    import pandas as pd
    if printout:
        print('Collection Size start: ' + str(collection.shape[0]))
    # First remove any games that have no User Rating and no BGG rating
    #Grab indices of N/A User and BGG rating rows
    na_idx = collection[(collection['User rating'].isnull()) &
                        (collection['BGG rating'].isnull())].index.tolist()
    collection.drop(collection.index[na_idx], inplace=True)

    if printout:
        print('Collection Size after removing N/A ratings: ' +
                str(collection.shape[0]))

    # Remove any games with User rating < 7
    #Ratings are stored as strings to account for N/A rating
    #Convert column to numeric using pandas to_numeric function
    #Set errors to 'coerce' which turns non-numeric strings to NaN values, which will work for numeric comparisons

    #Keep if User rating == N/A (for BGG based comparison) and User rating >= 7
    collection = collection[(collection['User rating'].isnull()) |
                            (pd.to_numeric(collection['User rating'],
                            errors='coerce') >= 7)]

    if printout:
        print('Collection Size after removing low User ratings: ' +
                str(collection.shape[0]))

    #Reset index
    collection.reset_index(drop=True, inplace=True)

    # Remove any games with BGG rating < 7 (if there is no User Rating)
    #Similarly, convert BGG rating column to_numeric
    #Find row indices where User rating == N/A and BGG rating < 7
    #Drop relevant rows from DataFrame
    low_bgg_idx = collection[(collection['User rating'].isnull()) &
                                (pd.to_numeric(collection['BGG rating'],
                                                errors='coerce') < 7)].index.tolist()
    collection.drop(collection.index[low_bgg_idx],inplace=True)
    collection.reset_index(drop=True, inplace=True)
    if printout:
        print('Collection Size after removing low BGG ratings (if no User rating available): ' +
                str(collection.shape[0]))
    return collection

# Find mean TSNE Similarity of a game collection -------------------------------
def collection_distance(games, tsne):
    '''Provided a username in our playerlist and the tsne/game coordinate file,
    return the average distance/dissimilarity score of the user's collection

    Input: List of games,
        Dataframe of dimensions for each game (df)
    Output: Mean distance of the games in the list
    '''
    import pandas as pd
    import numpy as np

    games = list(games)
    dist = []
    for i in np.arange(0,len(games)):
        g1 = games[i]
        if g1 in tsne['Game name'].values:
            g1_x = tsne[tsne['Game name'] == g1]['Ax1'].iloc[0]
            g1_y = tsne[tsne['Game name'] == g1]['Ax2'].iloc[0]
            for j in np.arange(i+1, len(games)):
                g2 = games[j]
                if g2 in tsne['Game name'].values:
                    g2_x = tsne[tsne['Game name'] == g2]['Ax1'].iloc[0]
                    g2_y = tsne[tsne['Game name'] == g2]['Ax2'].iloc[0]

                    d = (g1_x - g2_x)**2 + (g1_y - g2_y)**2
                    dist = dist + [d]
    return np.mean(dist)

# Find mean Euclidean distance of a game collection ---------------------------
def collection_euc_distance(games, game_attributes):
    '''Provided a username in our playerlist and the tsne/game coordinate file,
    return the average distance/dissimilarity score of the user's collection.

    Input: List of games,
        Dataframe of dimensions for each game (df)
    Output: Mean distance of the games in the list
    '''
    import pandas as pd
    import numpy as np

    games = list(games)
    dist = []
    for i in np.arange(0,len(games)):
        g1 = games[i]
        if g1 in game_attributes['Game name'].values:
            g1_coord = game_attributes[game_attributes['Game name'] == g1].iloc[0,1:].as_matrix()
            for j in np.arange(i+1, len(games)):
                g2 = games[j]
                if g2 in game_attributes['Game name'].values:
                    g2_coord = game_attributes[game_attributes['Game name']==g2].iloc[0,1:].as_matrix()
                    d = np.sqrt(np.square(g1_coord-g2_coord).sum()) # Distance calculation
                    dist = dist + [d]
    return np.mean(dist)

# Calculate distances ----------------------------------------------------------
def calc_distance(target_coord, all_coord):
    ''' Take a set of target coordinates and a matrix containing all sets of
    coordinates. Calculate the distance the target from all other points in space.

    Input: Target coordinate (df row),
        Array of all other coordinates (df)
    Output: Array of calculated distances (df)
    '''
    import numpy as np
    # Calculate distance
    summed_distance = np.square(all_coord-target_coord).sum(axis=1) #Euclidean distance formula
    distance = np.sqrt(summed_distance.astype(np.float64)) #Need to convert summed_distance to np.float64 for sqrt function to work

    return distance

# Find Closest Games -----------------------------------------------------------
def find_closest_game(game, game_coordinates,n=5):
    '''Take the target game name (string), the dataframe of coordinates (pandas
    dataframe), and number of games to return (int). Game_coordinates can be
    euclidean or t-sne or any other coordinates. First column must be game name.
    All other columns are coordinates in multi-dimensional space.

    Input: Game name (string)
        Dataframe of coordinates (dataframe)
        Number of games to return (int)
    Output: Dataframe with CLOSEST boardgame names and calculated distances
    '''
    import pandas as pd
    import numpy as np

    # Get the coordinates of target game, converted to matrix
    target_coord = game_coordinates[game_coordinates['Game name'] == game].iloc[0,1:].as_matrix()

    # Convert the coordinate dataframe to matrix of coordinates
    all_coord = game_coordinates.iloc[:,1:].as_matrix()

    # Calculate distances
    distances = calc_distance(target_coord, all_coord)
    # Get the n+1 closest games, which includes the target game itself
    closest_idx = np.argpartition(distances, n+1)[:n+1]

    # Get the game names and concatenate to the distances
    #  Need to reset index for concatenation
    found = pd.concat([game_coordinates.iloc[closest_idx,0].reset_index(drop=True),
                       pd.DataFrame(distances[closest_idx],columns=['Dissimilarity/Distance'])],
                     axis=1)

    # Remove the target game from the output
    found = found[found['Game name'] != game]

    #Sort output by calculated distance
    return found.sort_values(by=['Dissimilarity/Distance']).reset_index(drop=True)

# Find Furthest Games ----------------------------------------------------------
def find_furthest_game(game, game_coordinates, k=5):
    '''Take the target game name (string), the dataframe of coordinates (pandas
    dataframe), and number of games to return (int). Game_coordinates can be
    euclidean or t-sne or any other coordinates. First column must be game name.
    All other columns are coordinates in multi-dimensional space.

    Input: Game name (string)
        Dataframe of coordinates (dataframe)
        Number of games to return (int)
    Output: Dataframe with FURTHEST boardgame names and calculated distances
    '''
    import pandas as pd
    import numpy as np

    # Get the coordinates of target game, converted to matrix
    target_coord = game_coordinates[game_coordinates['Game name'] == game].iloc[0,1:].as_matrix()

    # Convert the coordinate dataframe to matrix of coordinates
    all_coord = game_coordinates.iloc[:,1:].as_matrix()

    # Calculate distances
    distances = calc_distance(target_coord, all_coord)
    # Get the k furthest games
    closest_idx = np.argsort(distances)[-k:]

    # Get the game names and concatenate to the distances
    #  Need to reset index for concatenation
    found = pd.concat([game_coordinates.iloc[closest_idx,0].reset_index(drop=True),
                       pd.DataFrame(distances[closest_idx],columns=['Dissimilarity/Distance'])],
                     axis=1)

    # Remove the target game from the output
    found = found[found['Game name'] != game]

    #Sort output by calculated distance
    return found.sort_values(by=['Dissimilarity/Distance'], ascending=False).reset_index(drop=True)

# Get User Collection ----------------------------------------------------------
def UserCollection(user):
    '''Take a user name (string) and pull their collection off boardgamegeekself.

    Input: Username (string)
    Ouptut: Dataframe with user's boardgame collection
    '''
    # Import necessary libraries
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd

    # Set up pd dataframe
    col = ['User','Game','User rating', 'BGG rating']
    games = pd.DataFrame(columns = col)

    # Pull webpage
    url = "https://boardgamegeek.com/collection/user/" + user + "?own=1&subtype=boardgame&ff=1"
    r = requests.get(url)

    data = r.text

    soup = BeautifulSoup(data, "lxml")

    # Get all the "Objects" that contain each boardgame in the collection.
    #Use the fact that the boardgame collection is the only one with the attribute "id".
    #All other tables on the page have no additional attributes
    all_games = soup.find_all(lambda tag:tag.name=='tr' and len(tag.attrs) > 0)

    for i in range(0,len(all_games)):
        # Start with 1 boardgame "Object"
        current_game = all_games[i]

        # To return Game name:
        #t is a bs4 tag
        #collection rating is stored under td tag, class = collection_objectname , a
        t_name = current_game.find_all('td', {'class':'collection_objectname '})[0]
        name = t_name.a.contents[0]


        # To return (user) Collection Rating:
        #t is a bs4 tag
        #collection rating is stored under td tag, class = collection_rating , div class=ratingtext
        t_UserRating = current_game.find_all('td',{'class':'collection_rating '})[0]
        t_UserRating = t_UserRating.find_all('div',{'class':'ratingtext'})

        # It's possible the user does not provide a rating, so we should check for that
        if t_UserRating:
            t_UserRating = t_UserRating[0]
            UserRating = t_UserRating.contents[0]
        else:
            UserRating = 'N/A'


        # To return BGG Rating:
        #t is a bs4 tag
        #collection rating is stored under td tag, class = collection_bggrating
        t_BggRating = current_game.find_all('td',{'class':'collection_bggrating'})

        #Can't call the td attribute for some reason. Use 'text' to pull the text string
        rate_text = t_BggRating[0].text

        #The text string includes \n and \t spacing. Get just the rating number (a float)
        #Find first index of a numerical digit
        #This can be done using the regular expression library 're'
        import re
        start = re.search("\d",rate_text) #\d re flag for any decimal digit [0-9]

        # It's possible bgg does not provide a rating, so we should check
        if start is None:
            #Start is currently set as a NoneType if regex does not find a numerical character
            start = rate_text.find('N') #Use standard str.find to locate index for "N/A"
        else:
            #Regex does return an output
            start = re.search("\d",rate_text).start()

        #Find the end of the rating string
        end = rate_text.find('\t',start)

        #Get the decimal rating
        BggRating = rate_text[start:end]


        # Final Result
        print('User: ' + user)
        print('Game name: ' + name)
        print('User rating: ' + UserRating)
        print('BGG rating: ' + BggRating)
        print('\n')

        games = games.append(pd.DataFrame([[user,name, UserRating, BggRating]], columns=col))

    # Reset the index of the pd dataframe since the rows get appended all with index=0
    games.reset_index(drop=True, inplace=True)

    # Export collection as a csv for later
    games.to_csv(user+'_raw.csv',sep='\t')

    return games

# Games to Vector --------------------------------------------------------------
def games2vec(user_collection, games_dict):
    '''Take the user collection (game titles series) and game title dictionary
        as input. Output a one-hot vector representation of the collection. All
        mismatched game titles will be ignored.

    Input: Dataframe containing user's collection (df),
      Games dictionary (df)
    Output: Vector representation fo the collection
    '''

    # Use numpy to generate the vector
    import numpy as np

    # Set up unpopulated vector of 0's
    collection = np.zeros((1,len(games_dict)))

    for i in range(0,len(user_collection)):
        current_game = user_collection.iloc[i]

        if current_game in games_dict:
            current_idx = games_dict[current_game]
            collection[0, current_idx] = 1

    return collection

# Create boardgame dictionary --------------------------------------------------
def bgg_dict(min_rank=-1):
    '''Create dictionary of all ranked games on BGG to a number; the number is
    the index for the game used in the data sets. Also returns a reverse
    dictionary of number to game for decoding.
    Input: Minimum rank to put in the dicionary (higher rank = lower number; lower
      rank = higher number)
     Returns: Games dictionary, Games_reverse dictionary
    '''

    import pandas as pd
    # Create dictionary of all games. From previous webscraping, we have a csv with all ranked games on BGG. We'll limit our recommendations to games on the list.
    games_list = pd.read_csv('bgg_id_output.csv')

    # Remove all NaN rows
    games_list.dropna(axis=0,how='any',inplace=True)
    games_list.reset_index(drop=True,inplace=True)

    # There are repeat titles in the list. Remove them.
    rep_games_idx = games_list[games_list['Game'].duplicated()].index.tolist() #Returns the indices of all repeat titles. This list does NOT include the first appearance
    games_list.drop(games_list.index[rep_games_idx],inplace=True)
    games_list.reset_index(drop=True,inplace=True)

    # Get just the titles
    games = games_list['Game']

    # Create empty dictionary
    games_idx = dict()

    # Establish minimum game rank to consider; default -1 considers all games
    if min_rank == -1:
        min_rank = len(games)
    elif min_rank >= len(games):
        min_rank = len(games)
    else:
        min_rank = min_rank+1

    # The dictionary games_idx will use Game Titles as the key and a number as the value
    for i in range(0,min_rank):
        game_title = games[i]
        games_idx[game_title] = i


    games_reverse = dict((y,x) for x,y in games_idx.items())

    return games_idx, games_reverse

# Create training set ----------------------------------------------------------
def generate_training(collection, games_dict):
    """Takes a user's collection as input along with the games dictionary.
    Outputs a matrix where each row is the training vector (predictor) for a game
    and another matrix where each row is the expected vector (predicted)
    Input: User collection (df), games dictionary to set length of each row (df)
    Output: Two matrices: one predictor and one predicted matrix
    """

    import numpy as np

    # Find all indices in the collection array with value=1
    in_collection = np.nonzero(collection)[0] #returns index of nonzeros (aka value=1) of collection array

    # Create predictor vectors and predicted vectors
    #Iterate through every game in the collection, indices given by in_collection
    #Set value at the current index to 0 for predictor vector
    #Set a predicted vector to have 1 at the current index

    # Create two empty np arrays. #Rows = #Games in collection. #Columns = #Game Titles
    predictor = np.empty([len(in_collection), len(games_dict)])
    predicted = np.zeros([len(in_collection), len(games_dict)])

    for i in range(0,len(in_collection)):
        # Index of current game
        current_game = in_collection[i]

    #     # Create copy of the collection
    #     curr_predictor = np.copy(collection)
    #    # Set current boardgame value to 0 in predictor
    #    curr_predictor[0,current_game] = 0

        # Set values in current predictor row to collection
        predictor[i,:] = collection
        # Set current boardgame value to 0 in predictor
        predictor[i,current_game] = 0

        # Set current boardgame in predicted vector to 1
        predicted[i,current_game] = 1

    return predictor, predicted
