import graph_tool.all as gt
import pandas as pd
import numpy as np
import requests
import json

def get_graph(team, ssn, web, directed=False, verbose=False):
    """Scrape stats.nba.com to create a graph of a team's passing or assist connections.

    Parameters
    ----------
    team : str
        The three letter abbreviation of an NBA team.
    ssn : str
        The season in which you'd like to create the network.
    web : str
        The type of network you'd like to create. 'pass' for a passing network, 'assist' for an assist network.
    directed : bool, optional
        Whether or not you'd like the graph to be directed. The default is False.
    verbose : bool, optional
        Whether or not you'd like to print the steps of the function. The default is False.

    Returns
    -------
    g : graph_tool.Graph
        The graph of the team's passing or assist connections.

    Examples
    --------
    >>> get_graph('GSW','2016-17','AST')
    """

    if verbose == True:
        print('Team: ' + team)
        print('Season: ' + ssn)
        print('Network: ' + web)
        print('Verbose: ' + str(verbose))
        print('-------------------------')
    
    if (web in ['ast','AST','assist','assists','assisting','ASSIST','ASSISTS']):
        web = 'assist'
    elif (web in ['pass','passes','passing','PASS','PASSES']):
        web = 'pass'
    else:
        return print('Error: Third parameter requires "assist" or "pass" input')
    
    headers = {'Host': 'stats.nba.com','Accept': 'application/json, text/plain, */*','x-nba-stats-token': 'true','User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Mobile Safari/537.36','x-nba-stats-origin': 'stats','Origin': 'https://www.nba.com','Referer': 'https://www.nba.com/','Accept-Encoding': 'gzip, deflate, br','Accept-Language': 'en-US,en;q=0.9}'}
    
    ### scrape stats.nba.com to find the team_id for the inputted team 
    
    url = 'https://stats.nba.com/stats/leaguedashptstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode=PerGame&Period=0&PlayerExperience=&PlayerOrTeam=Team&PlayerPosition=&PtMeasureType=Drives&Season=' + str(ssn) + '&SeasonSegment=&SeasonType=Regular+Season&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight='

    json = requests.get(url, headers=headers).json()

    data = json['resultSets'][0]['rowSet']
    columns = json['resultSets'][0]['headers']

    tms = pd.DataFrame.from_records(data, columns=columns)

    team_id = tms[tms.TEAM_ABBREVIATION == team].reset_index(drop=True).TEAM_ID[0]
    team_name = tms[tms.TEAM_ABBREVIATION == team].reset_index(drop=True).TEAM_NAME[0]
    
    ### using the scraped team_id, find all players who accumulated at least 10 assists with that team in the inputted season

    url = 'https://stats.nba.com/stats/leaguedashplayerstats?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&GameScope=&GameSegment=&Height=&LastNGames=0&LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&PerMode=Totals&Period=0&PlayerExperience=&PlayerPosition=&PlusMinus=N&Rank=N&Season=' + str(ssn) + '&SeasonSegment=&SeasonType=Regular+Season&ShotClockRange=&StarterBench=&TeamID=' + str(team_id) + '&TwoWay=0&VsConference=&VsDivision=&Weight='
    json = requests.get(url, headers=headers).json()

    data = json['resultSets'][0]['rowSet']
    columns = json['resultSets'][0]['headers']

    df = pd.DataFrame.from_records(data, columns=columns)
    
    df = df[df.AST > 10]

    players = df.PLAYER_ID.unique()
    minutes_played = df.MIN.unique()
    
    ### using the players previously found, record all of their pass connections while on the team

    if verbose == True:
        print('# of players: ' + str(len(players)))
        print('-------------------------')



    df_list = []

    for x in players:

        url = 'https://stats.nba.com/stats/playerdashptpass?DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PerMode=Totals&Period=0&PlayerID=' + str(x) + '&Season=' + str(ssn) + '&SeasonSegment=&SeasonType=Regular+Season&TeamID=0&VsConference=&VsDivision='

        json = requests.get(url, headers=headers).json()

        data = json['resultSets'][0]['rowSet']
        columns = json['resultSets'][0]['headers']

        df_list.append(pd.DataFrame.from_records(data, columns=columns))

    df = pd.concat(df_list)

    df = df[['TEAM_ABBREVIATION','PLAYER_NAME_LAST_FIRST','PASS_TO','PASS','AST']]
    df.columns = ['tm','passer','receiver','passes','ast']
    
    ### clean up the format for the name

    def fix_name(name):
        if ',' in name:
            return name.split(", ")[1][:1] + "." + name.split(", ")[0]
        else:
            return name
    df.passer = np.vectorize(fix_name)(df.passer)
    df.receiver = np.vectorize(fix_name)(df.receiver)

    df = df[df.receiver.isin(df.passer.unique())].reset_index(drop=True)

    players = df.passer.unique()
    
    ### making list with assist totals for each player
    
    ast_list = []
    for x in players:
        tf = df[df.passer == x].reset_index(drop=True)
        ast_list.append(tf.ast.sum())
        
    ### creating adjacency matrix

    adf = pd.DataFrame(index=players,columns=players)

    adf.values[[np.arange(adf.shape[0])]*2] = 0
    
    if web == 'pass':
        for x in players:
            for y in range(0,len(players)):
                tf1 = df[(df.passer == x) & (df.receiver == adf.columns[y])].reset_index(drop=True)
                tf2 = df[(df.passer == adf.columns[y]) & (df.receiver == x)].reset_index(drop=True)
                if (tf1.shape[0] == 1) & (tf2.shape[0] == 0):
                    adf.at[x,adf.columns[y]] = tf1.passes[0]
                elif (tf1.shape[0] == 0) & (tf2.shape[0] == 1):
                    adf.at[x,adf.columns[y]] = tf2.passes[0]
                elif (tf1.shape[0] == 1) & (tf2.shape[0] == 1):
                    adf.at[x,adf.columns[y]] = tf1.passes[0] + tf2.passes[0]
                else:
                    adf.at[x,adf.columns[y]] = 0
                    
    else:
        for x in players:
            for y in range(0,len(players)):
                tf1 = df[(df.passer == x) & (df.receiver == adf.columns[y])].reset_index(drop=True)
                tf2 = df[(df.passer == adf.columns[y]) & (df.receiver == x)].reset_index(drop=True)
                if (tf1.shape[0] == 1) & (tf2.shape[0] == 0):
                    adf.at[x,adf.columns[y]] = tf1.ast[0]
                elif (tf1.shape[0] == 0) & (tf2.shape[0] == 1):
                    adf.at[x,adf.columns[y]] = tf2.ast[0]
                elif (tf1.shape[0] == 1) & (tf2.shape[0] == 1):
                    adf.at[x,adf.columns[y]] = tf1.ast[0] + tf2.ast[0]
                else:
                    adf.at[x,adf.columns[y]] = 0
                    
    ### creating the graph
    g = gt.Graph(directed=directed)
    g.add_vertex(len(adf.index))

    if directed == False:
        # simmetrize the adjacency matrix
        adf = (adf + adf.T)  / 2
        # iterate over the upper triangle of the adjacency matrix
        for i in range(len(adf.index)):
            for j in range(i+1, len(adf.columns)):
                if adf.iloc[i,j] > 0:
                    g.add_edge(g.vertex(i), g.vertex(j)) 
    else:
        for i in range(len(adf.index)):
            for j in range(len(adf.columns)):
                if adf.iloc[i,j] > 0:
                    g.add_edge(g.vertex(i), g.vertex(j))

    weights = g.new_edge_property("int")
    for x in g.edges():
        weights[x] = adf.iloc[int(x.source()), int(x.target())]
    g.ep['weights'] = weights

    names = g.new_vertex_property("string")
    for x in range(0,len(adf.index)):
        names[g.vertex(x)] = adf.index[x]
    g.vp['names'] = names

    minutes = g.new_vertex_property("int")
    try:
        minutes.a = minutes_played
    except:
        for x in range(0,len(adf.index)):
            minutes[g.vertex(x)] = minutes_played[x]

    g.vp['minutes'] = minutes
    
    if verbose == True:
        print('Nodes: ', g.num_vertices())
        print('Edges: ', g.num_edges())
        print('-------------------------')
        print('Edge weights (number of connections):')
        for x in g.edges():
            print(g.vp['names'][x.source()], ' - ',g.vp['names'][x.target()], ': ', g.ep['weights'][x])
        print('-------------------------')
        print('Minutes played:')
        for x in g.vertices():
            print(g.vp['names'][x], ': ', g.vp['minutes'][x])
        print('-------------------------')

    if verbose == True:
        print(g)

    return g