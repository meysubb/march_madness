import pandas as pd
import numpy as np
import math
from trueskill import TrueSkill, Rating, rate_1vs1

def calc_ts(dataframe):
    df = dataframe
    ts = TrueSkill()
    
    teams = list(set(df['HTeamID'].append(df['ATeamID'])))
    ts_dict = {}
    
    for team in teams:
        ts_dict[team] = {'TS':ts.Rating()}
        
    for index, row in df.iterrows():
        home_rating = ts_dict[row['HTeamID']]['TS']
        away_rating = ts_dict[row['ATeamID']]['TS']
        if row['HScore'] > row['AScore']:
            home_rating, away_rating = ts.rate_1vs1(home_rating, away_rating)
        elif row['AScore'] > row['HScore']:
            away_rating, home_rating = ts.rate_1vs1(away_rating, home_rating)
        else:
            home_rating, away_rating = ts.rate_1vs1(home_rating, away_rating, drawn = True)

        ts_dict[row['HTeamID']]['TS'] = home_rating
        ts_dict[row['ATeamID']]['TS'] = away_rating

    ts_list = [ts_dict[team]['TS'].mu for team in teams]
    
    return pd.DataFrame({'TS':ts_list},index = teams)

def calc_colley(dataframe):
    df = dataframe
    
    teams = list(set(df['HTeamID'].append(df['ATeamID'])))
    n_teams = len(teams)
    c_mat = np.identity(n_teams) * 2
    b_mat = np.zeros(n_teams)
    
    for index, row in df.iterrows():
        home_team_ndx = teams.index(row['HTeamID'])
        away_team_ndx = teams.index(row['ATeamID'])
        spread = row['HScore'] - row['AScore']

        c_mat[home_team_ndx, home_team_ndx] += 1
        c_mat[away_team_ndx, away_team_ndx] += 1
        c_mat[home_team_ndx, away_team_ndx] -= 1
        c_mat[away_team_ndx, home_team_ndx] -= 1

        b_mat[home_team_ndx] += (row['HScore'] > row['AScore']) * 1 + (row['AScore'] > row['HScore']) * -1
        b_mat[away_team_ndx] += (row['AScore'] > row['HScore']) * 1 + (row['HScore'] > row['AScore']) * -1

    b_mat = b_mat / 2 + 1
    ranks = np.linalg.solve(c_mat,b_mat).tolist()
    return pd.DataFrame({'Colley':ranks},index = teams)

# mod comesfrom R function
def calc_lrmc(dataframe,mod):
    df = dataframe
  
    h = mod['h']
    a = mod['a']
    b = mod['b']
    
    teams = list(set(df['HTeamID'].append(df['ATeamID'])))
    
    n_teams = len(teams)
    p = np.zeros((n_teams,n_teams))
    n_games = np.zeros(n_teams)
    
    for index, row in df.iterrows():
        home_team_ndx = teams.index(row['HTeamID'])
        away_team_ndx = teams.index(row['ATeamID'])
        
        # calculate r_x
        spread = row['HScore'] - row['AScore'] + h * (row['NeutralFlag'])
        r_x = math.exp(a * spread + b) / (1 + math.exp(a * spread + b))
        
        # update respective matrices
        n_games[home_team_ndx] += 1
        n_games[away_team_ndx] += 1
        
        p[home_team_ndx, away_team_ndx] += 1 - r_x
        p[away_team_ndx, home_team_ndx] += r_x
        p[home_team_ndx, home_team_ndx] += r_x
        p[away_team_ndx, away_team_ndx] += 1 - r_x
        
    # solve matrix
    p = p / n_games[:,None]
    prior = n_teams - np.array(list(range(n_teams)))
    steady_state = np.linalg.matrix_power(p, 1000)
    rating = prior.dot(steady_state)
    
    return pd.DataFrame({'LRMC':rating}, index = teams)


def calc_elo(dataframe, avg_elo = 1500, elo_width = 400, mov_factor = 8, hfa = 135, hfa_rescore = False, ot_rescore = False):
    df = dataframe
    
    df = df.sort_values(by=['DayNum'])
    
    teams = list(set(df['HTeamID'].append(df['ATeamID'])))
    elos = {team:avg_elo for team in teams}
    
    for index, row in df.iterrows():
        home_elo = elos[row['HTeamID']]
        away_elo = elos[row['ATeamID']]
        
        home_elo_adj = home_elo
        if not row['NeutralFlag'] and not hfa_rescore and not ot_rescore:
            home_elo_adj += hfa
            
        # calculate expected score
        home_prob = 1.0/(1+10**((away_elo - home_elo_adj)/elo_width))
        away_prob = 1 - home_prob
        
        # calculate k factor as a function of margin of the log of margin of victory
        home_diff = row['HScore'] - row['AScore']
        k_factor = math.log(abs(home_diff)+1) * mov_factor
        
        home_win = (row['HScore'] > row['AScore']) * 1
        away_win = (row['AScore'] > row['HScore']) * 1
        
        # update elos
        elos[row['HTeamID']] = home_elo + k_factor * (home_win - home_prob)
        elos[row['ATeamID']] = away_elo + k_factor * (away_win - away_prob)
    
    elo_df = pd.DataFrame(elos,index=[0]).transpose()
    elo_df.columns = ['Elo']
    
    return elo_df
