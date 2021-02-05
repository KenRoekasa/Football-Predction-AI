import random

import pandas as pd


def get_team_home_games(table, team):
    return table[table["home team"].str.contains(team)].sort_values(by=['date'])


def get_team_away_games(table, team):
    return table[table["away team"].str.contains(team)].sort_values(by=['date'])


def get_all_team_games(table, team):
    return pd.concat([get_team_home_games(table, team), get_team_away_games(table, team)]).sort_values(by=['date'])


def select_random_consecutive_games(table, team, n):  # select random n consecutive games
    all_games = get_all_team_games(table, team).sort_values(by=['date'])

    first_index = random.randrange(0, len(all_games) - n)
    return all_games.iloc[first_index:first_index + n]


def get_previous_n_games(table, team, n, game):
    all_games = get_all_team_games(table, team).sort_values(by=['date'])
    all_games = all_games.reset_index(drop=True)
    print(all_games)

    index = all_games[all_games['link'] == game['link']].index[0]

    previous = index - n
    return all_games.iloc[previous:index]


def format_data(data):
    # Change date to date format
    data['date'] = pd.to_datetime(data["date"])

    # Make a subset of the table to include the fields we need
    data_subset = data[
        ['date', 'link', 'home team', 'away team', 'home score', 'away score', 'home total shots', 'away total shots',
         'home shots on target', 'away shots on target', 'home possession', 'away possession',
         'home total conversion rate',
         'away total conversion rate', 'home fouls', 'away fouls', 'home yellow cards', 'away yellow cards',
         'home red cards', 'away red cards', 'home total passes', 'away total passes', 'home accurate passess',
         'away accurate passess', 'home open play conversion rate', 'away open play conversion rate',
         'home set piece conversion', 'away set piece conversion', 'home counter atack shots',
         'away counter atack shots',
         'home counter attack goals', 'away counter attack goals', 'home key passes', 'away key passes',
         'home dribbles attempted', 'away dribbles attempted', 'home dribble success', 'away dribble success',
         'home aerials won%', 'away aerials won%', 'home tackles attempted', 'away tackles attempted',
         'home tackles success %', 'away tackles success %', 'home was dribbled', 'away was dribbled',
         'home interceptions',
         'away interceptions', 'home dispossesssed', 'away dispossesssed', 'home errors', 'away errors'
         ]]
    return data_subset


def get_training_data(
        data):  # Input formatted data and will give an an array of vectors to be inputted into the neural network

    # Select a random game

    n = 6
    random_game = data.iloc[random.randrange(n, len(data))]

    teama = random_game['home team']
    teamb = random_game['away team']

    print(random_game)
    # find previous n games for each team
    team_a_previous_games = get_previous_n_games(data, teama, n, random_game)
    team_b_previous_games = get_previous_n_games(data, teamb, n, random_game)

    # Get home statistics
    # Get all home games
    team_a_home_games = get_team_home_games(team_a_previous_games, teama)

    home_mean_error = team_a_home_games["home dispossesssed"].mean()
    print(home_mean_error)

    # Get away statistics
    get_team_away_games(team_a_previous_games, teama)

    # combine statistics if needed

    # Same for team b


if __name__ == '__main__':
    data = pd.read_csv("../data/whoscored/premierleague-20192020.csv")
data_subset = format_data(data)

all_games = get_all_team_games(data_subset, 'Manchester United')
print(all_games)
rand_games = select_random_consecutive_games(data_subset, 'Manchester United', 6)
print(rand_games)

print(get_team_home_games(rand_games, 'Manchester United'))

get_training_data(data_subset)
