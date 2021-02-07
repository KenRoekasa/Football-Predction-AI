import json
import pickle
import random

import numpy
import pandas as pd


# pd.set_option("display.max_rows", None, "display.max_columns", None)


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
    all_games = get_all_team_games(table, team).sort_values(by=['date']).copy()
    all_games = all_games.reset_index(drop=True)
    # print(all_games)

    index = all_games[all_games['link'] == game['link']].index[0]

    if index > n:
        previous = index - n

    else:
        previous = 0

    previous_games = all_games.iloc[previous:index]
    return previous_games


def format_data(data):
    # Change date to date format
    data['date'] = pd.to_datetime(data["date"])

    # Make a subset of the table to include the fields we need
    data_subset = data[
        ['date', 'link', 'home team', 'away team', 'home score', 'away score', 'home total shots', 'away total shots',
         'home shots on target', 'away shots on target', 'home possession', 'away possession',
         'home total conversion rate',
         'away total conversion rate', 'home fouls', 'away fouls', 'home yellow cards', 'away yellow cards',
         'home red cards', 'away red cards', 'home total passes', 'away total passes', 'home accurate passes',
         'away accurate passes', 'home open play conversion rate', 'away open play conversion rate',
         'home set piece conversion', 'away set piece conversion', 'home counter attack shots',
         'away counter attack shots',
         'home counter attack goals', 'away counter attack goals', 'home key passes', 'away key passes',
         'home dribbles attempted', 'away dribbles attempted', 'home dribble success', 'away dribble success',
         'home aerials won%', 'away aerials won%', 'home tackles attempted', 'away tackles attempted',
         'home tackles success %', 'away tackles success %', 'home was dribbled', 'away was dribbled',
         'home interceptions',
         'away interceptions', 'home dispossessed', 'away dispossessed', 'home errors', 'away errors'
         ]].copy()

    # data_subset = data[
    #     ['date', 'link', 'home team', 'away team', 'home score', 'away score', 'home possession', 'away possession',
    #      'home total conversion rate',
    #      'away total conversion rate', 'home accurate passes',
    #      'away accurate passes', 'home open play conversion rate', 'away open play conversion rate',
    #      'home set piece conversion', 'away set piece conversion',
    #      'home counter attack goals', 'away counter attack goals', 'home key passes', 'away key passes', 'home dribble success', 'away dribble success',
    #      'home aerials won%', 'away aerials won%', 'home tackles attempted', 'away tackles attempted',
    #      'home tackles success %', 'away tackles success %', 'home was dribbled', 'away was dribbled',
    #      'home interceptions',
    #      'away interceptions', 'home dispossessed', 'away dispossessed', 'home errors', 'away errors'
    #      ]].copy()

    # remove percentages symbol
    percentage_column = ['home total conversion rate',
                         'away total conversion rate', 'home open play conversion rate',
                         'away open play conversion rate', 'home set piece conversion', 'away set piece conversion']
    for column in percentage_column:
        data_subset[column] = data_subset[column].str.rstrip('%').astype('float64') / 100.0
    data_subset.dropna(inplace=True)
    data_subset = data_subset.sort_values(by=['date'])
    data_subset = data_subset.reset_index(drop=True)
    return data_subset


def get_training_data(
        data):  # TODO comment functions
    n = 6  # n is the last previous games to get the average from
    training_data = []
    for i in range(20, 2260):
        # Select a random team
        # table_of_teams = data['home team'].unique()
        # random_team = data.iloc[random.randrange(0, len(table_of_teams))]['home team']
        # Select a random game
        random_game = data.iloc[i]

        teama = random_game['home team']
        teamb = random_game['away team']

        home_goals = random_game['home score']
        away_goals = random_game['away score']

        if home_goals > away_goals:
            classification_label = 0  # win
        elif home_goals == away_goals:
            classification_label = 1  # draw
        else:
            classification_label = 2  # lose

        # print(random_game)
        # find previous n games for each team

        teama_previous_games = get_previous_n_games(data, teama, n, random_game)
        # if previous games is empty recently promoted to the league
        # if len(teama_previous_games) == 0:
        #     teama_previous_games = pd.DataFrame(0, index=numpy.arange(1), columns=teama_previous_games.columns)

        # print(teama_previous_games)
        teamb_previous_games = get_previous_n_games(data, teamb, n, random_game)
        # print(teamb_previous_games)

        teama_mean = get_mean_stats(teama_previous_games, teama)
        teamb_mean = get_mean_stats(teamb_previous_games, teamb)
        # print(teama_mean)
        # print(teamb_mean)
        teama_mean_array = teama_mean.array.to_numpy(copy=True)
        teamb_mean_array = teamb_mean.array.to_numpy(copy=True)
        # print(teama_mean_array)
        # print(teamb_mean_array)

        mean_data_array = numpy.append(teama_mean_array, teamb_mean_array)
        # print(mean_data_array)

        training_data.append([mean_data_array, classification_label])
    #     print(training_data)

    return training_data


def get_random_game(csvfile):
    data = pd.read_csv(csvfile)
    data_subset = format_data(data)
    random_game = data_subset.iloc[random.randrange(0, len(data) - 20)]
    teama = random_game['home team']
    teamb = random_game['away team']
    n = 6
    home_goals = random_game['home score']
    away_goals = random_game['away score']

    if home_goals > away_goals:
        classification_label = 0  # win
    elif home_goals == away_goals:
        classification_label = 1  # lose
    else:
        classification_label = 2  # draw

    # print(random_game)
    # find previous n games for each team

    teama_previous_games = get_previous_n_games(data_subset, teama, n, random_game)





    teamb_previous_games = get_previous_n_games(data_subset, teamb, n, random_game)
    # print(teamb_previous_games)
    teama_mean = get_mean_stats(teama_previous_games, teama)
    teamb_mean = get_mean_stats(teamb_previous_games, teamb)
    # print(teama_mean)
    # print(teamb_mean)
    teama_mean_array = teama_mean.array.to_numpy(copy=True)
    teamb_mean_array = teamb_mean.array.to_numpy(copy=True)
    # print(teama_mean_array)
    # print(teamb_mean_array)

    mean_data_array = numpy.append(teama_mean_array, teamb_mean_array)
    # print(mean_data_array)

    return [teama, teamb, home_goals, away_goals, mean_data_array, classification_label]


def get_mean_stats(previous_games, team):
    # Get home statistics
    # Get all home games
    home_games = get_team_home_games(previous_games, team)
    home_games = home_games.filter(regex='home')
    # print(home_games)
    #
    # print(home_games.mean(numeric_only=True))
    home_mean = home_games.mean(numeric_only=True)
    # print(home_mean)
    # Get away statistics
    away_games = get_team_away_games(previous_games, team)
    away_games = away_games.filter(regex='away')
    away_mean = away_games.mean(numeric_only=True)
    # print(away_mean)
    # Combine away and home statistics to get the final
    away_mean.index = home_mean.index
    combined_table = pd.DataFrame([home_mean, away_mean])
    combined_mean = combined_table.mean(numeric_only=True)
    combined_mean.fillna(0,inplace=True)
    return combined_mean


def merge_premier_league_table():
    data = pd.read_csv("../data/whoscored/premierleague-20132014.csv")
    data = data.append(pd.read_csv("../data/whoscored/premierleague-20142015.csv"), ignore_index=True)
    data = data.append(pd.read_csv("../data/whoscored/premierleague-20152016.csv"), ignore_index=True)
    data = data.append(pd.read_csv("../data/whoscored/premierleague-20162017.csv"), ignore_index=True)
    data = data.append(pd.read_csv("../data/whoscored/premierleague-20172018.csv"), ignore_index=True)
    data = data.append(pd.read_csv("../data/whoscored/premierleague-20182019.csv"), ignore_index=True)
    data = data.append(pd.read_csv("../data/whoscored/premierleague-20192020.csv"), ignore_index=True)
    return format_data(data)


def load_premier_league_data():
    with open("../data/whoscored/trainingdata.pickle", "rb") as f:
        training_data = pickle.load(f)
        return training_data


def generate_premier_league_data():
    training_data = get_training_data(merge_premier_league_table())
    with open('../data/whoscored/trainingdata.pickle', 'wb+') as file:
        pickle.dump(training_data, file)


def load_training_data(data):
    pass


if __name__ == '__main__':
    generate_premier_league_data()
