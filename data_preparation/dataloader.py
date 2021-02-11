import glob
import os
import pickle
import random

import numpy
import pandas as pd

import model.config as config

pd.set_option("display.max_rows", None, "display.max_columns", None)


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
        config.COLUMNS].copy()

    # remove percentages symbol
    # percentage_column = ['home total conversion rate',
    #                      'away total conversion rate', 'home open play conversion rate',
    #                      'away open play conversion rate', 'home set piece conversion', 'away set piece conversion']

    data_subset = data_subset.loc[:, ~data_subset.columns.duplicated()]

    # data_subset['home possession'] = data_subset['home possession'].astype('float64') / 100.0
    # data_subset['away possession'] = data_subset['away possession'].astype('float64') / 100.0
    # for column in percentage_column:
    #     data_subset[column] = data_subset[column].str.rstrip('%').astype('float64') / 100.0
    data_subset.dropna(inplace=True)
    data_subset = data_subset.sort_values(by=['date'])
    data_subset = data_subset.reset_index(drop=True)
    return data_subset


def create_training_data(
        data):  # TODO comment functions
    n = config.N_PREVIOUS_GAMES  # n is the last previous games to get the average from
    training_data = []
    for i in range(20, len(data)):
        # Select a random team
        # table_of_teams = data['home team'].unique()
        # random_team = data.iloc[random.randrange(0, len(table_of_teams))]['home team']
        # Select a random game
        random_game = data.iloc[i]

        teama = random_game['home team']
        teamb = random_game['away team']

        home_goals = random_game['home score']
        away_goals = random_game['away score']

        home_elo = random_game['home elo']
        away_elo = random_game['away elo']

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

        # print(teama_mean)
        # print(teamb_mean)
        teama_mean_array = teama_mean.array.to_numpy(copy=True)

        teamb_mean_array = teamb_mean.array.to_numpy(copy=True)

        teama_mean_array = numpy.append(teama_mean_array, home_elo)
        teamb_mean_array = numpy.append(teamb_mean_array, away_elo)

        mean_array_sum = (teamb_mean_array + teama_mean_array)
        teama_mean_array_norm = numpy.divide(teama_mean_array, mean_array_sum,where=mean_array_sum!=0)
        teamb_mean_array_norm = numpy.divide(teamb_mean_array, mean_array_sum,where=mean_array_sum!=0)

        # print(teama_mean_array)
        # print(teamb_mean_array)
        # mean_data_array = teama_mean_array - teamb_mean_array
        mean_data_array = config.combination_of_means(teama_mean_array_norm, teamb_mean_array_norm)
        # print(mean_data_array)

        training_data.append([mean_data_array, classification_label])

    return training_data


def get_random_game(csvfile):
    data = pd.read_csv(csvfile)
    data_subset = format_data(data)
    random_game = data_subset.iloc[random.randrange(0, len(data) - 20)]
    teama = random_game['home team']
    teamb = random_game['away team']
    n = 3
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

    # normalise values
    teama_mean_array_norm = teama_mean_array / (teamb_mean_array + teama_mean_array)
    teamb_mean_array_norm = teamb_mean_array / (teamb_mean_array + teama_mean_array)

    mean_data_array = numpy.append(teama_mean_array_norm, teamb_mean_array_norm)
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
    combined_mean.fillna(0, inplace=True)
    return combined_mean


# Merge the csv files for one league to be one dataset
def merge_seasons(path, csv):
    owd = os.getcwd()
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    data = pd.read_csv(result[0])
    for i, r in enumerate(result):
        if i > 0:
            print()
            read_csv = pd.read_csv(r)
            print(str(r) + str(len(read_csv)))
            data = data.append(read_csv)

    data['date'] = pd.to_datetime(data["date"])
    data.sort_values(by=['date'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.to_csv(csv, index=False)
    os.chdir(owd)


def load_premier_league_data():
    with open("../data/whoscored/trainingdata.pickle", "rb") as f:
        training_data = pickle.load(f)
        return training_data


def generate_premier_league_data():
    training_data = create_training_data(merge_seasons())
    with open('../data/whoscored/premierleague/trainingdata.pickle', 'wb+') as file:
        pickle.dump(training_data, file)


# from the csv with all the data create the training data and save it to a binary file using pickle
def generate_training_data():
    data = pd.read_csv("../data/whoscored/all.csv")
    data = format_data(data)
    training_data = create_training_data(data)
    with open('../data/whoscored/alltrainingdata.pickle', 'wb+') as file:
        pickle.dump(training_data, file)


# load a pickle file of the training data
def load_training_data(path):
    with open(path, "rb") as f:
        training_data = pickle.load(f)
        return training_data


if __name__ == '__main__':
    generate_training_data()
