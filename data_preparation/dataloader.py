import glob
import os
import pickle
import random

import numpy
import pandas as pd

from tqdm import tqdm

import model.config as config


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
        config.COLUMNS[config.columns_selector]].copy()

    data_subset.dropna(inplace=True)
    data_subset = data_subset.loc[:, ~data_subset.columns.duplicated()]  # removes duplicates

    if config.columns_selector == 'pi-rating' or config.columns_selector == 'elo':
        # remove percentages symbol
        percentage_column = ['home total conversion rate',
                             'away total conversion rate', 'home open play conversion rate',
                             'away open play conversion rate', 'home set piece conversion', 'away set piece conversion']



        data_subset['home possession'] = data_subset['home possession'].astype(int)
        data_subset['away possession'] = data_subset['away possession'].astype(int)
        for column in percentage_column:
            data_subset[column] = data_subset[column].str.rstrip('%').astype(int)  # strip the percentage symbol

    # Set types of each column
    # data_subset = data_subset.astype(
    #     {"home score": int, "away score": int, 'home total shots': int, 'away total shots': int,
    #      'home shots on target': int, 'away shots on target': int, 'home possession': int, 'away possession': int,
    #      'home total conversion rate': int,
    #      'away total conversion rate': int, 'home fouls': int, 'away fouls': int, 'home yellow cards': int,
    #      'away yellow cards': int,
    #      'home red cards': int, 'away red cards': int, 'home total passes': int, 'away total passes': int,
    #      'home accurate passes': int,
    #      'away accurate passes': int, 'home open play conversion rate': int, 'away open play conversion rate': int,
    #      'home set piece conversion': int, 'away set piece conversion': int, 'home counter attack shots': int,
    #      'away counter attack shots': int,
    #      'home counter attack goals': int, 'away counter attack goals': int, 'home key passes': int,
    #      'away key passes': int,
    #      'home dribbles attempted': int, 'away dribbles attempted': int, 'home dribble success': int,
    #      'away dribble success': int,
    #      'home aerials won%': int, 'away aerials won%': int, 'home tackles attempted': int,
    #      'away tackles attempted': int,
    #      'home tackles success %': int, 'away tackles success %': int, 'home was dribbled': int,
    #      'away was dribbled': int,
    #      'home interceptions': int,
    #      'away interceptions': int, 'home dispossessed': int, 'away dispossessed': int, 'home errors': int,
    #      'away errors': int,
    #      'home elo': int,
    #      'away elo': int})

    data_subset = data_subset.sort_values(by=['date'])
    data_subset = data_subset.reset_index(drop=True)
    return data_subset


def create_training_data(data):  # TODO comment functions

    n = config.N_PREVIOUS_GAMES  # n is the last previous games to get the average from
    training_data = []

    for i in tqdm(range(0, len(data))):
        # Select a random team
        # table_of_teams = data['home team'].unique()
        # random_team = data.iloc[random.randrange(0, len(table_of_teams))]['home team']
        # Select a random game
        random_game = data.iloc[i]

        teama = random_game['home team']
        teamb = random_game['away team']

        home_goals = random_game['home score']
        away_goals = random_game['away score']



        if config.columns_selector == 'pi-rating' or 'pi-rating only':

            min_rating = data[['home home pi rating','home away pi rating','away home pi rating','away away pi rating']].min().min()
            max_rating =data[['home home pi rating','home away pi rating','away home pi rating','away away pi rating']].max().max()

            # find max to normalise the values
            home_rating = (((random_game['home home pi rating'] + random_game['home away pi rating']) / 2) - min_rating) /(max_rating-min_rating)
            away_rating = (((random_game['away home pi rating'] + random_game['away away pi rating']) / 2) - min_rating) /(max_rating-min_rating)
        else:
            home_rating = random_game['home elo']
            away_rating = random_game['away elo']

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



        mean_array_sum = (teamb_mean_array + teama_mean_array)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            teama_mean_array_norm = numpy.true_divide(teama_mean_array, mean_array_sum)
            teamb_mean_array_norm = numpy.true_divide(teamb_mean_array, mean_array_sum)

            teama_mean_array_norm[teama_mean_array_norm == numpy.inf] = 0
            teama_mean_array_norm = numpy.nan_to_num(teama_mean_array_norm)

            teamb_mean_array_norm[teamb_mean_array_norm == numpy.inf] = 0
            teamb_mean_array_norm = numpy.nan_to_num(teamb_mean_array_norm)

        teama_mean_array_norm = numpy.append(teama_mean_array_norm, home_rating)
        teamb_mean_array_norm = numpy.append(teamb_mean_array_norm, away_rating)

        # print(teama_mean_array)
        # print(teamb_mean_array)
        # mean_data_array = teama_mean_array - teamb_mean_array

        mean_data_array = config.combination_of_means(teama_mean_array_norm, teamb_mean_array_norm)

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
    if config.columns_selector == 'pi-rating' or config.columns_selector == 'pi-rating only':  # remove ratings from the mean calculation
        home_games.drop('home home pi rating', axis=1, inplace=True)
        home_games.drop('home away pi rating', axis=1, inplace=True)
        home_games.drop('away home pi rating', axis=1, inplace=True)



    # print(home_games)
    #
    # print(home_games.mean(numeric_only=True))
    home_mean = home_games.mean(numeric_only=True)
    # print(home_mean)
    # Get away statistics
    away_games = get_team_away_games(previous_games, team)
    away_games = away_games.filter(regex='away')

    if config.columns_selector == 'pi-rating'or config.columns_selector == 'pi-rating only':  # remove ratings from the mean calculation
        away_games.drop('home away pi rating', axis=1, inplace=True)
        away_games.drop('away home pi rating', axis=1, inplace=True)
        away_games.drop('away away pi rating', axis=1, inplace=True)

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
            print(str(r) + " " + str(len(read_csv)))
            data = data.append(read_csv)

    data['date'] = pd.to_datetime(data["date"])
    data.sort_values(by=['date'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.to_csv(csv, index=False)
    os.chdir(owd)


def merge_leagues():
    data = pd.read_csv('../data/whoscored/premierleague/allpremierleague.csv')
    laliga = pd.read_csv('../data/whoscored/laliga/all-laliga.csv')
    data = data.append(laliga)

    data['date'] = pd.to_datetime(data["date"])
    data.sort_values(by=['date'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.to_csv('../data/whoscored/all-leagues.csv', index=False)


def load_premier_league_data():
    with open("../data/whoscored/trainingdata.pickle", "rb") as f:
        training_data = pickle.load(f)
        return training_data


def generate_premier_league_data():
    training_data = create_training_data(merge_seasons())
    with open('../data/whoscored/premierleague/trainingdata.pickle', 'wb+') as file:
        pickle.dump(training_data, file)


# from the csv with all the data create the training data and save it to a binary file using pickle
def generate_training_data(csv, pickle_path):
    data = pd.read_csv(csv)
    data = format_data(data)
    training_data = create_training_data(data)

    with open(pickle_path, 'wb+') as file:
        pickle.dump(training_data, file)


# load a pickle file of the training data
def load_training_data(path):
    with open(path, "rb") as f:
        training_data = pickle.load(f)

        x = []  # features set
        y = []  # label set
        for features, label in training_data:
            x.append(features)
            y.append(label)
        X = numpy.array(x)
        y = numpy.array(y)
        return X, y


if __name__ == '__main__':
    generate_training_data("../data/whoscored/all-leagues.csv", '../data/whoscored/alltrainingdata-pi-rating-only.pickle')
