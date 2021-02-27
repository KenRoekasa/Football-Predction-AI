import glob
import os
import pickle
import random

import numpy
import pandas as pd
from sklearn.utils import resample

from tqdm import tqdm

import model.config as config


# pd.set_option("display.max_rows", None, "display.max_columns", None)
def get_random_game(csvfile):
    data = pd.read_csv(csvfile)
    data_subset = format_data(data)

    training_data = []

    n = config.N_PREVIOUS_GAMES  # n is the last previous games to get the average from
    # Select a random team
    # table_of_teams = data['home team'].unique()
    # random_team = data.iloc[random.randrange(0, len(table_of_teams))]['home team']
    # Select a random game
    random_game = data_subset.iloc[random.randrange(0, len(data_subset))]

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

    if config.normalise == 0:
        if 'pi-rating' in config.columns_selector:
            min_rating = random_game[['home home pi rating', 'home away pi rating', 'away home pi rating',
                                      'away away pi rating']].min().min()
            max_rating = random_game[['home home pi rating', 'home away pi rating', 'away home pi rating',
                                      'away away pi rating']].max().max()

            # find max to normalise the values

            home_rating = (((random_game['home home pi rating'] + random_game[
                'home away pi rating']) / 2) - min_rating) / (max_rating - min_rating)
            away_rating = (((random_game['away home pi rating'] + random_game[
                'away away pi rating']) / 2) - min_rating) / (max_rating - min_rating)
        elif 'elo' in config.columns_selector:
            min_rating = random_game[['home elo', 'away elo']].min().min()
            max_rating = random_game[['home elo', 'away elo']].max().max()

            # find max to normalise the values
            home_rating = random_game['home elo']
            away_rating = random_game['away elo']
            home_rating = (home_rating - min_rating) / (max_rating - min_rating)
            away_rating = (away_rating - min_rating) / (max_rating - min_rating)

        # find previous n games for each team

        teama_previous_games = get_previous_n_games(data_subset, teama, n, random_game)

        teama_winstreak = get_winstreak(teama_previous_games, teama)

        teamb_previous_games = get_previous_n_games(data_subset, teamb, n, random_game)
        # print(teamb_previous_games)
        teamb_winstreak = get_winstreak(teamb_previous_games, teamb)

        teama_mean = get_mean_stats(teama_previous_games, teama)
        teamb_mean = get_mean_stats(teamb_previous_games, teamb)

        teama_mean_array = teama_mean.array.to_numpy(copy=True)
        teamb_mean_array = teamb_mean.array.to_numpy(copy=True)

        teama_mean_array = numpy.append(teama_mean_array, teama_winstreak)
        teamb_mean_array = numpy.append(teamb_mean_array, teamb_winstreak)

        mean_array_sum = (teamb_mean_array + teama_mean_array)

        with numpy.errstate(divide='ignore', invalid='ignore'):
            teama_mean_array_norm = numpy.true_divide(teama_mean_array, mean_array_sum)
            teamb_mean_array_norm = numpy.true_divide(teamb_mean_array, mean_array_sum)

            teama_mean_array_norm[teama_mean_array_norm == numpy.inf] = 0.5
            teama_mean_array_norm = numpy.nan_to_num(teama_mean_array_norm)

            teamb_mean_array_norm[teamb_mean_array_norm == numpy.inf] = 0.5
            teamb_mean_array_norm = numpy.nan_to_num(teamb_mean_array_norm)

        teama_mean_array_norm = numpy.append(teama_mean_array_norm, home_rating)
        teamb_mean_array_norm = numpy.append(teamb_mean_array_norm, away_rating)

        # print(teama_mean_array)
        # print(teamb_mean_array)
        # mean_data_array = teama_mean_array - teamb_mean_array

        mean_data_array = config.combination_of_means(teama_mean_array_norm, teamb_mean_array_norm)

        training_data.append([mean_data_array, classification_label])
    elif config.normalise == 1:
        if 'pi-rating' in config.columns_selector:
            # find max to normalise the values
            home_rating = (random_game['home home pi rating'] + random_game[
                'home away pi rating']) / 2
            away_rating = (random_game['away home pi rating'] + random_game[
                'away away pi rating']) / 2
        elif 'elo' in config.columns_selector:
            home_rating = random_game['home elo']
            away_rating = random_game['away elo']

        # print(random_game)
        # find previous n games for each team

        teama_previous_games = get_previous_n_games(random_game, teama, n, random_game)

        # if previous games is empty recently promoted to the league
        # if len(teama_previous_games) == 0:
        #     teama_previous_games = pd.DataFrame(0, index=numpy.arange(1), columns=teama_previous_games.columns)

        # print(teama_previous_games)

        teamb_previous_games = get_previous_n_games(random_game, teamb, n, random_game)
        # print(teamb_previous_games)

        teama_mean = get_mean_stats(teama_previous_games, teama)
        teamb_mean = get_mean_stats(teamb_previous_games, teamb)

        # print(teama_mean)

        # print(teama_mean)
        # print(teamb_mean)
        teama_mean_array = teama_mean.array.to_numpy(copy=True)
        teamb_mean_array = teamb_mean.array.to_numpy(copy=True)

        teama_mean_array = numpy.append(teama_mean_array, home_rating)
        teamb_mean_array = numpy.append(teamb_mean_array, away_rating)

        mean_data_array = teamb_mean_array - teama_mean_array

        training_data.append([mean_data_array, classification_label])

    return [teama, teamb, home_goals, away_goals, mean_data_array, classification_label]


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
    data['date'] = pd.to_datetime(data["date"], dayfirst=True, format="%d/%m/%Y")

    # Make a subset of the table to include the fields we need
    data_subset = data[
        config.COLUMNS[config.columns_selector]].copy()

    data_subset = data_subset.loc[:, ~data_subset.columns.duplicated()]  # removes duplicates

    if config.columns_selector == 'pi-rating' or config.columns_selector == 'elo' or config.columns_selector == 'pi-rating+':
        # remove percentages symbol
        percentage_column = ['home total conversion rate',
                             'away total conversion rate', 'home open play conversion rate',
                             'away open play conversion rate', 'home set piece conversion', 'away set piece conversion']

        for column in percentage_column:
            try:
                data_subset[column] = data_subset[column].str.rstrip('%').astype(int)  # strip the percentage symbol
            except KeyError:
                print('%s column not found so skipping' % column)

    data_subset.dropna(inplace=True)
    data_subset = data_subset.sort_values(by=['date'])
    data_subset = data_subset.reset_index(drop=True)
    return data_subset


def get_winstreak(previous_games, team):
    counter = 0

    n = len(previous_games)
    if n == 0:
        return 0
    for i in range(0, n):
        game = previous_games.iloc[i]
        home_goals = game['home score']
        away_goals = game['away score']

        if (home_goals > away_goals and game['home team'] == team) or (
                away_goals > home_goals and game['away team'] == team):  # If team the team wins
            counter += 1
        else:  # team loses or draw
            counter = 0

    return counter


def create_training_data(data):  # TODO comment functions

    # Split into leagues and seasons
    leagues = data['league'].unique()

    training_data = []

    for league in tqdm(leagues):

        data_league = data[data['league'] == league]

        seasons = data_league['season'].unique()

        for season in seasons:  # skip the first season
            data_season = data_league[data_league['season'] == season]

            n = config.N_PREVIOUS_GAMES  # n is the last previous games to get the average from

            for i in range(20, len(data_season)):  # remove the first game with no previous data
                # Select a random team
                # table_of_teams = data['home team'].unique()
                # random_team = data.iloc[random.randrange(0, len(table_of_teams))]['home team']
                # Select a random game
                random_game = data_season.iloc[i]

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

                if config.normalise == 0:
                    if 'pi-rating' in config.columns_selector:
                        min_rating = data_season[['home home pi rating', 'home away pi rating', 'away home pi rating',
                                                  'away away pi rating']].min().min()
                        max_rating = data_season[['home home pi rating', 'home away pi rating', 'away home pi rating',
                                                  'away away pi rating']].max().max()

                        # find max to normalise the values

                        home_rating = (((random_game['home home pi rating'] + random_game[
                            'home away pi rating']) / 2) - min_rating) / (max_rating - min_rating)
                        away_rating = (((random_game['away home pi rating'] + random_game[
                            'away away pi rating']) / 2) - min_rating) / (max_rating - min_rating)
                    elif 'elo' in config.columns_selector:
                        min_rating = data_season[['home elo', 'away elo']].min().min()
                        max_rating = data_season[['home elo', 'away elo']].max().max()

                        # find max to normalise the values
                        home_rating = random_game['home elo']
                        away_rating = random_game['away elo']
                        home_rating = (home_rating - min_rating) / (max_rating - min_rating)
                        away_rating = (away_rating - min_rating) / (max_rating - min_rating)

                    # find previous n games for each team

                    teama_previous_games = get_previous_n_games(data_season, teama, n, random_game)

                    teama_winstreak = get_winstreak(teama_previous_games, teama)

                    teamb_previous_games = get_previous_n_games(data_season, teamb, n, random_game)
                    # print(teamb_previous_games)
                    teamb_winstreak = get_winstreak(teamb_previous_games, teamb)

                    teama_mean = get_mean_stats(teama_previous_games, teama)
                    teamb_mean = get_mean_stats(teamb_previous_games, teamb)

                    teama_mean_array = teama_mean.array.to_numpy(copy=True)
                    teamb_mean_array = teamb_mean.array.to_numpy(copy=True)

                    teama_mean_array = numpy.append(teama_mean_array, teama_winstreak)
                    teamb_mean_array = numpy.append(teamb_mean_array, teamb_winstreak)

                    mean_array_sum = (teamb_mean_array + teama_mean_array)

                    with numpy.errstate(divide='ignore', invalid='ignore'):
                        teama_mean_array_norm = numpy.true_divide(teama_mean_array, mean_array_sum)
                        teamb_mean_array_norm = numpy.true_divide(teamb_mean_array, mean_array_sum)

                        teama_mean_array_norm = numpy.nan_to_num(teama_mean_array_norm, nan=0.5)
                        # teama_mean_array_norm[teama_mean_array_norm == numpy.nan] = 0.5

                        teamb_mean_array_norm = numpy.nan_to_num(teamb_mean_array_norm, nan=0.5)
                        # teamb_mean_array_norm[teamb_mean_array_norm == numpy.nan] = 0.5

                    teama_mean_array_norm = numpy.append(teama_mean_array_norm, home_rating)
                    teamb_mean_array_norm = numpy.append(teamb_mean_array_norm, away_rating)

                    # print(teama_mean_array)
                    # print(teamb_mean_array)
                    # mean_data_array = teama_mean_array - teamb_mean_array

                    mean_data_array = config.combination_of_means(teama_mean_array_norm, teamb_mean_array_norm)

                    training_data.append([mean_data_array, classification_label])
                elif config.normalise == 1:
                    if 'pi-rating' in config.columns_selector:
                        # find max to normalise the values
                        home_rating = (random_game['home home pi rating'] + random_game[
                            'home away pi rating']) / 2
                        away_rating = (random_game['away home pi rating'] + random_game[
                            'away away pi rating']) / 2
                    elif 'elo' in config.columns_selector:
                        home_rating = random_game['home elo']
                        away_rating = random_game['away elo']

                    # print(random_game)
                    # find previous n games for each team

                    teama_previous_games = get_previous_n_games(data_season, teama, n, random_game)

                    # if previous games is empty recently promoted to the league
                    # if len(teama_previous_games) == 0:
                    #     teama_previous_games = pd.DataFrame(0, index=numpy.arange(1), columns=teama_previous_games.columns)

                    # print(teama_previous_games)

                    teamb_previous_games = get_previous_n_games(data_season, teamb, n, random_game)
                    # print(teamb_previous_games)

                    teama_mean = get_mean_stats(teama_previous_games, teama)
                    teamb_mean = get_mean_stats(teamb_previous_games, teamb)

                    # print(teama_mean)

                    # print(teama_mean)
                    # print(teamb_mean)
                    teama_mean_array = teama_mean.array.to_numpy(copy=True)
                    teamb_mean_array = teamb_mean.array.to_numpy(copy=True)

                    teama_mean_array = numpy.append(teama_mean_array, home_rating)
                    teamb_mean_array = numpy.append(teamb_mean_array, away_rating)

                    mean_data_array = teamb_mean_array - teama_mean_array

                    training_data.append([mean_data_array, classification_label])

    return training_data


def get_mean_stats(previous_games, team):
    # Get home statistics
    # Get all home games
    home_games = get_team_home_games(previous_games, team)
    home_games = home_games.filter(regex='home')
    if 'pi-rating' in config.columns_selector:  # remove ratings from the mean calculation
        home_games.drop('home home pi rating', axis=1, inplace=True)
        home_games.drop('home away pi rating', axis=1, inplace=True)
        home_games.drop('away home pi rating', axis=1, inplace=True)
    elif config.columns_selector == 'elo' or config.columns_selector == 'elo only':
        home_games.drop('home elo', axis=1, inplace=True)

    # print(home_games)
    #
    # print(home_games.mean(numeric_only=True))
    home_mean = home_games.mean(numeric_only=True)
    # print(home_mean)
    # Get away statistics
    away_games = get_team_away_games(previous_games, team)
    away_games = away_games.filter(regex='away')

    if 'pi-rating' in config.columns_selector:  # remove ratings from the mean calculation
        away_games.drop('home away pi rating', axis=1, inplace=True)
        away_games.drop('away home pi rating', axis=1, inplace=True)
        away_games.drop('away away pi rating', axis=1, inplace=True)
    elif 'elo' in config.columns_selector:
        away_games.drop('away elo', axis=1, inplace=True)

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

    data['date'] = pd.to_datetime(data["date"], dayfirst=True, format="%a, %d-%b-%y")
    data.sort_values(by=['date'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.to_csv(csv, index=False)
    os.chdir(owd)


def merge_leagues():
    data = pd.read_csv('../data/whoscored/premierleague/all-premierleague.csv')
    laliga = pd.read_csv('../data/whoscored/laliga/all-laliga.csv')
    bundesliga = pd.read_csv('../data/whoscored/bundesliga/all-bundesliga.csv')
    seriea = pd.read_csv('../data/whoscored/seriea/all-seriea.csv')

    data = data.append(laliga)
    data = data.append(bundesliga)
    data = data.append(seriea)

    data['date'] = pd.to_datetime(data["date"], dayfirst=True)
    data.sort_values(by=['date'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.to_csv('../data/whoscored/all-leagues.csv', index=False)


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

        df = pd.DataFrame(training_data)

        draws = df[df[1] == 1].copy()
        wins = df[df[1] == 0].copy()
        loss = df[df[1] == 2].copy()

        # print(len(draws))
        # print(len(wins))
        # print(len(loss))

        # wins = wins[:-2326].copy()

        # wins_downsampled = resample(wins,
        #                             replace=False,  # sample without replacement
        #                             n_samples=len(draws),  # to match minority class
        #                             random_state=123)
        # loss_downsampled = resample(loss,
        #                             replace=False,  # sample without replacement
        #                             n_samples=len(draws),  # to match minority class
        #                             random_state=123)

        draws_upsampled = resample(draws, replace=True, n_samples=len(wins), random_state=123)

        loss_upsampled = resample(loss, replace=True, n_samples=len(wins), random_state=123)

        frames = [wins, draws_upsampled, loss_upsampled]
        result = pd.concat(frames)

        training_data = result.values.tolist()

        random.shuffle(training_data)

        # split into train and test

        x = []  # features set
        y = []  # label set

        counter_0 = 0
        counter_1 = 0
        counter_2 = 0

        for features, label in training_data:
            # balance data
            x.append(features)
            y.append(label)

        X = numpy.array(x)
        if config.normalise == 1:  # normalise further
            X = (X - numpy.min(X)) / (numpy.max(X) - numpy.min(X))

        y = numpy.array(y)
        return X, y


if __name__ == '__main__':
    generate_training_data('../data/whoscored/all-leagues.csv',
                           '../data/whoscored/trainingdata/alltrainingdata-%d-%s.pickle' % (
                               config.N_PREVIOUS_GAMES, config.columns_selector))

    # merge_seasons('../data/whoscored/seriea/datascraped','../all-seriea.csv')
    # merge_leagues()
