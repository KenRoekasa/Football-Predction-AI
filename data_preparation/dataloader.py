import glob
import os
import pickle
import random

import numpy
import pandas as pd
from sklearn.preprocessing import normalize
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

    if config.normalise_rating == 0:
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

        teama_mean = get_mean_stats(teama_previous_games, teama, settings)
        teamb_mean = get_mean_stats(teamb_previous_games, teamb, settings)

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
    elif config.normalise_rating == 1:
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
    return table[table["home team"].str.contains(team)]


def get_team_away_games(table, team):
    return table[table["away team"].str.contains(team)]


def get_all_team_games(table, team):
    return pd.concat([get_team_home_games(table, team), get_team_away_games(table, team)]).sort_values(by=['date'])


def select_random_consecutive_games(table, team, n):  # select random n consecutive games
    all_games = get_all_team_games(table, team)

    first_index = random.randrange(0, len(all_games) - n)
    return all_games.iloc[first_index:first_index + n]


def get_previous_n_games(table, team, n, game):
    all_games = get_all_team_games(table, team)
    all_games.reset_index(drop=True, inplace=True)

    gamelink = str(game['link'])
    the_game = all_games[all_games['link'] == gamelink]
    index = the_game.index[0]

    if index > n:
        previous = index - n

    else:
        previous = 0

    previous_games = all_games.iloc[previous:index]
    return previous_games


def format_data(data, settings):
    # Change date to date format
    data['date'] = pd.to_datetime(data["date"], dayfirst=True, format="%d/%m/%Y")

    # Make a subset of the table to include the fields we need
    data_subset = data[
        config.COLUMNS[settings['columns']]].copy()

    data_subset = data_subset.loc[:, ~data_subset.columns.duplicated()]  # removes duplicates

    if settings['columns'] == 'pi-rating' or settings['columns'] == 'elo' or settings[
        'columns'] == 'pi-rating+' or 'both' in settings['columns']:
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


def create_training_data(data, settings):  # TODO comment functions
    n = settings['n']
    # Split into leagues and seasons
    leagues = data['league'].unique()

    training_data = []

    for league in tqdm(leagues):

        data_league = data[data['league'] == league]

        seasons = data_league['season'].unique()

        for season in seasons:  # get training data for each season
            data_season = data_league[data_league['season'] == season]
            data_season.reset_index(drop=True, inplace=True)
            for i in range(20, len(data_season)):  # remove the first game with no previous data

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

                teama_previous_games = get_previous_n_games(data_season, teama, n, random_game)
                teamb_previous_games = get_previous_n_games(data_season, teamb, n, random_game)

                if teama_previous_games.size == 0 | teamb_previous_games.size ==0: # if no previous games are found skip
                    break;

                away_rating, home_rating = normalise_ratings(data_league, teama, teamb, settings, teama_previous_games,
                                                             teamb_previous_games)

                teama_mean_array = get_mean_array(teama,teama_previous_games)
                teamb_mean_array = get_mean_array(teamb,teamb_previous_games)

                # append the ratings
                teama_mean_array = numpy.append(teama_mean_array, home_rating)
                teamb_mean_array = numpy.append(teamb_mean_array, away_rating)

                if settings['combination'] == 'append':
                    mean_data_array = numpy.append(teama_mean_array, teamb_mean_array)
                if settings['combination'] == 'diff':
                    mean_data_array = teama_mean_array - teamb_mean_array

                training_data.append([mean_data_array, classification_label])

    return training_data


def normalise_ratings(data_league, teama, teamb, settings, teama_previous, teamb_previous):
    if settings['rating normalisation'] == 'min-max':

        if 'pi-rating' in settings['columns']:  # min max rating for ratings
            home_rating, away_rating = get_ratings(teama, teamb, 'pi rating', teama_previous, teamb_previous)

            # Get the minimum value for that league
            min_rating = data_league[['home pi rating', 'away pi rating']].min().min()
            # Get the maximum value for that league
            max_rating = data_league[['home pi rating', 'away pi rating']].max().max()

            away_rating, home_rating = min_max_normalisation(away_rating, home_rating, max_rating, min_rating)


        elif 'elo' in settings['columns']:  # same for elo

            home_rating, away_rating = get_ratings(teama, teamb, 'elo', teama_previous, teamb_previous)

            min_rating = data_league[['home elo', 'away elo']].min().min()
            max_rating = data_league[['home elo', 'away elo']].max().max()

            # find max to normalise the values

            away_rating, home_rating = min_max_normalisation(away_rating, home_rating,
                                                             max_rating, min_rating)

        elif 'both' in settings['columns']:  # setup both

            home_rating, away_rating = get_ratings(teama, teamb, 'pi rating', teama_previous, teamb_previous)

            # Get the minimum value for that league
            pi_rating_min_rating = data_league[['home pi rating', 'away pi rating']].min().min()
            # Get the maximum value for that league
            pi_rating_max_rating = data_league[['home pi rating', 'away pi rating']].max().max()

            pi_rating_away_rating, pi_rating_home_rating = min_max_normalisation(away_rating,
                                                                                 home_rating,
                                                                                 pi_rating_max_rating,
                                                                                 pi_rating_min_rating)

            elo_min_rating = data_league[['home elo', 'away elo']].min().min()
            elo_max_rating = data_league[['home elo', 'away elo']].max().max()

            # find max to normalise the values
            home_rating, away_rating = get_ratings(teama, teamb, 'elo', teama_previous, teamb_previous)
            elo_away_rating, elo_home_rating = min_max_normalisation(away_rating, home_rating,
                                                                     elo_max_rating, elo_min_rating)

            away_rating, home_rating = [elo_away_rating, pi_rating_away_rating], [elo_home_rating,
                                                                                  pi_rating_home_rating]

    if settings['rating normalisation'] == 'ratio':

        # ratio normalisation
        if 'pi-rating' in settings['columns']:
            home_rating, away_rating = get_ratings(teama, teamb, 'pi rating', teama_previous, teamb_previous)
            away_rating, home_rating = ratio_normalisation(away_rating, home_rating)

        elif 'elo' in settings['columns']:  # same for elo
            # find max to normalise the values
            home_rating, away_rating = get_ratings(teama, teamb, 'elo', teama_previous, teamb_previous)
            away_rating, home_rating = ratio_normalisation(away_rating, home_rating)

        elif 'both' in settings['columns']:
            home_rating, away_rating = get_ratings(teama, teamb, 'pi rating', teama_previous, teamb_previous)
            pi_rating_away_rating, pi_rating_home_rating = ratio_normalisation(away_rating, home_rating)

            home_rating, away_rating = get_ratings(teama, teamb, 'elo', teama_previous, teamb_previous)
            elo_away_rating, elo_home_rating = ratio_normalisation(away_rating, home_rating)

            away_rating, home_rating = [elo_away_rating, pi_rating_away_rating], [elo_home_rating,
                                                                                  pi_rating_home_rating]

    if settings['rating normalisation'] == 'none':

        # ratio normalisation
        if 'pi-rating' in settings['columns']:
            home_rating, away_rating = get_ratings(teama, teamb, 'pi rating', teama_previous, teamb_previous)

        elif 'elo' in settings['columns']:  # same for elo

            # find max to normalise the values
            home_rating, away_rating = get_ratings(teama, teamb, 'elo', teama_previous, teamb_previous)

    return away_rating, home_rating


def get_ratings(home_team, away_team, rating, teama_previous, teamb_previous):
    if rating == 'pi rating':
        # get the previous game
        home_previous = teama_previous.iloc[-1]

        if home_previous.empty:  # if no previous rating available rating is 0
            home_rating = 0
        else:
            # if rating team is home team get home rating
            if home_previous['home team'] == home_team:
                home_rating = home_previous['home pi rating']
            # if rating team is away team get away rating
            elif home_previous['away team'] == home_team:
                home_rating = home_previous['away pi rating']

        # get the previous game
        away_previous = teamb_previous.iloc[-1]

        if home_previous.empty:  # if no previous rating available rating is 0
            away_rating = 0
        else:
            # if rating team is home team get home rating
            if away_previous['home team'] == away_team:
                away_rating = away_previous['home pi rating']
            # if rating team is away team get away rating
            elif away_previous['away team'] == away_team:
                away_rating = away_previous['away pi rating']


    elif rating == 'elo':
        # get the previous game
        home_previous = teama_previous.iloc[-1]

        if home_previous.empty:  # if no previous rating available rating is 0
            home_rating = 1000
        else:
            # if rating team is home team get home rating
            if home_previous['home team'] == home_team:
                home_rating = home_previous['home elo']
            # if rating team is away team get away rating
            elif home_previous['away team'] == home_team:
                home_rating = home_previous['away elo']

        # get the previous game
        away_previous = teamb_previous.iloc[-1]

        if away_previous.empty:  # if no previous rating available rating is 0
            away_rating = 1000
        else:
            # if rating team is home team get home rating
            if away_previous['home team'] == away_team:
                away_rating = away_previous['home elo']
            # if rating team is away team get away rating
            elif away_previous['away team'] == away_team:
                away_rating = away_previous['away elo']

    return float(home_rating), float(away_rating)


def min_max_normalisation(away_rating, home_rating, max_rating, min_rating):
    home_rating = (home_rating - min_rating) / (max_rating - min_rating)
    away_rating = (away_rating - min_rating) / (max_rating - min_rating)
    return away_rating, home_rating


def ratio_normalisation(away_rating, home_rating):
    # find max to normalise the values

    sum_rating = home_rating + away_rating
    home_rating = home_rating / sum_rating
    away_rating = away_rating / sum_rating

    return away_rating, home_rating


def normalise_mean_array(teama_mean_array, teamb_mean_array, norm):
    if norm == 'ratio':
        mean_array_sum = teamb_mean_array + teama_mean_array
        with numpy.errstate(divide='ignore', invalid='ignore'):
            teama_mean_array_norm = numpy.true_divide(teama_mean_array, mean_array_sum)
            teamb_mean_array_norm = numpy.true_divide(teamb_mean_array, mean_array_sum)

            teama_mean_array_norm = numpy.nan_to_num(teama_mean_array_norm, nan=0.5)

            teamb_mean_array_norm = numpy.nan_to_num(teamb_mean_array_norm, nan=0.5)

    if norm == 'min-max':
        teama_mean_array_norm = normalize(teama_mean_array, axis=0, norm='max')
        teamb_mean_array_norm = normalize(teamb_mean_array, axis=0, norm='max')

    return teama_mean_array_norm, teamb_mean_array_norm


def get_mean_array(team, previous_games):
    teama_winstreak = get_winstreak(previous_games, team)
    teama_mean = get_mean_stats(previous_games, team)
    teama_mean_array = teama_mean.array.to_numpy(copy=True)
    teama_mean_array = numpy.append(teama_mean_array, teama_winstreak)
    return teama_mean_array


def get_mean_stats(previous_games, team):
    # Get home statistics
    # Get all home games
    home_games = get_team_home_games(previous_games, team)
    home_games = home_games.filter(regex='home')

    try:
        home_games.drop('home pi rating', axis=1, inplace=True)
    except KeyError:
        pass
    try:
        home_games.drop('home elo', axis=1, inplace=True)
    except KeyError:
        pass

    # print(home_games)
    #
    # print(home_games.mean(numeric_only=True))
    home_mean = home_games.mean(numeric_only=True)
    # print(home_mean)
    # Get away statistics
    away_games = get_team_away_games(previous_games, team)
    away_games = away_games.filter(regex='away')

    try:
        away_games.drop('away pi rating', axis=1, inplace=True)
    except KeyError:
        pass
    try:
        away_games.drop('away elo', axis=1, inplace=True)
    except KeyError:
        pass

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
def generate_training_data(csv, pickle_path, settings):
    data = pd.read_csv(csv)
    data = format_data(data, settings)
    training_data = create_training_data(data, settings)

    with open(pickle_path, 'wb+') as file:
        pickle.dump(training_data, file)


# load a pickle file of the training data
def load_training_data(path):
    with open(path, "rb") as f:
        training_data = pickle.load(f)

        # df = pd.DataFrame(training_data)
        #
        # draws = df[df[1] == 1].copy()
        # wins = df[df[1] == 0].copy()
        # loss = df[df[1] == 2].copy()
        #
        # # print(len(draws))
        # # print(len(wins))
        # # print(len(loss))
        #
        # # wins = wins[:-2326].copy()
        # sample =0
        # if sample == 0:
        #
        #     wins_downsampled = resample(wins,
        #                                 replace=False,  # sample without replacement
        #                                 n_samples=len(draws),  # to match minority class
        #                                 random_state=123)
        #     loss_downsampled = resample(loss,
        #                                 replace=False,  # sample without replacement
        #                                 n_samples=len(draws),  # to match minority class
        #                                 random_state=123)
        #
        #     frames = [wins_downsampled, draws, loss_downsampled]
        #
        # else:
        #
        #     draws_upsampled = resample(draws, replace=True, n_samples=len(wins), random_state=123)
        #
        #     loss_upsampled = resample(loss, replace=True, n_samples=len(wins), random_state=123)
        #     frames = [wins, draws_upsampled, loss_upsampled]
        #
        #
        # result = pd.concat(frames)
        #
        # training_data = result.values.tolist()

        random.shuffle(training_data)

        # split into train and test

        x = []  # features set
        y = []  # label set

        for features, label in training_data:
            # balance data
            x.append(features)
            y.append(label)

        X = numpy.array(x)

        y = numpy.array(y)
        return X, y


if __name__ == '__main__':
    for combination in ['append', 'diff']:
        settings = {'n': 8, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': combination}

        generate_training_data('../data/whoscored/all-leagues.csv',
                               '../data/whoscored/trainingdata/unnormalised/alltrainingdata-%d-%s-%s-%s.pickle' % (
                                   settings['n'], settings['columns'], settings['rating normalisation'],
                                   settings['combination']), settings)

    # merge_seasons('../data/whoscored/seriea/datascraped','../all-seriea.csv')
    # merge_leagues()
