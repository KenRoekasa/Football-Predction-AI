import glob
import os
import pickle
import random

import numpy
import pandas as pd
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.utils import resample
import tensorflow as tf
from tqdm import tqdm

import model.config as config


# pd.set_option("display.max_rows", None, "display.max_columns", None)

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
    data['date'] = pd.to_datetime(data["date"], dayfirst=True, format="%Y-%m-%d")

    # Make a subset of the table to include the fields we need
    data_subset = data[
        config.COLUMNS[settings['columns']]].copy()

    data_subset = data_subset.loc[:, ~data_subset.columns.duplicated()]  # removes duplicates

    # remove % sign from certain columns
    data_object = data_subset.select_dtypes(['object'])
    data_object = data_object.iloc[:, 5:]
    data_subset[data_object.columns] = data_object.apply(lambda x: x.str.rstrip('%').astype(float))

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


def get_losestreak(previous_games, team):
    counter = 0

    n = len(previous_games)
    if n == 0:
        return 0
    for i in range(0, n):
        game = previous_games.iloc[i]
        home_goals = game['home score']
        away_goals = game['away score']

        if (home_goals < away_goals and game['home team'] == team) or (
                away_goals < home_goals and game['away team'] == team):  # If team the team wins
            counter += 1
        else:  # team loses or draw
            counter = 0

    return counter


def create_training_data(data, settings):  # TODO comment functions
    n = settings['n']
    # Split into leagues and seasons
    leagues = data['league'].unique()  # change this to the league i want
    # leagues = ['premier-league']
    # leagues = ['seriea']
    # leagues = ['bundesliga']
    # leagues = ['ligue1']
    # leagues = ['laliga']

    training_data = []
    pbar = tqdm(total=len(data.index))
    for league in leagues:

        data_league = data[data['league'] == league]

        seasons = data_league['season'].unique()

        for season in seasons:  # get training data for each season
            data_season = data_league[data_league['season'] == season]
            data_season.reset_index(drop=True, inplace=True)
            for i in range(0, len(data_season)):  # remove the first game with no previous data

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

                if teama_previous_games.size == 0 or teamb_previous_games.size == 0:  # if no previous games are found skip
                    continue

                away_rating, home_rating = normalise_ratings(data_league, teama, teamb, settings, teama_previous_games,
                                                             teamb_previous_games)

                teama_mean_array = get_mean_array(teama, teama_previous_games)
                teamb_mean_array = get_mean_array(teamb, teamb_previous_games)

                # append the ratings
                teama_mean_array = numpy.append(teama_mean_array, home_rating)
                teamb_mean_array = numpy.append(teamb_mean_array, away_rating)

                if settings['combination'] == 'append':
                    mean_data_array = numpy.append(teama_mean_array, teamb_mean_array)
                if settings['combination'] == 'diff':
                    mean_data_array = teama_mean_array - teamb_mean_array

                final = numpy.insert(mean_data_array, 0, classification_label)
                final = numpy.append(final, league)
                training_data.append(final)
                pbar.update(1)
    pbar.close()
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

        # if rating team is home team get home rating
        if home_previous['home team'] == home_team:
            home_rating = home_previous['home pi rating']
        # if rating team is away team get away rating
        elif home_previous['away team'] == home_team:
            home_rating = home_previous['away pi rating']

        # get the previous game
        away_previous = teamb_previous.iloc[-1]

        # if rating team is home team get home rating
        if away_previous['home team'] == away_team:
            away_rating = away_previous['home pi rating']
        # if rating team is away team get away rating
        elif away_previous['away team'] == away_team:
            away_rating = away_previous['away pi rating']


    elif rating == 'elo':
        # get the previous game
        home_previous = teama_previous.iloc[-1]

        # if rating team is home team get home rating
        if home_previous['home team'] == home_team:
            home_rating = home_previous['home elo']
        # if rating team is away team get away rating
        elif home_previous['away team'] == home_team:
            home_rating = home_previous['away elo']

        # get the previous game
        away_previous = teamb_previous.iloc[-1]

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


def normalise_input_array(array, norm):
    x = numpy.copy(array)
    if norm == 'ratio':
        mean_array_sum = array[:, 1:-4:2] + array[:, 0:-4:2]
        with numpy.errstate(divide='ignore', invalid='ignore'):
            x[:, 0:-4:2] = numpy.true_divide(x[:, 0:-4:2], mean_array_sum)
            x[:, 1:-4:2] = numpy.true_divide(x[:, 1:-4:2], mean_array_sum)

            x[:, 0:-4:2] = numpy.nan_to_num(x[:, 0:-4:2], nan=0.5)
            x[:, 1:-4:2] = numpy.nan_to_num(x[:, 1:-4:2], nan=0.5)

    if norm == 'l1' or norm == 'l2':
        x = normalize(x, axis=1, norm=norm)
    elif norm == 'max':
        scaler = MinMaxScaler()
        scaler.fit(x[:, 0:-4])
        x[:, 0:-4] = scaler.transform(x[:, 0:-4])

    return x


def get_mean_array(team, previous_games):
    winstreak = get_winstreak(previous_games, team)
    losestreak = get_losestreak(previous_games, team)
    mean = get_mean_stats(previous_games, team)
    mean_array = numpy.append(mean, winstreak)
    mean_array = numpy.append(mean_array, losestreak)
    return mean_array


def get_mean_stats(previous_games, team):
    # Get home statistics
    # Get all home games
    home_games = get_team_home_games(previous_games, team)
    home_goals_conceded = home_games['away score'].to_numpy(dtype=float)
    home_games = home_games.filter(regex='home', axis=1)

    try:
        home_games.drop('home pi rating', axis=1, inplace=True)
    except KeyError:
        pass
    try:
        home_games.drop('home elo', axis=1, inplace=True)
    except KeyError:
        pass

    home_games.drop('home team', axis=1, inplace=True)
    # print(home_games.columns)
    home_mean = home_games.to_numpy(dtype=float)
    home_mean = numpy.insert(home_mean, numpy.shape(home_mean)[1], home_goals_conceded, axis=1)

    # print(home_mean)
    # Get away statistics
    away_games = get_team_away_games(previous_games, team)
    away_goals_conceded = away_games['home score'].to_numpy(dtype=float)
    away_games = away_games.filter(regex='away', axis=1)

    try:
        away_games.drop('away pi rating', axis=1, inplace=True)
    except KeyError:
        pass
    try:
        away_games.drop('away elo', axis=1, inplace=True)
    except KeyError:
        pass

    away_games.drop('away team', axis=1, inplace=True)
    away_mean = away_games.to_numpy(dtype=float)

    away_mean = numpy.insert(away_mean, numpy.shape(away_mean)[1], away_goals_conceded, axis=1)

    # Combine away and home statistics to get the final
    combined_mean = numpy.append(home_mean, away_mean, axis=0)
    combined_mean = numpy.mean(combined_mean, axis=0)

    # [columns] + goals conceded

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
    ligue1 = pd.read_csv('../data/whoscored/ligue1/all-ligue1.csv')

    data = data.append(laliga)
    data = data.append(bundesliga)
    data = data.append(seriea)
    data = data.append(ligue1)

    data['date'] = pd.to_datetime(data["date"], dayfirst=True, format="%Y-%m-%d")
    data.sort_values(by=['date'], inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.to_csv('../data/whoscored/all-leagues.csv', index=False)


# from the csv with all the data create the training data and save it to a binary file using pickle
def generate_training_data(csv, path, settings):
    data = pd.read_csv(csv)
    data = format_data(data, settings)
    training_data = create_training_data(data, settings)

    df = pd.DataFrame(data=training_data, dtype=float,
                      columns=['Outcome', 'home score', 'home total shots', 'home total conversion rate',
                               'home open play shots', 'home open play goals',
                               'home open play conversion rate', 'home set piece shots',
                               'home set piece goals', 'home set piece conversion',
                               'home counter attack shots', 'home counter attack goals',
                               'home counter attack conversion', 'home penalty shots',
                               'home penalty goals', 'home penalty conversion', 'home own goals shots',
                               'home own goals goals', 'home own goals conversion',
                               'home total passes', 'home total average pass streak', 'home crosses',
                               'home crosses average pass streak', 'home through balls',
                               'home through balls average streak', 'home long balls',
                               'home long balls average streak', 'home short passes',
                               'home short passes average streak', 'home cards', 'home fouls',
                               'home unprofessional', 'home dive', 'home other', 'home red cards',
                               'home yellow cards', 'home cards per foul', 'home woodwork',
                               'home shots on target', 'home shots off target', 'home shots blocked',
                               'home possession', 'home touches', 'home passes success',
                               'home accurate passes', 'home key passes', 'home dribbles won',
                               'home dribbles attempted', 'home dribbled past', 'home dribble success',
                               'home aerials won', 'home aerials won%', 'home offensive aerials',
                               'home defensive aerials', 'home successful tackles',
                               'home tackles attempted', 'home was dribbled', 'home tackles success %',
                               'home clearances', 'home interceptions', 'home corners',
                               'home corner accuracy', 'home dispossessed', 'home errors',
                               'home offsides',
                               'home goals conceded', 'home win streak', 'home lose streak', 'home elo',
                               'home pi rating', 'away score', 'away total shots', 'away total conversion rate',
                               'away open play shots', 'away open play goals',
                               'away open play conversion rate', 'away set piece shots',
                               'away set piece goals', 'away set piece conversion',
                               'away counter attack shots', 'away counter attack goals',
                               'away counter attack conversion', 'away penalty shots',
                               'away penalty goals', 'away penalty conversion', 'away own goals shots',
                               'away own goals goals', 'away own goals conversion',
                               'away total passes', 'away total average pass streak', 'away crosses',
                               'away crosses average pass streak', 'away through balls',
                               'away through balls average streak', 'away long balls',
                               'away long balls average streak', 'away short passes',
                               'away short passes average streak', 'away cards', 'away fouls',
                               'away unprofessional', 'away dive', 'away other', 'away red cards',
                               'away yellow cards', 'away cards per foul', 'away woodwork',
                               'away shots on target', 'away shots off target', 'away shots blocked',
                               'away possession', 'away touches', 'away passes success',
                               'away accurate passes', 'away key passes', 'away dribbles won',
                               'away dribbles attempted', 'away dribbled past', 'away dribble success',
                               'away aerials won', 'away aerials won%', 'away offensive aerials',
                               'away defensive aerials', 'away successful tackles',
                               'away tackles attempted', 'away was dribbled', 'away tackles success %',
                               'away clearances', 'away interceptions', 'away corners',
                               'away corner accuracy', 'away dispossessed', 'away errors',
                               'away offsides',
                               'away goals conceded', 'away win streak', 'away lose streak', 'away elo',
                               'away pi rating', 'league'])

    df.to_csv(path, index=False)


# load a pickle file of the training data
def load_training_data(path, features, league, return_columns=False):
    training_data = pd.read_csv(path)
    if league != 'all':
        training_data = training_data[training_data['league'] == league]

    training_data.drop(columns=['league'], inplace=True)
    training_data['Outcome'] = pd.Categorical(training_data['Outcome'])
    training_data = training_data.astype({"Outcome": int})

    target = training_data.pop('Outcome')
    training_data.astype('float64').dtypes

    temp = []

    if features != []:  # everything is included

        for f in features:
            temp.append('home %s' % f)
            temp.append('away %s' % f)


    else:  # interleave features
        l1 = training_data.columns[:len(training_data.columns) // 2]
        l2 = training_data.columns[len(training_data.columns) // 2:]

        temp = [val for pair in zip(l1, l2) for val in pair]

    training_data = training_data[temp]

    #
    # home_pi_rating_column = training_data.pop('home pi rating')
    # away_pi_rating_column = training_data.pop('away pi rating')
    #
    # home_elo_column = training_data.pop('home elo')
    # away_elo_column = training_data.pop('away elo')

    x = training_data.to_numpy()

    y = target.to_numpy()

    if return_columns:
        return x, y, temp

    return x, y


if __name__ == '__main__':
    for combination in ['append']:
        for column in ['everything both']:
            for i in range(10,0,-1):
                settings = {'n': i, 'columns': column, 'rating normalisation': 'min-max',
                            'combination': combination}
                generate_training_data('../data/whoscored/all-leagues.csv',
                                       '../data/whoscored/trainingdata/mean/alltrainingdata-%d.csv' % settings['n'],
                                       settings)

    # merge_seasons('../data/whoscored/premierleague/datascraped','../all-premierleague.csv')
    # merge_seasons('../data/whoscored/laliga/datascraped', '../all-laliga.csv')
    # merge_seasons('../data/whoscored/seriea/datascraped', '../all-seriea.csv')
    # merge_seasons('../data/whoscored/bundesliga/datascraped', '../all-bundesliga.csv')
    # merge_seasons('../data/whoscored/ligue1/datascraped', '../all-ligue1.csv')
    #
    # merge_leagues()
