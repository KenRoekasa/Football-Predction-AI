import unittest
import pandas as pd
from data_preparation.dataloader import get_previous_n_games, format_data, get_winstreak, normalise_ratings, \
    get_ratings, get_mean_array, create_training_data
from pandas._testing import assert_frame_equal
import numpy as np


class TestDataloader(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv('test.csv')

    def test_previous_games(self):
        settings = {'n': 3, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}
        data = format_data(self.data, settings)
        selected_game = data[
            data[
                'link'] == '/Matches/1080752/MatchReport/England-Premier-League-2016-2017-Crystal-Palace-Chelsea'].iloc[
            0]
        previous = get_previous_n_games(data, 'Chelsea', settings['n'], selected_game)

        actual_previous = pd.read_csv('previousgames.csv')
        actual_previous = format_data(actual_previous, settings)

        np.testing.assert_array_equal(previous.values, actual_previous.values)

        selected_game = data[
            data['link'] ==
            '/Matches/1080798/MatchReport/England-Premier-League-2016-2017-Liverpool-Bournemouth'].iloc[0]
        previous = get_previous_n_games(data, 'Liverpool', settings['n'], selected_game)
        actual_previous = pd.read_csv('previousgames2.csv')
        actual_previous = format_data(actual_previous, settings)

        np.testing.assert_array_equal(previous.values, actual_previous.values)

    def test_no_previous_games(self):
        settings = {'n': 3, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}
        data = format_data(self.data, settings)

        selected_game = data[
            data['link'] ==
            '/Matches/1080521/MatchReport/England-Premier-League-2016-2017-Liverpool-Middlesbrough'].iloc[0]
        previous = get_previous_n_games(data, 'Liverpool', settings['n'], selected_game)

        self.assertEqual(previous.size, 0)  # no previous game

    def test_get_winstreak(self):
        settings = {'n': 4, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}

        data = format_data(self.data, settings)

        selected_game = data[
            data[
                'link'] == '/Matches/1080752/MatchReport/England-Premier-League-2016-2017-Crystal-Palace-Chelsea'].iloc[
            0]
        self.assertEqual(get_winstreak(get_previous_n_games(data, 'Chelsea', settings['n'], selected_game),
                                       'Chelsea'),
                         3)

    def test_get_rating(self):
        settings = {'n': 4, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}
        data = format_data(self.data, settings)

        selected_game = data[
            data['link'] == '/Matches/1080661/MatchReport/England-Premier-League-2016-2017-Liverpool-Chelsea'].iloc[0]

        teama_previous = get_previous_n_games(data, 'Liverpool', settings['n'], selected_game)
        teamb_previous = get_previous_n_games(data, 'Chelsea', settings['n'], selected_game)

        home_rating, away_rating = get_ratings('Liverpool', 'Chelsea', 'pi rating', teama_previous, teamb_previous)

        self.assertAlmostEqual(float(home_rating), 1.893333329)
        self.assertAlmostEqual(float(away_rating), 2.66508901)

    def test_normalise_ratings(self):
        settings = {'n': 4, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}
        data = format_data(self.data, settings)

        selected_game = data[
            data['link'] == '/Matches/1080661/MatchReport/England-Premier-League-2016-2017-Liverpool-Chelsea'].iloc[0]

        teama_previous = get_previous_n_games(data, 'Liverpool', settings['n'], selected_game)
        teamb_previous = get_previous_n_games(data, 'Chelsea', settings['n'], selected_game)

        away_rating, home_rating = normalise_ratings(data, 'Liverpool', 'Chelsea', settings, teama_previous,
                                                     teamb_previous)

        self.assertAlmostEqual(float(home_rating), 0.760428496)
        self.assertAlmostEqual(float(away_rating), 0.908354822)

    def test_normalise_ratings_both(self):
        settings = {'n': 4, 'columns': 'both', 'rating normalisation': 'min-max',
                    'combination': 'append'}

        data = format_data(self.data, settings)

        selected_game = data[
            data['link'] == '/Matches/1080661/MatchReport/England-Premier-League-2016-2017-Liverpool-Chelsea'].iloc[0]

        teama_previous = get_previous_n_games(data, 'Liverpool', settings['n'], selected_game)
        teamb_previous = get_previous_n_games(data, 'Chelsea', settings['n'], selected_game)

        away_rating, home_rating = normalise_ratings(data, 'Liverpool', 'Chelsea', settings, teama_previous,
                                                     teamb_previous)

        self.assertAlmostEqual(float(home_rating[0]), 0.81054513)
        self.assertAlmostEqual(float(away_rating[0]), 0.91689008)

        self.assertAlmostEqual(float(home_rating[1]), 0.760428496)
        self.assertAlmostEqual(float(away_rating[1]), 0.908354822)

    def test_get_mean_data(self):
        settings = {'n': 3, 'columns': 'both', 'rating normalisation': 'min-max',
                    'combination': 'append'}

        data = format_data(self.data, settings)

        selected_game = data[
            data['link'] == '/Matches/1080661/MatchReport/England-Premier-League-2016-2017-Liverpool-Chelsea'].iloc[0]

        previous_games = get_previous_n_games(data, 'Liverpool', settings['n'], selected_game)
        array = get_mean_array('Liverpool', previous_games)

        np.testing.assert_array_equal(array, [2, 0])

    def test_create_training_data(self):
        settings = {'n': 3, 'columns': 'both', 'rating normalisation': 'min-max',
                    'combination': 'append'}

        data = format_data(self.data, settings)
        training_data = create_training_data(data, settings)

        print(training_data)

if __name__ == '__main__':
    unittest.main()
