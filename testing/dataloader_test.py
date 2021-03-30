import unittest
import pandas as pd
from data_preparation.dataloader import get_previous_n_games, format_data, get_winstreak, normalise_ratings, \
    get_ratings, get_mean_array, create_training_data, get_mean_stats, get_losestreak, normalise_input_array
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

        np.testing.assert_array_equal(array, [6, 6, 0, 1])

    def test_create_training_data(self):
        settings = {'n': 3, 'columns': 'both', 'rating normalisation': 'min-max',
                    'combination': 'append'}

        data = format_data(self.data, settings)
        training_data = create_training_data(data,settings)

        # print(training_data)

    def test_get_mean_stats(self):
        data = pd.DataFrame(np.array([['Chelsea', 'Arsenal', 1, 2, 3, 10], ['Liverpool', 'Chelsea', 4, 5, 6, 5],
                                      ['Chelsea', 'Man U', 7, 8, 9, 5]]),
                            columns=['home team', 'away team', 'home shots', 'home score',
                                     'away shots', 'away score'])

        mean_data = get_mean_stats(data, 'Chelsea')
        np.testing.assert_array_equal(mean_data, np.array([14, 15, 20]))

    def test_get_losestreak(self):
        previous = pd.DataFrame(np.array([['Chelsea', 'Arsenal', 1, 2, 3, 10],
                                          ['Arsenal', 'Liverpool', 4, 5, 6, 5],
                                          ['Arsenal', 'Man U', 0, 2, 4, 25],
                                          ['Chelsea', 'Arsenal', 2, 2, 3, 10],
                                          ['Arsenal', 'Man U', 0, 1, 25, 5],
                                          ['Watford', 'Arsenal', 3, 1, 4, 8],
                                          ['Tottenham', 'Arsenal', 7, 1, 2, 12]]),

                                columns=['home team', 'away team', 'home score', 'away score', 'home shots',
                                         'away shots'])

        self.assertEqual(get_losestreak(previous, 'Arsenal'),
                         3)

    def test_normalise_input_array(self):
        x = np.array([[2, 0, 4, 1, 5, 4, 100, 280, 1.2232, 1.2245, 1.00264, 0.212],
                      [6, 1, 0, 0, 2, 1, 96, 360, 1.22555, 0.245, 0.222, 1.666]])

        y = normalise_input_array(x, 'ratio')

        np.testing.assert_array_equal(y, np.array(
            [[1, 0, 0.8, 0.2, 5 / 9, 4 / 9, 100 / 380, 280 / 380, 1.2232, 1.2245, 1.00264, 0.212],
             [6 / 7, 1 / 7, 0.5, 0.5, 2 / 3, 1 / 3, 96 / 456, 360 / 456, 1.22555, 0.245, 0.222, 1.666]]))

        x = np.array([[2, 0, 4, 1, 5, 4, 100, 280, 1.2232, 1.2245, 1.00264, 0.212],
                      [6, 1, 0, 0, 2, 1, 96, 360, 1.22555, 0.245, 0.222, 1.666]])

        real = np.empty_like(x)
        real[0] = x[0] / np.sum(x[0])
        real[1] = x[1] / np.sum(x[1])

        y = normalise_input_array(x, 'l1')

        np.testing.assert_array_equal(y, real)

        x = np.array([[2, 0, 4, 1, 5, 4, 100, 280, 1.2232, 1.2245, 1.00264, 0.212],
                      [6, 1, 0, 0, 2, 1, 96, 360, 1.22555, 0.245, 0.222, 1.666]])

        real = np.empty_like(x)
        real[0] = x[0] / (np.sqrt(np.sum(np.square(x[0]))))
        real[1] = x[1] / (np.sqrt(np.sum(np.square(x[1]))))

        y = normalise_input_array(x, 'l2')

        np.testing.assert_array_equal(y, real)

        x = np.array([[2, 0, 4, 1, 5, 4, 100, 280, 1.2232, 1.2245, 1.00264, 0.212],
                  [6, 1, 0, 0, 2, 1, 96, 360, 1.456, 0.7856, 0.564, 1.456],
                  [4, 7, 0, 4, 1, 0, 25, 125, 1.22555, 0.245, 0.222, 1.666]])

        real = np.copy(x)

        real[:, :-4] = (x[:, :-4] - x[:, :-4].min(axis=0)) / (x[:, :-4].max(axis=0) - x[:, :-4].min(axis=0))

        y = normalise_input_array(x, 'max')

        np.testing.assert_array_equal(y, real)


if __name__ == '__main__':
    unittest.main()
