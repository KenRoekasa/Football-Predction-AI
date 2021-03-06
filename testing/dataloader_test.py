import unittest
import pandas as pd
from data_preparation.dataloader import get_previous_n_games, format_data, get_winstreak, normalise_ratings, get_ratings
from pandas._testing import assert_frame_equal


class TestDataloader(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_csv('test.csv')

    def test_get_winstreak(self):
        settings = {'n': 4, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}
        data = format_data(self.data, settings)

        self.assertEqual(get_winstreak(get_previous_n_games(data, 'Chelsea', settings['n'], data.iloc[14]), 'Chelsea'),
                         3)

    def test_get_rating(self):
        settings = {'n': 4, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}
        data = format_data(self.data, settings)

        home_rating, away_rating = get_ratings(data, data.iloc[9], 'pi rating')

        self.assertAlmostEqual(float(home_rating), 1.893333329)
        self.assertAlmostEqual(float(away_rating), 2.66508901)

    def test_normalise_ratings(self):
        settings = {'n': 4, 'columns': 'pi-rating only', 'rating normalisation': 'min-max',
                    'combination': 'append'}
        data = format_data(self.data, settings)

        away_rating, home_rating = normalise_ratings(data, data.iloc[9], settings)

        self.assertAlmostEqual(float(home_rating), 0.760428496)
        self.assertAlmostEqual(float(away_rating), 0.908354822)

    def test_normalise_ratings_both(self):
        settings = {'n': 4, 'columns': 'both', 'rating normalisation': 'min-max',
                    'combination': 'append'}

        data = format_data(self.data, settings)

        away_rating, home_rating = normalise_ratings(data, data.iloc[9], settings)

        self.assertAlmostEqual(float(home_rating[0]), 0.81054513)
        self.assertAlmostEqual(float(away_rating[0]), 0.91689008)

        self.assertAlmostEqual(float(home_rating[1]), 0.760428496)
        self.assertAlmostEqual(float(away_rating[1]), 0.908354822)


if __name__ == '__main__':
    unittest.main()
