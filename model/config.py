# Config file for the model of the neural network
import time
import numpy as np
from tensorboard.plugins.hparams import api as hp

INPUT_DIM = 4

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([5,4,10,20]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.9]))
HP_OPTIMISER = hp.HParam('optimiser', hp.Discrete(['sgd']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([0.001,0.005,0.01]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32,16]))
HP_MOMENTUM = hp.HParam('momentum', hp.Discrete([0.005,0.05]))
HP_REGULARISER_RATE = hp.HParam('regulariser_rate', hp.Discrete([0.0006]))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))

new_data = False



N_PREVIOUS_GAMES = 6

columns_selector = 'pi-rating only'  # Select which subset of fields to choose from below in the data

default_columns = ['date', 'link', 'home team', 'away team', 'home score', 'away score', 'league', 'season']

COLUMNS = {'elo only': default_columns + ['home elo', 'away elo'],

           'pi-rating only': default_columns +
                             ['home home pi rating', 'home away pi rating', 'away home pi rating',
                              'away away pi rating'],

           'pi-rating': default_columns + ['home total shots',
                                           'away total shots',
                                           'home shots on target', 'away shots on target', 'home possession',
                                           'away possession',
                                           'home total conversion rate',
                                           'away total conversion rate', 'home fouls', 'away fouls',
                                           'home yellow cards',
                                           'away yellow cards',
                                           'home red cards', 'away red cards', 'home total passes',
                                           'away total passes',
                                           'home accurate passes',
                                           'away accurate passes', 'home open play conversion rate',
                                           'away open play conversion rate',
                                           'home set piece conversion', 'away set piece conversion',
                                           'home counter attack shots',
                                           'away counter attack shots',
                                           'home counter attack goals', 'away counter attack goals',
                                           'home key passes', 'away key passes',
                                           'home dribbles attempted', 'away dribbles attempted',
                                           'home dribble success',
                                           'away dribble success',
                                           'home aerials won%', 'away aerials won%', 'home tackles attempted',
                                           'away tackles attempted',
                                           'home tackles success %', 'away tackles success %', 'home was dribbled',
                                           'away was dribbled',
                                           'home interceptions',
                                           'away interceptions', 'home dispossessed', 'away dispossessed',
                                           'home errors', 'away errors',
                                           'home home pi rating', 'home away pi rating', 'away home pi rating',
                                           'away away pi rating'
                                           ],
           'elo': default_columns + ['home total shots',
                                     'away total shots',
                                     'home shots on target', 'away shots on target', 'home possession',
                                     'away possession',
                                     'home total conversion rate',
                                     'away total conversion rate', 'home fouls', 'away fouls', 'home yellow cards',
                                     'away yellow cards',
                                     'home red cards', 'away red cards', 'home total passes', 'away total passes',
                                     'home accurate passes',
                                     'away accurate passes', 'home open play conversion rate',
                                     'away open play conversion rate',
                                     'home set piece conversion', 'away set piece conversion',
                                     'home counter attack shots',
                                     'away counter attack shots',
                                     'home counter attack goals', 'away counter attack goals', 'home key passes',
                                     'away key passes',
                                     'home dribbles attempted', 'away dribbles attempted', 'home dribble success',
                                     'away dribble success',
                                     'home aerials won%', 'away aerials won%', 'home tackles attempted',
                                     'away tackles attempted',
                                     'home tackles success %', 'away tackles success %', 'home was dribbled',
                                     'away was dribbled',
                                     'home interceptions',
                                     'away interceptions', 'home dispossessed', 'away dispossessed', 'home errors',
                                     'away errors',
                                     'home elo', 'away elo'
                                     ],

           'pi-rating+': default_columns + ['home total shots',
                                            'away total shots',
                                            'home shots on target', 'away shots on target', 'home possession',
                                            'away possession',
                                            'home total conversion rate',
                                            'away total conversion rate',
                                            'home home pi rating', 'home away pi rating', 'away home pi rating',
                                            'away away pi rating'], }
normalise = 0


def combination_of_means(teama_mean_array,
                         teamb_mean_array):  # How to combine the two means find the difference or put them all into the input
    mean_data_array = np.append(teama_mean_array, teamb_mean_array)
    # mean_data_array = teama_mean_array - teamb_mean_array
    return mean_data_array
