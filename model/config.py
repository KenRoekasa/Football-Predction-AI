# Config file for the model of the neural network
import time
import numpy as np
from tensorboard.plugins.hparams import api as hp

METRIC_ACCURACY = 'accuracy'


HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([24, 89]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete(np.round(np.arange(float(0.5), float(0.9), 0.1),1)))
# HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_LR = hp.HParam('learning_rate',hp.Discrete(np.round(np.arange(float(0.001), float(0.011), 0.001),3)))
HP_BATCH_SIZE = hp.HParam('batch_size',hp.Discrete([1,2,10,32,64,100]))





new_data = False

NETWORK_TOPOLOGY=[24]

DROPOUT_VALUE = 0.8

LEARNING_RATE = 0.001
MOMENTUM = 0.01
BATCH_SIZE = 50
time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
NAME = 'model-%s-lr=%f-momentum=%f-batch-size=%d-dropout=%d-24' % (time, LEARNING_RATE,MOMENTUM,BATCH_SIZE,DROPOUT_VALUE)


N_PREVIOUS_GAMES = 3


COLUMNS = ['date', 'link', 'home team', 'away team', 'home score', 'away score', 'home total shots', 'away total shots',
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
         'away interceptions', 'home dispossessed', 'away dispossessed', 'home errors', 'away errors',
        'home elo',
         'away elo'
         ]

INPUT_DIM=44

def combination_of_means(teama_mean_array,teamb_mean_array): # How to combine the two means find the difference or put them all into the input
    mean_data_array = np.append(teama_mean_array, teamb_mean_array)
    # mean_data_array = teama_mean_array - teamb_mean_array
    return mean_data_array


