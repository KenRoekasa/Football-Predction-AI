# Config file for the model of the neural network
from tensorboard.plugins.hparams import api as hp

from model.features_config import features_dict

league = 'all'  # premier-league la-liga bundesliga seriea ligue1
repeats = 1
number_of_parameters = 9
EPOCHS = 150
detail_view = 0 # 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
model_type = 'dropout'  # base l1 l2 dropout


HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([100]))
HP_NUM_UNITS2 = hp.HParam('num_units2', hp.Discrete([42]))
HP_NUM_UNITS3 = hp.HParam('num_units3', hp.Discrete([42]))
HP_NUM_UNITS4 = hp.HParam('num_units4', hp.Discrete([42]))


HP_OPTIMISER = hp.HParam('optimiser', hp.Discrete(['adam']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-5]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
HP_MOMENTUM = hp.HParam('momentum', hp.Discrete([0.05]))



if model_type == 'base':
    HP_REGULARISER_RATE = hp.HParam('regulariser_rate', hp.Discrete([0.1]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2]))
if model_type == 'l1' or model_type == 'l2':
    HP_REGULARISER_RATE = hp.HParam('regulariser_rate', hp.Discrete([1*10**(-exp) for exp in range(1,7)]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.2]))
if model_type == 'dropout':
    HP_REGULARISER_RATE = hp.HParam('regulariser_rate', hp.Discrete([1*10**(-exp) for exp in range(1,7)]))
    HP_DROPOUT = hp.HParam('dropout', hp.Discrete([i/10 for i in range(1,10)]))


# HP_REGULARISER_RATE = hp.HParam('regulariser_rate', hp.Discrete([0.1]))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
# HP_FEATURES = hp.HParam('features', hp.Discrete(features_dict.keys()))
HP_FEATURES = hp.HParam('features', hp.Discrete(['self_selected']))
# HP_PREVIOUS_GAMES = hp.HParam('previous_games', hp.Discrete([i for i in range(1, 10)]))
HP_PREVIOUS_GAMES = hp.HParam('previous_games', hp.Discrete([10]))
HP_NORMALISATION = hp.HParam('norm', hp.Discrete(['ratio']))
HP_RESAMPLING = hp.HParam('resampling',hp.Discrete(['oversample']))
# HP_combine = hp.HParam('combine', hp.Discrete(['append', 'diff']))
