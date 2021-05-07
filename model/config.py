# Config file for the model of the neural network
from tensorboard.plugins.hparams import api as hp
from model.features_config import features_dict


league = 'all'  # premier-league la-liga bundesliga seriea ligue1
repeats = 1
number_of_parameters = 144
EPOCHS = 200
detail_view = 0 # 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.


baselogdir = '../logs/final_final/new system/absolute_final/'

HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([64,100]))
HP_NUM_UNITS2 = hp.HParam('num_units2', hp.Discrete([42]))
HP_NUM_UNITS3 = hp.HParam('num_units3', hp.Discrete([42]))
HP_NUM_UNITS4 = hp.HParam('num_units4', hp.Discrete([42]))


HP_OPTIMISER = hp.HParam('optimiser', hp.Discrete(['adam']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([7e-04,8e-04,9e-04,1e-05,2e-05,3e-05]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([32]))
HP_MOMENTUM = hp.HParam('momentum', hp.Discrete([0.05]))

# HP_REGULARISER_RATE = hp.HParam('regulariser_rate', hp.Discrete([0.1]))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
# HP_FEATURES = hp.HParam('features', hp.Discrete(features_dict.keys()))
HP_FEATURES = hp.HParam('features', hp.Discrete(['self_selected','all']))
# HP_PREVIOUS_GAMES = hp.HParam('previous_games', hp.Discrete([i for i in range(1, 10)]))
HP_PREVIOUS_GAMES = hp.HParam('previous_games', hp.Discrete([7]))
HP_NORMALISATION = hp.HParam('norm', hp.Discrete(['ratio']))
HP_RESAMPLING = hp.HParam('resampling',hp.Discrete(['oversample','smote']))
# HP_combine = hp.HParam('combine', hp.Discrete(['append', 'diff']))
HP_MODEL = hp.HParam('model',hp.Discrete(['dropout']))


