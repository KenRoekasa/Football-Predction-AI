# Config file for the model of the neural network
from tensorboard.plugins.hparams import api as hp


n = 10
league = 'all' # premier-league la-liga bundesliga seriea ligue1
features = 'feature_importance'
normalisation = 'ratio'
resample = 'smote'

repeats = 1
number_of_parameters = 10

EPOCHS = 200


HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([42]))
HP_NUM_UNITS2 = hp.HParam('num_units2', hp.Discrete([42]))
HP_NUM_UNITS3 = hp.HParam('num_units3', hp.Discrete([42]))
HP_NUM_UNITS4 = hp.HParam('num_units4', hp.Discrete([42]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1]))
HP_OPTIMISER = hp.HParam('optimiser', hp.Discrete(['adam']))
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-1,1e-2,1e-3,1e-4,1e-5]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([16,100]))
HP_MOMENTUM = hp.HParam('momentum', hp.Discrete([0.05]))
HP_REGULARISER_RATE = hp.HParam('regulariser_rate', hp.Discrete([1e-1]))
HP_ACTIVATION = hp.HParam('activation', hp.Discrete(['relu']))
HP_previous_games = hp.HParam('num', hp.Discrete([i for i in range(1, 7)]))
HP_combine = hp.HParam('combine', hp.Discrete(['append', 'diff']))
HP_norm = hp.HParam('norm', hp.Discrete(['min-max', 'ratio']))

