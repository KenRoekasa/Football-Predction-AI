import os

from sklearn.utils import class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.regularizers import l1, l2
import sys

sys.path.append('..')
from model.features_config import features_dict
from model.confusion_callback import ConfusionCallbacck
from imblearn.under_sampling import RandomUnderSampler
import datetime
from tensorflow.keras.layers import Dense
from data_preparation.dataloader import load_training_data, normalise_input_array
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import model.config as cf

from model.config import league, number_of_parameters, repeats, EPOCHS, \
    detail_view, model_type

from tensorboard.plugins.hparams import api as hp
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.activations import softmax

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def train_test_model(logdir, hparams, x_train, y_train, x_valid, y_valid):
    if hparams[cf.HP_OPTIMISER] == "adam":
        optimiser = tf.keras.optimizers.Adam(learning_rate=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "sgd":
        optimiser = tf.keras.optimizers.SGD(lr=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "RMSprop":
        optimiser = tf.keras.optimizers.RMSprop(lr=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "Adagrad":
        optimiser = tf.keras.optimizers.Adagrad(lr=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "Adadelta":
        optimiser = tf.keras.optimizers.Adadelta(lr=hparams[cf.HP_LR])
    else:
        raise ValueError("unexpected optimiser name: %r" % (hparams[cf.HP_OPTIMISER],))

    if model_type == 'base':
        model = tf.keras.Sequential([
            Dense(hparams[cf.HP_NUM_UNITS1], activation=hparams[cf.HP_ACTIVATION],
                  ),
            Dense(hparams[cf.HP_NUM_UNITS2], activation=hparams[cf.HP_ACTIVATION],
                  ),
            Dense(hparams[cf.HP_NUM_UNITS3], activation=hparams[cf.HP_ACTIVATION],
                  ),
            Dense(hparams[cf.HP_NUM_UNITS4], activation=hparams[cf.HP_ACTIVATION],
                  ),
            Dense(3, activation=softmax)
        ])
    elif model_type == 'l1':  # l1
        model = tf.keras.Sequential([
            Dense(hparams[cf.HP_NUM_UNITS1], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l1(hparams[cf.HP_REGULARISER_RATE]),
                  ),
            Dense(hparams[cf.HP_NUM_UNITS2], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l1(hparams[cf.HP_REGULARISER_RATE])
                  ),
            Dense(hparams[cf.HP_NUM_UNITS3], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l1(hparams[cf.HP_REGULARISER_RATE])
                  ),
            Dense(hparams[cf.HP_NUM_UNITS4], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l1(hparams[cf.HP_REGULARISER_RATE])
                  ),
            Dense(3, activation=softmax)
        ])
    elif model_type == 'l2':  # l2
        model = tf.keras.Sequential([
            Dense(hparams[cf.HP_NUM_UNITS1], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l2(hparams[cf.HP_REGULARISER_RATE]),
                  ),
            Dense(hparams[cf.HP_NUM_UNITS2], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l2(hparams[cf.HP_REGULARISER_RATE])
                  ),
            Dense(hparams[cf.HP_NUM_UNITS3], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l2(hparams[cf.HP_REGULARISER_RATE])
                  ),
            Dense(hparams[cf.HP_NUM_UNITS4], activation=hparams[cf.HP_ACTIVATION],
                  kernel_regularizer=l2(hparams[cf.HP_REGULARISER_RATE])
                  ),
            Dense(3, activation=softmax)
        ])
    elif model_type == 'dropout':  # dropout
        model = tf.keras.Sequential([
            Dense(hparams[cf.HP_NUM_UNITS1], activation=hparams[cf.HP_ACTIVATION]),
            Dropout(hparams[cf.HP_DROPOUT]),
            Dense(hparams[cf.HP_NUM_UNITS2], activation=hparams[cf.HP_ACTIVATION]),
            Dropout(hparams[cf.HP_DROPOUT]),
            Dense(hparams[cf.HP_NUM_UNITS3], activation=hparams[cf.HP_ACTIVATION]),
            Dropout(hparams[cf.HP_DROPOUT]),
            Dense(hparams[cf.HP_NUM_UNITS4], activation=hparams[cf.HP_ACTIVATION]),
            Dropout(hparams[cf.HP_DROPOUT]),
            Dense(3, activation=softmax)
        ])

    model.compile(optimizer=optimiser, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')

    callbacks = [
        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
        ConfusionCallbacck(models=[model], x_valid=x_valid, y_valid=y_valid, file_writer_cm=file_writer_cm)

        # ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=30)
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=50, restore_best_weights=True)
        # early stopping
    ]
    if resample == 'class weights':
        weights = class_weight.compute_class_weight('balanced',
                                                    classes=np.unique(y_train),
                                                    y=y_train)
        weight_dict = {0: weights[0], 1: weights[1], 2: weights[2]}
    else:
        weight_dict = None

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=hparams[cf.HP_BATCH_SIZE], shuffle=True,
              verbose=detail_view,
              callbacks=callbacks, validation_data=(x_valid, y_valid), class_weight=weight_dict)

    _, accuracy = model.evaluate(x_train, y_train)

    tf.keras.models.save_model(model,
                               logdir + '/savedmodel' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    return 0


if __name__ == '__main__':
    with tqdm(total=number_of_parameters) as pbar:
        for features in cf.HP_FEATURES.domain.values:
            for n in cf.HP_PREVIOUS_GAMES.domain.values:
                for normalisation in cf.HP_NORMALISATION.domain.values:
                    for resample in cf.HP_RESAMPLING.domain.values:
                        x, y = load_training_data('../data/whoscored/trainingdata/mean/alltrainingdata-%d.csv' % n,
                                                  features_dict[features], league)

                        training_data_text = "%d-%s-%s-%s-%s" % (n, league, features, normalisation, resample)

                        print(y.tolist().count(0))
                        print(y.tolist().count(1))
                        print(y.tolist().count(2))

                        x = normalise_input_array(x, normalisation)

                        x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.30, shuffle=True)

                        if resample == 'smote':
                            oversample = SMOTE()
                            x_train, y_train = oversample.fit_resample(x_train, y_train)
                        if resample == 'oversample':
                            # oversample
                            oversample = RandomOverSampler(sampling_strategy='minority')
                            x_train, y_train = oversample.fit_resample(x_train, y_train)
                        if resample == 'undersample':
                            # undersample
                            undersample = RandomUnderSampler(sampling_strategy='majority')
                            x_train, y_train = undersample.fit_resample(x_train, y_train)

                        print(y_train.tolist().count(0))
                        print(y_train.tolist().count(1))
                        print(y_train.tolist().count(2))

                        print(y_valid.tolist().count(0))
                        print(y_valid.tolist().count(1))
                        print(y_valid.tolist().count(2))

                        session_num = 0

                        for i in range(0, repeats):  # repeats
                            for optimiser in cf.HP_OPTIMISER.domain.values:
                                for lr in cf.HP_LR.domain.values:
                                    for batch_size in cf.HP_BATCH_SIZE.domain.values:
                                        for activation in cf.HP_ACTIVATION.domain.values:
                                            for num_units1 in cf.HP_NUM_UNITS1.domain.values:
                                                for num_units2 in cf.HP_NUM_UNITS2.domain.values:
                                                    for num_units3 in cf.HP_NUM_UNITS3.domain.values:
                                                        for num_units4 in cf.HP_NUM_UNITS4.domain.values:
                                                            for momentum in cf.HP_MOMENTUM.domain.values:
                                                                for rr in cf.HP_REGULARISER_RATE.domain.values:
                                                                    for dropout in cf.HP_DROPOUT.domain.values:
                                                                        it_start_time = datetime.datetime.now()

                                                                        if model_type == 'base':
                                                                            hparams = {
                                                                                cf.HP_NUM_UNITS1: num_units1,
                                                                                cf.HP_NUM_UNITS2: num_units1,
                                                                                cf.HP_NUM_UNITS3: num_units1,
                                                                                cf.HP_NUM_UNITS4: num_units1,
                                                                                cf.HP_LR: lr,
                                                                                cf.HP_BATCH_SIZE: batch_size,
                                                                                cf.HP_OPTIMISER: optimiser,
                                                                                cf.HP_ACTIVATION: activation,
                                                                                # cf.HP_MOMENTUM: momentum,
                                                                                cf.HP_FEATURES: features,
                                                                                cf.HP_PREVIOUS_GAMES: n,
                                                                                cf.HP_NORMALISATION: normalisation,
                                                                                cf.HP_RESAMPLING: resample
                                                                            }
                                                                        elif model_type == 'l1' or model_type == 'l2':
                                                                            hparams = {
                                                                                cf.HP_NUM_UNITS1: num_units1,
                                                                                cf.HP_NUM_UNITS2: num_units1,
                                                                                cf.HP_NUM_UNITS3: num_units1,
                                                                                cf.HP_NUM_UNITS4: num_units1,
                                                                                cf.HP_LR: lr,
                                                                                cf.HP_BATCH_SIZE: batch_size,
                                                                                cf.HP_OPTIMISER: optimiser,
                                                                                cf.HP_ACTIVATION: activation,
                                                                                # cf.HP_MOMENTUM: momentum,
                                                                                cf.HP_REGULARISER_RATE: rr,
                                                                                cf.HP_FEATURES: features,
                                                                                cf.HP_PREVIOUS_GAMES: n,
                                                                                cf.HP_NORMALISATION: normalisation,
                                                                                cf.HP_RESAMPLING: resample

                                                                            }
                                                                        elif model_type == 'dropout':
                                                                            hparams = {
                                                                                cf.HP_NUM_UNITS1: num_units1,
                                                                                cf.HP_NUM_UNITS2: num_units1,
                                                                                cf.HP_NUM_UNITS3: num_units1,
                                                                                cf.HP_NUM_UNITS4: num_units1,
                                                                                cf.HP_LR: lr,
                                                                                cf.HP_BATCH_SIZE: batch_size,
                                                                                cf.HP_OPTIMISER: optimiser,
                                                                                cf.HP_ACTIVATION: activation,
                                                                                cf.HP_DROPOUT: dropout,
                                                                                cf.HP_FEATURES: features,
                                                                                cf.HP_PREVIOUS_GAMES: n,
                                                                                cf.HP_NORMALISATION: normalisation,
                                                                                cf.HP_RESAMPLING: resample
                                                                            }

                                                                        run_name = "run-%d" % session_num
                                                                        print('--- Starting trial: %s' % run_name)
                                                                        print({h.name: hparams[h] for h in hparams})
                                                                        today = datetime.date.today()

                                                                        logdir = '../logs/final_final/regularisation/' +str(model_type)+"/"+ str(
                                                                            today) + '/epoch' + str(
                                                                            EPOCHS) + str(
                                                                            datetime.datetime.now().strftime(
                                                                                "%Y%m%d-%H%M%S")) + str(
                                                                            training_data_text) + '-' + '-'.join(
                                                                            [str(lr), str(batch_size), str(optimiser)])

                                                                        model = train_test_model(
                                                                            logdir, hparams, x_train, y_train, x_valid,
                                                                            y_valid)

                                                                        pbar.update(1)

                                                                        session_num += 1
