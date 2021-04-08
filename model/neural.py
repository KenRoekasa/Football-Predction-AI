import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sklearn

import sys

sys.path.append('..')
from model.confusion_callback import ConfusionCallbacck
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight

from graphing.confusion_matrix import plot_confusion_matrix, plot_to_image
from model import dataset
import pandas as pd

from tensorflow.python.keras.layers import Dropout

import csv

import datetime
from tensorflow.keras.layers import Dense

from data_preparation.dataloader import load_training_data, normalise_input_array
from tqdm import tqdm
import tensorflow as tf
import model.config as cf
from tensorboard.plugins.hparams import api as hp

from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from tensorflow.python.keras.activations import softmax, relu

import re
import numpy as np
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def train_test_model(logdir, hparams, x_train, y_train, x_valid, y_valid):
    if hparams[cf.HP_OPTIMISER] == "adam":
        optimiser = tf.keras.optimizers.Adam(learning_rate=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "sgd":
        optimiser = tf.keras.optimizers.SGD(lr=hparams[cf.HP_LR], momentum=hparams[cf.HP_MOMENTUM])
    elif hparams[cf.HP_OPTIMISER] == "RMSprop":
        optimiser = tf.keras.optimizers.RMSprop(lr=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "Adagrad":
        optimiser = tf.keras.optimizers.Adagrad(lr=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "Adadelta":
        optimiser = tf.keras.optimizers.Adadelta(lr=hparams[cf.HP_LR])
    else:
        raise ValueError("unexpected optimiser name: %r" % (hparams[cf.HP_OPTIMISER],))

    model = tf.keras.Sequential([
        Dense(hparams[cf.HP_NUM_UNITS1], activation=hparams[cf.HP_ACTIVATION]),
        # Dropout(hparams[cf.HP_DROPOUT]),
        Dense(hparams[cf.HP_NUM_UNITS2], activation=hparams[cf.HP_ACTIVATION]),
        # Dropout(hparams[cf.HP_DROPOUT]),
        Dense(hparams[cf.HP_NUM_UNITS3], activation=hparams[cf.HP_ACTIVATION]),
        # Dropout(hparams[cf.HP_DROPOUT]),
        Dense(hparams[cf.HP_NUM_UNITS4], activation=hparams[cf.HP_ACTIVATION]),
        # Dropout(hparams[cf.HP_DROPOUT]),
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

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=hparams[cf.HP_BATCH_SIZE], shuffle=True, verbose=0,
              callbacks=callbacks, validation_data=(x_valid, y_valid))

    _, accuracy = model.evaluate(x_train, y_train)
    tf.keras.models.save_model(model,
                               logdir + '/savedmodel' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    return 0


if __name__ == '__main__':
    EPOCHS = 160

    training_data_text = 'alltraining mean'

    features = ['score', 'total shots', 'total conversion rate', 'open play shots', 'open play goals',
                'open play conversion rate', 'set piece shots', 'set piece goals', 'set piece conversion',
                'counter attack shots', 'counter attack goals', 'counter attack conversion', 'total passes',
                'total average pass streak', 'crosses', 'crosses average pass streak', 'through balls',
                'through balls average streak', 'long balls', 'long balls average streak', 'short passes',
                'short passes average streak', 'fouls', 'red cards', 'yellow cards', 'cards per foul', 'woodwork',
                'shots on target', 'shots off target', 'shots blocked', 'possession', 'touches', 'passes success',
                'accurate passes', 'key passes', 'dribbles won', 'dribbles attempted', 'dribbled past',
                'dribble success', 'aerials won', 'aerials won%', 'offensive aerials', 'defensive aerials',
                'successful tackles', 'tackles attempted', 'was dribbled', 'tackles success %', 'clearances',
                'interceptions', 'corners', 'corner accuracy', 'dispossessed', 'errors', 'offsides', 'goals conceded',
                'win streak', 'lose streak', 'elo', 'pi rating']

    # features = [
    #     'score',
    #     'goals conceded',
    #     'pi rating', 'elo']

    x, y = load_training_data('../data/whoscored/trainingdata/mean/alltrainingdata-10.csv',
                              [], 'all')

    print(y.tolist().count(0))
    print(y.tolist().count(1))
    print(y.tolist().count(2))

    x = normalise_input_array(x, 'ratio')

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.30, shuffle=True)

    oversample = SMOTE()
    x_train, y_train = oversample.fit_resample(x_train, y_train)

    # oversample
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # x_train, y_train = oversample.fit_resample(x_train, y_train)

    # undersample
    # undersample = RandomUnderSampler(sampling_strategy='majority')
    # x_train, y_train = undersample.fit_resample(x_train, y_train)

    # weights = class_weight.compute_class_weight('balanced',
    #                                             np.unique(y_train),
    #                                             y_train)
    #
    #     # class_weight = {0: weights[0], 1: weights[1], 2: weights[2]}

    print(y_train.tolist().count(0))
    print(y_train.tolist().count(1))
    print(y_train.tolist().count(2))

    print(y_valid.tolist().count(0))
    print(y_valid.tolist().count(1))
    print(y_valid.tolist().count(2))

    start_time = datetime.datetime.now()
    start_time_f = start_time.strftime("%H:%M:%S")
    print(start_time_f)
    session_num = 0

    number_of_parameters = 10
    with tqdm(total=number_of_parameters) as pbar:
        for i in range(0, 1):  # repeats
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
                                                        hparams = {
                                                            cf.HP_NUM_UNITS1: num_units1,
                                                            cf.HP_NUM_UNITS2: num_units2,
                                                            cf.HP_NUM_UNITS3: num_units3,
                                                            cf.HP_NUM_UNITS4: num_units4,
                                                            cf.HP_LR: lr,
                                                            cf.HP_BATCH_SIZE: batch_size,
                                                            cf.HP_OPTIMISER: optimiser,
                                                            cf.HP_ACTIVATION: activation,
                                                            # cf.HP_MOMENTUM: momentum,
                                                            # cf.HP_REGULARISER_RATE: rr,
                                                            # cf.HP_DROPOUT: dropout,
                                                        }
                                                        run_name = "run-%d" % session_num
                                                        print('--- Starting trial: %s' % run_name)
                                                        print({h.name: hparams[h] for h in hparams})
                                                        today = datetime.date.today()

                                                        logdir = '../logs/sum or mean/' + str(
                                                            today) + '/epoch' + str(
                                                            EPOCHS) + str(
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + str(
                                                            training_data_text)

                                                        model = train_test_model(
                                                            logdir, hparams, x_train, y_train, x_valid, y_valid)

                                                        # predictions = model.predict(x_valid)
                                                        # results = []
                                                        # for i in predictions:
                                                        #     results.append(np.argmax(i))
                                                        #
                                                        # print(predictions)
                                                        # print(y_valid)
                                                        # confusion = tf.math.confusion_matrix(labels=y_valid,
                                                        #                                      predictions=results)
                                                        # confusion = confusion.numpy()
                                                        #
                                                        # fig = plot_confusion_matrix(confusion, ['Win', 'Draw', 'Lose'])
                                                        # cm_image = plot_to_image(fig)
                                                        #
                                                        #
                                                        # file_writer_cm = tf.summary.create_file_writer(logdir + '/cm')
                                                        # with file_writer_cm.as_default():
                                                        #     tf.summary.image("Confusion Matrix", cm_image,step=0)
                                                        #
                                                        #
                                                        # results = []
                                                        # for i in predictions:
                                                        #     results.append(np.argmax(i))
                                                        #
                                                        # count = [results.count(0), results.count(1), results.count(2)]
                                                        # print(count)
                                                        # # plt.bar(['0', '1', '2'], count)
                                                        # # plt.show()
                                                        # #
                                                        # # plt.savefig(str(logdir) + '.png')
                                                        #
                                                        # with open(logdir + '/predictions.csv', 'w+') as f:
                                                        #     # using csv.writer method from CSV package
                                                        #     write = csv.writer(f)
                                                        #     write.writerow(count)
                                                        pbar.update(1)
                                                        it_end_time = datetime.datetime.now()

                                                        time_elapsed = (it_end_time - it_start_time)

                                                        print(str(time_elapsed) + ' elapsed')

                                                        session_num += 1

    end_time = datetime.datetime.now()

    time_elapsed = (end_time - start_time)

    print('Program took ' + str(time_elapsed) + ' to finish')
