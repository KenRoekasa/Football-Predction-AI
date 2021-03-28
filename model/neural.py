import os

from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import class_weight

from model import dataset
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.keras.layers import Dropout

import csv
import sys

sys.path.append('..')

import datetime
from tensorflow.keras.layers import Dense

from data_preparation.dataloader import load_training_data, get_random_game, normalise_mean_array
from tqdm import tqdm
import tensorflow as tf
import model.config as cf
from tensorboard.plugins.hparams import api as hp

import numpy
from imblearn.over_sampling import RandomOverSampler
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
        optimiser = tf.keras.optimizers.Adam(learning_rate=hparams[cf.HP_LR], beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                             amsgrad=False, )
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
        Dense(hparams[cf.HP_NUM_UNITS2], activation=hparams[cf.HP_ACTIVATION]),
        Dense(hparams[cf.HP_NUM_UNITS3], activation=hparams[cf.HP_ACTIVATION]),
        Dense(hparams[cf.HP_NUM_UNITS4], activation=hparams[cf.HP_ACTIVATION]),
        Dense(3, activation=softmax)
    ])

    model.compile(optimizer=optimiser, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
        # ReduceLROnPlateau(monitor='val_loss', factor=0.0001, patience=30)
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=50, restore_best_weights=True)
        # early stopping
    ]

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=hparams[cf.HP_BATCH_SIZE], shuffle=True, verbose=1,
              callbacks=callbacks, validation_data=(x_valid, y_valid))

    _, accuracy = model.evaluate(x_train, y_train)
    tf.keras.models.save_model(model,
                               logdir + '/savedmodel' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    return model


if __name__ == '__main__':
    EPOCHS = 160
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    numb = 10
    combination = 'append'
    norm = 'ratio'

    if combination == 'diff':
        norm == 'min-max'  # if combination type is diff only use min max normalisation

    training_data_path = '../data/whoscored/trainingdata/sum/'
    training_data = 'alltrainingdata-%d-both andrew-min-max-%s.pickle' % (numb, combination)

    x_train, y_train = load_training_data(training_data_path + training_data)

    f = re.search('(append|diff)', training_data)

    combination_type = f.group(0)

    # if append
    if combination == 'append':
        size = len(x_train[0])

        number_rating = 2  # this is base on how many ratings there are
        half = int((size / 2))

        x_train[:, 0:half - number_rating], x_train[:, half:-number_rating] = normalise_mean_array(
            x_train[:, 0:half - number_rating],
            x_train[:,
            half:-number_rating],
            norm)  # change normalisation type

    # if min max and diff
    if combination == 'diff':
        x_train = normalize(x_train, axis=0, norm='max')

    p = re.search('^.*(?=(\.pickle))', training_data)
    training_data_text = p.group(0)  # get the text so i can output using the same name

    data = dataset.Dataset('../data/book.csv')

    # train_results = data.processed_results
    #
    # train_df = pd.DataFrame.from_dict(train_results)
    #
    # train_np = train_df.values.tolist()
    #
    # # split into train and test
    #
    # x = []  # features set
    # y = []  # label set
    #
    # for i in train_np:
    #     # balance data
    #     x.append(i[4:])
    #     if i[0] == 'H':
    #         y.append(0)
    #     elif i[0] == 'D':
    #         y.append(1)
    #     elif i[0] == 'A':
    #         y.append(2)
    #
    # x_train = np.array(x)
    # y_train = np.array(y)
    #
    # x_train = normalize(x_train, axis=1, norm='l1')
    #
    #
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.30, shuffle=True)

    # oversample
    # oversample = RandomOverSampler(sampling_strategy='minority')
    # x_train, y_train = oversample.fit_resample(x_train, y_train)

    # # undersample
    # undersample = RandomUnderSampler(sampling_strategy='majority')
    # x_train, y_train = undersample.fit_resample(x_train, y_train)

    # weights = class_weight.compute_class_weight('balanced',
    #                                             np.unique(y_train),
    #                                             y_train)
    #
    # class_weight = {0: weights[0], 1: weights[1], 2: weights[0]}

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

    number_of_parameters = 5
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
                                                            # cf.HP_DROPOUT:dropout,
                                                        }
                                                        run_name = "run-%d" % session_num
                                                        print('--- Starting trial: %s' % run_name)
                                                        print({h.name: hparams[h] for h in hparams})
                                                        today = datetime.date.today()

                                                        logdir = 'logs/andrew/' + str(
                                                            today) + '/epoch' + str(
                                                            EPOCHS) + str(
                                                            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + str(
                                                            training_data_text)

                                                        model = train_test_model(
                                                            logdir, hparams, x_train, y_train, x_valid, y_valid)

                                                        predictions = model.predict(x_valid)
                                                        results = []
                                                        for i in predictions:
                                                            results.append(np.argmax(i))

                                                        count = [results.count(0), results.count(1), results.count(2)]
                                                        print(count)
                                                        # plt.bar(['0', '1', '2'], count)
                                                        # plt.show()
                                                        #
                                                        # plt.savefig(str(logdir) + '.png')

                                                        with open(logdir + '/predictions.csv', 'w+') as f:
                                                            # using csv.writer method from CSV package
                                                            write = csv.writer(f)
                                                            write.writerow(count)
                                                        pbar.update(1)
                                                        it_end_time = datetime.datetime.now()

                                                        time_elapsed = (it_end_time - it_start_time)

                                                        print(str(time_elapsed) + ' elapsed')

                                                        session_num += 1

    end_time = datetime.datetime.now()

    time_elapsed = (end_time - start_time)

    print('Program took ' + str(time_elapsed) + ' to finish')
