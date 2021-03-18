import csv
import datetime
import os

import numpy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from data_preparation.dataloader import load_training_data, get_random_game, normalise_mean_array
from tqdm import tqdm
import tensorflow as tf
import model.config as cf
from tensorboard.plugins.hparams import api as hp
import matplotlib.pyplot as plt
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def train_test_model(logdir, hparams):

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

    model = Sequential()

    model.add(Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))
    model.add(Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))
    model.add(Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))

    # # model.add(tf.keras.layers.Dropout(hparams[cf.HP_DROPOUT]))
    # model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))
    # # model.add(tf.keras.layers.Dropout(hparams[cf.HP_DROPOUT]))
    # model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))

    # model.add(tf.keras.layers.Dropout(hparams[cf.HP_DROPOUT]))

    #                                 kernel_regularizer=regularizers.l1(hparams[cf.HP_REGULARISER_RATE])))
    # model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=tf.nn.sigmoid,
    #                                 kernel_regularizer=regularizers.l1(hparams[cf.HP_REGULARISER_RATE])))
    # model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=tf.nn.sigmoid,
    #                                 kernel_regularizer=regularizers.l1(hparams[cf.HP_REGULARISER_RATE])))
    # model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=tf.nn.sigmoid,
    #                                 kernel_regularizer=regularizers.l1(hparams[cf.HP_REGULARISER_RATE])))
    # model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=tf.nn.sigmoid,
    #                                 kernel_regularizer=regularizers.l1(hparams[cf.HP_REGULARISER_RATE])))
    # model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=tf.nn.sigmoid,
    #                                 kernel_regularizer=regularizers.l1(hparams[cf.HP_REGULARISER_RATE])))
    # model.add(tf.keras.layers.Dropout(hparams[cf.HP_DROPOUT]))

    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

    model.compile(optimizer=optimiser, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
        # ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=10)
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=50, restore_best_weights=True)
        # early stopping
    ]

    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=hparams[cf.HP_BATCH_SIZE], shuffle=True, verbose=0,
              callbacks=callbacks, validation_data=(x_valid, y_valid))

    _, accuracy = model.evaluate(x_train, y_train)
    tf.keras.models.save_model(model,
                               logdir + '/savedmodel' + str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    return model


if __name__ == '__main__':

    EPOCHS = 250
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


    for numb in cf.HP_previous_games.domain.values:
        for combination in cf.HP_combine.domain.values:
            for norm in cf.HP_norm.domain.values:
                if combination == 'diff':
                    norm == 'min-max'  # if combination type is diff only use min max normalastion

                training_data_path = '../data/whoscored/trainingdata/unnormalised/'
                training_data = 'alltrainingdata-%d-both+-min-max-%s.pickle' % (numb, combination)

                x_train, y_train = load_training_data(training_data_path + training_data)

                # f = re.search('(append|diff)', training_data)
                #
                # combination_type = f.group(0)

                # # if append
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

                x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.30, shuffle=True)

                print(y_train.tolist().count(0))
                print(y_train.tolist().count(1))
                print(y_train.tolist().count(2))

                start_time = datetime.datetime.now()
                start_time_f = start_time.strftime("%H:%M:%S")
                print(start_time_f)
                session_num = 0

                for i in range(0, 3):  # repeats
                    for optimiser in cf.HP_OPTIMISER.domain.values:
                        for lr in cf.HP_LR.domain.values:
                            for batch_size in cf.HP_BATCH_SIZE.domain.values:
                                for activation in cf.HP_ACTIVATION.domain.values:
                                    for num_units in cf.HP_NUM_UNITS.domain.values:
                                        for momentum in cf.HP_MOMENTUM.domain.values:
                                            for rr in cf.HP_REGULARISER_RATE.domain.values:
                                                for dropout in cf.HP_DROPOUT.domain.values:
                                                    it_start_time = datetime.datetime.now()
                                                    hparams = {
                                                        cf.HP_NUM_UNITS: num_units,
                                                        cf.HP_LR: lr,
                                                        cf.HP_BATCH_SIZE: batch_size,
                                                        cf.HP_OPTIMISER: optimiser,
                                                        cf.HP_ACTIVATION: activation,
                                                        cf.HP_MOMENTUM: momentum,
                                                        cf.HP_REGULARISER_RATE: rr,
                                                        cf.HP_previous_games: numb,
                                                        cf.HP_combine: combination,
                                                        cf.HP_norm: norm,
                                                    }
                                                    run_name = "run-%d" % session_num
                                                    print('--- Starting trial: %s' % run_name)
                                                    print({h.name: hparams[h] for h in hparams})
                                                    today = datetime.date.today()

                                                    logdir = 'logs/allratings/' + str(
                                                        today) + '/' + norm + '/' + 'epoch' + str(
                                                        EPOCHS) + str(
                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")) + str(
                                                        training_data_text)

                                                    model = train_test_model(
                                                        logdir, hparams)

                                                    predictions = model.predict(x_valid)
                                                    results = []
                                                    for i in predictions:
                                                        results.append(np.argmax(i))

                                                    count = [results.count(0), results.count(1), results.count(2)]

                                                    # plt.bar(['0', '1', '2'], count)
                                                    # plt.show()
                                                    #
                                                    # plt.savefig(str(logdir) + '.png')

                                                    with open(logdir + '/predictions.csv', 'w+') as f:
                                                        # using csv.writer method from CSV package
                                                        write = csv.writer(f)
                                                        write.writerow(count)

                                                    it_end_time = datetime.datetime.now()

                                                    time_elapsed = (it_end_time - it_start_time)

                                                    print(str(time_elapsed) + ' elapsed')

                                                    session_num += 1

                end_time = datetime.datetime.now()

                time_elapsed = (end_time - start_time)

                print('Program took ' + str(time_elapsed) + ' to finish')

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)
#
