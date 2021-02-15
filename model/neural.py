import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np

from data_preparation.dataloader import load_training_data

import tensorflow as tf
import tensorflow_cloud as tfc
import model.config as cf
from tensorboard.plugins.hparams import api as hp

# tfc.run()

# gpus = tf.config.experimental.list_physical_devices('GPU')
#
# try:
#     for gpu in gpus:
#         tf.config.experimental.set_memory_growth(gpu, True)
# except RuntimeError as e:
#     print(e)

# tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(1)


def prepare_data():
    training_data = load_training_data('../data/whoscored/alltrainingdata.pickle')  # Get premier league data
    # random.shuffle(training_data)
    x = []  # features set
    y = []  # label set
    for features, label in training_data:
        x.append(features)
        y.append(label)
    X = np.array(x)

    # Normalise the values

    # X = (X-X.min(axis=0))/ (X.max(axis=0)-X.min(axis=0))

    # print(X[0])

    y = np.array(y)
    # X = tf.keras.utils.normalize(X, axis=1)
    return X, y


x_train, y_train = prepare_data()


# x_test, y_test = prepare_data()


def train_test_model(logdir, hparams):

    if hparams[cf.HP_OPTIMISER] == "adam":
        optimiser = tf.keras.optimizers.Adam(learning_rate=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "sgd":
        optimiser = tf.keras.optimizers.SGD(lr=hparams[cf.HP_LR], momentum=cf.MOMENTUM)
    else:
        raise ValueError("unexpected optimiser name: %r" % (hparams[cf.HP_OPTIMISER],))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=tf.nn.relu, input_dim=cf.INPUT_DIM))
    model.add(tf.keras.layers.Dropout(hparams[cf.HP_DROPOUT]))
    model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

    model.compile(optimizer=optimiser, loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(logdir,histogram_freq=1),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
    ]

    # model.save('saved_models/')
    model.fit(x_train, y_train, epochs=250, batch_size=hparams[cf.HP_BATCH_SIZE], shuffle=True, verbose=1,
              callbacks=callbacks, validation_split=0.1)
    _, accuracy = model.evaluate(x_train, y_train)

    return accuracy


session_num = 0
for optimiser in cf.HP_OPTIMISER.domain.values:
    for lr in cf.HP_LR.domain.values:
        for batch_size in cf.HP_BATCH_SIZE.domain.values:
            for num_units in cf.HP_NUM_UNITS.domain.values:
                for dropout_rate in cf.HP_DROPOUT.domain.values:
                    hparams = {
                        cf.HP_NUM_UNITS: num_units,
                        cf.HP_DROPOUT: dropout_rate,
                        cf.HP_LR: lr,
                        cf.HP_BATCH_SIZE: batch_size,
                        cf.HP_OPTIMISER: optimiser
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    train_test_model('logs/hparam_tuning/ratio-norm/elo/dropout/single-neuron/' + run_name, hparams)
                    session_num += 1

# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)
#
# random_game = get_random_game("../data/whoscored/premierleague-20182019.csv")
# print(random_game)
# test_game = np.array(random_game[4])
# print(test_game)
# predictions = model.predict(np.array([test_game]))
# print(predictions)
