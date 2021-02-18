import datetime
import os



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from data_preparation.dataloader import load_training_data
from tqdm import tqdm
import tensorflow as tf
import model.config as cf
from tensorboard.plugins.hparams import api as hp
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
EPOCHS = 300

gpus = tf.config.experimental.list_physical_devices('GPU')

try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

# tf.get_logger().setLevel('ERROR')
# tf.autograph.set_verbosity(1)

x_train, y_train = load_training_data('../data/whoscored/alltrainingdata-pi-rating-only-all-ratio-norm.pickle')


# x_test, y_test = prepare_data()


def train_test_model(logdir, hparams):
    if hparams[cf.HP_OPTIMISER] == "adam":
        optimiser = tf.keras.optimizers.Adam(learning_rate=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "sgd":
        optimiser = tf.keras.optimizers.SGD(lr=hparams[cf.HP_LR],momentum=hparams[cf.HP_MOMENTUM])
    elif hparams[cf.HP_OPTIMISER] == "RMSprop":
        optimiser = tf.keras.optimizers.RMSprop(lr=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "Adagrad":
        optimiser = tf.keras.optimizers.Adagrad(lr=hparams[cf.HP_LR])
    elif hparams[cf.HP_OPTIMISER] == "Adadelta":
        optimiser = tf.keras.optimizers.Adadelta(lr=hparams[cf.HP_LR])
    else:
        raise ValueError("unexpected optimiser name: %r" % (hparams[cf.HP_OPTIMISER],))

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))
    model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))
    model.add(tf.keras.layers.Dense(hparams[cf.HP_NUM_UNITS], activation=hparams[cf.HP_ACTIVATION]))

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

    model.compile(optimizer=optimiser, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    callbacks = [
        tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
        # ReduceLROnPlateau(monitor='val_loss', factor=0.001, patience=10)
        # tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=50, restore_best_weights=True)
        # early stopping
    ]

    # model.save('saved_models/')
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=hparams[cf.HP_BATCH_SIZE], shuffle=True, verbose=1,
              callbacks=callbacks, validation_split=0.05)
    _, accuracy = model.evaluate(x_train, y_train)

    return accuracy


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    start_time_f = start_time.strftime("%H:%M:%S")
    print(start_time_f)
    session_num = 0

    for optimiser in cf.HP_OPTIMISER.domain.values:
        for lr in cf.HP_LR.domain.values:
            for batch_size in cf.HP_BATCH_SIZE.domain.values:
                for activation in cf.HP_ACTIVATION.domain.values:
                    for num_units in cf.HP_NUM_UNITS.domain.values:
                        for momentum in cf.HP_MOMENTUM.domain.values:

                            it_start_time = datetime.datetime.now()

                            hparams = {
                                cf.HP_NUM_UNITS: num_units,
                                cf.HP_LR: lr,
                                cf.HP_BATCH_SIZE: batch_size,
                                cf.HP_OPTIMISER: optimiser,
                                cf.HP_ACTIVATION: activation,
                                cf.HP_MOMENTUM: momentum
                            }
                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            train_test_model(
                                'logs/hparam_tuning/min-max-all/pi-rating-only/3-hidden/' + 'epoch' + str(
                                    EPOCHS) + str(
                                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), hparams)

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
# random_game = get_random_game("../data/whoscored/premierleague-20182019.csv")
# print(random_game)
# test_game = np.array(random_game[4])
# print(test_game)
# predictions = model.predict(np.array([test_game]))
# print(predictions)
