
import numpy as np

from data_preparation.dataloader import load_training_data
import tensorflow as tf
import config
from tensorboard.plugins.hparams import api as hp

gpus = tf.config.experimental.list_physical_devices('GPU')

try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(config.NAME))

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([24, 90]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.5, 0.8))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )



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

model = tf.keras.models.Sequential()




for layer in config.NETWORK_TOPOLOGY:
    model.add(tf.keras.layers.Dense(layer, activation=tf.nn.relu, input_dim=config.INPUT_DIM))

model.add(tf.keras.layers.Dropout(config.DROPOUT_VALUE))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))


# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(optimizer=config.OPT, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=config.BATCH_SIZE, shuffle=True, callbacks=[tensorboard], validation_split=0.1)





# val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)
#
# random_game = get_random_game("../data/whoscored/premierleague-20182019.csv")
# print(random_game)
# test_game = np.array(random_game[4])
# print(test_game)
# predictions = model.predict(np.array([test_game]))
# print(predictions)
