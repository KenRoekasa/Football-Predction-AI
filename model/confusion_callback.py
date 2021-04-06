import sklearn
import tensorflow as tf
from tensorflow.python.keras.callbacks import Callback
import numpy as np

from graphing.confusion_matrix import plot_confusion_matrix, plot_to_image


class ConfusionCallbacck(Callback):
    """ NewCallback descends from Callback
    """
    def __init__(self, models,x_valid,y_valid,file_writer_cm):
        """ Save params in constructor
        """

        self.models = models
        self.y_valid = y_valid
        self.x_valid = x_valid
        self.file_writer_cm = file_writer_cm


    def on_epoch_end(self, epoch, logs={}):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.models[0].predict(self.x_valid)
        test_pred = np.argmax(test_pred_raw, axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(self.y_valid, test_pred)
        # Log the confusion matrix as an image summary.
        figure = plot_confusion_matrix(cm, class_names=['Win', 'Draw', 'Lose'])
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)